"""Post-answer, center-only verification using the repository's parallel RL contract.

Never use hint selection, weighted votes, peer correctness or teacher targets.
Verifier/data failures stop the batch; they are not fabricated incorrect answers.
The legacy SigmaRewardManager remains available for OLD experiment configs.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import torch

from feedback_state.memory_generator import grade
from feedback_state.tasks import code_extract_answer, get_task, task_type_of
from training.sigma_rl.outcome_protocol import PROTOCOL
from training.sigma_rl.reward import SigmaRewardManager, stdin_to_devnull

REFERENCE_KEYS = frozenset({
    "task_type", "problem", "question", "answer", "answer_aliases", "aliases",
    "choices", "choice_labels", "code_format", "test_cases", "test", "test_setup", "entry_point",
})


def verifier_record(record: Mapping) -> dict:
    """A private verifier payload with NO peer responses or peer annotations."""
    rec = {key: record[key] for key in REFERENCE_KEYS if key in record}
    task = task_type_of(rec)
    try:
        get_task(task)
    except KeyError as exc:
        raise ValueError(f"no verifier registered for task {task!r}") from exc
    if task != "code":
        if rec.get("answer") is None or str(rec["answer"]).strip() == "":
            raise ValueError("center outcome verification requires a final reference answer")
        return rec
    fmt = str(rec.get("code_format", "asserts"))
    if fmt not in ("io", "functional", "asserts", "unittest"):
        raise ValueError(f"unsupported code verifier format: {fmt}")
    if fmt in ("io", "functional"):
        cases = rec.get("test_cases")
        if not isinstance(cases, (list, tuple)) or not cases:
            raise ValueError("code verification requires nonempty test cases")
        if any(not isinstance(case, Mapping) or "input" not in case or "output" not in case for case in cases):
            raise ValueError("each code test must contain input and expected output")
        if fmt == "functional" and not rec.get("entry_point"):
            raise ValueError("functional verification requires entry_point")
    elif not str(rec.get("test") or "").strip():
        raise ValueError("code verification requires nonempty tests")
    return rec


def center_outcome(record: Mapping, response: str, *, code_timeout: float = 10.0) -> float:
    rec = verifier_record(record)
    if task_type_of(rec) != "code":
        return float(grade(rec, response, code_timeout=code_timeout))
    from data.builders.common.code_grading import score_code_record

    program = code_extract_answer(response)
    if not program.strip():
        return 0.0
    result = score_code_record(rec, program, timeout=code_timeout)
    if result.error.startswith("runner-error:"):
        raise RuntimeError(result.error)
    return float(result.passed)


def compute_center_score(data_source: str, solution_str: str, ground_truth, extra_info=None, **kwargs) -> dict:
    """verl's custom reward signature; no unverified/pseudo-reward branch."""
    info = dict(extra_info or {})
    if info.get("protocol") != PROTOCOL:
        raise ValueError("new outcome reward requires explicit peer_outcome_v1 data, not legacy guided rows")
    if info.get("verified", True) is not True:
        raise ValueError("no post-answer verifier: refusing a peer-vote fallback")
    record = info.get("record")
    if isinstance(record, str):
        record = json.loads(record)
    if record is None:
        record = {"task_type": data_source, "answer": ground_truth}
    if task_type_of(record) != str(data_source).lower():
        raise ValueError("data_source and private verifier task_type disagree")
    reward = center_outcome(record, solution_str, code_timeout=float(kwargs.get("code_timeout", 10.0)))
    return {"score": reward, "acc": reward, "verified": 1.0, "pseudo": 0.0}


def _score_in_code_worker(job: dict) -> dict:
    stdin_to_devnull()
    return compute_center_score(**job)


class OutcomeRewardManager(SigmaRewardManager):
    """Same DataProto interface as SigmaRewardManager, with strict outcome semantics.

    Non-code verifiers run in a thread pool; code uses isolated Ray tasks like
    the repository's existing parallel trainer. Those tasks invoke the EXISTING
    code harness, which is resource-limited but is not a security sandbox.
    """
    def __init__(self, tokenizer, *, max_workers: int = 32, code_timeout: float = 10.0):
        super().__init__(tokenizer, compute_score=compute_center_score, max_workers=max_workers, code_timeout=code_timeout)

    def __call__(self, data, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            raise ValueError("precomputed reward-model scores cannot replace post-answer center verification")
        responses = data.batch["responses"]
        prompt_length = data.batch["prompts"].shape[-1]
        mask = data.batch["attention_mask"][:, prompt_length:]
        if mask.shape != responses.shape or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("invalid center response attention mask")
        if mask.shape[-1] > 1 and (mask[:, 1:] > mask[:, :-1]).any():
            raise ValueError("center responses must be right-padded")
        lengths = mask.sum(-1).long().tolist()
        if any(length < 1 for length in lengths):
            raise ValueError("missing generated response tokens; no outcome can be assigned")
        jobs = []
        for i, length in enumerate(lengths):
            info = dict(data.non_tensor_batch["extra_info"][i])
            # Keep the controller's other diagnostics out of the verifier too.
            rec = info.get("record")
            if isinstance(rec, str):
                rec = json.loads(rec)
            if rec is not None:
                rec = verifier_record(rec)
                if task_type_of(rec) != str(data.non_tensor_batch["data_source"][i]).lower():
                    raise ValueError("data_source and private verifier task_type disagree")
            jobs.append({
                "data_source": str(data.non_tensor_batch["data_source"][i]).lower(),
                "solution_str": self.tokenizer.decode(responses[i, :length], skip_special_tokens=True),
                "ground_truth": data.non_tensor_batch["reward_model"][i]["ground_truth"],
                "extra_info": {"protocol": info.get("protocol"), "verified": info.get("verified", True), "record": rec},
                "code_timeout": self.code_timeout,
            })
        import ray

        code_indices = [i for i, job in enumerate(jobs) if job["data_source"] == "code"]
        if code_indices and not ray.is_initialized():
            raise RuntimeError("code outcome grading requires the initialized parallel Ray runtime")
        code_refs = {}
        if code_indices:
            worker = ray.remote(num_cpus=1)(_score_in_code_worker)
            code_refs = {i: worker.remote(jobs[i]) for i in code_indices}
        scores = {}
        local_indices = [i for i in range(len(jobs)) if i not in code_refs]
        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as pool:
            for i, score in zip(local_indices, pool.map(lambda i: compute_center_score(**jobs[i]), local_indices)):
                scores[i] = score
        if code_refs:
            scores.update(zip(code_refs, ray.get(list(code_refs.values()))))
        rewards = torch.zeros_like(responses, dtype=torch.float32)
        extras = {key: [] for key in ("score", "acc", "verified", "pseudo")}
        for i, length in enumerate(lengths):
            rewards[i, length - 1] = scores[i]["score"]
            for key in extras:
                extras[key].append(scores[i][key])
        if return_dict:
            return {"reward_tensor": rewards, "reward_extra_info": extras}
        return rewards
