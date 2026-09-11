"""Rewards for the multi-GPU trainer.

compute_score        the task verifier (math_equal / QA match / sandboxed code tests) when the prompt is
                     verified, the memory's reliability-weighted peer vote when it is not
SigmaRewardManager   verl reward manager that grades a batch in parallel and returns per-sample
                     extras (acc, verified, pseudo) for logging and validation metrics
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

from feedback_state.verdict import parse_verdict, record_is_flat, strip_verdict, verdict_agreement
from concurrent.futures import ThreadPoolExecutor

import torch

from verl import DataProto

from feedback_state.memory_generator import grade, strip_thinking
from feedback_state.memory_rl import memory_pseudo_reward, peer_texts_in_prompt_order


def stdin_to_devnull() -> None:
    """Point fd 0 at /dev/null so sandboxed programs that call input() hit EOF instead of blocking
    on the never-closing pipe a Ray worker inherits."""
    try:
        fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(fd, 0)
        os.close(fd)
        sys.stdin = open(os.devnull)
    except Exception:
        pass


def _record(info: dict, data_source: str, ground_truth) -> dict:
    rec = info.get("record")
    if isinstance(rec, str):
        return json.loads(rec)
    if isinstance(rec, dict):
        return rec
    return {"task_type": data_source, "answer": ground_truth}


def verifier_acc(record: dict, text: str, code_timeout: float = 10.0) -> float:
    try:
        return float(bool(grade(record, text, code_timeout=code_timeout)))
    except Exception:
        return 0.0


def _grade_code_in_worker(record_json: str, text: str, code_timeout: float) -> float:
    """Runs inside a Ray task: a single-threaded worker process that forks one sandbox at a time.
    (Forking from the many-threaded trainer actor can leave children stuck before exec.)"""
    stdin_to_devnull()
    return verifier_acc(json.loads(record_json), text, code_timeout)


_grade_code_remote = None


def grade_code_remote(record_json: str, text: str, code_timeout: float):
    global _grade_code_remote
    import ray

    if _grade_code_remote is None:
        _grade_code_remote = ray.remote(num_cpus=1)(_grade_code_in_worker)
    return _grade_code_remote.remote(record_json, text, code_timeout)


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None, code_timeout: float = 10.0, acc: float | None = None, **_) -> dict:
    """verl custom-reward signature.  Returns score (the training reward) plus logging fields.
    ``acc`` may be precomputed (the reward manager grades code in Ray workers)."""
    info = dict(extra_info or {})
    rec = _record(info, data_source, ground_truth)
    if acc is None:
        acc = verifier_acc(rec, solution_str, code_timeout)
    if bool(info.get("verified", True)):
        return {"score": acc, "acc": acc, "verified": 1.0, "pseudo": 0.0}
    try:
        peers = peer_texts_in_prompt_order(rec, [int(p) for p in info["peer_order"]])
        pr = memory_pseudo_reward(rec, peers, [float(p) for p in info["memory_prob"]], strip_thinking(solution_str))
    except Exception:
        pr = None
    return {"score": float(pr) if pr is not None else 0.0, "acc": acc, "verified": 0.0, "pseudo": 1.0 if pr is not None else 0.0}


class SigmaRewardManager:
    """Same contract as verl's NaiveRewardManager, graded with a thread pool (code tests run in
    sandboxed subprocesses, so threads give real parallelism)."""

    def __init__(self, tokenizer, num_examine: int = 0, compute_score=None, reward_fn_key: str = "data_source", max_workers: int = 32, code_timeout: float = 10.0, verdict_lambda: float = 0.0, verdict_flat: float = 0.1) -> None:
        self.verdict_lambda, self.verdict_flat = float(verdict_lambda), float(verdict_flat)   # method A: reward the Trust line for agreeing with the record
        self.tokenizer = tokenizer
        self.code_timeout = float(code_timeout)
        self.num_examine = int(num_examine)
        self.compute_score = compute_score or globals()["compute_score"]
        self.reward_fn_key = reward_fn_key
        self.max_workers = int(max_workers)

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            return {"reward_tensor": data.batch["rm_scores"]} if return_dict else data.batch["rm_scores"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        prompt_length = data.batch["prompts"].shape[-1]
        jobs = []
        for i in range(len(data)):
            item = data[i]
            valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum())
            response_str = self.tokenizer.decode(item.batch["responses"][:valid_response_length], skip_special_tokens=True)
            info = item.non_tensor_batch.get("extra_info", None)
            # method A (verdict): the reply may open with a 'Trust: a > b > ...' line; it is scored against the record's
            # estimate and stripped before the answer is graded
            verdict_rank, verdict_probs = None, None
            if self.verdict_lambda > 0 and info is not None and info.get("prompt_source", "peers") != "solo":
                verdict_probs = [float(x) for x in (info.get("memory_prob") or [])]
                verdict_rank = parse_verdict(response_str, len(verdict_probs)) if verdict_probs else None
                response_str = strip_verdict(response_str)
            jobs.append((i, valid_response_length, response_str, str(item.non_tensor_batch[self.reward_fn_key]),
                         item.non_tensor_batch["reward_model"]["ground_truth"], info, verdict_rank, verdict_probs))

        # code: sandboxed execution in Ray task workers (separate single-threaded processes); other tasks: threads
        acc: dict[int, float] = {}
        try:
            import ray

            use_ray = ray.is_initialized()
        except Exception:
            use_ray = False
        if use_ray:
            refs = {}
            for i, _, text, source, _, info, _, _ in jobs:
                rec_json = (info or {}).get("record")
                if source == "code" and isinstance(rec_json, str):
                    refs[i] = grade_code_remote(rec_json, text, self.code_timeout)
            if refs:
                for i, val in zip(refs.keys(), ray.get(list(refs.values()))):
                    acc[i] = float(val)

        def run(job):
            i, _, text, source, gt, info, _, _ = job
            return self.compute_score(data_source=source, solution_str=text, ground_truth=gt, extra_info=info, acc=acc.get(i))

        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as pool:
            scores = list(pool.map(run, jobs))
        extra = defaultdict(list)
        printed: dict[str, int] = defaultdict(int)
        for (i, length, text, source, gt, _, vrank, vprobs), s in zip(jobs, scores):
            if isinstance(s, dict):
                reward = float(s["score"])
                for k, v in s.items():
                    extra[k].append(float(v))
            else:
                reward = float(s)
            if self.verdict_lambda > 0:
                active = bool(vprobs) and not record_is_flat(vprobs, self.verdict_flat)
                agree = verdict_agreement(vrank, vprobs) if active else 0.0
                reward += self.verdict_lambda * agree if active else 0.0
                extra["verdict_active"].append(float(active)); extra["verdict_parsed"].append(float(vrank is not None))
                extra["verdict_agreement"].append(float(agree))   # 0 on flat records, so the mean stays a number
            reward_tensor[i, max(length - 1, 0)] = reward
            if printed[source] < self.num_examine:
                printed[source] += 1
                print(f"[reward] source={source} ground_truth={str(gt)[:80]!r} score={s}\n[response] {text[:600]}")
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(extra)}
        return reward_tensor
