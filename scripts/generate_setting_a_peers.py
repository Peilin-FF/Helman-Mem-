from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm.auto import tqdm

from feedback_state.adversarial import make_numeric_perturbation_response
from feedback_state.datasets import REGISTRY as MATH_REGISTRY
from feedback_state.datasets import load_math_dataset
from feedback_state.eval_datasets import REGISTRY as EVAL_REGISTRY
from feedback_state.eval_datasets import load_eval_dataset, task_type_for
from feedback_state.generation import GenerationConfig, TextGenerator, render_instruction_prompt
from feedback_state.tasks import build_peer_prompt as task_peer_prompt
from feedback_state.utils import (
    append_jsonl,
    completed_ids,
    load_config,
    merge_args_with_config,
)


def load_any_dataset(dataset_key: str, *, split, start_index, max_samples, cfg) -> list[dict]:
    """Load math, RAG, or code records, all tagged with task_type.

    Math goes through the math registry (keeps adversarial-math behaviour); RAG and
    code go through eval_datasets. A dataset name not in either registry falls back
    to the math loader (custom HF path).
    """
    key = dataset_key.lower().replace("-", "_")
    if key in {k.replace("-", "_") for k in EVAL_REGISTRY}:
        records = load_eval_dataset(
            dataset_key, split=split, start_index=int(start_index or 0),
            max_samples=int(max_samples) if max_samples is not None else None,
            cache_dir=cfg.get("hf_cache_dir"),
        )
        return records
    records = load_math_dataset(
        dataset_key,
        split=split,
        config=cfg.get("dataset_config"),
        start_index=int(start_index or 0),
        max_samples=int(max_samples) if max_samples is not None else None,
        hf_path_override=cfg.get("hf_path_override"),
        field_overrides=cfg.get("field_overrides"),
        cache_dir=cfg.get("hf_cache_dir"),
    )
    for record in records:
        record.setdefault("task_type", "math")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline peer responses for Setting A.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Registered dataset name (deepmath, math500, aime2024, amc, "
             "olympiadbench, minerva, college_math). Defaults to config or 'deepmath'.",
    )
    return parser.parse_args()


def build_peer_prompt(problem: str) -> str:
    return (
        "Solve this math problem. Give a concise solution and end with the final answer.\n\n"
        f"Problem:\n{problem}"
    )


def build_adversarial_prompt(problem: str, answer: str) -> str:
    return (
        "Produce a plausible but incorrect solution to this math problem. "
        f"The final answer must not equal {answer}.\n\nProblem:\n{problem}"
    )


def main() -> None:
    args = parse_args()
    cfg = merge_args_with_config(args, load_config(args.config))
    split = str(cfg.get("split", "train"))
    output = Path(cfg.get("output", f"data/deepmath_setting_a_{split}.jsonl"))
    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)

    dataset_key = str(cfg.get("dataset", cfg.get("dataset_name", "deepmath")))
    if "/" in dataset_key:
        # Backwards-compatible: a full HF path string still works for DeepMath etc.
        dataset_key = {"zwhe99/DeepMath-103K": "deepmath"}.get(dataset_key, dataset_key)
    max_samples = cfg.get("max_samples", cfg.get(f"max_{split}_samples", None))
    start_index = int(cfg.get("start_index", 0))
    records = load_any_dataset(
        dataset_key, split=split, start_index=start_index, max_samples=max_samples, cfg=cfg
    )
    shard_index = int(cfg.get("shard_index", 0))
    num_shards = int(cfg.get("num_shards", 1))

    peer_models = list(cfg.get("peer_models", []))
    if not peer_models:
        raise ValueError("Config must define peer_models")
    adversarial_rates = cfg.get("adversarial_rate", 0.0)
    if not isinstance(adversarial_rates, list):
        adversarial_rates = [float(adversarial_rates)] * len(peer_models)
    adversarial_mode = str(cfg.get("adversarial_mode", "numeric_perturb"))
    # Peers that receive NO retrieved context for RAG records (e.g. Gemma is kept
    # bad at RAG-QA by withholding documents, while staying good at math). This is
    # the mechanism that creates per-task reliability differences for the trust
    # state to discover. Compared by substring so "google/gemma-3-4b-it" matches.
    deprive_context_models = [str(m).lower() for m in cfg.get("deprive_context_models", [])]

    def peer_gets_context(model_name: str) -> bool:
        name = str(model_name).lower()
        return not any(token in name for token in deprive_context_models)

    done = completed_ids(output)
    gen_cfg = GenerationConfig(
        max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        temperature=float(cfg.get("temperature", 0.2)),
        top_p=float(cfg.get("top_p", 0.95)),
        dtype=str(cfg.get("dtype", "bfloat16")),
        device=str(cfg.get("device", "cuda:0")),
        use_vllm=bool(cfg.get("use_vllm", True)),
        local_files_only=bool(cfg.get("local_files_only", False)),
    )

    pending: list[dict] = []
    for index, base in enumerate(records):
        if index % num_shards != shard_index:
            continue
        if base["id"] in done:
            continue
        # Keep every standardised field (context/test/entry_point/aliases/...) so
        # downstream scoring (RAG metrics, offline code exec) has what it needs.
        record = dict(base)
        record.setdefault("source", dataset_key)
        record.setdefault("task_type", task_type_for(dataset_key))
        record["peer_responses"] = {}
        record["peer_metadata"] = {}
        pending.append(record)
    if not pending:
        return

    # Process records in chunks so progress is flushed to disk regularly (resumable).
    chunk_size = int(cfg.get("write_chunk_size", 64))
    batch_size = int(cfg.get("generation_batch_size", 4))
    for chunk_start in range(0, len(pending), chunk_size):
        chunk = pending[chunk_start : chunk_start + chunk_size]
        for peer_index, model_name in enumerate(peer_models):
            generator = TextGenerator(str(model_name), gen_cfg)
            try:
                prompts: list[str] = []
                adversarial_flags: list[bool] = []
                for record in chunk:
                    task_type = str(record.get("task_type", "math"))
                    # Adversarial corruption is a math-only control; skip for rag/code.
                    rate = float(adversarial_rates[min(peer_index, len(adversarial_rates) - 1)])
                    is_adv = task_type == "math" and rng.random() < rate
                    adversarial_flags.append(is_adv)
                    if is_adv and adversarial_mode == "numeric_perturb":
                        prompts.append("")
                    elif is_adv:
                        prompts.append(render_instruction_prompt(generator.tokenizer, build_adversarial_prompt(record["problem"], record["answer"])))
                    else:
                        # Task-aware peer prompt; RAG context withheld from deprived peers.
                        with_context = peer_gets_context(model_name)
                        prompt_text = task_peer_prompt(record, with_context=with_context)
                        prompts.append(render_instruction_prompt(generator.tokenizer, prompt_text))
                # samples_per_peer > 1 builds a per-peer SAMPLE POOL (peer_samples),
                # enabling counterfactual same-question pairing in the counter-trust
                # scenario. Needs temperature > 0 to get distinct samples. Default 1
                # reproduces the original single-response behaviour exactly.
                samples_per_peer = max(1, int(cfg.get("samples_per_peer", 1)))
                per_row_samples: list[list[str]] = [[] for _ in chunk]
                active_indices = [i for i, prompt in enumerate(prompts) if prompt]
                for sample_idx in range(samples_per_peer):
                    generated = [""] * len(prompts)
                    for start in tqdm(
                        range(0, len(active_indices), batch_size),
                        desc=f"{model_name} chunk {chunk_start // chunk_size} s{sample_idx}",
                    ):
                        chunk_indices = active_indices[start : start + batch_size]
                        chunk_prompts = [prompts[i] for i in chunk_indices]
                        chunk_outputs = generator.generate(chunk_prompts)
                        for idx, text in zip(chunk_indices, chunk_outputs):
                            generated[idx] = text
                    for row_index, record in enumerate(chunk):
                        is_adv = adversarial_flags[row_index]
                        text = generated[row_index]
                        if is_adv and not text:
                            text = make_numeric_perturbation_response(record["answer"], rng) or (
                                "A plausible but incorrect solution gives a different final answer."
                            )
                        per_row_samples[row_index].append(text)
                for row_index, record in enumerate(chunk):
                    is_adv = adversarial_flags[row_index]
                    samples = per_row_samples[row_index]
                    key = f"peer_{peer_index}"
                    record["peer_responses"][key] = samples[0]
                    if samples_per_peer > 1:
                        record.setdefault("peer_samples", {})[key] = samples
                    record["peer_metadata"][key] = {
                        "model": str(model_name),
                        "is_adversarial": bool(is_adv),
                        "known_incorrect": bool(is_adv),
                        "received_context": bool(
                            str(record.get("task_type", "math")) != "rag" or peer_gets_context(model_name)
                        ),
                        "num_samples": samples_per_peer,
                        "generation_params": {
                            "max_new_tokens": gen_cfg.max_new_tokens,
                            "temperature": gen_cfg.temperature,
                            "top_p": gen_cfg.top_p,
                            "backend": generator.backend,
                        },
                    }
            finally:
                generator.close()
        append_jsonl(output, chunk)


if __name__ == "__main__":
    main()
