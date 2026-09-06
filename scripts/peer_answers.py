"""Answers of a candidate peer model on the training stream, generated and graded like the current peers.

Prompt: feedback_state.tasks.build_peer_prompt(record, with_context=True), rendered with the model's own chat template
(one user turn, thinking off) as data/builders/common/peer_generation.render_instruction_prompt does.
Sampling: temperature 0.2, top_p 0.95 (the peer-generation configs); max_new_tokens per task as in those configs
(math 512, rag 256, code 768); ``--reasoning`` raises the budget to 4096 for models that think before answering and
grades the text after the last </think>.
Grading: the rule of data/builders/common/precompute_peer_correct.py, i.e. the task's target function rounded:
token-F1 >= 0.5 for reading, exact equality for math, executed tests for code (FEEDBACK_CODE_EXEC_ALLOW=1).

  PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 python scripts/peer_answers.py --model /mnt/data/peilin/HF_MODEL/Mistral-7B-Instruct-v0.3 \
      --records data/mixed_train_big/train.jsonl --output outputs/peergen_new/Mistral-7B-Instruct-v0.3
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import grade, strip_thinking
from feedback_state.tasks import build_peer_prompt, get_task, task_type_of

MAX_TOKENS = {"math": 512, "rag": 256, "code": 768, "boolqa": 96, "mcqa": 96, "shortqa": 96}   # the peer-generation configs (OOD peers were generated with 96)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reasoning", action="store_true", help="4096-token budget, grade after the last </think>")
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--sources", default=None, help="comma-separated source datasets to keep (e.g. apps)")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--enforce_eager", action="store_true")
    return p.parse_args()


def _target(args):
    record, text = args
    return float(get_task(task_type_of(record)).target_fn(text, record))


def _worker(args, q):
    try:
        q.put(_target(args))
    except Exception:
        q.put(0.0)


def graded_value(record: dict, text: str, timeout: float = 4.0) -> float:
    """precompute_peer_correct's rule: target_fn, math in a subprocess with a hard timeout (sympy can hang)."""
    task = task_type_of(record)
    if task == "code":
        return 1.0 if grade(record, text) else 0.0
    if task != "math":
        try:
            return _target((record, text))
        except Exception:
            return 0.0
    q = mp.Queue()
    proc = mp.Process(target=_worker, args=((record, text), q))
    proc.start(); proc.join(timeout)
    if proc.is_alive():
        proc.terminate(); proc.join()
        return 0.0
    try:
        return q.get_nowait()
    except Exception:
        return 0.0


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args.output.mkdir(parents=True, exist_ok=True)
    records = [r for i, r in enumerate(JsonlDataset(args.records).records) if i % args.num_shards == args.shard_index]
    if args.sources:
        keep = set(args.sources.split(","))
        records = [r for r in records if r.get("source") in keep]
    if args.max_examples:
        records = records[: args.max_examples]
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=args.trust_remote_code)
    prompts, params = [], []
    for r in records:
        content = build_peer_prompt(r, with_context=True)
        try:
            text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except (TypeError, ValueError):
            text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        prompts.append(text)
        budget = 4096 if args.reasoning else MAX_TOKENS.get(task_type_of(r), 512)
        params.append(SamplingParams(temperature=0.2, top_p=0.95, max_tokens=budget, seed=0))
    print(f"[peer-answers] {args.model}: {len(records)} records (shard {args.shard_index}/{args.num_shards})", flush=True)
    t0 = time.time()
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
              enable_prefix_caching=True, trust_remote_code=args.trust_remote_code, enforce_eager=args.enforce_eager, seed=0)
    outs = llm.generate(prompts, params, use_tqdm=True)
    texts = [o.outputs[0].text for o in outs]
    print(f"[peer-answers] generated in {time.time() - t0:.0f}s; grading", flush=True)
    del llm
    rows, by_source = [], collections.defaultdict(lambda: [0, 0.0])
    for r, text in zip(records, texts):
        answer = strip_thinking(text) if args.reasoning else text
        v = graded_value(r, answer)
        rows.append({"id": r.get("id"), "source": r.get("source"), "task_type": task_type_of(r), "response": text, "target": v, "correct": int(round(v))})
        by_source[r.get("source")][0] += 1; by_source[r.get("source")][1] += int(round(v))
    tag = ("." + args.sources.replace(",", "_")) if args.sources else ""
    stem = args.records.stem   # train / test
    with (args.output / (f"{stem}{tag}.jsonl" if args.num_shards == 1 else f"{stem}{tag}.shard{args.shard_index}of{args.num_shards}.jsonl")).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"model": args.model, "n": len(rows), "reasoning": args.reasoning, "by_source": {s: {"n": n, "accuracy": 100 * k / n} for s, (n, k) in by_source.items()}, "seconds": time.time() - t0}
    (args.output / (f"summary_{stem}{tag}.json" if args.num_shards == 1 else f"summary_{stem}{tag}.shard{args.shard_index}of{args.num_shards}.json")).write_text(json.dumps(summary, indent=1))
    print("[peer-answers] " + ", ".join(f"{s}: {v['accuracy']:.1f}% ({v['n']})" for s, v in summary["by_source"].items()) + f" ({summary['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
