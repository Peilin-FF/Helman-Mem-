"""Read-only data preflight for the new protocol; no rows are selected or rewritten.

The output is an audit, NOT a training dataset or precomputed memory trajectory.
It records exact rendered question+ALL-peer token lengths with the local model's
tokenizer, so an 8192-token budget cannot silently remove peers or long tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from feedback_state.memory_generator import render_prompt
from feedback_state.tasks import task_type_of
from training.sigma_rl.outcome_protocol import MAX_PROMPT_LENGTH, PROTOCOL, public_messages
from training.sigma_rl.outcome_reward import verifier_record


def audit(records: Path, tokenizer, *, max_length: int, chunk_size: int = 64) -> dict:
    if max_length < 1 or chunk_size < 1:
        raise ValueError("max_length and chunk_size must be positive")
    lengths, identifiers = [], []
    tasks, peer_counts, code_formats = defaultdict(list), Counter(), Counter()
    errors, private_reference_errors = [], []
    invalid_public, invalid_reference = 0, 0
    digest = hashlib.sha256()
    pending = []
    count = 0
    start = time.perf_counter()

    def flush():
        if not pending:
            return
        encoded = tokenizer([item[2] for item in pending], add_special_tokens=False, truncation=False, padding=False)["input_ids"]
        for (task, rid, _), ids in zip(pending, encoded):
            lengths.append(len(ids))
            identifiers.append(rid)
            tasks[task].append(len(ids))
        pending.clear()

    with records.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            count += 1
            rid = str(record.get("id") or record.get("uid") or f"line:{line_number}")
            task = task_type_of(record)
            peers = record.get("peer_responses") or {}
            if not isinstance(peers, dict):
                raise ValueError(f"{rid}: peer_responses must map canonical peer keys to text")
            keys = sorted(peers)
            peer_counts[len(keys)] += 1
            if task == "code":
                code_formats[str(record.get("code_format", "asserts"))] += 1
            try:
                # Reference validation is a controller-only diagnostic. It is
                # never used to choose or reorder any peer in the public prompt.
                verifier_record(record)
            except ValueError as exc:
                invalid_reference += 1
                if len(private_reference_errors) < 10:
                    private_reference_errors.append({"id": rid, "error": str(exc)})
            try:
                messages = public_messages(record, [str(peers[key]) for key in keys])
            except ValueError as exc:
                invalid_public += 1
                if len(errors) < 10:
                    errors.append({"id": rid, "error": str(exc)})
                continue
            pending.append((task, rid, render_prompt(tokenizer, messages)))
            if len(pending) >= chunk_size:
                flush()
            if count % 1000 == 0:
                print(f"[audit] records={count} elapsed_s={time.perf_counter() - start:.1f}", flush=True)
    flush()

    def summary(values):
        return {
            "count": len(values), "over_budget": sum(n > max_length for n in values),
            "min": min(values) if values else None, "max": max(values) if values else None,
            "p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "p99": float(np.percentile(values, 99)) if values else None,
        }

    return {
        "protocol": PROTOCOL, "scope": "data audit only; no filtering, generation, memory writes or training",
        "records_path": str(records.resolve()), "records_sha256": digest.hexdigest(), "records": count,
        "max_prompt_length": max_length, "tokenizer": tokenizer.name_or_path,
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "enable_thinking": False, "peer_text_truncated": False, "context_truncated": False,
        "lengths": summary(lengths), "by_task": {task: summary(values) for task, values in sorted(tasks.items())},
        "peer_count_histogram": dict(peer_counts), "code_formats": dict(code_formats),
        "invalid_public_records": invalid_public, "public_error_examples": errors,
        "invalid_reference_records": invalid_reference, "reference_error_examples": private_reference_errors,
        "over_budget_examples": [{"id": rid, "tokens": n} for rid, n in zip(identifiers, lengths) if n > max_length][:10],
        "elapsed_s": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--tokenizer", default="/mnt/data/peilin/HF_MODEL/Qwen3-4B")
    parser.add_argument("--max-prompt-length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"audit output exists: {args.out}; use a new path")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = audit(args.records, tokenizer, max_length=args.max_prompt_length)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
