"""Build the APPS solvable-id allowlist (run once, offline).

Many APPS stdin/stdout problems accept multiple valid outputs, so the exact-match
check_io grader rejects a correct answer. Those problems are false-negatives for
EVERY peer and flatten the per-peer trust gradient. We keep only problems whose own
reference solution passes check_io -- i.e. problems the grader can actually score.

Writes data/apps_solvable_ids.json (a list of APPS ids). load_apps() then filters
to this set automatically (see APPS_SOLVABLE_IDS). Re-run only if the slice filter
(stdin-only / APPS_MAX_SOL_CHARS) changes.

    FEEDBACK_CODE_EXEC_ALLOW=1 PYTHONPATH=. python scripts/build_apps_solvable_ids.py
"""
from __future__ import annotations

import argparse
import json
import os

from tqdm.auto import tqdm

from feedback_state.code_exec import check_io
from feedback_state.eval_datasets import APPS_MAX_SOL_CHARS, _dataset_path, _io_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify APPS reference solutions pass check_io.")
    p.add_argument("--split", default="train")
    p.add_argument("--output", default="data/apps_solvable_ids.json")
    p.add_argument("--timeout", type=float, default=8.0)
    p.add_argument("--max_cases", type=int, default=8, help="cap test cases checked per problem")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = _dataset_path("codeparrot/apps")
    jsonl = os.path.join(local_dir, f"{args.split or 'train'}.jsonl")
    rows = []
    with open(jsonl) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    solvable: list[str] = []
    considered = 0
    for ex in tqdm(rows, desc="verify apps refs"):
        try:
            io = json.loads(ex["input_output"]) if isinstance(ex.get("input_output"), str) else (ex.get("input_output") or {})
        except Exception:
            continue
        inputs, outputs = io.get("inputs") or [], io.get("outputs") or []
        if not inputs or "fn_name" in io:  # stdin-style only
            continue
        try:
            sols = json.loads(ex["solutions"]) if isinstance(ex.get("solutions"), str) else (ex.get("solutions") or [])
        except Exception:
            sols = []
        if not sols or min(len(s) for s in sols) > APPS_MAX_SOL_CHARS:
            continue
        considered += 1
        cases = [{"input": _io_text(i), "output": _io_text(o)} for i, o in zip(inputs, outputs)][: args.max_cases]
        ref = min(sols, key=len)
        if check_io(ref, cases, timeout=args.timeout).passed:
            solvable.append(str(ex.get("id")))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(solvable, f)
    print(f"solvable {len(solvable)}/{considered} -> {args.output}")


if __name__ == "__main__":
    main()
