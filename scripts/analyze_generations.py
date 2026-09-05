"""Why does a fine-tuned generator lose math? Copy rates, own-solving rates and output length per arm.

  PYTHONPATH=. python scripts/analyze_generations.py --prompts outputs/gen/q3_4b/prompts_indist_shuffled0.jsonl \
      --records data/indist/test.jsonl --runs solo=outputs/gen/q3_4b/frozen/eval_indist_shuffled0_solo ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feedback_state.data import JsonlDataset
from feedback_state.tasks import _normalize_qa, qa_extract_answer, task_type_of
from feedback_state.utils import extract_final_answer, math_equal


def final_of(task: str, text: str) -> str:
    if task == "math":
        return extract_final_answer(text)
    if task == "rag":
        return _normalize_qa(qa_extract_answer(text))
    return ""


def same(task: str, a: str, b: str) -> bool:
    if not a or not b:
        return False
    return math_equal(a, b) if task == "math" else a == b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir pairs (dir holds generations.jsonl)")
    args = ap.parse_args()
    records = {str(r.get("id")): r for r in JsonlDataset(args.records).records}
    prompts = {}
    for line in args.prompts.open():
        r = json.loads(line)
        prompts[str(r["id"])] = r
    for spec in args.runs:
        name, d = spec.split("=", 1)
        gens = [json.loads(l) for l in (Path(d) / "generations.jsonl").open()]
        ids = {g["id"] for g in gens}
        print(f"\n=== {name}  ({len(gens)} events)")
        for task in ("math", "rag", "code"):
            rows = [g for g in gens if g["task_type"] == task]
            if not rows:
                continue
            acc = np.mean([g["correct"] for g in rows])
            lens = np.mean([len(g["generation"]) for g in rows])
            all_wrong = [g for g in rows if max(g["peer_correct"]) == 0]
            some_right = [g for g in rows if max(g["peer_correct"]) == 1]
            acc_aw = np.mean([g["correct"] for g in all_wrong]) if all_wrong else float("nan")
            acc_sr = np.mean([g["correct"] for g in some_right]) if some_right else float("nan")
            line = f"  {task:5s} acc={100*acc:5.1f}  n={len(rows)}  mean_len={lens:6.0f} chars | all-peers-wrong n={len(all_wrong)} acc={100*acc_aw:5.1f} | some-peer-right n={len(some_right)} acc={100*acc_sr:5.1f}"
            if task in ("math", "rag"):
                copy = copy_wrong = copy_right = disagree_right = 0
                for g in rows:
                    p = prompts[str(g["id"])]
                    rec = records[str(g["id"])]
                    peers_texts = []
                    # peer texts in prompt order come from the messages (peer blocks); recover from the record via peer_order
                    keys = sorted(rec.get("peer_responses", {}))
                    for slot, pid in enumerate(p["peer_order"]):
                        peers_texts.append((str(rec["peer_responses"][keys[pid]]), int(p["peer_correct"][slot])))
                    own = final_of(task, g["generation"])
                    match = [c for t, c in peers_texts if same(task, own, final_of(task, t))]
                    if match:
                        copy += 1
                        if max(match) == 1:
                            copy_right += 1
                        else:
                            copy_wrong += 1
                    elif g["correct"]:
                        disagree_right += 1
                n = len(rows)
                line += f" | answer matches a peer: {100*copy/n:4.1f}% (matched peer right {100*copy_right/n:4.1f}%, wrong {100*copy_wrong/n:4.1f}%) | own answer, no peer match, correct: {100*disagree_right/n:4.1f}%"
            print(line)


if __name__ == "__main__":
    main()
