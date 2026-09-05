"""Own-words solutions with the most reliable verified-correct peer's solution as a hint (memory-ranked).

For every event with at least one correct peer, the student is shown that peer's solution (the one with the
highest memory reliability among the correct ones) and asked to solve in its own words; the output is graded.
Correct outputs are on-policy, self-contained, peer-informed targets for question-only distillation.

  PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 python scripts/generate_hinted.py --central_model ... \
      --prompts outputs/gen/q3_4b/prompts_train_fixed.jsonl --records data/mixed_train_big/train.jsonl --output outputs/gen/q3_4b/frozen/gen_train_hinted
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from feedback_state.newarch_loader import apply_torch_fp8_shim, dtype_from_name, load_central_model

apply_torch_fp8_shim()

from transformers import AutoTokenizer

from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import grade, render_prompt
from feedback_state.train_rlvr import hint_messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--central_model", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--hint_choice", choices=["memory", "random"], default="memory", help="memory = most reliable verified-correct peer; random = a random verified-correct peer (control)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import random as _random
    rng = _random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.central_model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load_central_model(args.central_model, dtype=dtype_from_name("bfloat16"), local_files_only=True).to("cuda").eval()
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    rows = [json.loads(l) for l in args.prompts.open()]
    todo = []
    for r in rows:
        correct = [s for s in range(len(r["peer_correct"])) if int(r["peer_correct"][s]) == 1]
        if not correct:
            continue
        s = max(correct, key=lambda k: r["memory_prob"][k]) if args.hint_choice == "memory" else rng.choice(correct)
        rec = records[str(r["id"])]
        keys = sorted(rec.get("peer_responses", {}))
        hint = str(rec["peer_responses"][keys[r["peer_order"][s]]])
        todo.append((r, s, render_prompt(tok, hint_messages(rec, hint, float(r["memory_prob"][s]), float(r["memory_evidence"][s])))))
    if args.max_examples:
        todo = todo[: args.max_examples]
    print(f"[hinted] events with a correct peer: {len(todo)} of {len(rows)}", flush=True)
    outputs = [""] * len(todo)
    order = sorted(range(len(todo)), key=lambda i: len(todo[i][2]))
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), args.batch_size):
            ids = order[b : b + args.batch_size]
            enc = tok([todo[i][2] for i in ids], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            for i, t in zip(ids, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                outputs[i] = t
            if (b // args.batch_size) % 25 == 0:
                print(f"[hinted] {min(b + args.batch_size, len(order))}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
    hits = []
    with (args.output / "generations.jsonl").open("w") as f:
        for (r, s, _), text in zip(todo, outputs):
            ok = grade(records[str(r["id"])], text)
            hits.append(int(ok))
            f.write(json.dumps({"pos": r["pos"], "id": r["id"], "task_type": r["task_type"], "source": r["source"], "correct": int(ok),
                                "hint_slot": s, "hint_prob": r["memory_prob"][s], "peer_correct": r["peer_correct"], "generation": text}) + "\n")
    by = {t: float(np.mean([h for h, (r, _, _) in zip(hits, todo) if r["task_type"] == t])) for t in sorted({r["task_type"] for r, _, _ in todo})}
    (args.output / "eval_metrics.json").write_text(json.dumps({"accuracy": float(np.mean(hits)), "num_samples": len(hits), "by_task": by, "mode": "hinted"}, indent=1))
    print(f"[hinted] accuracy of own-words hinted solutions: {100 * np.mean(hits):.2f} by_task={ {k: round(100 * v, 1) for k, v in by.items()} } ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
