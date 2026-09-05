"""Verified generations  ->  SFT parquet (prompt = question-only chat, response = own-words solution).

Sources of responses, in this order of preference per row:
  --generations   generations.jsonl of scripts/generate_hinted.py (hinted, own-words answers; the
                  hint was chosen by the memory) or of any solo evaluation; kept when correct
  --prompts       the prompt file's ``target`` (peer / self-distilled targets), when no generation

  PYTHONPATH=. python -m training.sigma_rl.build_sft_data --prompts outputs/gen/q3_4b/prompts_train_fixed.jsonl \
      --records data/mixed_train_big/train.jsonl --generations outputs/gen/q3_4b/hinted_memory/generations.jsonl \
      --out outputs/rl/data/q3_4b/sft_hinted_memory.parquet
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import grade


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--generations", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--use_targets", choices=["on", "off"], default="off", help="on = also use the prompt file's target for rows without a correct generation")
    ap.add_argument("--verify", choices=["on", "off"], default="on", help="re-grade generations (off = trust the 'correct' field)")
    ap.add_argument("--max_chars", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(l) for l in args.prompts.open()]
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    gens: dict[str, dict] = {}
    if args.generations:
        for l in args.generations.open():
            g = json.loads(l)
            gens[str(g["id"])] = g
    out, sources = [], collections.Counter()
    for r in rows:
        rid = str(r["id"])
        rec = records[rid]
        response, src = None, None
        g = gens.get(rid)
        if g and str(g.get("generation", "")).strip():
            ok = bool(grade(rec, g["generation"])) if args.verify == "on" else bool(g.get("correct", 0))
            if ok:
                response, src = str(g["generation"]).strip(), "generation"
        if response is None and args.use_targets == "on" and r.get("target"):
            response, src = str(r["target"]).strip(), f"target:{r.get('target_source', '?')}"
        if response is None or len(response) > args.max_chars:
            continue
        sources[src] += 1
        out.append({"prompt": r["messages_solo"], "response": response, "data_source": str(r["task_type"]), "id": rid, "source": src})
        if args.limit and len(out) >= args.limit:
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(args.out, index=False)
    print(f"[build-sft-data] wrote {len(out)} rows to {args.out}: sources={dict(sources)} tasks={dict(collections.Counter(x['data_source'] for x in out))}")


if __name__ == "__main__":
    main()
