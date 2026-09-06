"""Merge candidate peers' answers into a stream, producing a k-peer stream file (the original is never modified).

Each new peer's answers come from scripts/peer_answers.py (rows {id, response, target, correct}); they are appended as
peer_<k> with the same fields the existing peers carry: peer_responses, peer_metadata, peer_correct (the graded target
value) and correctness_by_peer (rounded).  Events missing an answer from any new peer are dropped and counted.

  PYTHONPATH=. python scripts/merge_peers.py --records data/mixed_train_big/train.jsonl --out data/mixed_train_big5/train.jsonl \
      --peer Meta-Llama-3.1-8B-Instruct=outputs/peergen_new/Meta-Llama-3.1-8B-Instruct/train.jsonl \
      --peer DeepSeek-Coder-V2-Lite-Instruct=outputs/peergen_new/DeepSeek-Coder-V2-Lite-Instruct/train.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--peer", action="append", required=True, help="name=path (path may be a glob over shard files)")
    ap.add_argument("--model_root", default="/mnt/data/peilin/HF_MODEL")
    args = ap.parse_args()
    new_peers = []
    for spec in args.peer:
        name, path = spec.split("=", 1)
        rows = {}
        for f in sorted(glob.glob(path)):
            for line in open(f):
                r = json.loads(line)
                rows[str(r["id"])] = r
        new_peers.append((name, rows))
        print(f"[merge] {name}: {len(rows)} answers from {path}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    dropped = collections.Counter()
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    with args.out.open("w") as f:
        for line in args.records.open():
            rec = json.loads(line)
            n_in += 1
            rid = str(rec["id"])
            keys = sorted(rec["peer_responses"])
            ok = True
            for name, rows in new_peers:
                if rid not in rows:
                    dropped[name] += 1
                    ok = False
            if not ok:
                continue
            for name, rows in new_peers:
                k = f"peer_{len(keys)}"
                r = rows[rid]
                rec["peer_responses"][k] = r["response"]
                meta = dict(next(iter(rec.get("peer_metadata", {}).values()), {}))
                meta.update({"model": f"{args.model_root}/{name}", "received_context": True, "num_samples": 1})
                rec.setdefault("peer_metadata", {})[k] = meta
                rec.setdefault("peer_correct", {})[k] = float(r.get("target", r["correct"]))
                if "correctness_by_peer" in rec:   # only where the record already carries the rounded labels (train streams)
                    rec["correctness_by_peer"][k] = int(r["correct"])
                keys.append(k)
            labels = rec.get("correctness_by_peer") or rec.get("peer_correct") or {}
            for k in keys:
                acc[rec.get("source")][k][0] += 1
                acc[rec.get("source")][k][1] += int(round(float(labels.get(k, 0))))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"[merge] {n_in} events in, {n_out} out, dropped for missing answers: {dict(dropped)} -> {args.out}")
    for src, per in acc.items():
        print(f"   {src}: " + ", ".join(f"{k} {100 * v[1] / v[0]:.0f}%" for k, v in sorted(per.items())))


if __name__ == "__main__":
    main()
