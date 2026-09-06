"""Accuracy table of the current and candidate peers on the training stream, with "at least one" columns.

Current peers: correctness_by_peer stored in the stream records (peer_0 gemma-3-4b-it, peer_1 Phi-4-mini-instruct,
peer_2 Qwen2.5-Coder-7B-Instruct).  Candidates: outputs/peergen_new/<model>/train*.jsonl from scripts/peer_answers.py,
graded with the same rule.  Per dataset: each model's accuracy, "at least one" over the current three, over the
current three plus each candidate, and over all.

  PYTHONPATH=. python scripts/peer_table.py --records data/mixed_train_big/train.jsonl --candidates outputs/peergen_new/*
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

CURRENT = {"peer_0": "Gemma-3-4b-it", "peer_1": "Phi-4-mini", "peer_2": "Qwen2.5-Coder-7B"}
ORDER = {"gsm8k": "GSM8K", "squad": "SQuAD", "apps": "APPS"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--candidates", nargs="*", type=Path, default=[])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    ok: dict[str, dict[str, int]] = collections.defaultdict(dict)   # model -> id -> correct
    source: dict[str, str] = {}
    for line in args.records.open():
        r = json.loads(line)
        source[str(r["id"])] = r.get("source")
        for p, name in CURRENT.items():
            ok[name][str(r["id"])] = int(round(float(r.get("correctness_by_peer", r.get("peer_correct", {})).get(p, 0))))
    cands = []
    for d in args.candidates:
        rows = [json.loads(l) for f in sorted(glob.glob(str(d / "train*.jsonl"))) for l in open(f)]
        if not rows:
            continue
        name = d.name
        cands.append(name)
        for row in rows:
            ok[name][str(row["id"])] = int(row["correct"])
    models = list(CURRENT.values()) + cands
    ids_by_source = collections.defaultdict(list)
    for rid, s in source.items():
        ids_by_source[s].append(rid)
    header = ["Dataset", "Number"] + models + ["At least one (current 3)"] + [f"+{c}" for c in cands] + (["All"] if cands else [])
    table = [header]
    for s in ORDER:
        ids = ids_by_source.get(s, [])
        row = [ORDER[s], str(len(ids))]
        for m in models:
            vals = [ok[m].get(i) for i in ids]
            have = [v for v in vals if v is not None]
            row.append(f"{100 * sum(have) / len(have):.0f}%" + ("" if len(have) == len(ids) else f" ({len(have)})") if have else "—")
        cur = list(CURRENT.values())
        def at_least_one(ms):
            vals = [any(ok[m].get(i, 0) for m in ms) for i in ids]
            return f"{100 * sum(vals) / len(vals):.0f}%" if vals else "—"
        row.append(at_least_one(cur))
        for c in cands:
            row.append(at_least_one(cur + [c]))
        if cands:
            row.append(at_least_one(cur + cands))
        table.append(row)
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    for r in table:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    if args.out:
        args.out.write_text(json.dumps({"header": header, "rows": table[1:]}, indent=1))


if __name__ == "__main__":
    main()
