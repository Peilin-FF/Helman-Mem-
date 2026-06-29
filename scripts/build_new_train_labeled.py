#!/usr/bin/env python3
"""Build the NEW 3-task (math+code+rag) training set from GSM8K + SQuAD + APPS.

Sources (graded, 3 peers = gemma/phi/qwen-coder):
  math : data/v3/peergen/gsm8k.graded.jsonl   (GSM8K train)
  rag  : data/v3/peergen/squad.graded.jsonl   (SQuAD train, gemma context-deprived)
  code : data/v3/peergen/apps.graded.jsonl    (APPS easy/stdin slice, execution-scored)

These train sources are disjoint from the v3_unified CF test streams by construction
(test math = math500/amc/..., test rag = hotpotqa/triviaqa, test code = humaneval/mbpp/lcb);
code still gets the explicit test-id exclusion to be safe.

Every source carries per-peer correctness (peer_correct or correctness_by_peer); we
normalize both into correctness_by_peer so train_symmetric_memory reads one field.

Balancing: --per_task N caps each task to N records (after seeded shuffle). Default 1000.
Output: data/v3/mixed_train_labeled.jsonl (shuffled across tasks, seeded)
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

TEST_STREAMS = [f"data/v3_unified/{p}.jsonl" for p in ("p0", "p50", "p70", "p90")]
MATH_SRC = "data/v3/peergen/gsm8k.graded.jsonl"
RAG_SRC = "data/v3/peergen/squad.graded.jsonl"
CODE_SRC = "data/v3/peergen/apps.graded.jsonl"
PEERS = ("peer_0", "peer_1", "peer_2")
OUT = "data/v3/mixed_train_labeled.jsonl"


def test_ids_by_dataset() -> dict:
    seen = {}
    for path in TEST_STREAMS:
        if not Path(path).exists():
            continue
        for line in open(path):
            r = json.loads(line)
            seen.setdefault(r.get("dataset"), set()).add(r.get("id"))
    return seen


def norm_label(rec: dict) -> dict | None:
    """Ensure correctness_by_peer exists; derive from peer_correct if needed. None if no label."""
    out = dict(rec)
    if "correctness_by_peer" in out:
        out["correctness_by_peer"] = {p: int(round(float(out["correctness_by_peer"].get(p, 0)))) for p in PEERS}
        return out
    if "peer_correct" in out:
        out["correctness_by_peer"] = {p: int(round(float(out["peer_correct"].get(p, 0)))) for p in PEERS}
        return out
    return None


def load_capped(paths, task, cap, exclude_by_ds, seed):
    recs = []
    for path in (paths if isinstance(paths, list) else [paths]):
        if not Path(path).exists():
            print(f"  SKIP missing {path}"); continue
        for line in open(path):
            r = json.loads(line)
            ds = r.get("dataset") or r.get("source")
            if exclude_by_ds and r.get("id") in exclude_by_ds.get(ds, set()):
                continue
            nr = norm_label(r)
            if nr is None:
                continue
            nr["task_type"] = task
            recs.append(nr)
    random.Random(seed).shuffle(recs)
    if cap and cap > 0:
        recs = recs[:cap]
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_task", type=int, default=1000, help="cap per task (0 = keep all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    test_ids = test_ids_by_dataset()
    print("test ids per dataset:", {k: len(v) for k, v in test_ids.items()})

    math = load_capped(MATH_SRC, "math", args.per_task, None, args.seed)
    code = load_capped(CODE_SRC, "code", args.per_task, test_ids, args.seed + 1)
    rag = load_capped(RAG_SRC, "rag", args.per_task, test_ids, args.seed + 2)
    print(f"loaded -> math={len(math)} code={len(code)} rag={len(rag)}")

    allrecs = math + code + rag
    random.Random(args.seed + 99).shuffle(allrecs)

    # per-peer correct counts (sanity: peer reliability should ROTATE across tasks)
    from collections import defaultdict
    bytask = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in allrecs:
        tt = r["task_type"]
        for p in PEERS:
            v = r["correctness_by_peer"][p]
            bytask[tt][p][0] += v; bytask[tt][p][1] += 1
    print("per-(task,peer) correctness in TRAIN:")
    for tt in ("math", "code", "rag"):
        row = [f"{p}={100*bytask[tt][p][0]/bytask[tt][p][1]:.0f}%" if bytask[tt][p][1] else f"{p}=NA" for p in PEERS]
        print(f"  {tt:5}: " + "  ".join(row))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in allrecs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(allrecs)} records -> {args.out}")


if __name__ == "__main__":
    main()
