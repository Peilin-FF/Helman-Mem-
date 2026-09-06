"""Screen candidate peers: accuracy per task and complementarity with the current peers and the central model.

For every candidate's alone run on a probe slice (evaluate_memory_generator --mode solo), join its per-event
correctness with the current peers' graded correctness (from the prompt rows, re-graded with the evaluator's rule)
and the central model's alone correctness, and report per task:
  accuracy; right where all current peers are wrong (rescue rate); right where the central model is wrong;
  error correlation with each current peer and with the central model; the oracle gain (some peer right) if added.

  PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 python scripts/peer_screen.py --records data/ood/test.jsonl \
      --prompts outputs/gen/q3_4b/steer/ood/prompts_number.jsonl --central outputs/gen/q3_4b/frozen/eval_ood_shuffled0_solo_vllm/generations.jsonl \
      --candidates outputs/gen/peer_screen/*/ood --out outputs/gen/peer_screen/screen_ood.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import grade
from feedback_state.memory_rl import peer_texts_in_prompt_order
from feedback_state.tasks import task_type_of


def phi(a: list[int], b: list[int]) -> float:
    """Phi coefficient between two 0/1 sequences (error correlation)."""
    n = len(a)
    if n == 0:
        return float("nan")
    pa, pb = sum(a) / n, sum(b) / n
    pab = sum(x and y for x, y in zip(a, b)) / n
    den = math.sqrt(pa * (1 - pa) * pb * (1 - pb))
    return (pab - pa * pb) / den if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--central", type=Path, required=True, help="the central model's alone generations (whole stream)")
    ap.add_argument("--candidates", nargs="+", type=Path, required=True, help="directories with generations.jsonl of each candidate on the same rows")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    rows = {str(r["id"]): r for r in map(json.loads, args.prompts.open())}
    central = {str(g["id"]): int(g["correct"]) for g in map(json.loads, args.central.open())}
    peer_names = None
    peers_ok: dict[str, list[int]] = {}
    for rid, r in rows.items():
        rec = records[rid]
        if peer_names is None:
            meta = rec.get("peer_metadata", {})
            peer_names = [meta.get(k, {}).get("model", k).split("/")[-1] for k in sorted(rec.get("peer_responses", {}))]
        texts = peer_texts_in_prompt_order(rec, r["peer_order"])
        peers_ok[rid] = [int(grade(rec, t)) for t in texts] if task_type_of(rec) != "code" else [int(c) for c in r["peer_correct"]]
    # prompt order differs per event; map back to canonical peer ids
    canon: dict[str, dict[int, int]] = {rid: {int(p): peers_ok[rid][slot] for slot, p in enumerate(rows[rid]["peer_order"])} for rid in rows}
    result = {}
    for cdir in args.candidates:
        gpath = cdir / "generations.jsonl"
        if not gpath.exists():
            continue
        name = cdir.parent.name
        gens = {str(g["id"]): (int(g["correct"]), g["task_type"]) for g in map(json.loads, gpath.open())}
        by_task = collections.defaultdict(list)
        for rid, (ok, task) in gens.items():
            if rid not in rows or rid not in central:
                continue
            by_task[task].append((ok, canon[rid], central[rid]))
        res = {}
        for task, items in sorted(by_task.items()):
            n = len(items)
            acc = sum(ok for ok, _, _ in items) / n
            all_wrong = [(ok, c) for ok, c, _ in items if not any(c.values())]
            rescue = sum(ok for ok, _ in all_wrong) / max(1, len(all_wrong))
            cm_wrong = [ok for ok, _, cm in items if not cm]
            rescue_cm = sum(cm_wrong) / max(1, len(cm_wrong))
            oracle_now = sum(1 for _, c, _ in items if any(c.values())) / n
            oracle_new = sum(1 for ok, c, _ in items if ok or any(c.values())) / n
            corr = {peer_names[p]: phi([ok for ok, _, _ in items], [c[p] for _, c, _ in items]) for p in range(len(peer_names))}
            corr["central model"] = phi([ok for ok, _, _ in items], [cm for _, _, cm in items])
            res[task] = {"n": n, "accuracy": 100 * acc, "n_all_peers_wrong": len(all_wrong), "rescue_rate": 100 * rescue, "n_central_wrong": len(cm_wrong), "right_where_central_wrong": 100 * rescue_cm,
                         "oracle_now": 100 * oracle_now, "oracle_with_candidate": 100 * oracle_new, "correlation": corr}
        allv = [v for v in res.values()]
        res["all"] = {"n": sum(v["n"] for v in allv), "accuracy": sum(v["accuracy"] * v["n"] for v in allv) / max(1, sum(v["n"] for v in allv)),
                      "oracle_now": sum(v["oracle_now"] * v["n"] for v in allv) / max(1, sum(v["n"] for v in allv)), "oracle_with_candidate": sum(v["oracle_with_candidate"] * v["n"] for v in allv) / max(1, sum(v["n"] for v in allv))}
        result[name] = res
        print(f"== {name}: " + " | ".join(f"{t}: acc {v['accuracy']:.1f}, rescues {v['rescue_rate']:.0f}% of {v['n_all_peers_wrong']} all-peers-wrong, right where central wrong {v['right_where_central_wrong']:.0f}% of {v['n_central_wrong']}, oracle {v['oracle_now']:.0f}->{v['oracle_with_candidate']:.0f}, corr " + ",".join(f"{k.split('-')[0]}:{c:.2f}" for k, c in v['correlation'].items()) for t, v in res.items() if t != "all"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"current_peers": peer_names, "candidates": result}, indent=1))


if __name__ == "__main__":
    main()
