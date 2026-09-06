"""Quality of the memory's record on a prompt stream, against trivial histories.

Per (event, peer) pair the record prints an estimate p; the label says whether the peer was right.  Reports AUC and
Brier of the estimate, the accuracy of the record's favourite on events where some peers are right and some wrong
("mixed"), the same for a running mean per peer and per (peer, task) and for hindsight tables, the partition of the
events (unanimous / flat record / favourite in the plurality / favourite in the minority) and the same along the
stream in windows.  Read-before-write is guaranteed by construction of the prompt files.

  PYTHONPATH=. python scripts/record_quality.py --prompts outputs/gen/q3_4b/prompts_ood5_shuffled0.jsonl --out outputs/gen/q3_4b/probe5/record_ood.json
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
from pathlib import Path


def auc(pairs):
    pos = sorted(p for p, y in pairs if y == 1)
    neg = sorted(p for p, y in pairs if y == 0)
    if not pos or not neg:
        return float("nan")
    return sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = sorted((json.loads(l) for l in args.prompts.open()), key=lambda r: r["pos"])
    tot = collections.defaultdict(lambda: [0, 0]); tot_t = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        for slot, pid in enumerate(r["peer_order"]):
            y = r["peer_correct"][slot]; tot[pid][0] += 1; tot[pid][1] += y; tot_t[(pid, r["task_type"])][0] += 1; tot_t[(pid, r["task_type"])][1] += y
    run = collections.defaultdict(lambda: [0, 0]); run_t = collections.defaultdict(lambda: [0, 0])
    keys = ("memory", "running_peer", "running_peer_task", "hindsight_peer_task")
    preds = {k: [] for k in keys}; mixed = {k: [0, 0] for k in keys}; per_task = collections.defaultdict(lambda: {k: [] for k in keys})
    n_mixed = 0; chance = 0.0
    part = collections.Counter(); minority_fav_right = [0, 0]
    win = []
    N = len(rows); edges = [round(i * N / args.windows) for i in range(args.windows + 1)]
    for i, r in enumerate(rows):
        ys = r["peer_correct"]; task = r["task_type"]; probs = list(r["memory_prob"])
        cand = {"memory": probs, "running_peer": [(run[p][1] + 1) / (run[p][0] + 2) for p in r["peer_order"]],
                "running_peer_task": [(run_t[(p, task)][1] + 1) / (run_t[(p, task)][0] + 2) for p in r["peer_order"]],
                "hindsight_peer_task": [tot_t[(p, task)][1] / max(1, tot_t[(p, task)][0]) for p in r["peer_order"]]}
        for k, ps in cand.items():
            for p, y in zip(ps, ys):
                preds[k].append((p, y)); per_task[task][k].append((p, y))
        if 0 < sum(ys) < len(ys):
            n_mixed += 1; chance += sum(ys) / len(ys)
            for k, ps in cand.items():
                top = max(range(len(ps)), key=lambda s: ps[s]); mixed[k][0] += 1; mixed[k][1] += ys[top]
        # partition by peer agreement on correctness (labels only; answer groups need the texts)
        flat = max(probs) - min(probs) <= 0.1
        if len(set(ys)) == 1:
            part["unanimous (all right or all wrong)"] += 1
        elif flat:
            part["split, flat record"] += 1
        else:
            top = max(range(len(probs)), key=lambda s: probs[s])
            maj = 1 if sum(ys) * 2 > len(ys) else 0
            if ys[top] == maj:
                part["split, favourite agrees with the majority label"] += 1
            else:
                part["split, favourite against the majority"] += 1
                minority_fav_right[0] += 1; minority_fav_right[1] += ys[top]
        for slot, pid in enumerate(r["peer_order"]):
            run[pid][0] += 1; run[pid][1] += ys[slot]; run_t[(pid, task)][0] += 1; run_t[(pid, task)][1] += ys[slot]
    res = {"n_events": N, "num_peers": len(rows[0]["peer_order"]), "n_mixed": n_mixed, "chance_on_mixed": 100 * chance / max(1, n_mixed),
           "peer_accuracy": {str(p): 100 * v[1] / v[0] for p, v in sorted(tot.items())},
           "peer_accuracy_by_task": {f"{p}/{t}": 100 * v[1] / v[0] for (p, t), v in sorted(tot_t.items())},
           "partition": dict(part), "minority_favourite_right": 100 * minority_fav_right[1] / max(1, minority_fav_right[0])}
    for k in keys:
        res[k] = {"auc": auc(preds[k]), "favourite_acc_mixed": 100 * mixed[k][1] / max(1, mixed[k][0]), "auc_by_task": {t: auc(per_task[t][k]) for t in sorted(per_task)}}
    # along the stream
    for a, b in zip(edges[:-1], edges[1:]):
        seg = rows[a:b]
        mp = [(p, y) for r in seg for p, y in zip(r["memory_prob"], r["peer_correct"])]
        mx = [r for r in seg if 0 < sum(r["peer_correct"]) < len(r["peer_correct"])]
        fav = sum(r["peer_correct"][max(range(len(r["memory_prob"])), key=lambda s: r["memory_prob"][s])] for r in mx) / max(1, len(mx))
        win.append({"events": f"{a}-{b - 1}", "memory_auc": auc(mp), "memory_favourite_right": 100 * fav, "n_mixed": len(mx), "mean_evidence": sum(e for r in seg for e in r["memory_evidence"]) / max(1, sum(len(r["memory_evidence"]) for r in seg))})
    res["windows"] = win
    print(f"== {args.prompts.name}: {N} events, {res['num_peers']} peers, mixed {n_mixed} ({100 * n_mixed / N:.1f}%), chance pick {res['chance_on_mixed']:.1f}%")
    print("   peer accuracy:", {k: round(v, 1) for k, v in res["peer_accuracy"].items()})
    for k in keys:
        print(f"   {k:20s} AUC {res[k]['auc']:.3f}  favourite right on mixed {res[k]['favourite_acc_mixed']:.1f}%  by task {{{', '.join(f'{t}: {v:.2f}' for t, v in res[k]['auc_by_task'].items())}}}")
    print("   partition:", res["partition"], f"| favourite against the majority is right {res['minority_favourite_right']:.1f}%")
    print("   along the stream: " + " | ".join(f"{w['events']}: AUC {w['memory_auc']:.3f}, fav {w['memory_favourite_right']:.1f}%, evidence {w['mean_evidence']:.0f}" for w in win))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
