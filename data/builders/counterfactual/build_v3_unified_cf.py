#!/usr/bin/env python3
"""Build a UNIFIED mixed-task counterfactual eval stream (one file per proportion).

Differences vs build_v3_counterfactuals.py (which this reuses):
  1. BALANCED weak target per dataset: among the records swapped at proportion p, the
     choice of WHICH wrong weak peer becomes correct is balanced across that dataset's
     weak peers (greedy least-used among each record's available wrong-weak options),
     instead of an independent per-record random pick. Slightly off-balance is fine.
  2. MIXED random stream: all datasets (math+code+rag) are pooled into ONE shuffled
     JSONL per proportion, so eval reads a single mixed-task stream and the online
     state S evolves continuously across task boundaries.

Strong/weak peer is still a PER-DATASET notion (argmax of per-peer accuracy on that
dataset), computed before pooling. Fixed eval set T per dataset (same record set for
every proportion) so accuracy-vs-proportion is a clean curve.

Output: data/counterfactual_3peer/cf_{0,50,70,90}.jsonl and build_stats.json
Every record carries task_type/dataset/source and (if swapped) the CF tags the
evaluators read (target_peers, counterfactual_type, strong_peer, swapped_with, ...).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# Reuse the validated swap mechanics from the per-dataset builder.
from data.builders.counterfactual.build_v3_counterfactuals import (
    DATASETS,
    PROPORTIONS,
    MAX_P,
    _seed,
    _load,
    peer_keys_of,
    binary_correct,
    per_peer_accuracy,
    is_swappable,
    mark_clean,
)


def make_swapped_with_target(
    record: dict[str, Any], strong: str, weaks: list[str], w_star: str
) -> dict[str, Any] | None:
    """Swap strong's correct response with a SPECIFIC wrong weak peer (w_star).

    Same as build_v3_counterfactuals.make_swapped but the weak target is given (for
    balanced assignment) rather than randomly chosen. Returns None if the swap does
    not yield strong_wrong_weak_correct.
    """
    keys = peer_keys_of(record)
    if strong not in keys or w_star not in keys:
        return None
    if not binary_correct(record, strong) or binary_correct(record, w_star):
        return None
    out = json.loads(json.dumps(record))  # deep copy
    resp = out["peer_responses"]
    resp[strong], resp[w_star] = resp[w_star], resp[strong]
    if isinstance(out.get("peer_correct"), dict) and strong in out["peer_correct"] and w_star in out["peer_correct"]:
        pc = out["peer_correct"]
        pc[strong], pc[w_star] = pc[w_star], pc[strong]
    # peer_metadata is NOT swapped: identity stays bound to the peer key.
    correct_after = [k for k in keys if binary_correct(out, k)]
    weak_correct = [k for k in weaks if k in correct_after]
    if binary_correct(out, strong) or not weak_correct:
        return None
    num_correct = len(correct_after)
    out["synthetic_counterfactual"] = True
    out["counter_trust"] = True
    out["counterfactual_type"] = "strong_wrong_weak_correct"
    out["strict_counterfactual"] = num_correct == 1
    out["strong_peer"] = strong
    out["weak_peers"] = list(weaks)
    out["target_peers"] = list(weak_correct)
    out["swapped_with"] = w_star
    out["is_swapped"] = True
    out["synthesis_mode"] = "response_swap"
    return out


def balanced_targets(records, swap_indices, strong, weaks, seed):
    """Assign a wrong-weak target to each swap index, balanced across weak peers.

    Greedy: process indices in a seeded order; for each, among its AVAILABLE wrong
    weak peers pick the one used least so far (tie-break by seeded order then key).
    Returns {index: w_star}. Indices whose available set is empty are skipped.
    """
    order = list(swap_indices)
    random.Random(seed).shuffle(order)
    used = {w: 0 for w in weaks}
    assign: dict[int, str] = {}
    for i in order:
        rec = records[i]
        keys = peer_keys_of(rec)
        avail = [w for w in weaks if w in keys and not binary_correct(rec, w)]
        if not avail:
            continue
        w_star = min(avail, key=lambda w: (used[w], w))
        assign[i] = w_star
        used[w_star] += 1
    return assign, used


def build_one_dataset(records, task, dataset, proportions, keep_all_records: bool = False):
    """Return {p_tag: [records]} for one dataset, with balanced weak targets, plus stats."""
    keys = peer_keys_of(records[0]) if records else []
    acc = per_peer_accuracy(records, keys)
    strong = max(keys, key=lambda k: (acc[k], k)) if keys else "peer_0"
    weaks = [k for k in keys if k != strong]

    swappable_idx = [i for i, r in enumerate(records) if is_swappable(r, strong, weaks)]
    S = len(swappable_idx)
    N = len(records)

    rng_t = random.Random(_seed(dataset, strong, "T", N, S))
    if keep_all_records:
        T = N
        n_swap_pool = S
    else:
        T = min(N, int(S / MAX_P)) if S > 0 else 0
        n_swap_pool = min(S, int(round(MAX_P * T)))
    n_clean_fill = max(0, T - n_swap_pool)
    swap_pool = list(swappable_idx)
    rng_t.shuffle(swap_pool)
    swap_pool = swap_pool[:n_swap_pool]
    non_swap = [i for i in range(N) if i not in set(swap_pool)]
    rng_t.shuffle(non_swap)
    clean_fill = non_swap[:n_clean_fill]

    out_by_p: dict[str, list[dict[str, Any]]] = {}
    stats_by_p: dict[str, Any] = {}
    for p in proportions:
        tag = f"p{int(round(p * 100))}"
        k_swap = min(len(swap_pool), int(round(p * T)))
        to_swap = swap_pool[:k_swap]
        clean_from_pool = swap_pool[k_swap:]
        # Balanced weak-target assignment over exactly the records we will swap.
        assign, used = balanced_targets(records, to_swap, strong, weaks, _seed(dataset, "bal", p))
        built: list[dict[str, Any]] = []
        for i in to_swap:
            w_star = assign.get(i)
            sw = make_swapped_with_target(records[i], strong, weaks, w_star) if w_star else None
            if sw is not None:
                sw["dataset"] = dataset
                sw["task_type"] = sw.get("task_type") or task
                built.append(sw)
            else:
                # Couldn't form the pattern -> keep clean so T stays fixed.
                cl = mark_clean(records[i]); cl["dataset"] = dataset
                cl["task_type"] = cl.get("task_type") or task
                built.append(cl)
        for i in list(clean_from_pool) + list(clean_fill):
            cl = mark_clean(records[i]); cl["dataset"] = dataset
            cl["task_type"] = cl.get("task_type") or task
            built.append(cl)
        n_sw = sum(1 for r in built if r.get("is_swapped"))
        # realized per-weak distribution among swapped records
        dist = defaultdict(int)
        for r in built:
            if r.get("is_swapped"):
                dist[r.get("swapped_with")] += 1
        out_by_p[tag] = built
        stats_by_p[tag] = {
            "requested_fraction": p,
            "num_total": len(built),
            "num_swapped": n_sw,
            "realized_fraction": (n_sw / len(built)) if built else 0.0,
            "weak_target_distribution": dict(dist),
        }
    stats = {
        "dataset": dataset, "task_type": task, "N": N, "peer_keys": keys,
        "per_peer_accuracy": acc, "strong_peer": strong, "weak_peers": weaks,
        "num_swappable": S, "fixed_eval_size_T": T,
        "keep_all_records": bool(keep_all_records),
        "proportions": stats_by_p,
    }
    return out_by_p, stats


TASK_OF = {
    "math500": "math", "amc": "math", "olympiadbench": "math", "college_math": "math",
    "humaneval": "code", "mbpp": "code", "livecodebench": "code",
    "hotpotqa": "rag", "triviaqa": "rag",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", default="data/counterfactual_source")
    ap.add_argument("--out_dir", default="data/counterfactual_3peer")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--proportions", nargs="*", type=float, default=PROPORTIONS)
    ap.add_argument(
        "--keep_all_records",
        action="store_true",
        help=(
            "Preserve every input record in each proportion. If a requested CF "
            "fraction is not fully feasible, keep the remaining records clean "
            "instead of shrinking the evaluation set."
        ),
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled: dict[str, list[dict[str, Any]]] = {f"p{int(round(p*100))}": [] for p in args.proportions}
    all_stats: dict[str, Any] = {}
    for ds in args.datasets:
        path = in_dir / f"{ds}.jsonl"
        if not path.exists():
            print(f"[unified] SKIP {ds}: {path} missing"); continue
        records = _load(path)
        # Stamp a globally-unique uid (some source files, e.g. amc, reuse `id` across
        # genuinely-different problems). uid = dataset:rowindex; original id untouched.
        for j, r in enumerate(records):
            r["uid"] = f"{ds}:{j}"
        task = TASK_OF.get(ds, "math")
        out_by_p, stats = build_one_dataset(
            records, task, ds, args.proportions, keep_all_records=args.keep_all_records
        )
        all_stats[ds] = stats
        for tag, recs in out_by_p.items():
            pooled[tag].extend(recs)
        print(f"[unified] {ds:<14} task={task} strong={stats['strong_peer']} T={stats['fixed_eval_size_T']}")

    # Pool + shuffle each proportion into one mixed-task stream.
    pooled_stats: dict[str, Any] = {}
    for tag, recs in pooled.items():
        random.Random(_seed("unified", "pool", tag)).shuffle(recs)
        outp = out_dir / f"cf_{tag.removeprefix('p')}.jsonl"
        with outp.open("w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_sw = sum(1 for r in recs if r.get("is_swapped"))
        tmix = defaultdict(int)
        for r in recs:
            tmix[r.get("task_type")] += 1
        pooled_stats[tag] = {
            "num_total": len(recs), "num_swapped": n_sw,
            "realized_fraction": (n_sw / len(recs)) if recs else 0.0,
            "task_mix": dict(tmix),
        }
        print(f"[unified] wrote {outp}  n={len(recs)} swapped={n_sw} ({100*n_sw/max(1,len(recs)):.1f}%)")

    (out_dir / "build_stats.json").write_text(
        json.dumps({"per_dataset": all_stats, "pooled": pooled_stats}, indent=2)
    )
    print(f"[unified] stats -> {out_dir/'build_stats.json'}")


if __name__ == "__main__":
    main()
