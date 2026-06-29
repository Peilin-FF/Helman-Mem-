#!/usr/bin/env python3
"""Build synthetic counterfactual eval sets by SWAPPING strong<->weak peer responses
based on correctness (task-agnostic; math + code + rag).

Idea (teacher's spec)
---------------------
For each dataset, the historically STRONG peer is fixed by task type
(math->gemma/peer_0, code->qwen/peer_2, rag->phi/peer_1). A *counterfactual* record
is one where the strong peer is CORRECT and at least one weak peer is WRONG; we SWAP
the strong peer's (correct) response with that wrong weak peer's response. After the
swap the strong slot shows WRONG content and the weak slot shows CORRECT content
(``strong_wrong_weak_correct``). Identity stays bound to the peer KEY (peer_metadata
is untouched), so a response-BLIND selector that trusts the strong peer's identity now
picks the wrong answer, while a response-READING selector can still find the correct
one in the weak slot.

Why swap ``peer_correct`` too: the eval reads ``record["peer_correct"]`` for math/code
(and soft-F1 for rag) via ``peer_target_value``. Swapping the response text WITHOUT
swapping ``peer_correct`` would desync the label from the content. We swap both, and
also set ``target_peers`` (the now-correct peer key[s]) so the evaluators define CF
accuracy by key membership regardless of the grading path.

Proportions: mix swapped (special) + clean (normal, sampled from the OTHER originals,
so no problem appears twice) so the swapped fraction is exactly p in {0.50,0.70,0.90}.
The 0% baseline is the existing clean eval (data/v3/graded or data/v3).

Output: data/v3_cf/p{50,70,90}/{dataset}.jsonl + data/v3_cf/cf_build_stats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from feedback_state.tasks import peer_is_correct, peer_target_value, task_type_of

# Strong peer is chosen PER DATASET as the argmax of each peer's own accuracy on that
# dataset (the spec: "strong/weak defined per dataset by each peer's own accuracy").
# This is the oracle best-fixed peer — the strongest response-blind target — so the
# counterfactual swap attacks exactly the peer a blind selector would trust. The
# typical outcome matches the illustrative domain map below (math->gemma, code->qwen,
# rag->phi) but per-dataset argmax also keeps low-overlap sets (e.g. livecodebench,
# where qwen scores ~0.01) feasible by donating from the genuinely-strong peer.
DOMAIN_STRONG = {"math": "peer_0", "code": "peer_2", "rag": "peer_1"}  # reported for context

DATASETS = [
    "math500", "amc", "olympiadbench", "college_math",   # math
    "humaneval", "mbpp", "livecodebench",                 # code
    "hotpotqa", "triviaqa",                               # rag
]
# 0.0 = on-T clean baseline (same record set as the CF sets); 0.5/0.7/0.9 = CF fractions.
PROPORTIONS = [0.0, 0.50, 0.70, 0.90]
MAX_P = 0.90  # the highest proportion drives the fixed-T size (T = floor(S / MAX_P))


def _seed(*parts: Any) -> int:
    h = hashlib.sha256("::".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def peer_keys_of(record: dict[str, Any]) -> list[str]:
    return sorted(dict(record.get("peer_responses", {})))


def binary_correct(record: dict[str, Any], key: str) -> bool:
    """Binary per-peer correctness on the SAME ruler the joint evaluator uses.

    The evaluator's ``correctness_by_peer`` is ``peer_target_value(...) > 0.5`` for
    every task: math/code read the precomputed binary ``peer_correct``; rag reads the
    soft token-F1 stored in ``peer_correct`` (so F1>0.5 == "correct" here too). We
    grade the swap, ``target_peers``, and feasibility on this exact rule so the labels
    we tag never disagree with how accuracy is later computed. (The reported-accuracy
    EM grader ``peer_is_correct`` is stricter on rag, but the swap moves whole
    responses+labels together, so strong-wrong / weak-correct holds under both.)
    """
    text = str(dict(record.get("peer_responses", {})).get(key, ""))
    return peer_target_value(record, key, text) > 0.5


def per_peer_accuracy(records: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    cor = defaultdict(float)
    tot = defaultdict(float)
    for r in records:
        for k in keys:
            if k in dict(r.get("peer_responses", {})):
                tot[k] += 1.0
                cor[k] += 1.0 if binary_correct(r, k) else 0.0
    return {k: (cor[k] / tot[k] if tot[k] else 0.0) for k in keys}


def is_swappable(record: dict[str, Any], strong: str, weaks: list[str]) -> bool:
    keys = peer_keys_of(record)
    if strong not in keys or len(keys) < 2:
        return False
    if not binary_correct(record, strong):
        return False  # strong must be correct to donate a correct response
    return any((w in keys) and (not binary_correct(record, w)) for w in weaks)


def make_swapped(record: dict[str, Any], strong: str, weaks: list[str], rng: random.Random) -> dict[str, Any] | None:
    """Swap strong's correct response with a wrong weak peer's response (+ labels)."""
    keys = peer_keys_of(record)
    wrong_weaks = [w for w in weaks if w in keys and not binary_correct(record, w)]
    if not (strong in keys and binary_correct(record, strong) and wrong_weaks):
        return None
    w_star = rng.choice(sorted(wrong_weaks))  # deterministic given the seed

    out = json.loads(json.dumps(record))  # deep copy
    resp = out["peer_responses"]
    resp[strong], resp[w_star] = resp[w_star], resp[strong]
    if isinstance(out.get("peer_correct"), dict) and strong in out["peer_correct"] and w_star in out["peer_correct"]:
        pc = out["peer_correct"]
        pc[strong], pc[w_star] = pc[w_star], pc[strong]
    # peer_metadata is NOT swapped: identity stays bound to the peer key.

    # Re-derive correctness AFTER the swap to label the counterfactual.
    correct_after = [k for k in keys if binary_correct(out, k)]
    weak_correct = [k for k in weaks if k in correct_after]
    if binary_correct(out, strong) or not weak_correct:
        return None  # swap did not produce strong_wrong_weak_correct -> discard
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


def mark_clean(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out["synthetic_counterfactual"] = False
    out["is_swapped"] = False
    # Leave any naturally-occurring counterfactual_type intact for diagnostics.
    return out


def build_dataset(records: list[dict[str, Any]], task: str, proportions: list[float]) -> dict[str, Any]:
    """Build nested CF eval sets over ONE FIXED record set T per dataset.

    To make "accuracy vs proportion" a clean curve, every proportion uses the SAME
    T records — only HOW MANY of them are swapped changes. T is sized so the highest
    proportion is reachable: T = min(N, floor(S / MAX_P)), split into a swappable
    pool (the first |T|*MAX_P swappable records) and a clean filler. At fraction p we
    swap the first round(p*|T|) records of the swappable pool and leave the rest clean;
    the clean filler is always clean. p=0 -> all-clean baseline on the same T.
    """
    keys = peer_keys_of(records[0]) if records else []
    acc = per_peer_accuracy(records, keys)
    # Strong = per-dataset argmax (the oracle best-fixed peer). Tie-break by key.
    strong = max(keys, key=lambda k: (acc[k], k)) if keys else "peer_0"
    weaks = [k for k in keys if k != strong]
    strong_argmax = strong
    strong_domain = DOMAIN_STRONG.get(task, "peer_0")

    swappable_idx = [i for i, r in enumerate(records) if is_swappable(r, strong, weaks)]
    S = len(swappable_idx)
    N = len(records)

    # Fixed eval set T (same indices for every proportion).
    rng_t = random.Random(_seed(task, strong, "T", N, S))
    T = min(N, int(S / MAX_P)) if S > 0 else 0
    n_swap_pool = min(S, int(round(MAX_P * T)))  # swappable slots inside T
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
        k_swap = min(len(swap_pool), int(round(p * T)))  # how many of T are swapped
        swapped_ids = set(swap_pool[:k_swap])
        clean_from_pool = swap_pool[k_swap:]  # remaining swappable, kept CLEAN

        built: list[dict[str, Any]] = []
        for i in swap_pool[:k_swap]:
            sw = make_swapped(records[i], strong, weaks,
                              random.Random(_seed(task, "rec", records[i].get("id", i))))
            if sw is not None:
                built.append(sw)
        for i in list(clean_from_pool) + list(clean_fill):
            built.append(mark_clean(records[i]))
        random.Random(_seed(task, "shuffle", p)).shuffle(built)

        n_sw = sum(1 for r in built if r.get("is_swapped"))
        out_by_p[tag] = built
        stats_by_p[tag] = {
            "requested_fraction": p,
            "num_total": len(built),
            "num_swapped": n_sw,
            "num_clean": len(built) - n_sw,
            "realized_fraction": (n_sw / len(built)) if built else 0.0,
        }

    stats = {
        "N": N,
        "task_type": task,
        "peer_keys": keys,
        "per_peer_accuracy": acc,
        "strong_peer": strong,
        "strong_peer_domain_hint": strong_domain,
        "strong_peer_argmax": strong_argmax,
        "num_swappable": S,
        "swappable_fraction": (S / N if N else 0.0),
        "fixed_eval_size_T": T,
        "max_proportion_feasible": (n_swap_pool / T if T else 0.0),
        "proportions": stats_by_p,
    }
    return {"out_by_p": out_by_p, "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", default="data/v3/graded", help="dir with graded {ds}.jsonl (has peer_correct)")
    ap.add_argument("--out_dir", default="data/v3_cf")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--proportions", nargs="*", type=float, default=PROPORTIONS)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    all_stats: dict[str, Any] = {}
    for ds in args.datasets:
        path = in_dir / f"{ds}.jsonl"
        if not path.exists():
            print(f"[cf] SKIP {ds}: {path} missing")
            continue
        records = _load(path)
        task = task_type_of(records[0]) if records else "math"
        result = build_dataset(records, task, args.proportions)
        for tag, built in result["out_by_p"].items():
            outp = out_dir / tag / f"{ds}.jsonl"
            outp.parent.mkdir(parents=True, exist_ok=True)
            with outp.open("w") as f:
                for r in built:
                    f.write(json.dumps(r) + "\n")
        all_stats[ds] = result["stats"]
        st = result["stats"]
        fr = " ".join(f"{tag}:{result['stats']['proportions'][tag]['num_total']}@{result['stats']['proportions'][tag]['realized_fraction']:.2f}"
                      for tag in result["out_by_p"])
        print(f"[cf] {ds:14} task={task:4} strong={st['strong_peer']}"
              f"(domain-hint {st['strong_peer_domain_hint']}) swappable={st['num_swappable']}/{st['N']}"
              f" ({st['swappable_fraction']*100:.1f}%)  -> {fr}")

    stats_path = out_dir / "cf_build_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(all_stats, indent=2))
    print(f"[cf] wrote stats -> {stats_path}")


if __name__ == "__main__":
    main()
