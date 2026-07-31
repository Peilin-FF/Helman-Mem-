#!/usr/bin/env python3
"""Evaluate mixed-trained discounted Beta B1 routing on canonical CF streams."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from feedback_state.ood_routing import DiscountedBetaRoute


PEER_KEYS = ("peer_0", "peer_1", "peer_2")
CF_SPLITS = ("cf_0", "cf_50", "cf_70", "cf_90")
DEFAULT_WARM_DATA = Path("data/mixed_train/train.jsonl")
DEFAULT_CF_DIR = Path("data/counterfactual_3peer")
DEFAULT_OUTPUT = Path("outputs/cf_beta_b1/summary.json")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peer_scores(record: Mapping[str, Any]) -> list[float]:
    table = record.get("correctness_by_peer") or record.get("peer_correct")
    if not isinstance(table, Mapping):
        raise ValueError(f"record {record.get('id', '<unknown>')} has no peer labels")
    values = [float(table[key]) for key in PEER_KEYS]
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("peer correctness scores must lie in [0, 1]")
    return values


def _mixed_correctness(record: Mapping[str, Any]) -> list[int]:
    values = _peer_scores(record)
    if any(value not in (0.0, 1.0) for value in values):
        raise ValueError("mixed warm-start requires binary correctness labels")
    return [int(value) for value in values]


def _cf_correctness(record: Mapping[str, Any]) -> list[int]:
    """Match the established CF evaluator's int(round(raw_score)) protocol."""

    return [int(round(value)) for value in _peer_scores(record)]


def _warm_state(
    records: Sequence[dict[str, Any]], *, gamma: float
) -> DiscountedBetaRoute:
    state = DiscountedBetaRoute(num_peers=len(PEER_KEYS), gamma=gamma)
    for record in records:
        state.update(_mixed_correctness(record))
    return state


def evaluate_split(
    warm_records: Sequence[dict[str, Any]],
    cf_records: Sequence[dict[str, Any]],
    *,
    gamma: float,
) -> dict[str, Any]:
    state = _warm_state(warm_records, gamma=gamma)
    warm_alpha, warm_beta = state.snapshot()
    correct = 0
    ties = 0
    picks: Counter[int] = Counter()
    by_task: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_dataset: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

    for record in cf_records:
        # The current question and peer responses are never passed to the router.
        decision = state.route()
        labels = _cf_correctness(record)
        selected_correct = labels[decision.peer]
        correct += selected_correct
        ties += int(decision.tied)
        picks[decision.peer] += 1
        for table, key in (
            (by_task, str(record.get("task_type") or "unknown")),
            (by_dataset, str(record.get("dataset") or record.get("source") or "unknown")),
        ):
            table[key][0] += selected_correct
            table[key][1] += 1
        state.update(labels)

    total = len(cf_records)
    final_alpha, final_beta = state.snapshot()

    def summarize(groups: Mapping[str, list[int]]) -> dict[str, Any]:
        return {
            name: {
                "correct": values[0],
                "total": values[1],
                "accuracy": values[0] / values[1],
            }
            for name, values in sorted(groups.items())
        }

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "selected_peers": [picks[index] for index in range(len(PEER_KEYS))],
        "all_equal_ties": ties,
        "by_task_type": summarize(by_task),
        "by_dataset": summarize(by_dataset),
        "warm_state": {
            "alpha": warm_alpha.tolist(),
            "beta": warm_beta.tolist(),
            "reliability": (warm_alpha / (warm_alpha + warm_beta)).tolist(),
        },
        "final_state": {
            "alpha": final_alpha.tolist(),
            "beta": final_beta.tolist(),
            "reliability": (final_alpha / (final_alpha + final_beta)).tolist(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-data", type=Path, default=DEFAULT_WARM_DATA)
    parser.add_argument("--cf-dir", type=Path, default=DEFAULT_CF_DIR)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warm_records = _read_jsonl(args.warm_data)
    streams = {
        split: _read_jsonl(args.cf_dir / f"{split}.jsonl")
        for split in CF_SPLITS
    }
    results = {
        split: evaluate_split(warm_records, records, gamma=args.gamma)
        for split, records in streams.items()
    }
    summary = {
        "experiment": "mixed_trained_discounted_beta_b1_cf_routing",
        "response_blind": True,
        "model_independent": True,
        "prior": {"alpha": 1.0, "beta": 1.0},
        "gamma": args.gamma,
        "update": (
            "alpha <- gamma * alpha + r; "
            "beta <- gamma * beta + (1-r)"
        ),
        "protocol": "mixed warm-start; reset per CF split; decide then update",
        "warm_data": {
            "path": str(args.warm_data),
            "sha256": _sha256(args.warm_data),
            "num_events": len(warm_records),
        },
        "cf_streams": {
            split: {
                "path": str(args.cf_dir / f"{split}.jsonl"),
                "sha256": _sha256(args.cf_dir / f"{split}.jsonl"),
                "num_events": len(records),
            }
            for split, records in streams.items()
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for split in CF_SPLITS:
        result = results[split]
        print(
            f"[cf-beta-b1] {split}: {result['correct']}/{result['total']} "
            f"({result['accuracy']:.2%}) picks={result['selected_peers']}"
        )


if __name__ == "__main__":
    main()
