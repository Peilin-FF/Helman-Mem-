"""Feedback-sparsity ablation for direct OOD M-route and M-vote.

The experiment reuses frozen question-only phi caches and learned gamma/eta from
the established OOD runs, including PIQA in the original OOD trajectory.  For a
randomly selected fraction of events, the post-decision correctness
vector contributes the usual rank-one innovation; unobserved events still
advance the learned per-event decay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from eval_ood_memory_routing import (
    CANONICAL_ORDERED_ID_SHA256,
    CANONICAL_STREAM_SHA256,
    CANONICAL_STREAM_SIZE,
    PAPER_OOD_GROUPS,
    PAPER_OOD_ORDERED_ID_SHA256,
    PAPER_OOD_SIZE,
    PEER_KEYS,
    RUN_PROFILES,
    _decision_record,
    _frozen_vote_correctness,
    _ordered_id_sha256,
    _record_answers,
    _record_labels,
    _sha256_file,
)
from feedback_state.data import JsonlDataset
from feedback_state.ood_routing import (
    OODRoutingState,
    _answer_tie_key,
    canonical_answer,
)


PROFILES = ("q35_4b", "q35_9b")
FEEDBACK_PERCENTS = (5, 10, 20, 50, 80, 100)
SEEDS = (0, 1, 2)
FULL_100_PERCENT_CORRECT = {
    "q35_4b": {"route": 10_558, "vote": 10_591},
    "q35_9b": {"route": 10_562, "vote": 10_592},
}
FULL_MAJORITY_CORRECT = 10_289


@dataclass(frozen=True)
class PreparedEvent:
    original_index: int
    identifier: str
    source: str
    paper_ood: bool
    canonical_answers: tuple[str, ...]
    answer_tie_keys: Mapping[str, tuple]
    correctness: np.ndarray
    correctness_signed: np.ndarray


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _mask_sha256(mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(mask.size).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.packbits(mask.astype(np.uint8)).tobytes())
    return digest.hexdigest()


def make_feedback_masks(
    num_events: int,
    *,
    percents: Sequence[int] = FEEDBACK_PERCENTS,
    seeds: Sequence[int] = SEEDS,
) -> dict[tuple[int, int], np.ndarray]:
    """Create exact-quota, nested event masks from one permutation per seed."""

    if num_events < 1:
        raise ValueError("num_events must be positive")
    normalized = tuple(int(percent) for percent in percents)
    if any(percent <= 0 or percent > 100 for percent in normalized):
        raise ValueError("feedback percents must lie in (0, 100]")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("feedback percents must be unique and increasing")

    masks: dict[tuple[int, int], np.ndarray] = {}
    for seed in seeds:
        permutation = np.random.default_rng(int(seed)).permutation(num_events)
        previous = np.zeros(num_events, dtype=bool)
        for percent in normalized:
            count = int(round(num_events * percent / 100.0))
            mask = np.zeros(num_events, dtype=bool)
            mask[permutation[:count]] = True
            if np.any(previous & ~mask):
                raise AssertionError("feedback masks must be nested within each seed")
            masks[(int(seed), percent)] = mask
            previous = mask
    return masks


def _load_and_prepare_events(
    path: Path,
) -> tuple[list[dict[str, Any]], list[PreparedEvent]]:
    if _sha256_file(path) != CANONICAL_STREAM_SHA256:
        raise ValueError("Canonical OOD stream SHA256 changed")
    records = JsonlDataset(path).records
    if len(records) != CANONICAL_STREAM_SIZE:
        raise ValueError("Canonical OOD stream size changed")
    if _ordered_id_sha256(records) != CANONICAL_ORDERED_ID_SHA256:
        raise ValueError("Canonical OOD stream order changed")

    events: list[PreparedEvent] = []
    for original_index, record in enumerate(records):
        source = str(record.get("source") or "")
        answers = _record_answers(record)
        decision_record = _decision_record(record)
        canonical = tuple(
            canonical_answer(decision_record, answer) for answer in answers
        )
        correctness, correctness_signed = _record_labels(record)
        events.append(
            PreparedEvent(
                original_index=original_index,
                identifier=str(record.get("id") or record.get("uid") or ""),
                source=source,
                paper_ood=source in PAPER_OOD_GROUPS,
                canonical_answers=canonical,
                answer_tie_keys={
                    answer: _answer_tie_key(decision_record, answer)
                    for answer in set(canonical)
                },
                correctness=correctness,
                correctness_signed=correctness_signed,
            )
        )
    return records, events


def _select_canonical_vote(event: PreparedEvent, weights: Sequence[float]) -> str:
    if len(weights) != len(event.canonical_answers):
        raise ValueError("vote weights do not match the peer count")
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for answer, weight in zip(event.canonical_answers, weights):
        value = float(weight)
        if not math.isfinite(value):
            raise ValueError("vote weights must be finite")
        grouped[answer].append(value)
    totals = {
        answer: float(math.fsum(values)) for answer, values in grouped.items()
    }
    best = max(totals.values())
    tied = [answer for answer, score in totals.items() if score == best]
    return min(tied, key=lambda answer: event.answer_tie_keys[answer])


def _empty_scope() -> dict[str, int]:
    return {"route_correct": 0, "vote_correct": 0, "total": 0}


def replay_sparse_feedback(
    *,
    events: Sequence[PreparedEvent],
    phis: np.ndarray,
    feedback_mask: np.ndarray,
    rank: int,
    gamma: float,
    eta: float,
    capture_trace: bool = False,
) -> dict[str, Any]:
    """Replay decisions on every event while masking only feedback innovations."""

    if feedback_mask.shape != (len(events),):
        raise ValueError("feedback mask length does not match the replay trajectory")
    state = OODRoutingState(
        num_peers=len(PEER_KEYS),
        rank=rank,
        gamma=gamma,
        eta=eta,
        gamma_g=0.9,
        eta_g=0.1,
    )
    scopes = {"full_stream": _empty_scope(), "paper_ood": _empty_scope()}
    selections: Counter[int] = Counter()
    route_ties = 0
    paper_feedback = 0
    trace: list[dict[str, Any]] | None = [] if capture_trace else None

    for event_index, event in enumerate(events):
        phi = np.array(phis[event.original_index], dtype=np.float64, copy=True)
        phi /= np.linalg.norm(phi)

        route = state.route(phi, event_index)
        vote_answer = _select_canonical_vote(event, route.scores)
        route_correct = int(event.correctness[route.peer])
        vote_correct = _frozen_vote_correctness(
            vote_answer, list(event.canonical_answers), event.correctness
        )[0]

        for scope_name in ("full_stream", "paper_ood"):
            if scope_name == "paper_ood" and not event.paper_ood:
                continue
            scopes[scope_name]["route_correct"] += route_correct
            scopes[scope_name]["vote_correct"] += vote_correct
            scopes[scope_name]["total"] += 1
        selections[route.peer] += 1
        route_ties += int(route.tied)
        observed = bool(feedback_mask[event_index])
        paper_feedback += int(observed and event.paper_ood)
        if trace is not None:
            trace.append(
                {
                    "id": event.identifier,
                    "route_peer": route.peer,
                    "route_scores": route.scores.tolist(),
                    "route_correct": route_correct,
                    "vote_answer": vote_answer,
                    "vote_correct": int(vote_correct),
                    "feedback_observed": observed,
                }
            )

        if observed:
            state.update(phi, event.correctness_signed)
        else:
            state.decay_without_feedback()

    final_m, _ = state.snapshot()
    metrics: dict[str, Any] = {}
    for scope_name, values in scopes.items():
        total = values["total"]
        metrics[scope_name] = {
            **values,
            "route_accuracy": values["route_correct"] / total,
            "vote_accuracy": values["vote_correct"] / total,
        }
    return {
        "metrics": metrics,
        "feedback_count": int(feedback_mask.sum()),
        "paper_ood_feedback_count": paper_feedback,
        "route_selected_peers": {
            str(peer): int(selections[peer]) for peer in range(len(PEER_KEYS))
        },
        "route_all_equal_ties": route_ties,
        "final_M_frobenius_norm": float(np.linalg.norm(final_m)),
        "trace": trace,
    }


def _majority_metrics(events: Sequence[PreparedEvent]) -> dict[str, Any]:
    weights = np.ones(len(PEER_KEYS), dtype=np.float64) / len(PEER_KEYS)
    scopes = {"full_stream": [0, 0], "paper_ood": [0, 0]}
    for event in events:
        answer = _select_canonical_vote(event, weights)
        correct = _frozen_vote_correctness(
            answer, list(event.canonical_answers), event.correctness
        )[0]
        scopes["full_stream"][0] += int(correct)
        scopes["full_stream"][1] += 1
        if event.paper_ood:
            scopes["paper_ood"][0] += int(correct)
            scopes["paper_ood"][1] += 1
    return {
        scope: {
            "correct": values[0],
            "total": values[1],
            "accuracy": values[0] / values[1],
        }
        for scope, values in scopes.items()
    }


def _load_profile_phis(profile_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    profile = RUN_PROFILES[profile_name]
    output = Path("outputs/ood_memory_routing") / profile_name
    phi_path = output / "phis.npy"
    manifest_path = output / "phi_manifest.json"
    if not phi_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete established OOD output for {profile_name}")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "offline_data_sha256": CANONICAL_STREAM_SHA256,
        "checkpoint_sha256": profile.checkpoint_sha256,
        "central_model": profile.central_model,
        "num_examples": CANONICAL_STREAM_SIZE,
        "gamma": profile.gamma,
        "eta": profile.eta,
    }
    changed = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if changed:
        raise ValueError(f"OOD phi provenance changed for {profile_name}: {changed}")
    if manifest.get("phi_file_sha256") != _sha256_file(phi_path):
        raise ValueError(f"OOD phi cache hash changed for {profile_name}")
    phis = np.load(phi_path, allow_pickle=False)
    if phis.shape[0] != CANONICAL_STREAM_SIZE or not np.isfinite(phis).all():
        raise ValueError(f"Invalid OOD phi cache for {profile_name}: {phis.shape}")
    return phis, manifest


def _curve_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in ("route", "vote"):
        by_percent: dict[str, Any] = {}
        for percent in FEEDBACK_PERCENTS:
            selected = [run for run in runs if int(run["feedback_percent"]) == percent]
            accuracies = np.asarray(
                [run[f"{method}_paper_ood_accuracy"] for run in selected],
                dtype=np.float64,
            )
            by_percent[str(percent)] = {
                "seed_accuracies": accuracies.tolist(),
                "mean_accuracy": float(accuracies.mean()),
                "std_accuracy": float(accuracies.std(ddof=0)),
                "min_accuracy": float(accuracies.min()),
                "max_accuracy": float(accuracies.max()),
            }
        result[method] = by_percent
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline-data",
        type=Path,
        default=Path(
            "data/peer_generalization/hf_generalization_all_canonical_peers/"
            "hf_generalization_all_peer012.labeled.jsonl"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/ood_feedback_sparsity_with_piqa")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, events = _load_and_prepare_events(args.offline_data)
    paper_records = [
        record
        for record in records
        if str(record.get("source") or "") in PAPER_OOD_GROUPS
    ]
    if len(paper_records) != PAPER_OOD_SIZE:
        raise ValueError("Paper OOD size changed")
    if _ordered_id_sha256(paper_records) != PAPER_OOD_ORDERED_ID_SHA256:
        raise ValueError("Paper OOD order changed")
    if not any(event.source == "piqa" for event in events):
        raise AssertionError("PIQA must remain in the replay and feedback trajectory")

    masks = make_feedback_masks(len(events))
    mask_manifest = {
        f"seed_{seed}_feedback_{percent}": {
            "feedback_count": int(mask.sum()),
            "realized_fraction": float(mask.mean()),
            "mask_sha256": _mask_sha256(mask),
            "paper_ood_feedback_count": int(
                sum(
                    bool(mask[index]) and event.paper_ood
                    for index, event in enumerate(events)
                )
            ),
        }
        for (seed, percent), mask in sorted(masks.items())
    }
    majority = _majority_metrics(events)
    if majority["paper_ood"]["correct"] != FULL_MAJORITY_CORRECT:
        raise ValueError("Majority control changed")
    profile_results: dict[str, Any] = {}
    all_run_rows: list[dict[str, Any]] = []

    for profile_name in PROFILES:
        print(f"[feedback-sparsity] running {profile_name}", flush=True)
        profile = RUN_PROFILES[profile_name]
        phis, phi_manifest = _load_profile_phis(profile_name)
        runs: list[dict[str, Any]] = []
        for seed in SEEDS:
            for percent in FEEDBACK_PERCENTS:
                mask = masks[(seed, percent)]
                replay = replay_sparse_feedback(
                    events=events,
                    phis=phis,
                    feedback_mask=mask,
                    rank=int(phis.shape[1]),
                    gamma=profile.gamma,
                    eta=profile.eta,
                )
                full = replay["metrics"]["full_stream"]
                paper = replay["metrics"]["paper_ood"]
                row = {
                    "profile": profile_name,
                    "seed": seed,
                    "feedback_percent": percent,
                    "feedback_count": replay["feedback_count"],
                    "paper_ood_feedback_count": replay["paper_ood_feedback_count"],
                    "mask_sha256": _mask_sha256(mask),
                    "route_full_accuracy": full["route_accuracy"],
                    "route_full_correct": full["route_correct"],
                    "route_paper_ood_accuracy": paper["route_accuracy"],
                    "route_paper_ood_correct": paper["route_correct"],
                    "vote_full_accuracy": full["vote_accuracy"],
                    "vote_full_correct": full["vote_correct"],
                    "vote_paper_ood_accuracy": paper["vote_accuracy"],
                    "vote_paper_ood_correct": paper["vote_correct"],
                    "route_selected_peers": replay["route_selected_peers"],
                    "route_all_equal_ties": replay["route_all_equal_ties"],
                    "final_M_frobenius_norm": replay["final_M_frobenius_norm"],
                }
                runs.append(row)
                all_run_rows.append(row)
                print(
                    f"[feedback-sparsity] {profile_name} seed={seed} "
                    f"feedback={percent}% route={paper['route_accuracy'] * 100:.2f}% "
                    f"vote={paper['vote_accuracy'] * 100:.2f}%",
                    flush=True,
                )
        for method in ("route", "vote"):
            full_feedback = [
                run[f"{method}_paper_ood_correct"]
                for run in runs
                if run["feedback_percent"] == 100
            ]
            if len(set(full_feedback)) != 1:
                raise ValueError(f"{profile_name} 100% {method} differs across seeds")
            expected_correct = FULL_100_PERCENT_CORRECT[profile_name][method]
            if full_feedback[0] != expected_correct:
                raise ValueError(
                    f"{profile_name} 100% {method} changed: "
                    f"{full_feedback[0]} != {expected_correct}"
                )
        profile_results[profile_name] = {
            "profile": {
                "central_model": profile.central_model,
                "checkpoint_sha256": profile.checkpoint_sha256,
                "phi_file_sha256": phi_manifest["phi_file_sha256"],
                "gamma": profile.gamma,
                "eta": profile.eta,
                "runtime_M_initialization": "zeros",
            },
            "runs": runs,
            "curve": _curve_summary(runs),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    runs_path = args.output / "runs.jsonl"
    temporary = runs_path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as handle:
        for row in all_run_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(runs_path)
    summary = {
        "status": "complete",
        "experiment": "ood_feedback_sparsity_direct_M_with_piqa",
        "profiles": list(PROFILES),
        "feedback_percents": list(FEEDBACK_PERCENTS),
        "seeds": list(SEEDS),
        "method": {
            "feedback_sampling_unit": "event correctness vector for all peers",
            "mask_construction": "exact quota; nested prefixes of one permutation per seed",
            "mask_shared_across_models": True,
            "unobserved_event_update": "decay only; label innovation masked",
            "decision_then_score_then_feedback_update": True,
            "phi_input": "question only",
            "checkpoint_runtime_M_G_loaded": False,
            "training_performed": False,
        },
        "stream": {
            "path": str(args.offline_data),
            "sha256": CANONICAL_STREAM_SHA256,
            "num_examples": CANONICAL_STREAM_SIZE,
            "ordered_id_sha256": CANONICAL_ORDERED_ID_SHA256,
            "piqa_included_in_decisions": True,
            "piqa_eligible_for_feedback": True,
            "paper_ood_num_examples": PAPER_OOD_SIZE,
            "paper_ood_ordered_id_sha256": PAPER_OOD_ORDERED_ID_SHA256,
            "physical_relative_order_preserved": True,
        },
        "feedback_masks": mask_manifest,
        "majority_control": majority,
        "full_100_percent_reference_correct": FULL_100_PERCENT_CORRECT,
        "results": profile_results,
        "runs_sha256": _sha256_file(runs_path),
    }
    _write_json_atomic(args.output / "summary.json", summary)
    print(f"[feedback-sparsity] wrote {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
