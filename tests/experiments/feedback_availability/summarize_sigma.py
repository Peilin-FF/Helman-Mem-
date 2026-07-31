"""Summarize selective-feedback Sigma-Mem OOD runs.

The score streams are produced by
``python -m tests.experiments.feedback_availability.run_sigma``. This
script then replays Sigma+joint-G hard selection while masking
which event correctness vectors are allowed to update the history and graph.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tests.experiments.feedback_availability.m_route_vote import (
    FEEDBACK_PERCENTS,
    SEEDS,
    _load_and_prepare_events,
    _mask_sha256,
    make_feedback_masks,
)
from tests.experiments.selection_mechanisms.m_route_vote import (
    PEER_KEYS,
    _ordered_id_sha256,
    _sha256_file,
)
from tests.experiments.common.model_profiles import MODEL_RUNS


PROFILES = ("q35_4b", "q35_9b")
DEFAULT_DATA = Path("data/ood/test.jsonl")
SIGMA_WITH_G_REPLAY = {
    "center_weight": 0.6,
    "sigma_weight": 0.4,
    "unary_weight": 0.7,
    "history_weight": 0.8,
    "graph_weight": 0.5,
    "history_decay": 0.9,
    "history_eta": 0.1,
    "graph_decay": 0.9,
    "graph_eta": 0.1,
    "centered_graph_update": True,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _number_keyed_vector(
    row: Mapping[str, Any], field: str, *, dtype: np.dtype[Any]
) -> np.ndarray:
    mapping = row.get(field)
    if not isinstance(mapping, Mapping):
        raise ValueError(f"Score row is missing {field}")
    values = []
    for peer in range(len(PEER_KEYS)):
        key: str | int = str(peer) if str(peer) in mapping else peer
        if key not in mapping:
            raise ValueError(f"{field} is missing peer {peer}")
        values.append(mapping[key])
    return np.asarray(values, dtype=dtype)


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _curve_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    curve: dict[str, Any] = {}
    for percent in FEEDBACK_PERCENTS:
        selected = [run for run in runs if int(run["feedback_percent"]) == int(percent)]
        values = np.asarray([run["accuracy"] for run in selected], dtype=np.float64)
        curve[str(percent)] = {
            "seed_accuracies": values.tolist(),
            "mean_accuracy": float(values.mean()),
            "std_accuracy": float(values.std(ddof=0)),
            "min_accuracy": float(values.min()),
            "max_accuracy": float(values.max()),
        }
    return curve


def replay_sparse_sigma_with_g(
    *,
    center_path: Path,
    sigma_path: Path,
    records: list[dict[str, Any]],
    feedback_mask: np.ndarray,
) -> dict[str, Any]:
    center_rows = _read_jsonl(center_path)
    sigma_rows = _read_jsonl(sigma_path)
    if len(center_rows) != len(records) or len(sigma_rows) != len(records):
        raise ValueError("center/sigma score streams do not match the input stream length")
    if feedback_mask.shape != (len(records),):
        raise ValueError("feedback mask length does not match the input stream length")

    states = np.asarray(
        [
            [1.0 if (mask >> peer) & 1 else -1.0 for peer in range(len(PEER_KEYS))]
            for mask in range(1 << len(PEER_KEYS))
        ],
        dtype=np.float32,
    )
    history = np.zeros(len(PEER_KEYS), dtype=np.float32)
    graph = np.zeros((len(PEER_KEYS), len(PEER_KEYS)), dtype=np.float32)
    cfg = SIGMA_WITH_G_REPLAY
    correct = 0
    selections: Counter[int] = Counter()

    for index, (record, center_row, sigma_row) in enumerate(
        zip(records, center_rows, sigma_rows)
    ):
        expected_id = str(record.get("id") or record.get("uid") or "")
        if str(center_row.get("id") or "") != expected_id:
            raise ValueError(f"center ID mismatch at {index}: {expected_id}")
        if str(sigma_row.get("id") or "") != expected_id:
            raise ValueError(f"sigma ID mismatch at {index}: {expected_id}")

        center_scores = _number_keyed_vector(
            center_row, "peer_scores", dtype=np.dtype(np.float32)
        )
        sigma_scores = _number_keyed_vector(
            sigma_row, "peer_scores", dtype=np.dtype(np.float32)
        )
        peer_correct = _number_keyed_vector(
            sigma_row, "peer_correct", dtype=np.dtype(np.int8)
        )
        center_correct = _number_keyed_vector(
            center_row, "peer_correct", dtype=np.dtype(np.int8)
        )
        if not np.array_equal(center_correct, peer_correct):
            raise ValueError(f"center/sigma labels differ at {expected_id}")
        correctness_signed = np.where(peer_correct == 1, 1.0, -1.0).astype(np.float32)

        unary = (
            float(cfg["center_weight"]) * center_scores
            + float(cfg["sigma_weight"]) * sigma_scores
        )
        standardized = (unary - unary.mean()) / (unary.std() + 1e-6)
        field = (
            float(cfg["unary_weight"]) * standardized
            + float(cfg["history_weight"]) * history
        )
        logits = states @ field
        logits += 0.5 * float(cfg["graph_weight"]) * np.einsum(
            "bi,ij,bj->b", states, graph, states
        )
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        marginal_scores = probabilities @ states
        selected_peer = int(np.argmax(marginal_scores))
        selected_correct = int(peer_correct[selected_peer])

        correct += selected_correct
        selections[selected_peer] += 1

        observed = bool(feedback_mask[index])
        if observed:
            centered = correctness_signed
            if bool(cfg["centered_graph_update"]):
                centered = centered - centered.mean()
            history = (
                float(cfg["history_decay"]) * history
                + float(cfg["history_eta"]) * correctness_signed
            )
            graph = (
                float(cfg["graph_decay"]) * graph
                + float(cfg["graph_eta"]) * np.outer(centered, centered)
            )
        else:
            history = float(cfg["history_decay"]) * history
            graph = float(cfg["graph_decay"]) * graph
        np.fill_diagonal(graph, 0.0)

    metrics = {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
    }
    return {
        "metrics": metrics,
        "feedback_count": int(feedback_mask.sum()),
        "selected_peers": {str(peer): int(selections[peer]) for peer in range(len(PEER_KEYS))},
        "final_history": history.astype(np.float64).tolist(),
        "final_graph": graph.astype(np.float64).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-root", type=Path, default=Path("outputs/ood_sigma_feedback_sparsity")
    )
    parser.add_argument(
        "--direct-m-root", type=Path, default=Path("outputs/ood_feedback_sparsity")
    )
    parser.add_argument("--offline-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ood_sigma_feedback_sparsity/summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, events = _load_and_prepare_events(args.offline_data)

    masks = make_feedback_masks(len(events))
    direct_summary_path = args.direct_m_root / "summary.json"
    direct_summary = json.loads(direct_summary_path.read_text())

    profile_results: dict[str, Any] = {}
    run_rows: list[dict[str, Any]] = []
    for profile_name in PROFILES:
        profile = MODEL_RUNS[profile_name]
        center_path = args.score_root / profile_name / "center" / "selections.jsonl"
        runs: list[dict[str, Any]] = []
        for seed in SEEDS:
            for percent in FEEDBACK_PERCENTS:
                sigma_seed = 0 if int(percent) == 100 else int(seed)
                sigma_path = (
                    args.score_root
                    / profile_name
                    / f"sigma_p{percent}_s{sigma_seed}"
                    / "selections.jsonl"
                )
                mask = masks[(int(seed), int(percent))]
                replay = replay_sparse_sigma_with_g(
                    center_path=center_path,
                    sigma_path=sigma_path,
                    records=records,
                    feedback_mask=mask,
                )
                metrics = replay["metrics"]
                row = {
                    "profile": profile_name,
                    "seed": int(seed),
                    "feedback_percent": int(percent),
                    "feedback_count": replay["feedback_count"],
                    "mask_sha256": _mask_sha256(mask),
                    "accuracy": metrics["accuracy"],
                    "correct": metrics["correct"],
                    "selected_peers": replay["selected_peers"],
                    "sigma_score_stream": str(sigma_path),
                    "center_score_stream": str(center_path),
                }
                runs.append(row)
                run_rows.append(row)
                print(
                    f"[sigma-sparsity/summary] {profile_name} seed={seed} "
                    f"feedback={percent}% sigma_wG={metrics['accuracy'] * 100:.2f}%",
                    flush=True,
                )
        profile_results[profile_name] = {
            "profile": {
                **direct_summary["results"][profile_name]["profile"],
            },
            "runs": runs,
            "curve": _curve_summary(runs),
            "direct_M_route_curve": direct_summary["results"][profile_name]["curve"]["route"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    runs_path = args.output.parent / "sigma_with_g_runs.jsonl"
    tmp = runs_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as handle:
        for row in run_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(runs_path)

    summary = {
        "status": "complete",
        "experiment": "ood_feedback_sparsity_sigma_with_g_and_direct_M",
        "profiles": list(PROFILES),
        "feedback_percents": list(FEEDBACK_PERCENTS),
        "seeds": list(SEEDS),
        "method": {
            "sigma_mem": "residual-steered Sigma score stream plus joint-G replay",
            "direct_M_route": "question-only M readout hard routing, no peer answers",
            "majority": "uniform answer voting control",
            "feedback_sampling_unit": "event correctness vector for all peers",
            "mask_construction": "exact quota; nested prefixes of one permutation per seed",
            "unobserved_event_update": "decay only; label innovation masked",
            "decision_then_score_then_feedback_update": True,
            "training_performed": False,
            "score_streams_regenerated_for_sigma_mem": True,
            "center_score_stream_regenerated_once_per_model": True,
        },
        "stream": {
            "path": str(args.offline_data),
            "sha256": _sha256_file(args.offline_data),
            "num_examples": len(records),
            "ordered_id_sha256": _ordered_id_sha256(records),
        },
        "replay": dict(SIGMA_WITH_G_REPLAY),
        "majority_control": direct_summary["majority_control"],
        "direct_M_summary": str(direct_summary_path),
        "results": profile_results,
        "runs_sha256": _sha256_file(runs_path),
    }
    _write_json_atomic(args.output, summary)
    print(f"[sigma-sparsity/summary] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
