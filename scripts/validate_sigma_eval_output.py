#!/usr/bin/env python3
"""Validate Base and Sigma CF selection outputs and their provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feedback_state.prompt_protocol import (  # noqa: E402
    candidate_tokenization_format,
    prompt_context_format,
    prompt_protocol_name,
)
from feedback_state.permutations import canonical_peer_view  # noqa: E402


ARMS = (
    "base",
    "sigma_wo_g",
    "sigma_w_g",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AssertionError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def _record_id(row: Mapping[str, Any], fallback: int) -> str:
    # Evaluators serialize ``id`` when both identifiers are present. Some CF
    # rows retain a different legacy ``uid``, which must not look like reordering.
    return str(row.get("id") or row.get("uid") or fallback)


def _validate_common(
    metrics: Mapping[str, Any],
    selections: list[dict[str, Any]],
    benchmark: list[dict[str, Any]],
    benchmark_path: Path,
    *,
    legacy_prompt_protocol: bool,
) -> None:
    if len(selections) != len(benchmark):
        raise AssertionError(
            f"selection rows={len(selections)}, benchmark rows={len(benchmark)}"
        )
    if int(metrics.get("num_samples", -1)) != len(benchmark):
        raise AssertionError("metrics num_samples differs from benchmark")
    if metrics.get("offline_data_sha256") != _sha256(benchmark_path):
        raise AssertionError("benchmark SHA256 differs from metrics offline_data_sha256")
    expected_max_length = 8192 if legacy_prompt_protocol else 12288
    if int(metrics.get("max_length", -1)) != expected_max_length:
        raise AssertionError(
            f"metrics max_length={metrics.get('max_length')!r}, "
            f"expected {expected_max_length}"
        )
    expected_prompt = {
        "legacy_prompt_protocol": "on" if legacy_prompt_protocol else "off",
        "prompt_protocol": prompt_protocol_name(legacy_prompt_protocol),
        "prompt_context_format": prompt_context_format(legacy_prompt_protocol),
        "candidate_tokenization_format": candidate_tokenization_format(
            legacy_prompt_protocol
        ),
    }
    for field, expected in expected_prompt.items():
        if str(metrics.get(field)) != str(expected):
            raise AssertionError(
                f"metrics {field}={metrics.get(field)!r}, expected {expected!r}"
            )
    correct = 0
    num_peers = int(metrics.get("num_peers", -1))
    if num_peers < 1:
        raise AssertionError("metrics num_peers must be positive")
    for index, (selection, source) in enumerate(zip(selections, benchmark, strict=True), 1):
        selection_id = _record_id(selection, index)
        source_id = _record_id(source, index)
        if selection_id != source_id:
            raise AssertionError(
                f"row {index}: selection id={selection_id!r}, benchmark id={source_id!r}"
            )
        selected = int(selection.get("selected_peer", -1))
        view = canonical_peer_view(source, num_peers, setting="A")
        if selected < 0 or selected >= int(view["real"]):
            raise AssertionError(
                f"row {index}: selected_peer={selected} is outside the benchmark peers"
            )
        selected_correct = int(selection.get("selected_correct", -1))
        if selected_correct not in {0, 1}:
            raise AssertionError(f"row {index}: selected_correct must be binary")
        source_correctness = (
            source.get("correctness_by_peer") or source.get("peer_correct") or {}
        )
        expected_labels = {
            peer: int(round(float(source_correctness.get(key, 0))))
            for peer, key in enumerate(view["keys"][: int(view["real"])])
        }
        expected_selected_correct = expected_labels[selected]
        if selected_correct != expected_selected_correct:
            raise AssertionError(
                f"row {index}: selected_correct={selected_correct}, "
                f"expected {expected_selected_correct} from benchmark"
            )
        correct += selected_correct
    expected_accuracy = correct / len(selections) if selections else 0.0
    if not math.isclose(
        float(metrics.get("accuracy", float("nan"))),
        expected_accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("metrics accuracy differs from serialized selections")


def _validate_arm(
    arm: str,
    metrics: Mapping[str, Any],
    rows: list[dict[str, Any]],
    checkpoint: Path | None,
) -> None:
    if arm == "base":
        if metrics.get("checkpoint_sha256") not in {None, "", "null"}:
            raise AssertionError("base checkpoint_sha256 must be null")
        if metrics.get("ablate_memory") is not True:
            raise AssertionError("base metrics ablate_memory must be true")
        if str(metrics.get("graph_posterior")) != "off":
            raise AssertionError("base graph_posterior must be off")
        for index, row in enumerate(rows, 1):
            if row.get("memory_used") is not False or row.get("score_source") != "center":
                raise AssertionError(f"row {index}: invalid base selection provenance")
    else:
        if checkpoint is None:
            raise AssertionError("--checkpoint is required for Sigma arms")
        state_path = checkpoint / "sym_memory.pt"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        if metrics.get("checkpoint_sha256") != _sha256(state_path):
            raise AssertionError("checkpoint SHA256 differs from metrics")
        if metrics.get("ablate_memory") is not False:
            raise AssertionError("Sigma metrics ablate_memory must be false")
        expected_graph = "off" if arm.endswith("wo_g") else "ising"
        if str(metrics.get("graph_posterior")) != expected_graph:
            raise AssertionError(
                f"{arm} graph_posterior={metrics.get('graph_posterior')!r}, "
                f"expected {expected_graph!r}"
            )
        expected_source = "sigma" if expected_graph == "off" else "ising"
        for index, row in enumerate(rows, 1):
            if row.get("memory_used") is not True:
                raise AssertionError(f"row {index}: Sigma memory_used must be true")
            if row.get("score_source") != expected_source:
                raise AssertionError(
                    f"row {index}: score_source={row.get('score_source')!r}, "
                    f"expected {expected_source!r}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--legacy-prompt-protocol", choices=("on", "off"), default="off"
    )
    args = parser.parse_args()
    metrics_path = args.run_dir / "eval_metrics.json"
    selections_path = args.run_dir / "selections.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text())
    rows = _jsonl(selections_path)
    benchmark = _jsonl(args.benchmark)
    _validate_common(
        metrics,
        rows,
        benchmark,
        args.benchmark,
        legacy_prompt_protocol=args.legacy_prompt_protocol == "on",
    )
    _validate_arm(
        args.arm,
        metrics,
        rows,
        args.checkpoint,
    )
    print(f"valid sigma evaluation: arm={args.arm}, rows={len(rows)}")


if __name__ == "__main__":
    main()
