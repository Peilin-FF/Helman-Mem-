"""Evaluate answer-free direct M-routing on the canonical CF streams.

The frozen Sigma checkpoint is used only for the question-derived competence
direction and its learned gamma/eta scalars.  Runtime memory starts from zero for
each CF ratio.  A peer is selected from ``phi.T @ M[p] @ phi`` before the current
event correctness labels are read; peer responses are never accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from tests.experiments.selection_mechanisms.m_route_vote import (
    PEER_KEYS,
    _extract_phis,
    _load_checkpoint_encoder,
    _sha256_file,
)
from tests.experiments.common.model_profiles import MODEL_RUNS
from feedback_state.data import JsonlDataset
from feedback_state.newarch_loader import dtype_from_name
from feedback_state.ood_routing import OODRoutingState


CF_SPLITS = ("cf_0", "cf_50", "cf_70", "cf_90")
PHI_INPUT_POLICY = "problem_only_v1"


def _uid(record: Mapping[str, Any]) -> str:
    value = str(record.get("uid") or "")
    if not value:
        raise ValueError("Every canonical CF record must have a nonempty uid")
    return value


def _ordered_uid_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_uid(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _question_map_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    pairs = sorted(
        (_uid(record), str(record.get("problem", record.get("question", ""))))
        for record in records
    )
    digest = hashlib.sha256()
    for uid, question in pairs:
        digest.update(uid.encode("utf-8"))
        digest.update(b"\0")
        digest.update(question.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_cf_streams(
    data_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    streams: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    reference_questions: dict[str, str] | None = None

    for split in CF_SPLITS:
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing CF stream: {path}")
        file_sha = _sha256_file(path)
        records = JsonlDataset(path).records
        uids = [_uid(record) for record in records]
        if len(set(uids)) != len(uids):
            raise ValueError(f"{split} contains duplicate uid values")
        questions = {
            _uid(record): str(record.get("problem", record.get("question", "")))
            for record in records
        }
        if reference_questions is None:
            reference_questions = questions
        elif questions != reference_questions:
            raise ValueError(
                f"{split} does not contain the same uid/question pairs as {CF_SPLITS[0]}"
            )

        streams[split] = records
        provenance[split] = {
            "path": str(path),
            "sha256": file_sha,
            "num_examples": len(records),
            "ordered_uid_sha256": _ordered_uid_sha256(records),
            "question_map_sha256": _question_map_sha256(records),
        }

    return streams, provenance


def _phi_cache_paths(output: Path) -> tuple[Path, Path]:
    return output / "phis_p0.npy", output / "phi_manifest.json"


def _load_phi_cache(
    output: Path, *, expected: Mapping[str, Any]
) -> np.ndarray | None:
    phi_path, manifest_path = _phi_cache_paths(output)
    if not phi_path.is_file() or not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    compared = (
        "phi_input_policy",
        "p0_sha256",
        "p0_ordered_uid_sha256",
        "question_map_sha256",
        "checkpoint_sha256",
        "train_config_sha256",
        "central_model",
        "max_length",
        "phi_layer_frac",
        "proto_tau",
        "rank",
        "num_examples",
        "dtype",
        "gamma",
        "eta",
    )
    if any(manifest.get(key) != expected.get(key) for key in compared):
        return None
    if manifest.get("phi_file_sha256") != _sha256_file(phi_path):
        return None
    phis = np.load(phi_path, allow_pickle=False)
    shape = (int(expected["num_examples"]), int(expected["rank"]))
    if phis.shape != shape or not np.isfinite(phis).all():
        return None
    print(f"[cf-m-route] reusing phi cache {phi_path}", flush=True)
    return phis


def _save_phi_cache(
    output: Path, *, phis: np.ndarray, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    phi_path, manifest_path = _phi_cache_paths(output)
    temporary = phi_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, phis, allow_pickle=False)
    temporary.replace(phi_path)
    saved = dict(manifest) | {
        "phi_file_sha256": _sha256_file(phi_path),
        "phi_norm_min": float(np.linalg.norm(phis, axis=1).min()),
        "phi_norm_max": float(np.linalg.norm(phis, axis=1).max()),
    }
    _write_json_atomic(manifest_path, saved)
    return saved


def _metric(correct: int, total: int) -> dict[str, int | float]:
    return {
        "correct": int(correct),
        "total": int(total),
        "accuracy": float(correct / total) if total else math.nan,
    }


def _cf_record_labels(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Match the established CF evaluator's ``int(round(raw_score))`` protocol."""

    table = record.get("correctness_by_peer") or record.get("peer_correct") or {}
    missing = [key for key in PEER_KEYS if key not in table]
    if missing:
        raise ValueError(f"Record {_uid(record)} is missing peer labels: {missing}")
    raw = np.asarray([float(table[key]) for key in PEER_KEYS], dtype=np.float64)
    if not np.isfinite(raw).all() or ((raw < 0.0) | (raw > 1.0)).any():
        raise ValueError(f"Record {_uid(record)} has labels outside [0, 1]")
    binary = np.asarray([int(round(float(value))) for value in raw], dtype=np.int8)
    return binary, np.where(binary > 0, 1.0, -1.0)


def replay_m_route(
    *,
    records: list[dict[str, Any]],
    phi_by_uid: Mapping[str, np.ndarray],
    rank: int,
    gamma: float,
    eta: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay one physical stream without ever reading peer response strings."""

    state = OODRoutingState(
        num_peers=len(PEER_KEYS),
        rank=rank,
        gamma=gamma,
        eta=eta,
    )
    correct = 0
    ties = 0
    selections: Counter[int] = Counter()
    by_dataset: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_task_type: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    rows: list[dict[str, Any]] = []

    for event_index, record in enumerate(records):
        uid = _uid(record)
        phi = np.asarray(phi_by_uid[uid], dtype=np.float64)
        phi /= np.linalg.norm(phi)

        # The router sees only phi, historical M, and the fixed tie-break index.
        decision = state.route(phi, event_index)

        # Current-event supervision is first accessed after peer selection.
        peer_correct, correctness_signed = _cf_record_labels(record)
        selected_correct = int(peer_correct[decision.peer])
        correct += selected_correct
        ties += int(decision.tied)
        selections[decision.peer] += 1
        dataset = str(record.get("dataset") or record.get("source") or "unknown")
        task_type = str(record.get("task_type") or "unknown")
        by_dataset[dataset][0] += selected_correct
        by_dataset[dataset][1] += 1
        by_task_type[task_type][0] += selected_correct
        by_task_type[task_type][1] += 1
        rows.append(
            {
                "t": event_index,
                "uid": uid,
                "id": str(record.get("id") or ""),
                "dataset": dataset,
                "task_type": task_type,
                "selected_peer": decision.peer,
                "scores": decision.scores.tolist(),
                "tied": decision.tied,
                "selected_correct": selected_correct,
            }
        )

        state.update(phi, correctness_signed)

    final_m = state.snapshot()
    total = len(records)
    summary = {
        "accuracy": _metric(correct, total),
        "selected_peers": {
            str(peer): int(selections[peer]) for peer in range(len(PEER_KEYS))
        },
        "all_equal_ties": ties,
        "by_dataset": {
            name: _metric(values[0], values[1])
            for name, values in sorted(by_dataset.items())
        },
        "by_task_type": {
            name: _metric(values[0], values[1])
            for name, values in sorted(by_task_type.items())
        },
        "final_M_frobenius_norm": float(np.linalg.norm(final_m)),
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(MODEL_RUNS), default="q3_4b")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/counterfactual_3peer")
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--force-recompute-phis", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("outputs/cf_memory_routing") / args.profile
    return args


def main() -> None:
    args = parse_args()
    profile = MODEL_RUNS[args.profile]
    streams, stream_provenance = _load_cf_streams(args.data_dir)
    p0_records = streams["cf_0"]

    checkpoint_file = profile.checkpoint / "sym_memory.pt"
    train_config_file = profile.checkpoint / "train_config.json"
    checkpoint_sha = _sha256_file(checkpoint_file)
    train_config_sha = _sha256_file(train_config_file)
    train_config = json.loads(train_config_file.read_text())
    checkpoint_payload = torch.load(checkpoint_file, map_location="cpu")
    rank = int(checkpoint_payload["mem"]["proto_proj.weight"].shape[0])
    gamma = float(
        torch.exp(
            -torch.nn.functional.softplus(checkpoint_payload["mem"]["_theta"])
        ).mean()
    )
    eta = float(
        torch.nn.functional.softplus(checkpoint_payload["mem"]["_eta_raw"])
    )
    del checkpoint_payload
    phi_layer_frac = float(train_config.get("phi_layer_frac", 0.5))
    proto_tau = float(train_config.get("proto_tau", 0.1))

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_base = {
        "phi_input_policy": PHI_INPUT_POLICY,
        "p0_sha256": stream_provenance["cf_0"]["sha256"],
        "p0_ordered_uid_sha256": stream_provenance["cf_0"]["ordered_uid_sha256"],
        "question_map_sha256": stream_provenance["cf_0"]["question_map_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "train_config_sha256": train_config_sha,
        "central_model": profile.central_model,
        "max_length": int(args.max_length),
        "phi_layer_frac": phi_layer_frac,
        "proto_tau": proto_tau,
        "rank": rank,
        "num_examples": len(p0_records),
        "dtype": str(args.dtype).lower(),
        "gamma": gamma,
        "eta": eta,
        "encoder_frozen": True,
        "training_performed": False,
        "checkpoint_runtime_state_ignored": ["M", "G"],
    }
    phis = None if args.force_recompute_phis else _load_phi_cache(
        args.output, expected=manifest_base
    )
    if phis is None:
        device = torch.device(args.device)
        dtype = dtype_from_name(str(args.dtype))
        base, tokenizer, memory, encoder_metadata = _load_checkpoint_encoder(
            checkpoint=profile.checkpoint,
            model_name=profile.central_model,
            device=device,
            dtype=dtype,
            local_files_only=True,
        )
        phis = _extract_phis(
            records=p0_records,
            base=base,
            tokenizer=tokenizer,
            memory=memory,
            device=device,
            max_length=int(args.max_length),
            layer_frac=phi_layer_frac,
        )
        if not math.isclose(
            float(encoder_metadata["gamma"]), gamma, rel_tol=0.0, abs_tol=1e-8
        ) or not math.isclose(
            float(encoder_metadata["eta"]), eta, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError("Checkpoint parameters changed while loading the encoder")
        phi_manifest = _save_phi_cache(
            args.output, phis=phis, manifest=manifest_base
        )
        del memory, base, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        phi_manifest = json.loads((args.output / "phi_manifest.json").read_text())

    phi_by_uid = {
        _uid(record): phis[index] for index, record in enumerate(p0_records)
    }
    split_summaries: dict[str, Any] = {}
    record_hashes: dict[str, str] = {}
    for split in CF_SPLITS:
        print(f"[cf-m-route] replaying {split}", flush=True)
        split_summary, rows = replay_m_route(
            records=streams[split],
            phi_by_uid=phi_by_uid,
            rank=rank,
            gamma=gamma,
            eta=eta,
        )
        records_path = args.output / f"records_{split}.jsonl"
        temporary = records_path.with_suffix(".jsonl.tmp")
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        temporary.replace(records_path)
        record_hashes[split] = _sha256_file(records_path)
        split_summaries[split] = split_summary
        accuracy = split_summary["accuracy"]["accuracy"]
        print(
            f"[cf-m-route] {split}: {accuracy * 100:.2f}% "
            f"selected={split_summary['selected_peers']}",
            flush=True,
        )

    summary = {
        "status": "complete",
        "experiment": "cf_direct_M_route",
        "profile": profile.name,
        "method": {
            "score": "phi(x)^T M_p phi(x)",
            "selection": "argmax_p score_p",
            "phi_input": "problem only",
            "peer_responses_accessed": False,
            "decision_then_update": True,
            "runtime_M_cold_start_per_split": True,
            "runtime_G_used": False,
            "residual_steering_used": False,
            "candidate_yes_no_scoring_used": False,
        },
        "checkpoint": {
            "path": str(profile.checkpoint),
            "central_model": profile.central_model,
            "checkpoint_sha256": checkpoint_sha,
            "train_config_sha256": train_config_sha,
            "rank": rank,
            "gamma": gamma,
            "eta": eta,
            "phi_manifest": phi_manifest,
        },
        "streams": stream_provenance,
        "results": split_summaries,
        "records_sha256": record_hashes,
    }
    _write_json_atomic(args.output / "summary.json", summary)
    print(f"[cf-m-route] wrote {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
