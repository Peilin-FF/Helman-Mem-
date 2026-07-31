"""Evaluate training-free OOD routing and voting baselines.

The frozen Sigma checkpoint is used only to construct the competence direction
``phi``. Runtime M state is rebuilt from a cold start, all
decisions are made before the event labels are read, and no residual steering or
candidate-scoring forward pass is performed.
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

from feedback_state.newarch_loader import (
    apply_torch_fp8_shim,
    dtype_from_name,
    load_central_model,
)

apply_torch_fp8_shim()

from transformers import AutoTokenizer

from feedback_state.data import JsonlDataset
from feedback_state.ood_routing import (
    INVALID_ANSWER,
    OODRoutingState,
    canonical_answer,
    canonical_gold_answer,
    vote_is_correct,
)
from feedback_state.symmetric_memory import (
    DEFAULT_TASK_TYPES,
    SymmetricTrustMemory,
    cm_context_vector,
)
from tests.experiments.common.model_profiles import MODEL_RUNS


BENCHMARK_NAMES = {
    "piqa": "PIQA",
    "mmlu": "MMLU",
    "openbookqa": "OpenBookQA",
    "sciq": "SciQ",
    "bbh": "BBH",
    "superglue_wic": "SuperGLUE",
    "superglue_rte": "SuperGLUE",
    "superglue_cb": "SuperGLUE",
    "superglue_copa": "SuperGLUE",
    "superglue_wsc": "SuperGLUE",
    "superglue_multirc": "SuperGLUE",
}
PEER_KEYS = ("peer_0", "peer_1", "peer_2")


def benchmark_name(source: str) -> str:
    """Return a display group while preserving unknown benchmark names."""

    return BENCHMARK_NAMES.get(source, source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(MODEL_RUNS), default="q3_4b")
    parser.add_argument(
        "--offline-data",
        type=Path,
        default=Path("data/ood/test.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--central-model",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    args = parser.parse_args()
    profile = MODEL_RUNS[args.profile]
    if args.checkpoint is None:
        args.checkpoint = profile.checkpoint
    if args.central_model is None:
        args.central_model = profile.central_model
    if args.output is None:
        args.output = profile.routing_output
    return args


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_id_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        identifier = str(record.get("id") or record.get("uid") or "")
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_checkpoint_encoder(
    *,
    checkpoint: Path,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool,
) -> tuple[Any, Any, SymmetricTrustMemory, dict[str, Any]]:
    checkpoint_file = checkpoint / "sym_memory.pt"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Missing Sigma checkpoint: {checkpoint_file}")
    train_config_path = checkpoint / "train_config.json"
    train_config = (
        json.loads(train_config_path.read_text()) if train_config_path.is_file() else {}
    )
    payload = torch.load(checkpoint_file, map_location="cpu")
    memory_state = payload["mem"]
    proto_weight = memory_state["proto_proj.weight"]
    rank = int(proto_weight.shape[0])
    proto_tau = float(train_config.get("proto_tau", 0.1))
    task_types_raw = train_config.get("task_types", DEFAULT_TASK_TYPES)
    if isinstance(task_types_raw, str):
        import ast

        task_types_raw = ast.literal_eval(task_types_raw)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = load_central_model(
        model_name,
        dtype=dtype,
        local_files_only=local_files_only,
    ).to(device=device, dtype=dtype)
    base.eval()
    memory = SymmetricTrustMemory(
        num_peers=len(PEER_KEYS),
        rank=rank,
        task_types=tuple(task_types_raw),
        phi_mode="proto",
        phi_in_dim=int(base.config.hidden_size),
        proto_tau=proto_tau,
        device=device,
    ).to(device)

    whitened_centroids = memory_state["proto_centroids"]
    proto_mean = memory_state["proto_mean"]
    proto_std = memory_state["proto_std"]
    raw_centroids = whitened_centroids.to(proto_std.dtype) * (
        proto_std + 1e-6
    ) + proto_mean
    memory.set_prototypes(raw_centroids, proto_mean, proto_std)
    runtime_free_state = {
        key: value
        for key, value in memory_state.items()
        if key not in {"M", "G", "_theta_joint"}
    }
    current = memory.state_dict()
    incompatible = {}
    for key, value in runtime_free_state.items():
        if key not in current:
            incompatible[key] = (tuple(value.shape), None)
        elif tuple(value.shape) != tuple(current[key].shape):
            incompatible[key] = (tuple(value.shape), tuple(current[key].shape))
    if incompatible:
        raise ValueError(f"Checkpoint encoder shapes are incompatible: {incompatible}")
    missing, unexpected = memory.load_state_dict(runtime_free_state, strict=False)
    material_missing = [key for key in missing if key not in {"M", "G"}]
    if material_missing or unexpected:
        raise ValueError(
            "Incomplete checkpoint encoder load: "
            f"missing={material_missing}, unexpected={unexpected}"
        )
    memory.reset()
    memory.eval()
    metadata = {
        "rank": rank,
        "proto_tau": proto_tau,
        "phi_layer_frac": float(train_config.get("phi_layer_frac", 0.5)),
        "gamma": float(memory.gamma_scalar()),
        "eta": float(memory.eta.detach().cpu()),
        "checkpoint_sha256": _sha256_file(checkpoint_file),
    }
    return base, tokenizer, memory, metadata


def _phi_cache_paths(output: Path) -> tuple[Path, Path]:
    return output / "phis.npy", output / "phi_manifest.json"


def _load_cached_phis(
    *,
    output: Path,
    expected_manifest: Mapping[str, Any],
) -> np.ndarray | None:
    phi_path, manifest_path = _phi_cache_paths(output)
    if not phi_path.is_file() or not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    compared_keys = (
        "offline_data_sha256",
        "ordered_id_sha256",
        "checkpoint_sha256",
        "central_model",
        "max_length",
        "phi_layer_frac",
        "rank",
        "num_examples",
        "dtype",
        "proto_tau",
        "train_config_sha256",
        "gamma",
        "eta",
    )
    if any(manifest.get(key) != expected_manifest.get(key) for key in compared_keys):
        return None
    if manifest.get("phi_file_sha256") != _sha256_file(phi_path):
        return None
    phis = np.load(phi_path, allow_pickle=False)
    expected_shape = (
        int(expected_manifest["num_examples"]),
        int(expected_manifest["rank"]),
    )
    if phis.shape != expected_shape or not np.isfinite(phis).all():
        return None
    print(f"[ood-routing] reusing phi cache {phi_path}", flush=True)
    return phis


def _save_phi_cache(
    *, output: Path, phis: np.ndarray, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    phi_path, manifest_path = _phi_cache_paths(output)
    temporary = phi_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, phis, allow_pickle=False)
    temporary.replace(phi_path)
    saved_manifest = dict(manifest) | {"phi_file_sha256": _sha256_file(phi_path)}
    _write_json_atomic(manifest_path, saved_manifest)
    return saved_manifest


@torch.no_grad()
def _extract_phis(
    *,
    records: list[dict[str, Any]],
    base: Any,
    tokenizer: Any,
    memory: SymmetricTrustMemory,
    device: torch.device,
    max_length: int,
    layer_frac: float,
) -> np.ndarray:
    phis = np.empty((len(records), memory.rank), dtype=np.float32)
    for index, record in enumerate(records):
        question = str(record.get("problem", record.get("question", "")))
        context = cm_context_vector(
            base,
            tokenizer,
            question,
            device=device,
            max_length=max_length,
            layer_frac=layer_frac,
        )
        phi = memory.phi_of(context).detach().float().cpu().numpy()
        phis[index] = phi
        count = index + 1
        if count == 1 or count % 250 == 0 or count == len(records):
            print(f"[ood-routing/phi] {count}/{len(records)}", flush=True)
    norms = np.linalg.norm(phis.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"Extracted phi vectors are not normalized: {norms.min()}..{norms.max()}"
        )
    return phis


def _decision_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove all current-event supervision before calling a decision method."""

    return {
        key: value
        for key, value in record.items()
        if key not in {"answer", "peer_correct", "correctness_by_peer"}
    }


def _record_answers(record: Mapping[str, Any]) -> list[str]:
    responses = record.get("peer_responses") or {}
    missing = [key for key in PEER_KEYS if key not in responses]
    if missing:
        raise ValueError(f"Record {record.get('id')} is missing peers: {missing}")
    return [str(responses[key]) for key in PEER_KEYS]


def _record_labels(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    table = record.get("peer_correct") or {}
    values = np.asarray([float(table[key]) for key in PEER_KEYS], dtype=np.float64)
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError(f"Non-binary peer labels in record {record.get('id')}")
    return values.astype(np.int8), np.where(values > 0.5, 1.0, -1.0)


def _frozen_vote_correctness(
    selected_answer: str,
    canonical_answers: list[str],
    peer_correct: np.ndarray,
) -> tuple[int, int]:
    """Score a voted answer under the frozen external-label protocol.

    The lowest peer index is the fixed representative of the winning answer
    group.  This remains fixed even when historical parser bugs gave two peers
    in one normalized option group different labels; those conflicts are counted
    separately instead of being resolved in the new method's favor.
    """

    members = [
        peer for peer, answer in enumerate(canonical_answers)
        if answer == selected_answer
    ]
    if not members:
        raise ValueError(f"Voted answer {selected_answer!r} has no peer member")
    representative = min(members)
    return int(peer_correct[representative]), representative


def _scope_indices(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    sources = np.asarray([str(record.get("source") or "") for record in records])
    groups = np.asarray([benchmark_name(source) for source in sources])
    scopes: dict[str, np.ndarray] = {
        "overall": np.ones(len(records), dtype=bool),
    }
    for group in sorted(set(groups)):
        scopes[f"group:{group}"] = groups == group
    for source in sorted(set(sources)):
        scopes[f"source:{source}"] = sources == source
    return scopes


def _summarize_correctness(
    correctness: Mapping[str, list[int]],
    scopes: Mapping[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float | int]]]:
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for scope_name, mask in scopes.items():
        count = int(mask.sum())
        methods: dict[str, dict[str, float | int]] = {}
        for method, values in correctness.items():
            array = np.asarray(values, dtype=np.int8)
            correct = int(array[mask].sum())
            methods[method] = {
                "correct": correct,
                "total": count,
                "accuracy": correct / count if count else math.nan,
            }
        result[scope_name] = methods
    return result


def _selection_summary(
    selections: Mapping[str, list[int]],
    correctness: Mapping[str, list[int]],
    scopes: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method, chosen in selections.items():
        chosen_array = np.asarray(chosen, dtype=np.int8)
        corr_array = np.asarray(correctness[method], dtype=np.int8)
        per_scope: dict[str, Any] = {}
        for scope, mask in scopes.items():
            rows = {}
            total = int(mask.sum())
            for peer in range(len(PEER_KEYS)):
                peer_mask = mask & (chosen_array == peer)
                count = int(peer_mask.sum())
                rows[str(peer)] = {
                    "selected": count,
                    "share": count / total if total else math.nan,
                    "accuracy_when_selected": (
                        int(corr_array[peer_mask].sum()) / count
                        if count
                        else math.nan
                    ),
                }
            per_scope[scope] = rows
        result[method] = per_scope
    return result


def main() -> None:
    args = parse_args()
    profile = MODEL_RUNS[args.profile]
    if (
        args.max_examples is not None
        and args.output.resolve() == profile.routing_output.resolve()
    ):
        raise ValueError("A truncated run requires an explicit --output directory")

    input_sha = _sha256_file(args.offline_data)
    records = JsonlDataset(args.offline_data).records
    if args.max_examples is not None:
        records = records[: int(args.max_examples)]
    ordered_id_sha = _ordered_id_sha256(records)
    checkpoint_sha = _sha256_file(args.checkpoint / "sym_memory.pt")
    train_config_sha = _sha256_file(args.checkpoint / "train_config.json")
    checkpoint_payload = torch.load(
        args.checkpoint / "sym_memory.pt", map_location="cpu"
    )
    rank = int(checkpoint_payload["mem"]["proto_proj.weight"].shape[0])
    scalar_device = torch.device(args.device)
    checkpoint_gamma = float(
        torch.exp(
            -torch.nn.functional.softplus(
                checkpoint_payload["mem"]["_theta"].to(scalar_device)
            )
        ).mean().cpu()
    )
    checkpoint_eta = float(
        torch.nn.functional.softplus(
            checkpoint_payload["mem"]["_eta_raw"].to(scalar_device)
        ).cpu()
    )
    del checkpoint_payload
    train_config = json.loads((args.checkpoint / "train_config.json").read_text())
    phi_layer_frac = float(train_config.get("phi_layer_frac", 0.5))
    proto_tau = float(train_config.get("proto_tau", 0.1))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").unlink(missing_ok=True)
    (args.output / "records.jsonl.tmp").unlink(missing_ok=True)

    cache_manifest_base = {
        "offline_data_sha256": input_sha,
        "ordered_id_sha256": ordered_id_sha,
        "checkpoint_sha256": checkpoint_sha,
        "central_model": str(args.central_model),
        "max_length": int(args.max_length),
        "phi_layer_frac": phi_layer_frac,
        "rank": rank,
        "num_examples": len(records),
        "encoder_frozen": True,
        "training_performed": False,
        "dtype": str(args.dtype).lower(),
        "proto_tau": proto_tau,
        "train_config_sha256": train_config_sha,
        "gamma": checkpoint_gamma,
        "eta": checkpoint_eta,
    }
    phis = _load_cached_phis(
        output=args.output, expected_manifest=cache_manifest_base
    )
    encoder_metadata: dict[str, Any]
    if phis is None:
        device = torch.device(args.device)
        dtype = dtype_from_name(str(args.dtype))
        base, tokenizer, memory, encoder_metadata = _load_checkpoint_encoder(
            checkpoint=args.checkpoint,
            model_name=str(args.central_model),
            device=device,
            dtype=dtype,
            local_files_only=bool(args.local_files_only),
        )
        if encoder_metadata["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Checkpoint changed while loading")
        phis = _extract_phis(
            records=records,
            base=base,
            tokenizer=tokenizer,
            memory=memory,
            device=device,
            max_length=int(args.max_length),
            layer_frac=phi_layer_frac,
        )
        cache_manifest = cache_manifest_base | {
            **encoder_metadata,
            "phi_norm_min": float(np.linalg.norm(phis, axis=1).min()),
            "phi_norm_max": float(np.linalg.norm(phis, axis=1).max()),
        }
        cache_manifest = _save_phi_cache(
            output=args.output, phis=phis, manifest=cache_manifest
        )
        del memory, base, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        cache_manifest = json.loads((args.output / "phi_manifest.json").read_text())
        encoder_metadata = {
            "rank": rank,
            "proto_tau": proto_tau,
            "phi_layer_frac": phi_layer_frac,
            "gamma": checkpoint_gamma,
            "eta": checkpoint_eta,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_runtime_state_ignored": ["M", "G"],
        }

    if not math.isclose(
        float(encoder_metadata["gamma"]), checkpoint_gamma, rel_tol=0.0, abs_tol=1e-8
    ) or not math.isclose(
        float(encoder_metadata["eta"]), checkpoint_eta, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("Loaded gamma/eta do not match the frozen checkpoint")

    gamma = float(encoder_metadata["gamma"])
    eta = float(encoder_metadata["eta"])
    state = OODRoutingState(
        num_peers=len(PEER_KEYS),
        rank=rank,
        gamma=gamma,
        eta=eta,
    )
    correctness: defaultdict[str, list[int]] = defaultdict(list)
    selections: defaultdict[str, list[int]] = defaultdict(list)
    route_ties = 0
    parser_label_mismatches = 0
    answer_group_label_conflicts = 0
    invalid_peer_answers = 0
    invalid_gold_answers = 0
    n_distinct_counts: Counter[int] = Counter()
    records_tmp = args.output / "records.jsonl.tmp"
    with records_tmp.open("w") as record_handle:
        for event_index, (record, phi_raw) in enumerate(zip(records, phis)):
            phi = np.asarray(phi_raw, dtype=np.float64)
            phi /= np.linalg.norm(phi)
            source = str(record.get("source") or "")

            # Decisions happen before peer_correct is accessed for this event.
            route = state.route(phi, event_index)
            answers = _record_answers(record)
            decision_record = _decision_record(record)
            main_votes = state.votes(
                phi,
                decision_record,
                answers,
            )
            canonical_answers = [
                canonical_answer(decision_record, answer) for answer in answers
            ]
            n_distinct = len(set(canonical_answers))
            n_distinct_counts[n_distinct] += 1
            route_ties += int(route.tied)
            invalid_peer_answers += sum(
                answer == INVALID_ANSWER for answer in canonical_answers
            )
            # Current-event supervision is read only after every arm has decided.
            peer_correct, correctness_signed = _record_labels(record)
            gold = canonical_gold_answer(record)
            invalid_gold_answers += int(gold == INVALID_ANSWER)
            parser_label_mismatches += sum(
                int(vote_is_correct(record, answer)) != int(label)
                for answer, label in zip(canonical_answers, peer_correct)
            )
            answer_group_label_conflicts += sum(
                len({
                    int(peer_correct[peer])
                    for peer, answer in enumerate(canonical_answers)
                    if answer == group
                }) > 1
                for group in set(canonical_answers)
            )

            vote_correct = {
                arm: _frozen_vote_correctness(
                    main_votes.answers[arm], canonical_answers, peer_correct
                )[0]
                for arm in ("maj", "M")
            }
            event_correct = {
                "route_M": int(peer_correct[route.peer]),
                "vote_majority": vote_correct["maj"],
                "vote_M": vote_correct["M"],
            }
            for method, value in event_correct.items():
                correctness[method].append(value)
            selections["route_M"].append(route.peer)

            row = {
                "t": event_index,
                "id": str(record.get("id") or record.get("uid") or ""),
                "source": source,
                "benchmark": benchmark_name(source),
                "n_distinct": n_distinct,
                "canonical_answers": canonical_answers,
                "canonical_gold": gold,
                "peer_correct": peer_correct.tolist(),
                "route_M_peer": route.peer,
                "route_M_scores": route.scores.tolist(),
                "vote_answers": dict(main_votes.answers),
                "vote_weights": {
                    name: weights.tolist()
                    for name, weights in main_votes.weights.items()
                },
                "correct": event_correct,
            }
            record_handle.write(json.dumps(row, sort_keys=True) + "\n")

            # Exactly one state write per event, after decision and logging.
            state.update(phi, correctness_signed)
            count = event_index + 1
            if count == 1 or count % 1000 == 0 or count == len(records):
                running = sum(correctness["route_M"]) / count
                print(
                    f"[ood-routing/replay] {count}/{len(records)} "
                    f"route_M={running * 100:.2f}%",
                    flush=True,
                )
    records_tmp.replace(args.output / "records.jsonl")

    scopes = _scope_indices(records)
    accuracy = _summarize_correctness(correctness, scopes)
    final_m = state.snapshot()
    summary = {
        "status": "complete",
        "experiment": "training_free_ood_memory_readout",
        "profile": profile.name,
        "training_performed": False,
        "residual_steering_used": False,
        "candidate_yes_no_scoring_used": False,
        "decision_then_update": True,
        "stream": {
            "path": str(args.offline_data),
            "sha256": input_sha,
            "num_examples": len(records),
            "ordered_id_sha256": ordered_id_sha,
            "physical_order_preserved": True,
            "benchmarks": sorted(
                {benchmark_name(str(record.get("source") or "")) for record in records}
            ),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            **encoder_metadata,
            "central_model": str(args.central_model),
            "phi_cache": str(args.output / "phis.npy"),
            "train_config_sha256": train_config_sha,
            "dtype": str(args.dtype).lower(),
            "max_length": int(args.max_length),
        },
        "fixed_constants": {
            "num_peers": len(PEER_KEYS),
            "rank": rank,
            "gamma": gamma,
            "eta": eta,
            "route_tie_epsilon": 1e-9,
            "main_weights": "raw_signed",
            "clipping": "appendix_only",
            "vote_tie_rule": (
                "lowest declared option index; then lexical canonical answer; "
                "invalid last"
            ),
            "winning_group_score_rule": (
                "frozen peer_correct of the lowest-index peer in the group"
            ),
        },
        "accuracy": accuracy,
        "selection": _selection_summary(selections, correctness, scopes),
        "diagnostics": {
            "route_all_equal_ties": route_ties,
            "n_distinct_answer_counts": dict(sorted(n_distinct_counts.items())),
            "invalid_peer_answers": invalid_peer_answers,
            "invalid_gold_answers": invalid_gold_answers,
            "parser_vs_external_label_mismatches": parser_label_mismatches,
            "canonical_group_label_conflicts": answer_group_label_conflicts,
            "final_M_frobenius_norm": float(np.linalg.norm(final_m)),
        },
        "artifacts": {
            "records": str(args.output / "records.jsonl"),
            "phis": str(args.output / "phis.npy"),
            "phi_manifest": str(args.output / "phi_manifest.json"),
        },
    }
    _write_json_atomic(args.output / "summary.json", summary)
    print(
        f"[ood-routing] complete: {args.output / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
