"""Evaluate the training-free OOD memory readout from ``instruction.md``.

The frozen Sigma checkpoint is used only to construct the competence direction
``phi``.  Runtime M/G state is rebuilt from the teacher-specified cold starts, all
decisions are made before the event labels are read, and no residual steering or
candidate-scoring forward pass is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from feedback_state.newarch_loader import apply_torch_fp8_shim, load_central_model

apply_torch_fp8_shim()

from transformers import AutoTokenizer

from feedback_state.data import JsonlDataset
from feedback_state.generation import dtype_from_name
from feedback_state.ood_routing import (
    DEFAULT_RIDGE,
    INVALID_ANSWER,
    ROBUSTNESS_RIDGES,
    DecayedLabelDictionary,
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


CANONICAL_STREAM_SHA256 = (
    "92102a2352aec0f78188ac334f1a3d74b08f87eccb06bca577285cecdb771539"
)
CANONICAL_ORDERED_ID_SHA256 = (
    "2cfafc668a740ebe5ee4744efb5411fa7857b4563a0e4c686f7d3489bace570c"
)
PAPER_OOD_ORDERED_ID_SHA256 = (
    "0b58e65516b838a9c6a74089ba3bb4d7598cb2067faa2810a8e3141a8d0d05f8"
)
CANONICAL_STREAM_SIZE = 23_460
PAPER_OOD_SIZE = 17_403
PAPER_OOD_GROUPS = {
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
MAIN_METHODS = (
    "route_M",
    "route_MG",
    "route_dictionary",
    "vote_majority",
    "vote_G",
    "vote_M",
    "vote_MG",
)
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


@dataclass(frozen=True)
class OODRunProfile:
    name: str
    central_model: str
    checkpoint: Path
    center_selections: Path
    sigma_selections: Path
    output: Path
    checkpoint_sha256: str
    train_config_sha256: str
    center_selections_sha256: str
    sigma_selections_sha256: str
    gamma: float
    eta: float
    sigma_with_g_full_correct: int
    sigma_with_g_paper_ood_correct: int
    sigma_with_g_selected_peers: tuple[int, int, int]


@dataclass(frozen=True)
class HistoricalJointGReplay:
    correctness: list[int]
    selections: list[int]
    final_history: list[float]
    final_graph: list[list[float]]


RUN_PROFILES = {
    "q3_0_6b": OODRunProfile(
        name="q3_0_6b",
        central_model="/mnt/data/peilin/HF_MODEL/Qwen3-0.6B",
        checkpoint=Path("outputs/sigma_candidate_yesno_q3_0.6b/proto"),
        center_selections=Path(
            "outputs/eval_generalization_3peer_peer012_qwen3/"
            "q3_0.6b_center/selections.jsonl"
        ),
        sigma_selections=Path(
            "outputs/eval_generalization_3peer_peer012_qwen3/"
            "q3_0.6b_sigma/selections.jsonl"
        ),
        output=Path("outputs/ood_memory_routing/q3_0_6b"),
        checkpoint_sha256=(
            "bac0f33ddf5f6f7ddbcc8609fbdcc90960ea5e39df3180c2907ad86bcd65e36f"
        ),
        train_config_sha256=(
            "25649484b1b83426eee42e5756e8bfb57ef34377f89f613f5ead7b0e1a8e88f5"
        ),
        center_selections_sha256=(
            "d33b76a74538ae92c0523d47cf4da14227bb06b852539ef58c4d857bd7c5cfaa"
        ),
        sigma_selections_sha256=(
            "c42bc1a9141135300c743c8d1a3d691012a109c0eba8e703705d33abf985c192"
        ),
        gamma=0.8749558925628662,
        eta=0.43567895889282227,
        sigma_with_g_full_correct=15_033,
        sigma_with_g_paper_ood_correct=10_422,
        sigma_with_g_selected_peers=(11_422, 6_185, 5_853),
    ),
    "q3_4b": OODRunProfile(
        name="q3_4b",
        central_model="/mnt/data/peilin/HF_MODEL/Qwen3-4B",
        checkpoint=Path("outputs/sigma_candidate_yesno_q3_4b/proto"),
        center_selections=Path(
            "outputs/eval_generalization_3peer_peer012_qwen3/"
            "q3_4b_center/selections.jsonl"
        ),
        sigma_selections=Path(
            "outputs/eval_generalization_3peer_peer012_qwen3/"
            "q3_4b_sigma/selections.jsonl"
        ),
        output=Path("outputs/ood_memory_routing/q3_4b"),
        checkpoint_sha256=(
            "ecc70510d9b519aeb9b2206aa3347580addf770e8020a93f7109187eebe5165d"
        ),
        train_config_sha256=(
            "3dc5565bf10a5f210ec4826f5d5db695d4fb8ce82fc95706c56168fca9477004"
        ),
        center_selections_sha256=(
            "53978193330e340f136df0ba819dd189157c8242819f8df84e3215286175640e"
        ),
        sigma_selections_sha256=(
            "0a77d5779da366d8a824db6eae38f65c5c2be5cb170039baead268deeee06664"
        ),
        gamma=0.8773147463798523,
        eta=0.4097904562950134,
        sigma_with_g_full_correct=15_165,
        sigma_with_g_paper_ood_correct=10_536,
        sigma_with_g_selected_peers=(11_491, 7_754, 4_215),
    ),
    "q3_8b": OODRunProfile(
        name="q3_8b",
        central_model="/mnt/data/peilin/HF_MODEL/Qwen3-8B",
        checkpoint=Path("outputs/sigma_candidate_yesno_q3_8b/proto"),
        center_selections=Path(
            "outputs/eval_generalization_3peer_peer012_qwen3/"
            "q3_8b_center/selections.jsonl"
        ),
        sigma_selections=Path(
            "outputs/eval_generalization_3peer_peer012_qwen3/"
            "q3_8b_sigma/selections.jsonl"
        ),
        output=Path("outputs/ood_memory_routing/q3_8b"),
        checkpoint_sha256=(
            "8a5386104caee0ef1de3d26866a18d86d39a45e70b117107c18ea98e156a1b63"
        ),
        train_config_sha256=(
            "d22a58225c3fb551106d7c5538248857af30393492de3f948973a3525904aa1e"
        ),
        center_selections_sha256=(
            "d3507a725d4cbf59b8bc3bfb4ebd43e1dd30247ec4b1d2918f48463d35134060"
        ),
        sigma_selections_sha256=(
            "b8cb95eb192a5194fbfce2eb1c92e7871c81422e7831571372b8a8cb8b736b99"
        ),
        gamma=0.8791462182998657,
        eta=0.4159024953842163,
        sigma_with_g_full_correct=15_211,
        sigma_with_g_paper_ood_correct=10_528,
        sigma_with_g_selected_peers=(14_178, 5_842, 3_440),
    ),
    "q35_4b": OODRunProfile(
        name="q35_4b",
        central_model="/mnt/data/peilin/HF_MODEL/Qwen3.5-4B",
        checkpoint=Path("outputs/sigma_candidate_yesno_q35_4b/proto"),
        center_selections=Path(
            "outputs/eval_generalization_3peer_peer012_qwen3/"
            "q35_4b_center/selections.jsonl"
        ),
        sigma_selections=Path(
            "outputs/eval_generalization_3peer_peer012_qwen3/"
            "q35_4b_sigma/selections.jsonl"
        ),
        output=Path("outputs/ood_memory_routing/q35_4b"),
        checkpoint_sha256=(
            "7d399de393415d2d45bcdd99f03dcfbdcaa5e87044740c6f53ac45ba5a46e5ff"
        ),
        train_config_sha256=(
            "8dbffbe7bb9628d3d3be93e4d8c5d5d4002b51cdfc428a62eedb4efda42b82af"
        ),
        center_selections_sha256=(
            "d2e78385cba98e3db66a29f451929d83fb5cf06ec4612de34b629442994a3500"
        ),
        sigma_selections_sha256=(
            "18d84b2ae160768c0d3ce36ed374dcd2255e6988988959f1d9c3e3f19363b61d"
        ),
        gamma=0.8776268362998962,
        eta=0.4006841480731964,
        sigma_with_g_full_correct=15_288,
        sigma_with_g_paper_ood_correct=10_594,
        sigma_with_g_selected_peers=(13_330, 6_998, 3_132),
    ),
    "q35_9b": OODRunProfile(
        name="q35_9b",
        central_model="/mnt/data/peilin/HF_MODEL/Qwen3.5-9B",
        checkpoint=Path("outputs/sigma_candidate_yesno_q35_9b/proto"),
        center_selections=Path(
            "outputs/eval_generalization_3peer_peer012_qwen3/"
            "q35_9b_center/selections.jsonl"
        ),
        sigma_selections=Path(
            "outputs/eval_generalization_3peer_peer012_qwen3/"
            "q35_9b_sigma/selections.jsonl"
        ),
        output=Path("outputs/ood_memory_routing/q35_9b"),
        checkpoint_sha256=(
            "12df397e35a5fd7318cd3edbfcde166b14b77e96823d0904714f77c4a55fdd92"
        ),
        train_config_sha256=(
            "1484f592cc36f669beae1611ab3f2319b464a6034261ea7b04b57025fd6b0b22"
        ),
        center_selections_sha256=(
            "c9229969b9b954d44d77812e102729d3b356cf66f6929b33aeb83409061b16cc"
        ),
        sigma_selections_sha256=(
            "b3f27fe18436f8422f823d1aa2497a5f9fc6861bcd7c8b34235dbe622b14c89f"
        ),
        gamma=0.8778956532478333,
        eta=0.40443554520606995,
        sigma_with_g_full_correct=15_385,
        sigma_with_g_paper_ood_correct=10_669,
        sigma_with_g_selected_peers=(15_953, 5_432, 2_075),
    ),
}


def _validate_output_target(
    *, profile_name: str, output: Path, strict_full_run: bool
) -> None:
    resolved = output.resolve()
    owner = next(
        (
            name
            for name, profile in RUN_PROFILES.items()
            if profile.output.resolve() == resolved
        ),
        None,
    )
    if owner is not None and owner != profile_name:
        raise ValueError(
            f"Profile {profile_name} cannot write into {owner}'s registered output: "
            f"{output}"
        )
    if owner is not None and not strict_full_run:
        raise ValueError(
            "Diagnostic/truncated runs require a non-registered --output directory"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(RUN_PROFILES), default="q3_4b")
    parser.add_argument(
        "--offline-data",
        type=Path,
        default=Path(
            "data/peer_generalization/hf_generalization_all_canonical_peers/"
            "hf_generalization_all_peer012.labeled.jsonl"
        ),
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
        "--center-selections",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--sigma-selections",
        type=Path,
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
    parser.add_argument("--gamma-g", type=float, default=0.9)
    parser.add_argument("--eta-g", type=float, default=0.1)
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument(
        "--allow-noncanonical-stream",
        action="store_true",
        help="Required for diagnostics on a truncated or different stream.",
    )
    args = parser.parse_args()
    profile = RUN_PROFILES[args.profile]
    if args.checkpoint is None:
        args.checkpoint = profile.checkpoint
    if args.central_model is None:
        args.central_model = profile.central_model
    if args.center_selections is None:
        args.center_selections = profile.center_selections
    if args.sigma_selections is None:
        args.sigma_selections = profile.sigma_selections
    if args.output is None:
        args.output = profile.output
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
        write_mode="addr",
        decay_mode=str(train_config.get("decay_mode", "scalar")),
        per_peer_decay=str(train_config.get("per_peer_decay", "off")).lower()
        in {"1", "true", "yes", "on"},
        use_joint=False,
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
        if key not in {"M", "G"}
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
        "checkpoint_runtime_state_ignored": ["M", "G"],
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
    groups = np.asarray([PAPER_OOD_GROUPS.get(source, "") for source in sources])
    scopes: dict[str, np.ndarray] = {
        "full_stream": np.ones(len(records), dtype=bool),
        "paper_ood": groups != "",
    }
    for group in sorted(set(PAPER_OOD_GROUPS.values())):
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


def _load_historical_method(
    *,
    name: str,
    path: Path,
    records: list[dict[str, Any]],
) -> list[int]:
    rows = _read_jsonl(path)
    if len(rows) != len(records):
        raise ValueError(
            f"{name} selections length {len(rows)} != stream length {len(records)}"
        )
    correctness: list[int] = []
    for index, (record, row) in enumerate(zip(records, rows)):
        expected_id = str(record.get("id") or record.get("uid") or "")
        if str(row.get("id")) != expected_id:
            raise ValueError(
                f"{name} ID mismatch at {index}: {row.get('id')} != {expected_id}"
            )
        peer = int(row["selected_peer"])
        expected = int(float(record["peer_correct"][f"peer_{peer}"]) > 0.5)
        if int(row.get("selected_correct", expected)) != expected:
            raise ValueError(
                f"{name} selected_correct mismatch at {expected_id}: "
                f"{row.get('selected_correct')} != {expected}"
            )
        correctness.append(expected)
    return correctness


def _number_keyed_vector(
    row: Mapping[str, Any], field: str, *, dtype: np.dtype[Any]
) -> np.ndarray:
    mapping = row.get(field)
    if not isinstance(mapping, Mapping):
        raise ValueError(f"Historical row is missing {field}")
    values = []
    for peer in range(len(PEER_KEYS)):
        if str(peer) in mapping:
            values.append(mapping[str(peer)])
        elif peer in mapping:
            values.append(mapping[peer])
        else:
            raise ValueError(f"Historical {field} is missing peer {peer}")
    return np.asarray(values, dtype=dtype)


def _replay_historical_sigma_with_g(
    *,
    center_path: Path,
    sigma_without_g_path: Path,
    records: list[dict[str, Any]],
) -> HistoricalJointGReplay:
    """Reconstruct the established Sigma w/G comparator from frozen score streams."""

    center_rows = _read_jsonl(center_path)
    sigma_rows = _read_jsonl(sigma_without_g_path)
    if len(center_rows) != len(records) or len(sigma_rows) != len(records):
        raise ValueError(
            "Sigma w/G source lengths differ from the canonical stream: "
            f"center={len(center_rows)}, sigma={len(sigma_rows)}, records={len(records)}"
        )

    states = np.asarray(
        [
            [1.0 if (mask >> peer) & 1 else -1.0 for peer in range(len(PEER_KEYS))]
            for mask in range(1 << len(PEER_KEYS))
        ],
        dtype=np.float32,
    )
    history = np.zeros(len(PEER_KEYS), dtype=np.float32)
    graph = np.zeros((len(PEER_KEYS), len(PEER_KEYS)), dtype=np.float32)
    correctness: list[int] = []
    selections: list[int] = []
    cfg = SIGMA_WITH_G_REPLAY

    for index, (record, center_row, sigma_row) in enumerate(
        zip(records, center_rows, sigma_rows)
    ):
        expected_id = str(record.get("id") or record.get("uid") or "")
        center_id = str(center_row.get("id") or "")
        sigma_id = str(sigma_row.get("id") or "")
        if center_id != expected_id or sigma_id != expected_id:
            raise ValueError(
                f"Sigma w/G ID mismatch at {index}: "
                f"center={center_id}, sigma={sigma_id}, expected={expected_id}"
            )

        center_scores = _number_keyed_vector(
            center_row, "peer_scores", dtype=np.dtype(np.float32)
        )
        sigma_scores = _number_keyed_vector(
            sigma_row, "peer_scores", dtype=np.dtype(np.float32)
        )
        peer_correct, correctness_signed = _record_labels(record)
        for source_name, source_row in (
            ("center", center_row),
            ("sigma_without_g", sigma_row),
        ):
            source_correct = _number_keyed_vector(
                source_row, "peer_correct", dtype=np.dtype(np.int8)
            )
            if not np.array_equal(source_correct, peer_correct):
                raise ValueError(
                    f"Sigma w/G {source_name} labels changed at {expected_id}"
                )

        # This is the frozen historical comparator: decide from prior h/G, then update.
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
        selections.append(selected_peer)
        correctness.append(int(peer_correct[selected_peer]))

        centered = correctness_signed.astype(np.float32)
        if bool(cfg["centered_graph_update"]):
            centered = centered - centered.mean()
        history = (
            float(cfg["history_decay"]) * history
            + float(cfg["history_eta"]) * correctness_signed.astype(np.float32)
        )
        graph = (
            float(cfg["graph_decay"]) * graph
            + float(cfg["graph_eta"]) * np.outer(centered, centered)
        )
        np.fill_diagonal(graph, 0.0)

    return HistoricalJointGReplay(
        correctness=correctness,
        selections=selections,
        final_history=history.astype(np.float64).tolist(),
        final_graph=graph.astype(np.float64).tolist(),
    )


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


def _paired_comparisons(
    correctness: Mapping[str, list[int]],
    scopes: Mapping[str, np.ndarray],
    *,
    reference: str = "original_sigma_with_g",
    seed: int = 0,
    bootstrap_replicates: int = 100_000,
) -> dict[str, Any]:
    """Paired win/loss counts and a multinomial paired-bootstrap interval."""

    if reference not in correctness:
        return {}
    methods = ("route_M", "route_MG", "vote_M", "vote_MG", "vote_majority")
    result: dict[str, Any] = {
        "reference": reference,
        "bootstrap_seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "scopes": {},
    }
    rng = np.random.default_rng(seed)
    reference_values = np.asarray(correctness[reference], dtype=np.int8)
    for scope_name in ("paper_ood", "full_stream"):
        mask = scopes[scope_name]
        old = reference_values[mask]
        scope_result = {}
        for method in methods:
            new = np.asarray(correctness[method], dtype=np.int8)[mask]
            wins = int(((new == 1) & (old == 0)).sum())
            losses = int(((new == 0) & (old == 1)).sum())
            ties = int(new.size - wins - losses)
            probabilities = np.asarray([losses, ties, wins], dtype=np.float64)
            probabilities /= probabilities.sum()
            draws = rng.multinomial(
                int(new.size), probabilities, size=int(bootstrap_replicates)
            )
            bootstrap_delta = (draws[:, 2] - draws[:, 0]) / int(new.size)
            interval = np.quantile(bootstrap_delta, [0.025, 0.975])
            scope_result[method] = {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "delta_accuracy": float((new - old).mean()),
                "paired_bootstrap_95ci": [float(interval[0]), float(interval[1])],
            }
        result["scopes"][scope_name] = scope_result
    return result


def main() -> None:
    args = parse_args()
    profile = RUN_PROFILES[args.profile]
    strict_full_run = args.max_examples is None and not args.allow_noncanonical_stream
    _validate_output_target(
        profile_name=profile.name,
        output=args.output,
        strict_full_run=strict_full_run,
    )

    input_sha = _sha256_file(args.offline_data)
    records = JsonlDataset(args.offline_data).records
    if args.max_examples is not None:
        records = records[: int(args.max_examples)]
    ordered_id_sha = _ordered_id_sha256(records)
    if strict_full_run:
        if input_sha != CANONICAL_STREAM_SHA256:
            raise ValueError(
                f"OOD stream SHA256 changed: {input_sha} != {CANONICAL_STREAM_SHA256}"
            )
        if len(records) != CANONICAL_STREAM_SIZE:
            raise ValueError(
                f"OOD stream size changed: {len(records)} != {CANONICAL_STREAM_SIZE}"
            )
        if ordered_id_sha != CANONICAL_ORDERED_ID_SHA256:
            raise ValueError("OOD stream physical order changed")

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
    if strict_full_run:
        fixed_values = {
            "checkpoint_sha256": (checkpoint_sha, profile.checkpoint_sha256),
            "train_config_sha256": (
                train_config_sha,
                profile.train_config_sha256,
            ),
            "center_selections_sha256": (
                _sha256_file(args.center_selections),
                profile.center_selections_sha256,
            ),
            "sigma_selections_sha256": (
                _sha256_file(args.sigma_selections),
                profile.sigma_selections_sha256,
            ),
            "central_model": (
                str(args.central_model),
                profile.central_model,
            ),
            "dtype": (str(args.dtype).lower(), "bfloat16"),
            "max_length": (int(args.max_length), 2048),
            "gamma_G": (float(args.gamma_g), 0.9),
            "eta_G": (float(args.eta_g), 0.1),
            "lambda": (float(args.ridge), 0.1),
            "gamma": (checkpoint_gamma, profile.gamma),
            "eta": (checkpoint_eta, profile.eta),
        }
        changed = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in fixed_values.items()
            if actual != expected
        }
        if changed:
            raise ValueError(f"Preregistered constants/provenance changed: {changed}")

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
        gamma_g=float(args.gamma_g),
        eta_g=float(args.eta_g),
    )
    dictionary = DecayedLabelDictionary(num_peers=len(PEER_KEYS), gamma=gamma)
    seeds = tuple(int(value) for value in str(args.random_seeds).split(","))
    if seeds != (0, 1, 2):
        raise ValueError("The preregistered random seeds are exactly 0,1,2")
    random_generators = {seed: np.random.default_rng(seed) for seed in seeds}

    correctness: defaultdict[str, list[int]] = defaultdict(list)
    selections: defaultdict[str, list[int]] = defaultdict(list)
    robustness_correctness: defaultdict[str, list[int]] = defaultdict(list)
    route_ties = 0
    dictionary_ties = 0
    parser_label_mismatches = 0
    answer_group_label_conflicts = 0
    invalid_peer_answers = 0
    invalid_gold_answers = 0
    n_distinct_counts: Counter[int] = Counter()
    flip_counts = Counter()
    maximum_condition = {str(ridge): 0.0 for ridge in ROBUSTNESS_RIDGES}
    records_tmp = args.output / "records.jsonl.tmp"
    with records_tmp.open("w") as record_handle:
        for event_index, (record, phi_raw) in enumerate(zip(records, phis)):
            phi = np.asarray(phi_raw, dtype=np.float64)
            phi /= np.linalg.norm(phi)
            source = str(record.get("source") or "")
            answers = _record_answers(record)
            decision_record = _decision_record(record)

            # Decisions happen before peer_correct is accessed for this event.
            route = state.route(phi, event_index)
            route_mg = state.route_mg(phi, event_index, ridge=float(args.ridge))
            dictionary_route = dictionary.route(source, event_index)
            main_votes = state.all_votes(
                phi,
                decision_record,
                answers,
                ridge=float(args.ridge),
                clipped=False,
            )
            robustness_votes = state.robustness_votes(
                phi, decision_record, answers, ridges=ROBUSTNESS_RIDGES
            )
            random_peers = {
                seed: int(generator.choice(len(PEER_KEYS)))
                for seed, generator in random_generators.items()
            }
            canonical_answers = [
                canonical_answer(decision_record, answer) for answer in answers
            ]
            n_distinct = len(set(canonical_answers))
            n_distinct_counts[n_distinct] += 1
            route_ties += int(route.tied)
            dictionary_ties += int(dictionary_route.tied)
            invalid_peer_answers += sum(
                answer == INVALID_ANSWER for answer in canonical_answers
            )
            for ridge in ROBUSTNESS_RIDGES:
                condition = float(
                    np.linalg.cond(
                        state.G + float(ridge) * np.eye(len(PEER_KEYS))
                    )
                )
                maximum_condition[str(ridge)] = max(
                    maximum_condition[str(ridge)], condition
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
                for arm in ("maj", "G", "M", "MG")
            }
            event_correct = {
                "route_M": int(peer_correct[route.peer]),
                "route_MG": int(peer_correct[route_mg.peer]),
                "route_dictionary": int(peer_correct[dictionary_route.peer]),
                "vote_majority": vote_correct["maj"],
                "vote_G": vote_correct["G"],
                "vote_M": vote_correct["M"],
                "vote_MG": vote_correct["MG"],
            }
            for method, value in event_correct.items():
                correctness[method].append(value)
            selections["route_M"].append(route.peer)
            selections["route_MG"].append(route_mg.peer)
            selections["route_dictionary"].append(dictionary_route.peer)
            for peer in range(len(PEER_KEYS)):
                correctness[f"fixed_peer_{peer}"].append(int(peer_correct[peer]))
            for seed, peer in random_peers.items():
                method = f"random_seed_{seed}"
                correctness[method].append(int(peer_correct[peer]))
                selections[method].append(peer)

            for (ridge, clipped), result in robustness_votes.items():
                for arm in ("G", "M", "MG"):
                    method = (
                        f"vote_{arm}_{'clipped' if clipped else 'raw'}_"
                        f"lambda_{ridge:g}"
                    )
                    robustness_correctness[method].append(
                        _frozen_vote_correctness(
                            result.answers[arm], canonical_answers, peer_correct
                        )[0]
                    )

            majority_correct = event_correct["vote_majority"]
            mg_correct = event_correct["vote_MG"]
            if main_votes.answers["maj"] != main_votes.answers["MG"]:
                if not majority_correct and mg_correct:
                    flip_counts["helpful"] += 1
                elif majority_correct and not mg_correct:
                    flip_counts["harmful"] += 1
                else:
                    flip_counts["same_correctness"] += 1
            else:
                flip_counts["same_answer"] += 1

            row = {
                "t": event_index,
                "id": str(record.get("id") or record.get("uid") or ""),
                "source": source,
                "paper_ood_group": PAPER_OOD_GROUPS.get(source),
                "n_distinct": n_distinct,
                "canonical_answers": canonical_answers,
                "canonical_gold": gold,
                "peer_correct": peer_correct.tolist(),
                "route_M_peer": route.peer,
                "route_M_scores": route.scores.tolist(),
                "route_MG_peer": route_mg.peer,
                "route_MG_scores": route_mg.scores.tolist(),
                "route_dictionary_peer": dictionary_route.peer,
                "random_peers": random_peers,
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
            dictionary.update(source, correctness_signed)
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
    if strict_full_run:
        paper_mask = scopes["paper_ood"]
        if int(paper_mask.sum()) != PAPER_OOD_SIZE:
            raise ValueError(
                f"Paper OOD mask changed: {int(paper_mask.sum())} != {PAPER_OOD_SIZE}"
            )
        if _ordered_id_sha256(
            record for record, keep in zip(records, paper_mask) if keep
        ) != PAPER_OOD_ORDERED_ID_SHA256:
            raise ValueError("Paper OOD relative order changed")

    if len(records) == CANONICAL_STREAM_SIZE:
        correctness["original_center"] = _load_historical_method(
            name="original_center",
            path=args.center_selections,
            records=records,
        )
        correctness["original_sigma_without_g"] = _load_historical_method(
            name="original_sigma_without_g",
            path=args.sigma_selections,
            records=records,
        )
        historical_sigma_with_g = _replay_historical_sigma_with_g(
            center_path=args.center_selections,
            sigma_without_g_path=args.sigma_selections,
            records=records,
        )
        correctness["original_sigma_with_g"] = (
            historical_sigma_with_g.correctness
        )
        selections["original_sigma_with_g"] = historical_sigma_with_g.selections
        if strict_full_run:
            observed = {
                "full_correct": sum(historical_sigma_with_g.correctness),
                "paper_ood_correct": int(
                    np.asarray(
                        historical_sigma_with_g.correctness, dtype=np.int8
                    )[paper_mask].sum()
                ),
                "selected_peers": tuple(
                    historical_sigma_with_g.selections.count(peer)
                    for peer in range(len(PEER_KEYS))
                ),
            }
            expected = {
                "full_correct": profile.sigma_with_g_full_correct,
                "paper_ood_correct": profile.sigma_with_g_paper_ood_correct,
                "selected_peers": profile.sigma_with_g_selected_peers,
            }
            if observed != expected:
                raise ValueError(
                    f"Historical Sigma w/G replay changed: {observed} != {expected}"
                )
    else:
        historical_sigma_with_g = None

    all_correctness = dict(correctness) | dict(robustness_correctness)
    for arm in ("G", "M", "MG"):
        main_name = f"vote_{arm}"
        robustness_name = f"vote_{arm}_raw_lambda_0.1"
        if correctness[main_name] != robustness_correctness[robustness_name]:
            raise ValueError(
                f"Main {main_name} differs from preregistered raw lambda=0.1 replay"
            )
    accuracy = _summarize_correctness(all_correctness, scopes)
    for scope, methods in accuracy.items():
        fixed = [methods[f"fixed_peer_{peer}"] for peer in range(len(PEER_KEYS))]
        best_peer = max(
            range(len(PEER_KEYS)), key=lambda peer: fixed[peer]["accuracy"]
        )
        methods["best_fixed_peer"] = dict(fixed[best_peer]) | {"peer": best_peer}
        random_accuracies = [
            methods[f"random_seed_{seed}"]["accuracy"] for seed in seeds
        ]
        methods["random_routing_3seed"] = {
            "seeds": list(seeds),
            "accuracies": random_accuracies,
            "mean_accuracy": float(np.mean(random_accuracies)),
            "std_accuracy": float(np.std(random_accuracies, ddof=0)),
            "total_per_seed": int(scopes[scope].sum()),
        }

    paper_mask = scopes["paper_ood"]
    paper_flip_counts = Counter()
    with (args.output / "records.jsonl").open() as handle:
        for keep, line in zip(paper_mask, handle):
            if not keep:
                continue
            row = json.loads(line)
            maj_answer = row["vote_answers"]["maj"]
            mg_answer = row["vote_answers"]["MG"]
            if maj_answer == mg_answer:
                paper_flip_counts["same_answer"] += 1
            elif not row["correct"]["vote_majority"] and row["correct"]["vote_MG"]:
                paper_flip_counts["helpful"] += 1
            elif row["correct"]["vote_majority"] and not row["correct"]["vote_MG"]:
                paper_flip_counts["harmful"] += 1
            else:
                paper_flip_counts["same_correctness"] += 1

    final_m, final_g = state.snapshot()
    summary = {
        "status": "complete" if strict_full_run else "diagnostic",
        "experiment": "teacher_training_free_ood_memory_readout",
        "profile": profile.name,
        "instruction_file": str(Path("instruction.md")),
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
            "paper_ood_num_examples": int(paper_mask.sum()),
            "paper_ood_ordered_id_sha256": _ordered_id_sha256(
                record for record, keep in zip(records, paper_mask) if keep
            ),
            "paper_ood_groups": sorted(set(PAPER_OOD_GROUPS.values())),
            "state_trajectory": "full_stream_then_reporting_mask",
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
            "gamma_G": float(args.gamma_g),
            "eta_G": float(args.eta_g),
            "lambda_main": float(args.ridge),
            "lambda_robustness": list(ROBUSTNESS_RIDGES),
            "random_seeds": list(seeds),
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
        "paired_comparisons": _paired_comparisons(
            correctness, scopes, reference="original_sigma_with_g"
        ),
        "selection": _selection_summary(selections, correctness, scopes),
        "diagnostics": {
            "route_all_equal_ties": route_ties,
            "dictionary_all_equal_ties": dictionary_ties,
            "n_distinct_answer_counts": dict(sorted(n_distinct_counts.items())),
            "invalid_peer_answers": invalid_peer_answers,
            "invalid_gold_answers": invalid_gold_answers,
            "parser_vs_external_label_mismatches": parser_label_mismatches,
            "canonical_group_label_conflicts": answer_group_label_conflicts,
            "majority_to_MG_flips_full": dict(flip_counts),
            "majority_to_MG_flips_paper_ood": dict(paper_flip_counts),
            "max_condition_number_G_plus_lambda_I": maximum_condition,
            "final_M_frobenius_norm": float(np.linalg.norm(final_m)),
            "final_G": final_g.tolist(),
        },
        "historical_label_protocol": {
            "frozen": True,
            "sigma_comparator": {
                "main_key": "original_sigma_with_g",
                "score_sources": {
                    "center": str(args.center_selections),
                    "residual_sigma_without_g": str(args.sigma_selections),
                },
                "replay": dict(SIGMA_WITH_G_REPLAY),
                "decision_then_update": True,
                "final_history": (
                    historical_sigma_with_g.final_history
                    if historical_sigma_with_g is not None
                    else None
                ),
                "final_graph": (
                    historical_sigma_with_g.final_graph
                    if historical_sigma_with_g is not None
                    else None
                ),
            },
            "known_issue": (
                "Canonical grouping preserves standalone BBH option tokens such as "
                "(A), but vote scoring and memory writes remain frozen to the "
                "lowest-index group member's historical peer_correct. Parser repairs "
                "therefore contribute no direct correctness-label gain."
            ),
        },
        "strict_provenance": strict_full_run,
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
