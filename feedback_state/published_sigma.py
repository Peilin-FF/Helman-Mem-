"""Fail-closed provenance checks for the five published Sigma checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from feedback_state.checkpoint_manifest import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_SIGMA_TRAIN_DATA = "data/mixed_train_labeled.jsonl"
PUBLISHED_SIGMA_MEMORY_KEYS = frozenset({
    "_eta_raw",
    "_theta",
    "_theta_joint",
    "proto_centroids",
    "proto_mean",
    "proto_std",
    "proto_ready",
    "M",
    "G",
    "proto_proj.weight",
})
PUBLISHED_SIGMA_STEERER_KEYS = frozenset({"gain", "proj.weight"})


@dataclass(frozen=True)
class PublishedSigmaSpec:
    name: str
    checkpoint_relative: str
    central_model: str
    state_sha256: str
    config_sha256: str

    @property
    def checkpoint_path(self) -> Path:
        return (REPOSITORY_ROOT / self.checkpoint_relative).resolve()


PUBLISHED_SIGMA_SPECS = (
    PublishedSigmaSpec(
        name="Qwen3-0.6B",
        checkpoint_relative="outputs/sigma_candidate_yesno_q3_0.6b/proto",
        central_model="/mnt/data/peilin/HF_MODEL/Qwen3-0.6B",
        state_sha256="bac0f33ddf5f6f7ddbcc8609fbdcc90960ea5e39df3180c2907ad86bcd65e36f",
        config_sha256="25649484b1b83426eee42e5756e8bfb57ef34377f89f613f5ead7b0e1a8e88f5",
    ),
    PublishedSigmaSpec(
        name="Qwen3-4B",
        checkpoint_relative="outputs/sigma_candidate_yesno_q3_4b/proto",
        central_model="/mnt/data/peilin/HF_MODEL/Qwen3-4B",
        state_sha256="ecc70510d9b519aeb9b2206aa3347580addf770e8020a93f7109187eebe5165d",
        config_sha256="3dc5565bf10a5f210ec4826f5d5db695d4fb8ce82fc95706c56168fca9477004",
    ),
    PublishedSigmaSpec(
        name="Qwen3-8B",
        checkpoint_relative="outputs/sigma_candidate_yesno_q3_8b/proto",
        central_model="/mnt/data/peilin/HF_MODEL/Qwen3-8B",
        state_sha256="8a5386104caee0ef1de3d26866a18d86d39a45e70b117107c18ea98e156a1b63",
        config_sha256="d22a58225c3fb551106d7c5538248857af30393492de3f948973a3525904aa1e",
    ),
    PublishedSigmaSpec(
        name="Qwen3.5-4B",
        checkpoint_relative="outputs/sigma_candidate_yesno_q35_4b/proto",
        central_model="/mnt/data/peilin/HF_MODEL/Qwen3.5-4B",
        state_sha256="7d399de393415d2d45bcdd99f03dcfbdcaa5e87044740c6f53ac45ba5a46e5ff",
        config_sha256="8dbffbe7bb9628d3d3be93e4d8c5d5d4002b51cdfc428a62eedb4efda42b82af",
    ),
    PublishedSigmaSpec(
        name="Qwen3.5-9B",
        checkpoint_relative="outputs/sigma_candidate_yesno_q35_9b/proto",
        central_model="/mnt/data/peilin/HF_MODEL/Qwen3.5-9B",
        state_sha256="12df397e35a5fd7318cd3edbfcde166b14b77e96823d0904714f77c4a55fdd92",
        config_sha256="1484f592cc36f669beae1611ab3f2319b464a6034261ea7b04b57025fd6b0b22",
    ),
)
PUBLISHED_SIGMA_BY_PATH = {
    spec.checkpoint_path: spec for spec in PUBLISHED_SIGMA_SPECS
}


@dataclass(frozen=True)
class PublishedSigmaCheckpoint:
    path: Path
    spec: PublishedSigmaSpec
    config: dict[str, Any]
    payload: dict[str, dict[str, torch.Tensor]]
    state_sha256: str
    config_sha256: str


def validate_published_sigma_checkpoint(
    checkpoint: Path,
) -> PublishedSigmaCheckpoint:
    """Load one canonical paper checkpoint and reject every substitution."""

    checkpoint = Path(checkpoint).resolve()
    try:
        spec = PUBLISHED_SIGMA_BY_PATH[checkpoint]
    except KeyError as exc:
        allowed = ", ".join(spec.checkpoint_relative for spec in PUBLISHED_SIGMA_SPECS)
        raise AssertionError(
            f"checkpoint is not a pinned published Sigma path: {checkpoint}; "
            f"allowed={allowed}"
        ) from exc

    state_path = checkpoint / "sym_memory.pt"
    config_path = checkpoint / "train_config.json"
    if not state_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("published Sigma checkpoint state/config is missing")
    state_sha = sha256_file(state_path)
    config_sha = sha256_file(config_path)
    if state_sha != spec.state_sha256:
        raise AssertionError(
            f"published Sigma state sha256={state_sha}, expected {spec.state_sha256}"
        )
    if config_sha != spec.config_sha256:
        raise AssertionError(
            f"published Sigma config sha256={config_sha}, expected {spec.config_sha256}"
        )

    config = json.loads(config_path.read_text())
    expected_config = {
        "central_model": spec.central_model,
        "num_peers": "3",
        "rank": "16",
        "phi_mode": "proto",
        "write_mode": "addr",
        "decay_mode": "scalar",
        "per_peer_decay": "off",
        "score_mode": "candidate_yesno",
        "peer_mode": "joint",
        "use_joint": "off",
        "include_context": "True",
        "max_length": "8192",
        "offline_data": PUBLISHED_SIGMA_TRAIN_DATA,
    }
    for field, expected in expected_config.items():
        if str(config.get(field)) != expected:
            raise AssertionError(
                f"published Sigma config {field}={config.get(field)!r}, expected {expected!r}"
            )

    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"mem", "steerer"}:
        raise AssertionError("published Sigma checkpoint top-level keys differ")
    if set(payload["mem"]) != PUBLISHED_SIGMA_MEMORY_KEYS:
        raise AssertionError("published Sigma memory keys differ")
    if set(payload["steerer"]) != PUBLISHED_SIGMA_STEERER_KEYS:
        raise AssertionError("published Sigma steerer keys differ")
    for group in ("mem", "steerer"):
        for name, value in payload[group].items():
            if not bool(torch.isfinite(torch.as_tensor(value).float()).all()):
                raise AssertionError(f"published Sigma {group}.{name} is non-finite")
    return PublishedSigmaCheckpoint(
        path=checkpoint,
        spec=spec,
        config=config,
        payload=payload,
        state_sha256=state_sha,
        config_sha256=config_sha,
    )
