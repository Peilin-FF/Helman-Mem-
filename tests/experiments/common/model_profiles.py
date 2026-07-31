"""Load central-model run locations from repository configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_CONFIG = REPO_ROOT / "configs/experiments/central_models.json"


@dataclass(frozen=True)
class ModelRun:
    name: str
    central_model: str
    checkpoint: Path
    routing_output: Path


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_model_runs(path: Path = DEFAULT_PROFILE_CONFIG) -> dict[str, ModelRun]:
    raw = json.loads(path.read_text())
    return {
        name: ModelRun(
            name=name,
            central_model=str(values["central_model"]),
            checkpoint=_repo_path(str(values["checkpoint"])),
            routing_output=_repo_path(str(values["routing_output"])),
        )
        for name, values in raw.items()
    }


MODEL_RUNS = load_model_runs()
