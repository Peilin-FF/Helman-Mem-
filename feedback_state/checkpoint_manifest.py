"""Tamper-evident provenance for a Sigma-Mem checkpoint directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = "checkpoint_manifest.json"
MANIFEST_SCHEMA = "sigma_mem_checkpoint_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FIELDS = frozenset({
    "schema",
    "origin",
    "sym_memory_sha256",
    "train_config_sha256",
    "train_data_path",
    "train_data_sha256",
    "train_data_rows",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one file or a directory tree, including relative file names."""
    path = path.resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"cannot hash empty directory: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = candidate.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            count += 1
    return count


def _manifest_data_path(path: Path) -> str:
    """Use repository-relative paths when possible so checkpoints stay portable."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_manifest_data_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def build_checkpoint_manifest(
    checkpoint: Path,
    train_data: Path,
    *,
    origin: str,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    train_data = train_data.resolve()
    state_path = checkpoint / "sym_memory.pt"
    config_path = checkpoint / "train_config.json"
    for path in (state_path, config_path, train_data):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "schema": MANIFEST_SCHEMA,
        "origin": origin,
        "sym_memory_sha256": sha256_file(state_path),
        "train_config_sha256": sha256_file(config_path),
        "train_data_path": _manifest_data_path(train_data),
        "train_data_sha256": sha256_file(train_data),
        "train_data_rows": jsonl_rows(train_data),
    }


def write_checkpoint_manifest(
    checkpoint: Path,
    train_data: Path,
    *,
    origin: str = "train_save",
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    if origin not in {"train_save", "backfill"}:
        raise ValueError(f"unsupported checkpoint manifest origin: {origin!r}")
    manifest = build_checkpoint_manifest(checkpoint, train_data, origin=origin)
    destination = checkpoint / MANIFEST_FILENAME
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return manifest


def validate_checkpoint_manifest(
    checkpoint: Path,
    *,
    expected_train_data: Path | None = None,
    required_origin: str | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    manifest_path = checkpoint / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"checkpoint manifest is missing: {manifest_path}; "
            "legacy checkpoints may omit the manifest"
        )
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        actual = sorted(manifest) if isinstance(manifest, dict) else type(manifest).__name__
        raise AssertionError(f"checkpoint manifest fields differ: {actual}")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise AssertionError(f"unsupported checkpoint manifest schema: {manifest.get('schema')!r}")
    if manifest.get("origin") not in {"train_save", "backfill"}:
        raise AssertionError(f"invalid checkpoint manifest origin: {manifest.get('origin')!r}")
    if required_origin is not None and manifest.get("origin") != required_origin:
        raise AssertionError(
            f"checkpoint manifest origin={manifest.get('origin')!r}, "
            f"required {required_origin!r}"
        )

    train_data = _resolve_manifest_data_path(str(manifest["train_data_path"]))
    if expected_train_data is not None and train_data != Path(expected_train_data).resolve():
        raise AssertionError(
            f"manifest train_data_path={train_data}, expected {Path(expected_train_data).resolve()}"
        )
    expected = build_checkpoint_manifest(
        checkpoint,
        train_data,
        origin=str(manifest["origin"]),
    )
    for field in MANIFEST_FIELDS:
        if manifest.get(field) != expected.get(field):
            raise AssertionError(
                f"checkpoint manifest {field}={manifest.get(field)!r}, "
                f"expected {expected.get(field)!r}"
            )
    return manifest
