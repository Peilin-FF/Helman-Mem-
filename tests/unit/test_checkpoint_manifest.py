from __future__ import annotations

import json

import pytest

import feedback_state.checkpoint_manifest as checkpoint_manifest
from feedback_state.checkpoint_manifest import (
    validate_checkpoint_manifest,
    write_checkpoint_manifest,
)


def _checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "sym_memory.pt").write_bytes(b"weights-v1")
    train_data = tmp_path / "train.jsonl"
    train_data.write_text(json.dumps({"id": "one"}) + "\n")
    (checkpoint / "train_config.json").write_text(json.dumps({
        "offline_data": str(train_data),
    }))
    return checkpoint, train_data


def test_train_save_manifest_binds_state_config_and_training_stream(tmp_path):
    checkpoint, train_data = _checkpoint(tmp_path)
    write_checkpoint_manifest(checkpoint, train_data)
    manifest = validate_checkpoint_manifest(
        checkpoint,
        expected_train_data=train_data,
        required_origin="train_save",
    )

    assert manifest["origin"] == "train_save"
    assert manifest["train_data_rows"] == 1

    (checkpoint / "train_config.json").write_text("{}")
    with pytest.raises(AssertionError, match="train_config_sha256"):
        validate_checkpoint_manifest(checkpoint, expected_train_data=train_data)


def test_manifest_rejects_weight_and_training_data_tampering(tmp_path):
    checkpoint, train_data = _checkpoint(tmp_path)
    write_checkpoint_manifest(checkpoint, train_data)

    (checkpoint / "sym_memory.pt").write_bytes(b"weights-v2")
    with pytest.raises(AssertionError, match="sym_memory_sha256"):
        validate_checkpoint_manifest(checkpoint, expected_train_data=train_data)

    (checkpoint / "sym_memory.pt").write_bytes(b"weights-v1")
    train_data.write_text(
        json.dumps({"id": "one"}) + "\n" + json.dumps({"id": "two"}) + "\n"
    )
    with pytest.raises(AssertionError, match="train_data"):
        validate_checkpoint_manifest(checkpoint, expected_train_data=train_data)


def test_backfilled_manifest_is_labeled_and_rejected_as_formal_training_save(tmp_path):
    checkpoint, train_data = _checkpoint(tmp_path)
    write_checkpoint_manifest(checkpoint, train_data, origin="backfill")

    assert validate_checkpoint_manifest(checkpoint)["origin"] == "backfill"
    with pytest.raises(AssertionError, match="origin='backfill'"):
        validate_checkpoint_manifest(checkpoint, required_origin="train_save")


def test_manifest_uses_repository_relative_training_data_path(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_manifest, "REPO_ROOT", tmp_path)
    checkpoint, train_data = _checkpoint(tmp_path)

    manifest = write_checkpoint_manifest(checkpoint, train_data)

    assert manifest["train_data_path"] == "train.jsonl"
    assert validate_checkpoint_manifest(
        checkpoint, expected_train_data=train_data
    )["train_data_path"] == "train.jsonl"
