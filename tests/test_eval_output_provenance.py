from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_sigma_eval_output.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _run_validator(
    run_dir: Path,
    benchmark: Path,
    arm: str,
    checkpoint: Path | None = None,
    legacy_prompt_protocol: str = "off",
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(VALIDATOR),
        "--run-dir",
        str(run_dir),
        "--benchmark",
        str(benchmark),
        "--arm",
        arm,
        "--legacy-prompt-protocol",
        legacy_prompt_protocol,
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _common_metrics(benchmark: Path) -> dict:
    return {
        "accuracy": 1.0,
        "num_samples": 1,
        "num_peers": 1,
        "max_length": 12288,
        "legacy_prompt_protocol": "off",
        "prompt_protocol": "current",
        "prompt_context_format": "rag_context_text_8000",
        "candidate_tokenization_format": "complete_prompt_fail_on_overflow",
        "offline_data_sha256": _sha256(benchmark),
    }


def test_base_validator_accepts_explicit_legacy_prompt_provenance(tmp_path: Path) -> None:
    benchmark = tmp_path / "p50.jsonl"
    _write_jsonl(benchmark, [{
        "uid": "legacy:0",
        "peer_responses": {"peer_0": "answer"},
        "peer_correct": {"peer_0": 1},
    }])
    run_dir = tmp_path / "base"
    run_dir.mkdir()
    _write_jsonl(run_dir / "selections.jsonl", [{
        "id": "legacy:0",
        "selected_peer": 0,
        "selected_correct": 1,
        "memory_used": False,
        "score_source": "center",
    }])
    metrics = _common_metrics(benchmark) | {
        "max_length": 8192,
        "legacy_prompt_protocol": "on",
        "prompt_protocol": "legacy_2685",
        "prompt_context_format": "legacy_python_str_context",
        "candidate_tokenization_format": "tokenizer_truncation_max_length_8192",
        "checkpoint_sha256": None,
        "ablate_memory": True,
        "graph_posterior": "off",
    }
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics))

    assert _run_validator(
        run_dir,
        benchmark,
        "base",
        legacy_prompt_protocol="on",
    ).returncode == 0
    wrong_protocol = _run_validator(run_dir, benchmark, "base")
    assert wrong_protocol.returncode != 0
    assert "expected 12288" in wrong_protocol.stderr


def test_base_provenance_requires_current_data_and_null_checkpoint(tmp_path: Path) -> None:
    benchmark = tmp_path / "p0.jsonl"
    _write_jsonl(benchmark, [{
        "uid": "example:0",
        "problem": "before",
        "peer_responses": {"peer_0": "answer"},
        "peer_correct": {"peer_0": 1},
    }])
    run_dir = tmp_path / "base"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "selections.jsonl",
        [{
            "id": "example:0",
            "selected_peer": 0,
            "selected_correct": 1,
            "memory_used": False,
            "score_source": "center",
        }],
    )
    metrics = _common_metrics(benchmark) | {
        "checkpoint_sha256": None,
        "ablate_memory": True,
        "graph_posterior": "off",
    }
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics))

    assert _run_validator(run_dir, benchmark, "base").returncode == 0

    _write_jsonl(benchmark, [{
        "uid": "example:0",
        "problem": "after",
        "peer_responses": {"peer_0": "answer"},
        "peer_correct": {"peer_0": 1},
    }])
    stale = _run_validator(run_dir, benchmark, "base")
    assert stale.returncode != 0
    assert "benchmark SHA256 differs" in stale.stderr

    metrics["offline_data_sha256"] = _sha256(benchmark)
    metrics["checkpoint_sha256"] = "not-null"
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics))
    invalid_base = _run_validator(run_dir, benchmark, "base")
    assert invalid_base.returncode != 0
    assert "checkpoint_sha256 must be null" in invalid_base.stderr


def test_sigma_provenance_requires_and_rehashes_checkpoint(tmp_path: Path) -> None:
    benchmark = tmp_path / "p50.jsonl"
    _write_jsonl(benchmark, [{
        "uid": "example:0",
        "peer_responses": {"peer_0": "answer"},
        "peer_correct": {"peer_0": 1},
    }])
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = checkpoint / "sym_memory.pt"
    state.write_bytes(b"checkpoint-v1")
    run_dir = tmp_path / "sigma"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "selections.jsonl",
        [{
            "id": "example:0",
            "selected_peer": 0,
            "selected_correct": 1,
            "memory_used": True,
            "score_source": "sigma",
        }],
    )
    metrics = _common_metrics(benchmark) | {
        "checkpoint_sha256": _sha256(state),
        "ablate_memory": False,
        "graph_posterior": "off",
    }
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics))

    missing_arg = _run_validator(run_dir, benchmark, "sigma_wo_g")
    assert missing_arg.returncode != 0
    assert "--checkpoint is required for Sigma arms" in missing_arg.stderr
    assert _run_validator(run_dir, benchmark, "sigma_wo_g", checkpoint).returncode == 0

    state.write_bytes(b"checkpoint-v2")
    stale = _run_validator(run_dir, benchmark, "sigma_wo_g", checkpoint)
    assert stale.returncode != 0
    assert "checkpoint SHA256 differs" in stale.stderr


def test_validator_recomputes_selected_correct_from_benchmark(tmp_path: Path) -> None:
    benchmark = tmp_path / "p0.jsonl"
    _write_jsonl(benchmark, [{
        "uid": "example:0",
        "peer_responses": {"peer_0": "right", "peer_1": "wrong"},
        "peer_correct": {"peer_0": 1, "peer_1": 0},
    }])
    run_dir = tmp_path / "base"
    run_dir.mkdir()
    valid_row = {
        "id": "example:0",
        "selected_peer": 0,
        "selected_correct": 1,
        "memory_used": False,
        "score_source": "center",
    }
    _write_jsonl(run_dir / "selections.jsonl", [valid_row])
    metrics = _common_metrics(benchmark) | {
        "num_peers": 2,
        "checkpoint_sha256": None,
        "ablate_memory": True,
        "graph_posterior": "off",
    }
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics))
    assert _run_validator(run_dir, benchmark, "base").returncode == 0

    _write_jsonl(
        run_dir / "selections.jsonl",
        [valid_row | {"selected_correct": 0}],
    )
    bad_correctness = _run_validator(run_dir, benchmark, "base")
    assert bad_correctness.returncode != 0
    assert "selected_correct=0, expected 1" in bad_correctness.stderr

    _write_jsonl(
        run_dir / "selections.jsonl",
        [valid_row | {"selected_peer": 1}],
    )
    bad_peer = _run_validator(run_dir, benchmark, "base")
    assert bad_peer.returncode != 0
    assert "selected_correct=1, expected 0" in bad_peer.stderr


def test_validator_prefers_id_when_legacy_uid_differs(tmp_path: Path) -> None:
    benchmark = tmp_path / "p50.jsonl"
    _write_jsonl(benchmark, [{
        "id": "current:0",
        "uid": "legacy:0",
        "peer_responses": {"peer_0": "right"},
        "peer_correct": {"peer_0": 1},
    }])
    run_dir = tmp_path / "base"
    run_dir.mkdir()
    _write_jsonl(run_dir / "selections.jsonl", [{
        "id": "current:0",
        "selected_peer": 0,
        "selected_correct": 1,
        "memory_used": False,
        "score_source": "center",
    }])
    metrics = _common_metrics(benchmark) | {
        "num_peers": 1,
        "checkpoint_sha256": None,
        "ablate_memory": True,
        "graph_posterior": "off",
    }
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics))

    valid = _run_validator(run_dir, benchmark, "base")
    assert valid.returncode == 0, valid.stderr


def test_validator_accepts_source_duplicate_ids_in_original_order(tmp_path: Path) -> None:
    benchmark = tmp_path / "p50.jsonl"
    source_rows = [
        {
            "id": "duplicate",
            "peer_responses": {"peer_0": "right"},
            "peer_correct": {"peer_0": 1},
        },
        {
            "id": "duplicate",
            "peer_responses": {"peer_0": "also right"},
            "peer_correct": {"peer_0": 1},
        },
    ]
    _write_jsonl(benchmark, source_rows)
    run_dir = tmp_path / "base"
    run_dir.mkdir()
    selections = [
        {
            "id": "duplicate",
            "selected_peer": 0,
            "selected_correct": 1,
            "memory_used": False,
            "score_source": "center",
        }
        for _ in source_rows
    ]
    _write_jsonl(run_dir / "selections.jsonl", selections)
    metrics = _common_metrics(benchmark) | {
        "accuracy": 1.0,
        "num_samples": 2,
        "num_peers": 1,
        "checkpoint_sha256": None,
        "ablate_memory": True,
        "graph_posterior": "off",
    }
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics))

    valid = _run_validator(run_dir, benchmark, "base")
    assert valid.returncode == 0, valid.stderr


def test_sigma_with_g_requires_ising_selection_provenance(tmp_path: Path) -> None:
    benchmark = tmp_path / "p90.jsonl"
    _write_jsonl(benchmark, [{
        "uid": "example:0",
        "peer_responses": {"peer_0": "answer"},
        "peer_correct": {"peer_0": 1},
    }])
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = checkpoint / "sym_memory.pt"
    state.write_bytes(b"checkpoint")
    run_dir = tmp_path / "sigma_w_g"
    run_dir.mkdir()
    row = {
        "id": "example:0",
        "selected_peer": 0,
        "selected_correct": 1,
        "memory_used": True,
        "score_source": "ising",
    }
    _write_jsonl(run_dir / "selections.jsonl", [row])
    metrics = _common_metrics(benchmark) | {
        "checkpoint_sha256": _sha256(state),
        "ablate_memory": False,
        "graph_posterior": "ising",
    }
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics))

    valid = _run_validator(run_dir, benchmark, "sigma_w_g", checkpoint)
    assert valid.returncode == 0, valid.stderr

    _write_jsonl(run_dir / "selections.jsonl", [row | {"score_source": "sigma"}])
    invalid = _run_validator(run_dir, benchmark, "sigma_w_g", checkpoint)
    assert invalid.returncode != 0
    assert "expected 'ising'" in invalid.stderr
