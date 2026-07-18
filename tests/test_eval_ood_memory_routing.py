from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from feedback_state.data import JsonlDataset
from eval_ood_memory_routing import (
    CANONICAL_ORDERED_ID_SHA256,
    CANONICAL_STREAM_SHA256,
    CANONICAL_STREAM_SIZE,
    PAPER_OOD_ORDERED_ID_SHA256,
    PAPER_OOD_SIZE,
    RUN_PROFILES,
    _decision_record,
    _frozen_vote_correctness,
    _ordered_id_sha256,
    _paired_comparisons,
    _replay_historical_sigma_with_g,
    _scope_indices,
    _sha256_file,
    _validate_output_target,
    parse_args,
)


def test_all_requested_model_profiles_are_registered() -> None:
    assert set(RUN_PROFILES) == {
        "q3_0_6b",
        "q3_4b",
        "q3_8b",
        "q35_4b",
        "q35_9b",
    }


def test_q35_profile_resolves_model_specific_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_ood_memory_routing.py", "--profile", "q35_4b"],
    )

    args = parse_args()
    profile = RUN_PROFILES["q35_4b"]

    assert args.central_model == profile.central_model
    assert args.checkpoint == profile.checkpoint
    assert args.center_selections == profile.center_selections
    assert args.sigma_selections == profile.sigma_selections
    assert args.output == profile.output
    assert profile.gamma == 0.8776268362998962
    assert profile.eta == 0.4006841480731964


def test_new_profile_defaults_are_isolated(monkeypatch) -> None:
    for name in ("q3_0_6b", "q3_8b", "q35_9b"):
        monkeypatch.setattr(
            sys,
            "argv",
            ["eval_ood_memory_routing.py", "--profile", name],
        )
        args = parse_args()
        profile = RUN_PROFILES[name]

        assert args.central_model == profile.central_model
        assert args.checkpoint == profile.checkpoint
        assert args.center_selections == profile.center_selections
        assert args.sigma_selections == profile.sigma_selections
        assert args.output == profile.output


def test_default_profile_remains_qwen3_4b(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["eval_ood_memory_routing.py"])

    args = parse_args()
    profile = RUN_PROFILES["q3_4b"]

    assert args.profile == "q3_4b"
    assert args.central_model == profile.central_model
    assert args.checkpoint == profile.checkpoint
    assert args.center_selections == profile.center_selections
    assert args.sigma_selections == profile.sigma_selections


def test_registered_outputs_are_isolated_by_profile() -> None:
    assert len({profile.output for profile in RUN_PROFILES.values()}) == len(
        RUN_PROFILES
    )
    with np.testing.assert_raises(ValueError):
        _validate_output_target(
            profile_name="q35_4b",
            output=RUN_PROFILES["q3_4b"].output,
            strict_full_run=True,
        )


def test_diagnostic_run_cannot_overwrite_registered_output() -> None:
    with np.testing.assert_raises(ValueError):
        _validate_output_target(
            profile_name="q35_4b",
            output=RUN_PROFILES["q35_4b"].output,
            strict_full_run=False,
        )


def test_profile_provenance_hashes_match_inputs() -> None:
    for profile in RUN_PROFILES.values():
        assert (
            _sha256_file(profile.checkpoint / "sym_memory.pt")
            == profile.checkpoint_sha256
        )
        assert (
            _sha256_file(profile.checkpoint / "train_config.json")
            == profile.train_config_sha256
        )
        assert (
            _sha256_file(profile.center_selections)
            == profile.center_selections_sha256
        )
        assert (
            _sha256_file(profile.sigma_selections)
            == profile.sigma_selections_sha256
        )


def test_decision_record_removes_current_event_supervision() -> None:
    record = {
        "problem": "question",
        "answer": "gold",
        "peer_correct": {"peer_0": 1},
        "correctness_by_peer": {"peer_0": 1},
        "peer_responses": {"peer_0": "candidate"},
    }

    decision = _decision_record(record)

    assert decision["problem"] == "question"
    assert decision["peer_responses"] == {"peer_0": "candidate"}
    assert "answer" not in decision
    assert "peer_correct" not in decision
    assert "correctness_by_peer" not in decision


def test_frozen_vote_scoring_uses_lowest_peer_in_winning_group() -> None:
    canonical = ["opt:a", "opt:b", "opt:a"]
    external = np.asarray([0, 1, 1], dtype=np.int8)

    correctness, representative = _frozen_vote_correctness(
        "opt:a", canonical, external
    )

    assert representative == 0
    assert correctness == 0


def test_paired_comparison_counts_wins_losses_and_ties() -> None:
    values = {
        "original_sigma_with_g": [1, 0, 1, 0],
        "route_M": [1, 1, 0, 0],
        "route_MG": [1, 1, 0, 0],
        "vote_M": [1, 1, 0, 0],
        "vote_MG": [1, 1, 0, 0],
        "vote_majority": [1, 1, 0, 0],
    }
    scopes = {
        "paper_ood": np.ones(4, dtype=bool),
        "full_stream": np.ones(4, dtype=bool),
    }

    result = _paired_comparisons(values, scopes, bootstrap_replicates=100)
    comparison = result["scopes"]["paper_ood"]["route_M"]

    assert comparison["wins"] == 1
    assert comparison["losses"] == 1
    assert comparison["ties"] == 2
    assert comparison["delta_accuracy"] == 0.0


def test_sigma_with_g_historical_replay_matches_pinned_results() -> None:
    path = Path(
        "data/peer_generalization/hf_generalization_all_canonical_peers/"
        "hf_generalization_all_peer012.labeled.jsonl"
    )
    records = JsonlDataset(path).records
    paper_mask = _scope_indices(records)["paper_ood"]

    for profile in RUN_PROFILES.values():
        replay = _replay_historical_sigma_with_g(
            center_path=profile.center_selections,
            sigma_without_g_path=profile.sigma_selections,
            records=records,
        )

        assert sum(replay.correctness) == profile.sigma_with_g_full_correct
        assert (
            int(np.asarray(replay.correctness, dtype=np.int8)[paper_mask].sum())
            == profile.sigma_with_g_paper_ood_correct
        )
        assert tuple(
            replay.selections.count(peer) for peer in range(3)
        ) == profile.sigma_with_g_selected_peers


def test_canonical_stream_and_paper_mask_are_pinned() -> None:
    path = Path(
        "data/peer_generalization/hf_generalization_all_canonical_peers/"
        "hf_generalization_all_peer012.labeled.jsonl"
    )
    records = JsonlDataset(path).records
    scopes = _scope_indices(records)
    paper_mask = scopes["paper_ood"]

    assert _sha256_file(path) == CANONICAL_STREAM_SHA256
    assert len(records) == CANONICAL_STREAM_SIZE
    assert _ordered_id_sha256(records) == CANONICAL_ORDERED_ID_SHA256
    assert int(paper_mask.sum()) == PAPER_OOD_SIZE
    assert _ordered_id_sha256(
        record for record, keep in zip(records, paper_mask) if keep
    ) == PAPER_OOD_ORDERED_ID_SHA256
