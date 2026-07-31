from __future__ import annotations

import sys

import numpy as np

from tests.experiments.common.model_profiles import MODEL_RUNS
from tests.experiments.selection_mechanisms.m_route_vote import (
    _decision_record,
    _frozen_vote_correctness,
    parse_args,
)


def test_all_requested_model_profiles_are_configured() -> None:
    assert set(MODEL_RUNS) == {
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
        ["tests.experiments.selection_mechanisms.m_route_vote", "--profile", "q35_4b"],
    )

    args = parse_args()
    profile = MODEL_RUNS["q35_4b"]

    assert args.central_model == profile.central_model
    assert args.checkpoint == profile.checkpoint
    assert args.output == profile.routing_output


def test_new_profile_defaults_are_isolated(monkeypatch) -> None:
    for name in ("q3_0_6b", "q3_8b", "q35_9b"):
        monkeypatch.setattr(
            sys,
            "argv",
            ["tests.experiments.selection_mechanisms.m_route_vote", "--profile", name],
        )
        args = parse_args()
        profile = MODEL_RUNS[name]

        assert args.central_model == profile.central_model
        assert args.checkpoint == profile.checkpoint
        assert args.output == profile.routing_output


def test_default_profile_remains_qwen3_4b(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["tests.experiments.selection_mechanisms.m_route_vote"]
    )

    args = parse_args()
    profile = MODEL_RUNS["q3_4b"]

    assert args.profile == "q3_4b"
    assert args.central_model == profile.central_model
    assert args.checkpoint == profile.checkpoint


def test_profile_outputs_are_isolated() -> None:
    assert len({profile.routing_output for profile in MODEL_RUNS.values()}) == len(
        MODEL_RUNS
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
