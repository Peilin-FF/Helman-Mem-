from __future__ import annotations

import copy

import numpy as np

from tests.experiments.counterfactual.m_route import _cf_record_labels, replay_m_route


def _records() -> list[dict]:
    return [
        {
            "uid": "event-0",
            "id": "event-0",
            "problem": "question zero",
            "dataset": "demo",
            "task_type": "mcqa",
            "peer_correct": {"peer_0": 1, "peer_1": 0, "peer_2": 0},
            "peer_responses": {"peer_0": "A", "peer_1": "B", "peer_2": "C"},
        },
        {
            "uid": "event-1",
            "id": "event-1",
            "problem": "question one",
            "dataset": "demo",
            "task_type": "mcqa",
            "peer_correct": {"peer_0": 1, "peer_1": 0, "peer_2": 0},
            "peer_responses": {"peer_0": "A", "peer_1": "B", "peer_2": "C"},
        },
    ]


def test_cf_m_route_is_invariant_to_peer_responses() -> None:
    records = _records()
    changed = copy.deepcopy(records)
    for record in changed:
        record["peer_responses"] = {
            "peer_0": "completely changed",
            "peer_1": "different content",
            "peer_2": "another answer",
        }
    phis = {
        "event-0": np.asarray([1.0, 0.0]),
        "event-1": np.asarray([1.0, 0.0]),
    }

    left_summary, left_rows = replay_m_route(
        records=records, phi_by_uid=phis, rank=2, gamma=0.8, eta=0.25
    )
    right_summary, right_rows = replay_m_route(
        records=changed, phi_by_uid=phis, rank=2, gamma=0.8, eta=0.25
    )

    assert left_summary == right_summary
    assert left_rows == right_rows
    assert [row["selected_peer"] for row in left_rows] == [0, 0]


def test_cf_m_route_decides_before_current_label_changes_future_state() -> None:
    records = _records()
    changed = copy.deepcopy(records)
    changed[0]["peer_correct"] = {"peer_0": 0, "peer_1": 1, "peer_2": 0}
    phis = {
        "event-0": np.asarray([1.0, 0.0]),
        "event-1": np.asarray([1.0, 0.0]),
    }

    _, left_rows = replay_m_route(
        records=records, phi_by_uid=phis, rank=2, gamma=0.8, eta=0.25
    )
    _, right_rows = replay_m_route(
        records=changed, phi_by_uid=phis, rank=2, gamma=0.8, eta=0.25
    )

    assert left_rows[0]["selected_peer"] == right_rows[0]["selected_peer"] == 0
    assert left_rows[1]["selected_peer"] == 0
    assert right_rows[1]["selected_peer"] == 1


def test_cf_labels_match_existing_rounding_protocol() -> None:
    record = _records()[0]
    record["peer_correct"] = {"peer_0": 0.5, "peer_1": 0.5001, "peer_2": 1.0}

    binary, signed = _cf_record_labels(record)

    np.testing.assert_array_equal(binary, [0, 1, 1])
    np.testing.assert_array_equal(signed, [-1.0, 1.0, 1.0])
