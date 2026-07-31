from __future__ import annotations

import json

import numpy as np

from tests.experiments.feedback_availability.m_route_vote import (
    PreparedEvent,
    make_feedback_masks,
    replay_sparse_feedback,
)
from tests.experiments.feedback_availability.summarize_sigma import (
    replay_sparse_sigma_with_g,
)


def _event(identifier: str, correctness: list[int]) -> PreparedEvent:
    binary = np.asarray(correctness, dtype=np.int8)
    return PreparedEvent(
        original_index=int(identifier),
        identifier=identifier,
        source="demo",
        canonical_answers=("a", "b", "c"),
        answer_tie_keys={"a": (0, 0, ""), "b": (0, 1, ""), "c": (0, 2, "")},
        correctness=binary,
        correctness_signed=np.where(binary > 0, 1.0, -1.0),
    )


def _event_with_answers(
    identifier: str, correctness: list[int], answers: tuple[str, str, str]
) -> PreparedEvent:
    event = _event(identifier, correctness)
    return PreparedEvent(
        original_index=event.original_index,
        identifier=event.identifier,
        source=event.source,
        canonical_answers=answers,
        answer_tie_keys={answer: (1, 0, answer) for answer in set(answers)},
        correctness=event.correctness,
        correctness_signed=event.correctness_signed,
    )


def test_feedback_masks_are_exact_reproducible_and_nested() -> None:
    left = make_feedback_masks(101, percents=(5, 10, 50, 100), seeds=(0, 1))
    right = make_feedback_masks(101, percents=(5, 10, 50, 100), seeds=(0, 1))

    for seed in (0, 1):
        previous = np.zeros(101, dtype=bool)
        for percent in (5, 10, 50, 100):
            mask = left[(seed, percent)]
            assert int(mask.sum()) == round(101 * percent / 100)
            assert np.all(previous <= mask)
            np.testing.assert_array_equal(mask, right[(seed, percent)])
            previous = mask


def test_unobserved_event_decays_but_does_not_write_its_label() -> None:
    events = [_event("0", [1, 0, 0]), _event("1", [0, 1, 0])]
    phis = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    first_only = replay_sparse_feedback(
        events=events,
        phis=phis,
        feedback_mask=np.asarray([True, False]),
        rank=2,
        gamma=0.8,
        eta=0.25,
    )
    all_feedback = replay_sparse_feedback(
        events=events,
        phis=phis,
        feedback_mask=np.asarray([True, True]),
        rank=2,
        gamma=0.8,
        eta=0.25,
    )

    # Event 1 sees only event 0 in both runs; its current label cannot change its decision.
    assert first_only["metrics"] == all_feedback["metrics"]
    assert first_only["final_M_frobenius_norm"] != all_feedback["final_M_frobenius_norm"]


def test_no_feedback_keeps_router_at_cold_tie_with_per_event_time() -> None:
    events = [
        _event("0", [1, 0, 0]),
        _event("1", [1, 0, 0]),
        _event("2", [1, 0, 0]),
    ]
    result = replay_sparse_feedback(
        events=events,
        phis=np.asarray([[1.0, 0.0]] * 3),
        feedback_mask=np.zeros(3, dtype=bool),
        rank=2,
        gamma=0.8,
        eta=0.25,
    )

    assert result["route_all_equal_ties"] == 3
    assert result["route_selected_peers"] == {"0": 1, "1": 1, "2": 1}
    assert result["final_M_frobenius_norm"] == 0.0


def test_route_decisions_and_scores_do_not_depend_on_answers() -> None:
    original = [
        _event_with_answers("0", [1, 0, 0], ("a", "b", "c")),
        _event_with_answers("1", [0, 1, 0], ("a", "b", "c")),
        _event_with_answers("2", [0, 0, 1], ("a", "b", "c")),
    ]
    rewritten = [
        _event_with_answers("0", [1, 0, 0], ("x", "x", "x")),
        _event_with_answers("1", [0, 1, 0], ("z", "y", "z")),
        _event_with_answers("2", [0, 0, 1], ("u", "v", "u")),
    ]
    phis = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    mask = np.ones(3, dtype=bool)
    left = replay_sparse_feedback(
        events=original,
        phis=phis,
        feedback_mask=mask,
        rank=2,
        gamma=0.8,
        eta=0.25,
        capture_trace=True,
    )
    right = replay_sparse_feedback(
        events=rewritten,
        phis=phis,
        feedback_mask=mask,
        rank=2,
        gamma=0.8,
        eta=0.25,
        capture_trace=True,
    )

    for left_row, right_row in zip(left["trace"], right["trace"]):
        assert left_row["route_peer"] == right_row["route_peer"]
        np.testing.assert_array_equal(
            left_row["route_scores"], right_row["route_scores"]
        )


def test_masked_current_label_cannot_change_current_or_future_route() -> None:
    baseline = [
        _event("0", [1, 0, 0]),
        _event("1", [0, 1, 0]),
        _event("2", [0, 0, 1]),
    ]
    changed = [
        _event("0", [1, 0, 0]),
        _event("1", [1, 0, 1]),
        _event("2", [0, 0, 1]),
    ]
    phis = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    mask = np.asarray([True, False, False])

    def replay(events: list[PreparedEvent]) -> dict:
        return replay_sparse_feedback(
            events=events,
            phis=phis,
            feedback_mask=mask,
            rank=2,
            gamma=0.8,
            eta=0.25,
            capture_trace=True,
        )

    left = replay(baseline)
    right = replay(changed)
    for left_row, right_row in zip(left["trace"], right["trace"]):
        assert left_row["route_peer"] == right_row["route_peer"]
        np.testing.assert_array_equal(
            left_row["route_scores"], right_row["route_scores"]
        )
    assert left["final_M_frobenius_norm"] == right["final_M_frobenius_norm"]


def test_sparse_sigma_with_g_masked_label_cannot_change_future(tmp_path) -> None:
    records = [
        {"id": "a", "source": "piqa"},
        {"id": "b", "source": "piqa"},
        {"id": "c", "source": "piqa"},
    ]
    center_rows = [
        {
            "id": record["id"],
            "peer_scores": {"0": 1.0, "1": 0.0, "2": -1.0},
            "peer_correct": {"0": 1, "1": 0, "2": 0},
        }
        for record in records
    ]
    sigma_rows = [
        {
            "id": record["id"],
            "peer_scores": {"0": 1.0, "1": 0.0, "2": -1.0},
            "peer_correct": {"0": 1, "1": 0, "2": 0},
        }
        for record in records
    ]
    changed_center_rows = [dict(row) for row in center_rows]
    changed_sigma_rows = [dict(row) for row in sigma_rows]
    changed_center_rows[1] = {
        **changed_center_rows[1],
        "peer_correct": {"0": 0, "1": 1, "2": 1},
    }
    changed_sigma_rows[1] = {
        **changed_sigma_rows[1],
        "peer_correct": {"0": 0, "1": 1, "2": 1},
    }

    def write_jsonl(path, rows) -> None:
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    center_path = tmp_path / "center.jsonl"
    sigma_path = tmp_path / "sigma.jsonl"
    changed_center_path = tmp_path / "changed_center.jsonl"
    changed_sigma_path = tmp_path / "changed_sigma.jsonl"
    write_jsonl(center_path, center_rows)
    write_jsonl(sigma_path, sigma_rows)
    write_jsonl(changed_center_path, changed_center_rows)
    write_jsonl(changed_sigma_path, changed_sigma_rows)

    mask = np.asarray([True, False, False])
    left = replay_sparse_sigma_with_g(
        center_path=center_path,
        sigma_path=sigma_path,
        records=records,
        feedback_mask=mask,
    )
    right = replay_sparse_sigma_with_g(
        center_path=changed_center_path,
        sigma_path=changed_sigma_path,
        records=records,
        feedback_mask=mask,
    )

    assert left["selected_peers"] == right["selected_peers"]
    assert left["final_history"] == right["final_history"]
    assert left["final_graph"] == right["final_graph"]
