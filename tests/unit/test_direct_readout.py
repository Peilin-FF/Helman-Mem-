import numpy as np
import pytest

from feedback_state.ood_routing import (
    DiscountedBetaRoute,
    INVALID_ANSWER,
    OODRoutingState,
    canonical_answer,
    vote,
)


def test_discounted_beta_b1_is_response_blind_and_decide_then_update():
    state = DiscountedBetaRoute(num_peers=3, gamma=0.9)

    initial = state.route()
    assert initial.peer == 0
    np.testing.assert_allclose(initial.scores, [0.5, 0.5, 0.5])

    state.update([0, 1, 0])
    alpha, beta = state.snapshot()
    np.testing.assert_allclose(alpha, [0.9, 1.9, 0.9])
    np.testing.assert_allclose(beta, [1.9, 0.9, 1.9])
    assert state.route().peer == 1


def _state() -> OODRoutingState:
    return OODRoutingState(num_peers=3, rank=2, gamma=0.8, eta=0.25)


def _mc_record():
    return {
        "task_type": "mcqa",
        "problem": "Which option?",
        "choices": ["first", "second", "third"],
        "choice_labels": ["A", "B", "C"],
        "answer": "B",
    }


def test_state_starts_with_zero_m():
    state = _state()

    assert state.M.dtype == np.float64
    np.testing.assert_array_equal(state.M, np.zeros((3, 2, 2)))


def test_route_is_read_only_and_cold_ties_use_round_robin():
    state = _state()
    before = state.snapshot()

    decision = state.route([1.0, 0.0], event_index=4)

    assert decision.peer == 1
    assert decision.tied
    np.testing.assert_array_equal(decision.scores, np.zeros(3))
    np.testing.assert_array_equal(state.M, before)


def test_route_uses_rayleigh_quotient_and_argmax():
    state = _state()
    state.M[0] = np.diag([0.2, 5.0])
    state.M[1] = np.diag([0.7, -3.0])
    state.M[2] = np.diag([-0.4, 9.0])

    decision = state.route([1.0, 0.0], event_index=0)

    np.testing.assert_allclose(decision.scores, [0.2, 0.7, -0.4])
    assert decision.peer == 1
    assert not decision.tied


def test_update_matches_signed_m_equation():
    state = _state()
    state.M[:] = 0.5
    old_m = state.snapshot()
    phi = np.array([0.6, 0.8])
    correctness = np.array([1.0, -1.0, 1.0])

    state.update(phi, correctness)

    outer = np.outer(phi, phi)
    expected = 0.8 * old_m + 0.25 * correctness[:, None, None] * outer[None]
    np.testing.assert_allclose(state.M, expected)


def test_missing_feedback_applies_decay_without_label_write():
    state = _state()
    state.M[:] = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    old_m = state.snapshot()

    state.decay_without_feedback()

    np.testing.assert_allclose(state.M, 0.8 * old_m)


def test_decisions_do_not_observe_current_correctness_before_update():
    left = _state()
    right = _state()
    phi = [1.0, 0.0]

    before_left = left.route(phi, 0)
    before_right = right.route(phi, 0)
    left.update(phi, [1, -1, -1])
    right.update(phi, [-1, 1, -1])

    assert before_left.peer == before_right.peer == 0
    assert left.route(phi, 1).peer == 0
    assert right.route(phi, 1).peer == 1


def test_votes_expose_only_majority_and_m_weights_without_mutation():
    state = _state()
    state.M[0, 0, 0] = 0.8
    state.M[1, 0, 0] = -0.2
    state.M[2, 0, 0] = 0.4
    before = state.snapshot()

    result = state.votes(
        [1.0, 0.0],
        _mc_record(),
        ["Final answer: A", "Final answer: B", "Final answer: B"],
    )

    assert set(result.answers) == {"maj", "M"}
    np.testing.assert_allclose(result.weights["maj"], np.ones(3) / 3)
    np.testing.assert_allclose(result.weights["M"], [0.8, -0.2, 0.4])
    np.testing.assert_array_equal(state.M, before)


def test_canonical_vote_groups_option_label_formats():
    record = _mc_record()
    answers = ["Final answer: (B) second", "Answer: B", "Final answer: A"]

    assert [canonical_answer(record, answer) for answer in answers] == ["b", "b", "a"]
    assert vote(record, answers, [1.0, 1.0, 1.0]) == "b"


def test_unparsed_answers_are_explicitly_invalid():
    assert canonical_answer(_mc_record(), "No final option here") == INVALID_ANSWER


def test_shortqa_option_a_is_preserved_for_vote_grouping():
    record = {
        "task_type": "shortqa",
        "problem": "Choose one:\n(A) first\n(B) second",
        "answer": "(A)",
    }

    assert canonical_answer(record, "Final answer: (A)") == "opt:a"
    assert canonical_answer(record, "A") == "opt:a"
    assert canonical_answer(record, "Final answer: `(B)`.") == "opt:b"
    assert canonical_answer(record, "Final answer: (B) because...") != "opt:b"
    assert vote(record, ["(B)", "(A)", "A"], [1.0, 1.0, 1.0]) == "opt:a"


def test_vote_preserves_raw_signed_weights():
    answers = ["Final answer: A", "Final answer: A", "Final answer: B"]
    assert vote(_mc_record(), answers, [-2.0, -1.0, 0.1]) == "b"


def test_vote_tie_uses_lowest_declared_option_index():
    assert vote(
        _mc_record(),
        ["Final answer: C", "Final answer: A", "Final answer: B"],
        [1.0, 1.0, 1.0],
    ) == "a"


def test_phi_and_correctness_contracts_are_enforced():
    state = _state()

    with pytest.raises(ValueError, match="L2-normalized"):
        state.route([2.0, 0.0], 0)
    with pytest.raises(ValueError, match=r"only -1 or \+1"):
        state.update([1.0, 0.0], [1, 0, -1])
