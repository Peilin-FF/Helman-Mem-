import numpy as np
import pytest

from feedback_state.ood_routing import (
    DEFAULT_RIDGE,
    INVALID_ANSWER,
    ROBUSTNESS_RIDGES,
    DecayedLabelDictionary,
    OODRoutingState,
    canonical_answer,
    gls_weights,
    vote,
)


def _state() -> OODRoutingState:
    return OODRoutingState(
        num_peers=3,
        rank=2,
        gamma=0.8,
        eta=0.25,
        gamma_g=0.9,
        eta_g=0.1,
    )


def _mc_record():
    return {
        "task_type": "mcqa",
        "problem": "Which option?",
        "choices": ["first", "second", "third"],
        "choice_labels": ["A", "B", "C"],
        "answer": "B",
    }


def test_state_starts_with_zero_m_and_identity_g():
    state = _state()

    assert state.M.dtype == np.float64
    assert state.G.dtype == np.float64
    np.testing.assert_array_equal(state.M, np.zeros((3, 2, 2)))
    np.testing.assert_array_equal(state.G, np.eye(3))


def test_route_is_read_only_and_cold_ties_use_round_robin():
    state = _state()
    before = state.snapshot()

    decision = state.route([1.0, 0.0], event_index=4)

    assert decision.peer == 1
    assert decision.tied
    np.testing.assert_array_equal(decision.scores, np.zeros(3))
    np.testing.assert_array_equal(state.M, before[0])
    np.testing.assert_array_equal(state.G, before[1])


def test_route_uses_rayleigh_quotient_and_argmax():
    state = _state()
    state.M[0] = np.diag([0.2, 5.0])
    state.M[1] = np.diag([0.7, -3.0])
    state.M[2] = np.diag([-0.4, 9.0])

    decision = state.route([1.0, 0.0], event_index=0)

    np.testing.assert_allclose(decision.scores, [0.2, 0.7, -0.4])
    assert decision.peer == 1
    assert not decision.tied


def test_route_mg_uses_gls_scores_without_mutating_state():
    state = _state()
    state.M[0, 0, 0] = 0.1
    state.M[1, 0, 0] = 0.4
    state.M[2, 0, 0] = 0.2
    state.G = np.asarray(
        [
            [1.0, 0.4, 0.0],
            [0.4, 1.0, 0.8],
            [0.0, 0.8, 1.0],
        ],
        dtype=np.float64,
    )
    before = state.snapshot()

    decision = state.route_mg([1.0, 0.0], event_index=7, ridge=DEFAULT_RIDGE)
    expected = gls_weights(state.G, state.reliability([1.0, 0.0]), DEFAULT_RIDGE)

    np.testing.assert_allclose(decision.scores, expected)
    assert decision.peer == int(np.argmax(expected))
    np.testing.assert_array_equal(state.M, before[0])
    np.testing.assert_array_equal(state.G, before[1])


def test_update_matches_signed_m_and_centered_g_equations():
    state = _state()
    state.M[:] = 0.5
    old_m, old_g = state.snapshot()
    phi = np.array([0.6, 0.8])
    correctness = np.array([1.0, -1.0, 1.0])

    state.update(phi, correctness)

    outer = np.outer(phi, phi)
    expected_m = (
        0.8 * old_m + 0.25 * correctness[:, None, None] * outer[None]
    )
    q = correctness - correctness.mean()
    expected_g = 0.9 * old_g + 0.1 * np.outer(q, q)
    np.fill_diagonal(expected_g, 1.0)
    np.testing.assert_allclose(state.M, expected_m)
    np.testing.assert_allclose(state.G, expected_g)
    np.testing.assert_allclose(state.G, state.G.T)
    np.testing.assert_array_equal(np.diag(state.G), np.ones(3))


def test_missing_feedback_advances_decay_without_label_write():
    state = _state()
    state.M[:] = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    state.G[:] = np.asarray(
        [[1.0, 0.2, -0.1], [0.2, 1.0, 0.3], [-0.1, 0.3, 1.0]]
    )
    old_m, old_g = state.snapshot()

    state.decay_without_feedback()

    np.testing.assert_allclose(state.M, 0.8 * old_m)
    expected_g = 0.9 * old_g
    np.fill_diagonal(expected_g, 1.0)
    np.testing.assert_allclose(state.G, expected_g)


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


def test_canonical_vote_groups_option_label_formats():
    record = _mc_record()
    answers = ["Final answer: (B) second", "Answer: B", "Final answer: A"]

    assert [canonical_answer(record, answer) for answer in answers] == ["b", "b", "a"]
    assert vote(record, answers, [1.0, 1.0, 1.0]) == "b"


def test_unparsed_answers_are_explicitly_invalid():
    record = _mc_record()

    assert canonical_answer(record, "No final option here") == INVALID_ANSWER


def test_shortqa_option_a_is_preserved_for_vote_grouping():
    record = {
        "task_type": "shortqa",
        "problem": "Choose one:\n(A) first\n(B) second",
        "answer": "(A)",
    }

    assert canonical_answer(record, "Final answer: (A)") == "opt:a"
    assert canonical_answer(record, "A") == "opt:a"
    assert canonical_answer(record, "Final answer: `(B)`." ) == "opt:b"
    assert canonical_answer(record, "Final answer: (B) because...") != "opt:b"
    assert vote(record, ["(B)", "(A)", "A"], [1.0, 1.0, 1.0]) == "opt:a"


def test_vote_preserves_raw_signed_weights():
    record = _mc_record()
    answers = ["Final answer: A", "Final answer: A", "Final answer: B"]

    # Clipping would pick A (0 + 0 versus 0.1); raw signed aggregation picks B.
    assert vote(record, answers, [-2.0, -1.0, 0.1]) == "b"


def test_vote_tie_uses_lowest_declared_option_index():
    record = _mc_record()

    assert vote(
        record,
        ["Final answer: C", "Final answer: A", "Final answer: B"],
        [1.0, 1.0, 1.0],
    ) == "a"


def test_gls_weights_are_exact_linear_solve():
    G = np.array(
        [[1.0, 0.2, -0.1], [0.2, 1.0, 0.3], [-0.1, 0.3, 1.0]],
        dtype=np.float64,
    )
    rhs = np.array([0.5, -0.25, 1.0])

    actual = gls_weights(G, rhs, DEFAULT_RIDGE)

    np.testing.assert_allclose((G + DEFAULT_RIDGE * np.eye(3)) @ actual, rhs)


def test_factorial_votes_expose_raw_weights_without_mutating_state():
    state = _state()
    state.M[0, 0, 0] = 0.8
    state.M[1, 0, 0] = -0.2
    state.M[2, 0, 0] = 0.4
    before = state.snapshot()

    result = state.all_votes(
        [1.0, 0.0],
        _mc_record(),
        ["Final answer: A", "Final answer: B", "Final answer: B"],
    )

    assert set(result.answers) == {"maj", "G", "M", "MG"}
    np.testing.assert_allclose(result.weights["maj"], np.ones(3) / 3)
    np.testing.assert_allclose(result.weights["M"], [0.8, -0.2, 0.4])
    np.testing.assert_array_equal(state.M, before[0])
    np.testing.assert_array_equal(state.G, before[1])


def test_robustness_arms_cover_declared_ridges_and_clipping():
    state = _state()
    results = state.robustness_votes(
        [1.0, 0.0],
        _mc_record(),
        ["Final answer: A", "Final answer: B", "Final answer: C"],
    )

    assert set(results) == {
        (ridge, clipped)
        for ridge in ROBUSTNESS_RIDGES
        for clipped in (False, True)
    }
    assert np.all(results[(0.1, True)].weights["M"] >= 0.0)


def test_dictionary_is_dataset_specific_and_updates_after_decision():
    dictionary = DecayedLabelDictionary(num_peers=3, gamma=0.8)

    initial = dictionary.route("dataset-a", event_index=2)
    dictionary.update("dataset-a", [-1, 1, -1])

    assert initial.peer == 2
    assert dictionary.route("dataset-a", event_index=0).peer == 1
    assert dictionary.route("dataset-b", event_index=0).peer == 0


def test_phi_and_correctness_contracts_are_enforced():
    state = _state()

    with pytest.raises(ValueError, match="L2-normalized"):
        state.route([2.0, 0.0], 0)
    with pytest.raises(ValueError, match=r"only -1 or \+1"):
        state.update([1.0, 0.0], [1, 0, -1])
