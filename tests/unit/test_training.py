import torch

from feedback_state.train_symmetric_memory import (
    _candidate_training_loss,
    _planned_backward_events,
    _skip_training_event,
)


def test_loss_uses_first_correct_peer():
    logits = torch.tensor([-0.5, 0.25, 1.75])

    actual = _candidate_training_loss(logits, [0, 1, 1])
    expected = torch.nn.functional.cross_entropy(
        logits.unsqueeze(0),
        torch.tensor([1]),
    )

    assert torch.allclose(actual, expected)


def test_event_without_correct_peer_has_no_objective():
    logits = torch.tensor([0.2, -0.4, 1.1])

    assert _candidate_training_loss(logits, [0, 0, 0]) is None
    assert _skip_training_event(3, None)
    assert _skip_training_event(0, None)


def test_scheduler_plan_counts_events_with_valid_targets():
    base = {"peer_responses": {f"peer_{i}": str(i) for i in range(3)}}
    records = [
        {**base, "correctness_by_peer": {f"peer_{i}": 0 for i in range(3)}},
        {**base, "correctness_by_peer": {"peer_0": 0, "peer_1": 1, "peer_2": 0}},
        {**base, "correctness_by_peer": {f"peer_{i}": 1 for i in range(3)}},
        {**base, "correctness_by_peer": {"peer_0": 1, "peer_1": 1, "peer_2": 0}},
    ]

    assert _planned_backward_events(records, 3, total_events=4) == 3
