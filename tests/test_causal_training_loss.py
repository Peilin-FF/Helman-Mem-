import pytest
import torch

from train_symmetric_memory import (
    LOSS_CAUSAL_MULTILABEL_BCE,
    LOSS_FIRST_CORRECT_CE,
    _candidate_training_loss,
    _effective_diff_write,
    _planned_backward_events,
    _skip_training_event,
    _validate_loss_semantics,
)


def test_causal_bce_scores_every_peer_on_multi_correct_event():
    logits = torch.zeros(3, requires_grad=True)
    labels = [1, 1, 0]

    loss = _candidate_training_loss(
        logits,
        labels,
        loss_mode=LOSS_CAUSAL_MULTILABEL_BCE,
    )

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        torch.tensor(labels, dtype=torch.float32),
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0] < 0
    assert logits.grad[1] < 0
    assert logits.grad[2] > 0


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1]])
def test_causal_bce_keeps_all_wrong_and_all_correct_events(labels):
    logits = torch.tensor([0.2, -0.4, 1.1], requires_grad=True)

    loss = _candidate_training_loss(
        logits,
        labels,
        loss_mode=LOSS_CAUSAL_MULTILABEL_BCE,
    )

    assert loss is not None
    assert torch.isfinite(loss)
    target = next((i for i, value in enumerate(labels) if value), None)
    assert not _skip_training_event(3, target, LOSS_CAUSAL_MULTILABEL_BCE)


def test_first_correct_ce_preserves_legacy_target_and_skip_behavior():
    logits = torch.tensor([-0.5, 0.25, 1.75])

    actual = _candidate_training_loss(
        logits,
        [0, 1, 1],
        loss_mode=LOSS_FIRST_CORRECT_CE,
    )
    expected = torch.nn.functional.cross_entropy(
        logits.unsqueeze(0),
        torch.tensor([1]),
    )

    assert torch.allclose(actual, expected)
    assert _candidate_training_loss(
        logits,
        [0, 0, 0],
        loss_mode=LOSS_FIRST_CORRECT_CE,
    ) is None
    assert _skip_training_event(3, None, LOSS_FIRST_CORRECT_CE)


def test_causal_mode_disables_current_label_diff_write():
    assert not _effective_diff_write(LOSS_CAUSAL_MULTILABEL_BCE, requested=True)
    assert not _effective_diff_write(LOSS_CAUSAL_MULTILABEL_BCE, requested=False)
    assert _effective_diff_write(LOSS_FIRST_CORRECT_CE, requested=True)

    _validate_loss_semantics(
        LOSS_CAUSAL_MULTILABEL_BCE,
        score_mode="candidate_yesno",
    )
    _validate_loss_semantics(
        LOSS_FIRST_CORRECT_CE,
        score_mode="peer_name",
    )
    with pytest.raises(ValueError, match="score_mode=candidate_yesno"):
        _validate_loss_semantics(
            LOSS_CAUSAL_MULTILABEL_BCE,
            score_mode="peer_name",
        )


def test_scheduler_plan_counts_backward_events_not_raw_events():
    base = {"peer_responses": {f"peer_{i}": str(i) for i in range(3)}}
    records = [
        {**base, "correctness_by_peer": {f"peer_{i}": 0 for i in range(3)}},
        {**base, "correctness_by_peer": {"peer_0": 0, "peer_1": 1, "peer_2": 0}},
        {**base, "correctness_by_peer": {f"peer_{i}": 1 for i in range(3)}},
        {**base, "correctness_by_peer": {"peer_0": 1, "peer_1": 1, "peer_2": 0}},
    ]

    assert _planned_backward_events(
        records, 3, total_events=4, loss_mode=LOSS_CAUSAL_MULTILABEL_BCE
    ) == 4
    assert _planned_backward_events(
        records, 3, total_events=4, loss_mode=LOSS_FIRST_CORRECT_CE
    ) == 3
