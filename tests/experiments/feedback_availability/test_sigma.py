from __future__ import annotations

import numpy as np
import torch

from tests.experiments.feedback_availability.m_route_vote import make_feedback_masks
from tests.experiments.common.evaluate_sigma import _selective_feedback_mask
from feedback_state.symmetric_memory import SymmetricTrustMemory


def test_decay_without_feedback_advances_only_the_event_decay() -> None:
    memory = SymmetricTrustMemory(
        num_peers=3,
        rank=2,
        task_types=("demo",),
        phi_in_dim=2,
        gamma_init=0.8,
        dtype=torch.float64,
    )
    with torch.no_grad():
        memory.M.copy_(
            torch.tensor(
                [
                    [[1.0, 2.0], [2.0, 4.0]],
                    [[-1.0, 0.5], [0.5, 3.0]],
                    [[2.0, -2.0], [-2.0, 1.0]],
                ],
                dtype=torch.float64,
            )
        )
    before_m = memory.M.clone()

    memory.decay_without_feedback()

    for peer in range(memory.num_peers):
        expected = memory.gamma().detach() * before_m[peer]
        torch.testing.assert_close(memory.M[peer], expected)


def test_sigma_evaluator_uses_the_exact_m_route_feedback_masks() -> None:
    expected = make_feedback_masks(
        101, percents=(5, 10, 20, 50, 80, 100), seeds=(0, 1, 2)
    )
    for (seed, percent), route_mask in expected.items():
        sigma_mask, _ = _selective_feedback_mask(
            101, percent=percent, seed=seed
        )
        np.testing.assert_array_equal(sigma_mask, route_mask)
