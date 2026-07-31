import torch

from tests.experiments.common.evaluate_sigma import _ising_marginal_scores, _zscore_1d


def test_zero_topology_reduces_to_sigma_unary_posterior():
    sigma_utilities = torch.tensor([-0.5, 0.25, 1.75])
    graph = torch.zeros(3, 3)
    unary_weight = 0.7

    actual = _ising_marginal_scores(
        sigma_utilities,
        graph,
        unary_weight=unary_weight,
        g_weight=0.0,
    )
    expected = torch.tanh(unary_weight * _zscore_1d(sigma_utilities))

    assert torch.allclose(actual, expected, atol=1e-6)


def test_pairwise_topology_preserves_peer_symmetry():
    sigma_utilities = torch.tensor([1.0, 1.0, -1.0])
    graph = torch.tensor(
        [
            [0.0, 0.8, -0.4],
            [0.8, 0.0, -0.4],
            [-0.4, -0.4, 0.0],
        ]
    )

    posterior = _ising_marginal_scores(
        sigma_utilities,
        graph,
        unary_weight=0.7,
        g_weight=0.5,
    )

    assert torch.allclose(posterior[0], posterior[1], atol=1e-6)
    assert posterior[0] > posterior[2]


def test_joint_posterior_is_permutation_equivariant():
    sigma_utilities = torch.tensor([1.8, -0.4, 0.3])
    graph = torch.tensor(
        [
            [0.0, 0.7, -0.2],
            [0.7, 0.0, 0.5],
            [-0.2, 0.5, 0.0],
        ]
    )
    posterior = _ising_marginal_scores(
        sigma_utilities,
        graph,
        unary_weight=0.7,
        g_weight=0.5,
    )

    permutation = torch.tensor([2, 0, 1])
    permuted = _ising_marginal_scores(
        sigma_utilities[permutation],
        graph.index_select(0, permutation).index_select(1, permutation),
        unary_weight=0.7,
        g_weight=0.5,
    )

    assert torch.allclose(permuted, posterior[permutation], atol=1e-6)
