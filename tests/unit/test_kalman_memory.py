import math

import torch

from feedback_state.kalman_memory import DeltaMemory, KalmanMemory


def _random_events(n: int, d: int, heads: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    psi = torch.randn(n, d, generator=g, dtype=torch.float64)
    s = torch.randint(0, 2, (n, heads), generator=g).to(torch.float64) * 2 - 1
    return psi, s


def test_sherman_morrison_matches_direct_posterior():
    d, heads, lam = 8, 2, 3.0
    psi, s = _random_events(20, d, heads)
    mem = KalmanMemory(d, heads, lam=lam)
    for x, y in zip(psi, s):
        mem.write(x, y)
    Lam = lam * torch.eye(d, dtype=torch.float64) + psi.T @ psi
    b = s.T @ psi
    q = torch.randn(3, d, dtype=torch.float64)
    mu, var = mem.read(q)
    assert torch.allclose(mu, q @ torch.linalg.solve(Lam, b.T), atol=1e-9)
    assert torch.allclose(var, (q @ torch.linalg.solve(Lam, q.T)).diagonal(), atol=1e-9)


def test_state_is_order_invariant():
    d, heads = 6, 3
    psi, s = _random_events(15, d, heads)
    a = KalmanMemory(d, heads, lam=2.0)
    b = KalmanMemory(d, heads, lam=2.0)
    for x, y in zip(psi, s):
        a.write(x, y)
    for i in torch.randperm(15, generator=torch.Generator().manual_seed(1)).tolist():
        b.write(psi[i], s[i])
    assert torch.allclose(a.P, b.P, atol=1e-9) and torch.allclose(a.b, b.b, atol=1e-9)


def test_forgetting_branch_matches_explicit_recursion():
    d, heads, lam, rho = 5, 1, 4.0, 0.9
    psi, s = _random_events(10, d, heads)
    mem = KalmanMemory(d, heads, lam=lam, rho=rho)
    Lam = lam * torch.eye(d, dtype=torch.float64)
    b = torch.zeros(heads, d, dtype=torch.float64)
    for x, y in zip(psi, s):
        mem.write(x, y)
        Lam = rho * Lam + (1 - rho) * lam * torch.eye(d, dtype=torch.float64) + torch.outer(x, x)
        b = rho * b + torch.outer(y, x)
    q = torch.randn(2, d, dtype=torch.float64)
    mu, var = mem.read(q)
    assert torch.allclose(mu, q @ torch.linalg.solve(Lam, b.T), atol=1e-9)


def test_cold_start_is_uninformative_and_evidence_grows():
    mem = KalmanMemory(4, 1, lam=10.0)
    q = torch.ones(1, 4, dtype=torch.float64)
    mu, var = mem.read(q)
    assert float(mu.abs().max()) == 0.0
    assert math.isclose(float(mem.prob(mu, var)[0, 0]), 0.5)
    assert float(mem.evidence(q, var)) == 0.0
    for _ in range(5):
        mem.write(q[0], torch.tensor([1.0]))
    mu2, var2 = mem.read(q)
    assert float(mem.evidence(q, var2)) > 1.0 and float(mem.prob(mu2, var2)[0, 0]) > 0.5


def test_delta_rule_moves_towards_target():
    mem = DeltaMemory(3, 1, beta=0.5)
    x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    for _ in range(20):
        mem.write(x, torch.tensor([1.0]))
    mu, _ = mem.read(x.unsqueeze(0))
    assert float(mu) > 0.99
