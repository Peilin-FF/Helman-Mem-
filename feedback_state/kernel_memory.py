"""Growing-capacity memories: episodic (nearest-neighbour) and random-feature kernel memories.

A d-dimensional linear memory saturates: its online regret is O(d log T), so its
learning curve flattens once T >> d.  A kernel memory is a linear memory in an
infinite feature space, so its posterior keeps contracting as evidence arrives
and the curve keeps rising.  Two finite realisations are kept here:

* ``EpisodicMemory``  Nadaraya-Watson over the stored past events (softmax over
  cosine similarity in the address space = one attention head over the feedback
  history; the transformer-native reading of the same memory).
* ``RandomFeatureMap``  random Fourier features of an RBF kernel; feeding them to
  ``KalmanMemory`` gives Bayesian kernel regression (a GP with an RBF kernel) in
  closed form with a fixed-size state.
* ``HybridMemory``  Kalman linear state + episodic residual correction: the
  linear part carries the global, order-invariant competence; the episodic part
  corrects it locally where the history disagrees with the linear prediction.
"""
from __future__ import annotations

import math

import torch

from feedback_state.kalman_memory import KalmanMemory


class EpisodicMemory:
    def __init__(self, dim: int, heads: int, *, capacity: int, k: int = 64, tau: float = 0.05, device=None, dtype=torch.float64) -> None:
        self.dim, self.heads, self.capacity, self.k, self.tau = int(dim), int(heads), int(capacity), int(k), float(tau)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        self.keys = torch.zeros(self.capacity, self.dim, device=self.device, dtype=self.dtype)
        self.vals = torch.zeros(self.capacity, self.heads, device=self.device, dtype=self.dtype)
        self.n = 0

    @staticmethod
    def _unit(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-9)

    @torch.no_grad()
    def neighbours(self, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """psi [B, dim] -> (indices [B, k'], weights [B, k']) of the stored neighbours (softmax over cosine / tau)."""
        q = self._unit(psi.to(self.device, self.dtype))
        if self.n == 0:
            return torch.zeros(q.shape[0], 0, device=self.device, dtype=torch.long), torch.zeros(q.shape[0], 0, device=self.device, dtype=self.dtype)
        sims = q @ self.keys[: self.n].T
        k = min(self.k, self.n)
        top, idx = sims.topk(k, dim=1)
        w = torch.softmax(top / self.tau, dim=1)
        return idx, w

    @torch.no_grad()
    def read(self, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu [B, heads] in [-1, 1], var [B]) where var shrinks with the neighbourhood's effective size."""
        idx, w = self.neighbours(psi)
        B = psi.shape[0]
        if idx.shape[1] == 0:
            return torch.zeros(B, self.heads, device=self.device, dtype=self.dtype), torch.ones(B, device=self.device, dtype=self.dtype)
        vals = self.vals[idx]                       # [B, k, heads]
        observed = (vals != 0).to(self.dtype)
        wsum = (w.unsqueeze(-1) * observed).sum(1)  # [B, heads]
        mu = (w.unsqueeze(-1) * vals).sum(1) / wsum.clamp_min(1e-9)
        n_eff = 1.0 / (w * w).sum(1).clamp_min(1e-9)   # effective number of neighbours
        var = 1.0 / n_eff
        return mu, var

    prob = staticmethod(KalmanMemory.prob)
    logit = staticmethod(KalmanMemory.logit)

    @torch.no_grad()
    def write(self, psi: torch.Tensor, s: torch.Tensor) -> None:
        if self.n >= self.capacity:
            raise RuntimeError("episodic memory is full")
        self.keys[self.n] = self._unit(psi.to(self.device, self.dtype).reshape(self.dim))
        self.vals[self.n] = s.to(self.device, self.dtype).reshape(self.heads)
        self.n += 1


class RandomFeatureMap:
    """phi(x) = sqrt(2/D) cos(W x + b), W ~ N(0, 1/ell^2): an RBF kernel exp(-|x-y|^2 / 2 ell^2) in expectation."""

    def __init__(self, dim_in: int, dim_out: int, *, lengthscale: float, seed: int = 0, device=None, dtype=torch.float64) -> None:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.W = (torch.randn(dim_in, dim_out, generator=g, dtype=dtype) / float(lengthscale)).to(device)
        self.b = (torch.rand(dim_out, generator=g, dtype=dtype) * 2 * math.pi).to(device)
        self.dim_out = int(dim_out)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(2.0 / self.dim_out) * torch.cos(x.to(self.W.device, self.W.dtype) @ self.W + self.b)


@torch.no_grad()
def median_lengthscale(x: torch.Tensor, samples: int = 4096, seed: int = 0) -> float:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    idx = torch.randperm(x.shape[0], generator=g)[:samples]
    sub = x[idx].to(torch.float64)
    d = torch.cdist(sub, sub)
    return float(d[d > 0].median())


class HybridMemory:
    """Kalman linear state + episodic residual: mu(x) = mu_lin(x) + sum_j w_j (s_j - mu_lin(psi_j))."""

    def __init__(self, dim: int, heads: int, *, lam: float, capacity: int, k: int = 64, tau: float = 0.05, mix: float = 1.0, device=None) -> None:
        self.lin = KalmanMemory(dim, heads, lam=lam, device=device)
        self.epi = EpisodicMemory(dim, heads, capacity=capacity, k=k, tau=tau, device=device)
        self.mix = float(mix)
        self.heads = int(heads)

    def reset(self) -> None:
        self.lin.reset(); self.epi.reset()

    @torch.no_grad()
    def read(self, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_lin, var = self.lin.read(psi)
        idx, w = self.epi.neighbours(psi)
        if idx.shape[1] == 0:
            return mu_lin, var
        keys = self.epi.keys[idx]                    # [B, k, dim] (unit norm; residuals are read at the stored raw scale below)
        raw = self.epi_raw[idx]                      # [B, k, dim]
        mu_nb, _ = self.lin.read(raw.reshape(-1, raw.shape[-1]))
        mu_nb = mu_nb.reshape(raw.shape[0], raw.shape[1], self.heads)
        vals = self.epi.vals[idx]
        observed = (vals != 0).to(vals.dtype)
        resid = (vals - mu_nb) * observed
        corr = (w.unsqueeze(-1) * resid).sum(1) / (w.unsqueeze(-1) * observed).sum(1).clamp_min(1e-9)
        return mu_lin + self.mix * corr, var

    prob = staticmethod(KalmanMemory.prob)
    logit = staticmethod(KalmanMemory.logit)

    def evidence(self, psi, var):
        return self.lin.evidence(psi, var)

    @torch.no_grad()
    def write(self, psi: torch.Tensor, s: torch.Tensor) -> None:
        if not hasattr(self, "epi_raw"):
            self.epi_raw = torch.zeros(self.epi.capacity, self.epi.dim, device=self.epi.device, dtype=self.epi.dtype)
        self.epi_raw[self.epi.n] = psi.to(self.epi.device, self.epi.dtype).reshape(-1)
        self.lin.write(psi, s)
        self.epi.write(psi, s)
