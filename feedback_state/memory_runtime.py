"""Online memory over one event stream: projected addresses + Kalman state + read/write helpers.

Shared by the judge trainer and evaluator so both run exactly the same
decide-then-update protocol: ``read(t)`` returns the memory's log-odds and
evidence counts for the real candidates of event ``t`` before any label of
that event is seen; ``write(t, labels)`` applies the feedback afterwards.
"""
from __future__ import annotations

import torch

from feedback_state.addresses import Projection, design_dim, design_rows, memory_evidence
from feedback_state.feature_streams import FeatureStream
from feedback_state.kalman_memory import KalmanMemory


class MemoryRuntime:
    def __init__(self, *, design: str, proj_q: Projection, proj_c: Projection, num_peers: int, lam: float, rho: float = 1.0, device=None) -> None:
        self.design, self.proj_q, self.proj_c = str(design), proj_q, proj_c
        self.P, self.lam, self.rho = int(num_peers), float(lam), float(rho)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.per_peer = self.design == "q"
        self.D = design_dim(self.design, self.P, proj_q.dim, proj_c.dim)
        self.mem = KalmanMemory(self.D, self.P if self.per_peer else 1, lam=self.lam, rho=self.rho, device=self.device)
        self.psi_q = self.psi_c = self.z = self.real = None

    def attach(self, fs: FeatureStream, *, q_source: str = "sem") -> None:
        src = fs.sem if q_source == "sem" else fs.q_mean
        N, P = len(fs), fs.num_peers
        self.psi_q = self.proj_q(src).to(torch.float64)
        self.psi_c = self.proj_c(fs.peer_hidden.reshape(N * P, -1)).reshape(N, P, -1).to(torch.float64)
        self.z = fs.margins.to(self.device, torch.float64)
        self.real = fs.real.tolist()

    def reset(self) -> None:
        self.mem.reset()

    def rows(self, t: int) -> torch.Tensor:
        r = self.real[t]
        return design_rows(self.design, self.psi_q[t], self.psi_c[t, :r], self.z[t, :r], list(range(r)), self.P)

    @torch.no_grad()
    def read(self, t: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (ell [r] log-odds, n_eff [r], evidence [r, 4], rows X) for the real candidates of event t."""
        r = self.real[t]
        X = self.rows(t)
        mu, var = self.mem.read(X)
        if self.per_peer:
            mu, var = mu[0, :r], var.expand(r)
            Xe = X.expand(r, -1)
        else:
            mu, Xe = mu[:, 0], X
        p = self.mem.prob(mu, var)
        ell = self.mem.logit(p)
        n_eff = self.mem.evidence(Xe, var)
        return ell, n_eff, memory_evidence(ell, n_eff), X

    @torch.no_grad()
    def write(self, t: int, X: torch.Tensor, labels) -> None:
        r = self.real[t]
        s = torch.tensor([1.0 if int(labels[c]) else -1.0 for c in range(r)], device=self.device, dtype=torch.float64)
        if self.per_peer:
            sv = torch.zeros(self.P, device=self.device, dtype=torch.float64)
            sv[:r] = s
            self.mem.write(X[0], sv)
        else:
            for c in range(r):
                self.mem.write(X[c], s[c : c + 1])

    def config(self) -> dict:
        return {"design": self.design, "num_peers": self.P, "lam": self.lam, "rho": self.rho, "dim_q": self.proj_q.dim, "dim_c": self.proj_c.dim}
