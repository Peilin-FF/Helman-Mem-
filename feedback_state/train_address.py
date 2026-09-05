"""Meta-learn the memory address: a projection trained so that the online Kalman memory learns fast.

Outer loop: the projections W_q, W_c (frozen-feature dim -> d).  Inner loop: the
Kalman memory run over an episode of the training stream, exactly as at test time
(cold start, decide-then-update).  The outer loss is the online selection loss
summed along the episode, i.e. the area above the learning curve; gradients flow
through the whole Kalman recursion (Sherman-Morrison is differentiable), so the
address is shaped for *online learnability*, not for offline classification.
This is the "learning to learn" reading of DeltaNet / TTT / Mesa layers: the
projections are the slow weights, the memory state the fast weights.

  PYTHONPATH=. python -m feedback_state.train_address --model q3_4b --dim 64 --episodes 400 --episode_len 512 \
      --out outputs/address/q3_4b/d64.pt
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from feedback_state.addresses import AddressMap, Projection
from feedback_state.feature_streams import load_stream


class LearnedProjection(torch.nn.Module):
    """Standardise (fixed) + linear map (trainable, initialised from PCA) + fixed global scale."""

    def __init__(self, pca: Projection) -> None:
        super().__init__()
        self.register_buffer("mean", pca.mean.clone())
        self.register_buffer("std", pca.std.clone())
        self.register_buffer("center", pca.center.clone())
        self.W = torch.nn.Parameter(pca.basis.clone())
        self.scale = float(pca.scale)
        self.dim = int(pca.basis.shape[1])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (((X.to(self.W.device, torch.float32) - self.mean) / self.std - self.center) @ self.W) / self.scale

    def to_projection(self, device) -> Projection:
        return Projection(state={"mean": self.mean.cpu(), "std": self.std.cpu(), "center": self.center.cpu(), "basis": self.W.detach().cpu(), "scale": self.scale}, device=device)


def episode_loss(psi_q: torch.Tensor, psi_c: torch.Tensor, labels: torch.Tensor, real: list[int], *, lam: float, P: int) -> tuple[torch.Tensor, float]:
    """Differentiable cold-start Kalman run over one episode (design qc, shared head).

    psi_q [L, dq], psi_c [L, P, dc] (float64, with grad), labels [L, P] in {0,1}.
    Returns (sum of selection losses, selection accuracy)."""
    L, dq, dc = psi_q.shape[0], psi_q.shape[1], psi_c.shape[2]
    D = P * dq + dc + 1
    dev = psi_q.device
    Pm = torch.eye(D, device=dev, dtype=torch.float64) / lam
    b = torch.zeros(D, device=dev, dtype=torch.float64)
    total = psi_q.new_zeros(())
    hits = 0
    counted = 0
    for t in range(L):
        r = real[t]
        if r < 1:
            continue
        blocks = torch.zeros(r, P * dq, device=dev, dtype=torch.float64)
        for p in range(r):
            blocks[p, p * dq : (p + 1) * dq] = psi_q[t]
        X = torch.cat([blocks, psi_c[t, :r], torch.ones(r, 1, device=dev, dtype=torch.float64)], 1)  # [r, D]
        PX = X @ Pm
        mu = PX @ b
        var = (PX * X).sum(1)
        z = mu / torch.sqrt(1.0 + var)
        prob = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        ell = torch.log(prob.clamp(1e-6, 1 - 1e-6)) - torch.log1p(-prob.clamp(1e-6, 1 - 1e-6))
        y = labels[t, :r]
        nc = int(y.sum())
        if 0 < nc < r:
            logp = torch.log_softmax(ell, 0)
            total = total - torch.logsumexp(logp[y.bool()], 0)
        hits += int(y[int(torch.argmax(ell.detach()))]); counted += 1   # accuracy over all events, as in memsim
        s = y.to(torch.float64) * 2 - 1
        for c in range(r):  # decide-then-update, one write per candidate
            x = X[c]
            px = Pm @ x
            k = px / (1.0 + x @ px)
            Pm = Pm - torch.outer(k, px)
            b = b + s[c] * x
    return total, hits / max(counted, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q3_4b")
    ap.add_argument("--stream", default="train")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--episode_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=float, default=0.1, help="fraction of the stream kept for validation episodes")
    ap.add_argument("--arch", choices=["linear", "orthogonal", "flow"], default="linear", help="address map family (see addresses.AddressMap)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    fs = load_stream(args.stream, args.model)
    N, P = len(fs), fs.num_peers
    pca_q = Projection(fs.sem, args.dim, device)
    pca_c = Projection(fs.peer_hidden.reshape(-1, fs.peer_hidden.shape[-1]), args.dim, device)
    proj_q, proj_c = AddressMap(pca_q, arch=args.arch).to(device), AddressMap(pca_c, arch=args.arch).to(device)
    sem = fs.sem.to(device); ph = fs.peer_hidden.to(device)
    labels = fs.labels.to(device); real = fs.real.tolist()
    rng = np.random.default_rng(args.seed)
    n_val = int(N * args.holdout)
    perm = rng.permutation(N)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    optim = torch.optim.Adam(list(proj_q.parameters()) + list(proj_c.parameters()), lr=args.lr)

    def run(idx: np.ndarray, grad: bool):
        idx_t = torch.as_tensor(idx, device=device)
        with torch.set_grad_enabled(grad):
            pq = proj_q.forward(sem[idx_t]).to(torch.float64)
            pc = proj_c.forward(ph[idx_t].reshape(len(idx) * P, -1)).reshape(len(idx), P, -1).to(torch.float64)
            return episode_loss(pq, pc, labels[idx_t], [real[i] for i in idx], lam=args.lam, P=P)

    def validate() -> float:
        accs = []
        for s in range(0, len(val_idx) - args.episode_len + 1, args.episode_len):
            _, acc = run(val_idx[s : s + args.episode_len], grad=False)
            accs.append(acc)
        return float(np.mean(accs)) if accs else float("nan")

    t0 = time.time()
    base_val = validate()
    print(f"[address] arch={args.arch} model={args.model} dim={args.dim} episodes={args.episodes} len={args.episode_len} val_episodes={len(val_idx) // args.episode_len} pca_val_acc={100 * base_val:.2f}", flush=True)
    history = []
    for ep in range(1, args.episodes + 1):
        start = int(rng.integers(0, len(train_idx) - args.episode_len))
        loss, acc = run(train_idx[start : start + args.episode_len], grad=True)
        optim.zero_grad(set_to_none=True)
        (loss / args.episode_len).backward()
        torch.nn.utils.clip_grad_norm_(list(proj_q.parameters()) + list(proj_c.parameters()), 1.0)
        optim.step()
        history.append((float(loss) / args.episode_len, acc))
        if ep % 20 == 0 or ep == args.episodes:
            recent = history[-20:]
            val = validate()
            print(f"[address] ep {ep}/{args.episodes} loss={np.mean([h[0] for h in recent]):.4f} train_ep_acc={100 * np.mean([h[1] for h in recent]):.2f} val_ep_acc={100 * val:.2f} ({time.time() - t0:.0f}s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "learned_address_v2", "model": args.model, "dim": args.dim, "lam": args.lam, "arch": args.arch,
                "proj_q": (proj_q.state() if args.arch != "linear" else proj_q.to_projection(device).state()),
                "proj_c": (proj_c.state() if args.arch != "linear" else proj_c.to_projection(device).state()),
                "pca_val_acc": base_val, "final_val_acc": validate(), "history": history}, args.out)
    print(f"[address] saved {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
