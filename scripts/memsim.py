"""Simulate the online competence memory on cached frozen-center-model features.

Every arm is decide-then-update from a cold start: at event t the memory reads
the addresses of the candidates, a peer is selected, then the correctness of all
peers is written.  No labels, dataset names or peer identities enter the
addresses.  Reported per arm: total accuracy, accuracy in consecutive windows
of the stream (the learning curve), and cumulative accuracy at checkpoints.

Address designs:
  q     per-peer heads on the question address                    "who is reliable on questions like this"
  c     one head on the candidate-conditioned address             "does this answer look right" (verification)
  cm    c + the judge's own Yes/No log-odds as a feature
  qc    peer-blocked question address + shared candidate address  both, in one linear model
  qcm   qc + judge log-odds (shared and per-peer)
  rq / rc / rqc   the same with random Fourier features of the addresses (RBF kernel memory)
Memory kinds:
  kalman  exact Bayesian linear posterior (RLS / Kalman gain)      fixed-size state, O(d^2) per write
  delta   one gradient step with a fixed rate (DeltaNet rule)      ablation
  knn     episodic Nadaraya-Watson over stored events              growing state
  hybrid  kalman + episodic residual correction                    growing state
Selection rules: route = argmax of the memory's predictive probability; vote =
Nitzan-Paroush weighted majority over candidates giving the same answer with
weights logit(p) (reduces to route when no two candidates agree).

  PYTHONPATH=. python scripts/memsim.py --model q3_4b --streams ood indist train --dim 256 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from feedback_state.addresses import Projection, design_dim, design_rows, load_address
from feedback_state.answer_groups import answer_groups
from feedback_state.feature_streams import FeatureStream, load_stream
from feedback_state.kalman_memory import DeltaMemory, KalmanMemory
from feedback_state.kernel_memory import EpisodicMemory, HybridMemory, RandomFeatureMap, median_lengthscale

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- features
class StreamFeatures:
    """Projected addresses of one stream on the device: psi_q [N, d], psi_c [N, P, d], z [N, P]."""

    def __init__(self, fs: FeatureStream, proj_q: Projection, proj_c: Projection, device, *, q_source: str = "sem",
                 rff_q: RandomFeatureMap | None = None, rff_c: RandomFeatureMap | None = None, judge_scores: dict | None = None) -> None:
        self.N, self.P = len(fs), fs.num_peers
        src = fs.sem if q_source == "sem" else fs.q_mean
        self.psi_q = proj_q(src).to(torch.float64)
        flat = fs.peer_hidden.reshape(self.N * self.P, -1)
        self.psi_c = proj_c(flat).reshape(self.N, self.P, -1).to(torch.float64)
        self.rff_q = rff_q(self.psi_q) if rff_q is not None else None
        self.rff_c = rff_c(self.psi_c.reshape(self.N * self.P, -1)).reshape(self.N, self.P, -1) if rff_c is not None else None
        z = fs.margins.clone()
        if judge_scores is not None:
            missing = 0
            for i, rid in enumerate(fs.ids):
                sc = judge_scores.get(str(rid))
                if sc is None:
                    missing += 1; continue
                z[i, : len(sc)] = torch.tensor(sc, dtype=z.dtype)
            print(f"[memsim] judge scores loaded for {len(fs) - missing}/{len(fs)} events", flush=True)
        self.z = z.to(device, torch.float64)
        self.labels = fs.labels.to(device)
        self.labels_np = fs.labels.numpy()
        self.real = fs.real.to(device)
        self.real_list = fs.real.tolist()
        self.groups = [answer_groups(rec, texts[: int(r)]) for rec, texts, r in zip(fs.records, fs.texts, fs.real.tolist())]
        self.task, self.source = fs.task, fs.source
        self.device = device

    def _addr(self, design: str):
        if design.startswith("r"):
            return self.rff_q, self.rff_c, design[1:]
        return self.psi_q, self.psi_c, design

    def design_dim(self, design: str) -> int:
        aq, ac, base = self._addr(design)
        return design_dim(base, self.P, aq.shape[1], ac.shape[2])

    def candidates(self, t: int, design: str) -> torch.Tensor:
        """Feature rows of the real candidates of event t: [real, D] (design 'q': [1, dq+1])."""
        aq, ac, base = self._addr(design)
        r = self.real_list[t]
        return design_rows(base, aq[t], ac[t, :r], self.z[t, :r], list(range(r)), self.P)


# ----------------------------------------------------------------------------- selection helpers
def select_route(ell: np.ndarray) -> int:
    return int(np.argmax(ell))


def select_vote(ell: np.ndarray, groups: list[int]) -> int:
    """Weighted majority: answer group with the largest sum of log-odds; inside it, the most reliable peer."""
    totals: dict[int, float] = {}
    for p, g in enumerate(groups):
        totals[g] = totals.get(g, 0.0) + float(ell[p])
    best = max(totals.values())
    members = [p for p, g in enumerate(groups) if totals[g] == best]
    return int(max(members, key=lambda p: ell[p]))


def curve(hits: np.ndarray, windows: int = 10) -> dict:
    n = len(hits)
    edges = np.linspace(0, n, windows + 1).astype(int)
    win = [float(hits[a:b].mean()) if b > a else float("nan") for a, b in zip(edges[:-1], edges[1:])]
    cum = {str(k): float(hits[:k].mean()) for k in (250, 500, 1000, 2000, 4000, 8000, 16000) if k <= n}
    return {"total": float(hits.mean()), "n": int(n), "windows": win,
            "first_half": float(hits[: n // 2].mean()), "second_half": float(hits[n // 2 :].mean()), "cumulative": cum}


# ----------------------------------------------------------------------------- arms
def make_memory(kind: str, D: int, heads: int, sf: StreamFeatures, design: str, args) -> object:
    cap = sf.N if design.endswith("q") else sf.N * sf.P
    if kind == "kalman":
        return KalmanMemory(D, heads, lam=args.lam, rho=args.rho, device=sf.device)
    if kind == "delta":
        return DeltaMemory(D, heads, beta=args.beta, alpha=args.rho, device=sf.device)
    if kind == "knn":
        return EpisodicMemory(D, heads, capacity=cap, k=args.knn_k, tau=args.tau, device=sf.device)
    if kind == "hybrid":
        return HybridMemory(D, heads, lam=args.lam, capacity=cap, k=args.knn_k, tau=args.tau, mix=args.mix, device=sf.device)
    raise ValueError(kind)


def run_memory_arm(sf: StreamFeatures, order: np.ndarray, design: str, kind: str, args) -> dict:
    D = sf.design_dim(design)
    per_peer = design.endswith("q")
    heads = sf.P if per_peer else 1
    mem = make_memory(kind, D, heads, sf, design, args)
    hits_route, hits_vote, evid = [], [], []
    for t in order.tolist():
        r = sf.real_list[t]
        if r < 1:
            continue
        X = sf.candidates(t, design)
        mu, var = mem.read(X)
        if per_peer:
            mu = mu[0, :r]
            var = var.expand(r)
        else:
            mu = mu[:, 0]
        p = mem.prob(mu, var)
        ell = mem.logit(p).cpu().numpy()
        y = sf.labels_np[t, :r]
        hits_route.append(int(y[select_route(ell)]))
        hits_vote.append(int(y[select_vote(ell, sf.groups[t])]))
        if hasattr(mem, "evidence"):
            evid.append(float(mem.evidence(X if not per_peer else X.expand(r, -1), var).mean()))
        s = torch.tensor([1.0 if y[p_] else -1.0 for p_ in range(r)], device=sf.device, dtype=torch.float64)
        if per_peer:
            sv = torch.zeros(sf.P, device=sf.device, dtype=torch.float64)
            sv[:r] = s
            mem.write(X[0], sv)
        else:
            for c in range(r):
                mem.write(X[c], s[c : c + 1])
    return {"route": curve(np.array(hits_route)), "vote": curve(np.array(hits_vote)),
            "mean_evidence_last_10pct": float(np.mean(evid[-max(1, len(evid) // 10):])) if evid else float("nan")}


def run_reference_arms(sf: StreamFeatures, order: np.ndarray) -> dict:
    labels = sf.labels_np
    z = sf.z.cpu().numpy()
    P = sf.P
    out = {k: [] for k in ("judge", "judge_vote", "majority", "oracle_event", "online_global", "online_source", "best_fixed_peer", "random")}
    alpha = np.ones(P); beta = np.ones(P)
    src_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(0)
    best_fixed = int(np.argmax(labels[order].sum(0)))
    for t in order.tolist():
        r = sf.real_list[t]
        if r < 1:
            continue
        y = labels[t, :r]
        zz = z[t, :r]
        out["judge"].append(int(y[int(np.argmax(zz))]))
        out["judge_vote"].append(int(y[select_vote(zz, sf.groups[t])]))
        out["majority"].append(int(y[select_vote(np.ones(r) + 1e-6 * zz, sf.groups[t])]))
        out["oracle_event"].append(int(y.max()))
        out["random"].append(int(y[rng.integers(r)]))
        out["best_fixed_peer"].append(int(y[min(best_fixed, r - 1)]))
        rel = (alpha / (alpha + beta))[:r]
        out["online_global"].append(int(y[int(np.argmax(rel))]))
        a_s, b_s = src_stats.setdefault(sf.source[t], (np.ones(P), np.ones(P)))
        out["online_source"].append(int(y[int(np.argmax((a_s / (a_s + b_s))[:r]))]))
        alpha[:r] += y; beta[:r] += 1 - y
        a_s[:r] += y; b_s[:r] += 1 - y
    return {k: curve(np.array(v)) for k, v in out.items()}


def hindsight_probe(sf: StreamFeatures, design: str, *, lam: float, folds: int = 5, fit_from: StreamFeatures | None = None) -> dict:
    """Ridge regression on the whole stream (in-sample and cross-validated) and, if given, fitted on another stream."""
    per_peer = design.endswith("q")

    def rows(s: StreamFeatures):
        X, y, ev = [], [], []
        for t in range(s.N):
            r = s.real_list[t]
            if r < 1:
                continue
            Xt = s.candidates(t, design)
            if per_peer:
                Xt = Xt.expand(r, -1)
            X.append(Xt); ev.append(torch.full((r,), t)); y.append(s.labels[t, :r].to(torch.float64) * 2 - 1)
        pi = torch.cat([torch.arange(s.real_list[t]) for t in range(s.N) if s.real_list[t] >= 1]).to(s.device)
        return torch.cat(X), torch.cat(y).to(s.device), torch.cat(ev), pi

    def fit(X, y, peer_index):
        D = X.shape[1]
        eye = torch.eye(D, device=X.device, dtype=X.dtype)
        if per_peer:
            W = []
            for p in range(sf.P):
                m = peer_index == p
                W.append(torch.linalg.solve(X[m].T @ X[m] + lam * eye, X[m].T @ y[m]))
            return torch.stack(W, 1)
        return torch.linalg.solve(X.T @ X + lam * eye, X.T @ y).unsqueeze(1)

    def predict(X, W, peer_index):
        if per_peer:
            return (X @ W)[torch.arange(X.shape[0], device=X.device), peer_index]
        return (X @ W)[:, 0]

    X, y, ev, peer_index = rows(sf)

    def acc_from_scores(scores: torch.Tensor) -> float:
        sc = scores.cpu().numpy(); evn = ev.numpy()
        hits, start = [], 0
        while start < len(sc):
            t = evn[start]; r = sf.real_list[t]
            hits.append(int(sf.labels_np[t, int(np.argmax(sc[start : start + r]))])); start += r
        return float(np.mean(hits))

    res = {"in_sample": acc_from_scores(predict(X, fit(X, y, peer_index), peer_index))}
    perm = np.random.default_rng(0).permutation(sf.N)
    fold_of_event = np.empty(sf.N, dtype=int); fold_of_event[perm] = np.arange(sf.N) % folds
    fold_rows = torch.as_tensor(fold_of_event[ev.numpy()], device=sf.device)
    scores = torch.zeros_like(y)
    for f in range(folds):
        m = fold_rows != f
        scores[~m] = predict(X[~m], fit(X[m], y[m], peer_index[m]), peer_index[~m])
    res["cv"] = acc_from_scores(scores)
    if fit_from is not None:
        Xo, yo, _, po = rows(fit_from)
        res["transfer"] = acc_from_scores(predict(X, fit(Xo, yo, po), peer_index))
    return res


# ----------------------------------------------------------------------------- main
def fmt(c: dict) -> str:
    cum = c["cumulative"]
    cs = " ".join(f"{k}:{100 * v:.1f}" for k, v in cum.items() if k in ("250", "1000", "4000"))
    return f"{100 * c['total']:6.2f}  {100 * c['first_half']:5.1f}->{100 * c['second_half']:5.1f} | " + " ".join(f"{100 * w:4.1f}" for w in c["windows"]) + f" | cum {cs}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q3_4b")
    ap.add_argument("--streams", nargs="+", default=["ood", "indist", "train"])
    ap.add_argument("--fit-stream", default="train", help="unlabeled stream used to fit the PCA projections")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--dim-c", type=int, default=None)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--rho", type=float, default=1.0, help="forgetting factor (kalman) / decay alpha (delta)")
    ap.add_argument("--beta", type=float, default=0.05, help="delta-rule step")
    ap.add_argument("--knn-k", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--mix", type=float, default=1.0)
    ap.add_argument("--rff", type=int, default=0, help="random Fourier feature dimension for the r* designs")
    ap.add_argument("--rff-scale", type=float, default=1.0, help="lengthscale multiplier on the median heuristic")
    ap.add_argument("--designs", nargs="+", default=["q", "c", "qc"])
    ap.add_argument("--kinds", nargs="+", default=["kalman"])
    ap.add_argument("--orders", nargs="+", default=["fixed", "shuffled0"])
    ap.add_argument("--probes", action="store_true", help="hindsight ridge probes (in-sample / cv / transfer from the fit stream)")
    ap.add_argument("--q-source", default="sem", choices=["sem", "q_mean"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--proj-ckpt", type=Path, default=None, help="learned address (train_address.py) instead of PCA projections")
    ap.add_argument("--judge-scores", type=Path, default=None,
                    help="selections.jsonl of a judge evaluation: its per-candidate judge_scores replace the frozen margins z (designs cm/qcm and the judge reference)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    device = torch.device(args.device)
    dim_c = args.dim_c or args.dim
    t0 = time.time()
    fit = load_stream(args.fit_stream, args.model, limit=args.limit)
    if args.proj_ckpt is not None:
        ck = torch.load(args.proj_ckpt, map_location="cpu", weights_only=False)
        proj_q = load_address(ck["proj_q"], device=device); proj_c = load_address(ck["proj_c"], device=device)
        args.dim = dim_c = proj_q.dim
        print(f"[memsim] learned address from {args.proj_ckpt} (dim={proj_q.dim})", flush=True)
    else:
        proj_q = Projection(fit.sem if args.q_source == "sem" else fit.q_mean, args.dim, device)
        proj_c = Projection(fit.peer_hidden.reshape(-1, fit.peer_hidden.shape[-1]), dim_c, device)
    rff_q = rff_c = None
    if args.rff > 0:
        lq = median_lengthscale(proj_q(fit.sem if args.q_source == "sem" else fit.q_mean)) * args.rff_scale
        lc = median_lengthscale(proj_c(fit.peer_hidden.reshape(-1, fit.peer_hidden.shape[-1]))) * args.rff_scale
        rff_q = RandomFeatureMap(args.dim, args.rff, lengthscale=lq, seed=0, device=device)
        rff_c = RandomFeatureMap(dim_c, args.rff, lengthscale=lc, seed=1, device=device)
        print(f"[memsim] rff dim={args.rff} lengthscales q={lq:.2f} c={lc:.2f}", flush=True)
    judge_scores = None
    if args.judge_scores is not None:
        judge_scores = {}
        for line in args.judge_scores.open():
            row = json.loads(line)
            judge_scores[str(row["id"])] = row.get("judge_scores") or row.get("scores")
    mk = lambda fs: StreamFeatures(fs, proj_q, proj_c, device, q_source=args.q_source, rff_q=rff_q, rff_c=rff_c,
                                   judge_scores=(judge_scores if fs is not fit or args.fit_stream in args.streams else None))
    fit_sf = mk(fit) if (args.probes or args.fit_stream in args.streams) else None
    print(f"[memsim] model={args.model} fit={args.fit_stream} n={len(fit)} dim={args.dim}/{dim_c} lam={args.lam} rho={args.rho} kinds={args.kinds} ({time.time() - t0:.0f}s)", flush=True)
    results = {"args": {k: str(v) for k, v in vars(args).items()}, "streams": {}}
    for name in args.streams:
        sf = fit_sf if name == args.fit_stream else mk(load_stream(name, args.model, limit=args.limit))
        print(f"\n=== {args.model} / {name}  events={sf.N}  ({time.time() - t0:.0f}s)", flush=True)
        results["streams"][name] = {}
        for order_name in args.orders:
            order = np.arange(sf.N) if order_name == "fixed" else np.random.default_rng(int(order_name.replace("shuffled", ""))).permutation(sf.N)
            block = {"reference": run_reference_arms(sf, order), "memory": {}}
            print(f"--- order={order_name}")
            for k, c in block["reference"].items():
                print(f"  ref {k:16s}          {fmt(c)}")
            for design in args.designs:
                for kind in args.kinds:
                    r = run_memory_arm(sf, order, design, kind, args)
                    block["memory"][f"{kind}:{design}"] = r
                    print(f"  {kind:7s} {design:5s} route     {fmt(r['route'])}   evid={r['mean_evidence_last_10pct']:.1f}", flush=True)
                    print(f"  {kind:7s} {design:5s} vote      {fmt(r['vote'])}")
            results["streams"][name][order_name] = block
            print(f"  ({time.time() - t0:.0f}s)", flush=True)
        if args.probes:
            probes = {}
            for design in args.designs:
                probes[design] = hindsight_probe(sf, design, lam=args.lam, fit_from=(fit_sf if name != args.fit_stream else None))
                print(f"  probe {design:6s} " + "  ".join(f"{k}={100 * v:.2f}" for k, v in probes[design].items()), flush=True)
            results["streams"][name]["probes"] = probes
    tag = args.tag or f"dim{args.dim}_lam{args.lam:g}_rho{args.rho:g}_{'-'.join(args.kinds)}" + (f"_rff{args.rff}" if args.rff else "") + ("_learned" if args.proj_ckpt else "") + ("_judgescores" if args.judge_scores else "")
    out = args.out or (ROOT / "outputs" / "memsim" / args.model / f"{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\n[memsim] wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
