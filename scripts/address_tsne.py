"""t-SNE maps of the three memory addresses on the OOD stream (the geometry behind the Method page's table).

  phi   the paper's soft task-prototype address: cosine of the whitened, mean-pooled mid-layer question vector to the
        three training-task centroids (code / math / rag), softmax at tau = 0.1, fixed random projection to 16 dims,
        L2-normalised (symmetric_memory.phi_of; the projection is random here instead of learned, which does not
        change the geometry: the address lives on a 2-simplex either way)
  psi   the new question+responses address: Projection (standardise + PCA-512 + global scale, fitted once on the
        2,963 mixed training events) of the candidate-judge hidden states averaged over candidates, plus a constant
  psi_w the same psi read in the metric of A = lam I + sum psi psi^T over the fixed-order OOD stream:  A^{-1/2} psi

Everything is label-free.  Colours are the source dataset, markers the task type, both used only for plotting.
Writes <out>/tsne_addresses.png, <out>/geometry.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from feedback_state.addresses import Projection

TASK_MARK = {"boolqa": "o", "mcqa": "^", "shortqa": "s", "math": "o", "rag": "^", "code": "s"}


def load(pattern: str):
    """One or several shards of one stream (glob), concatenated in stream order."""
    import glob
    shards = [torch.load(p, map_location="cpu", weights_only=False) for p in sorted(glob.glob(pattern))]
    if not shards:
        raise FileNotFoundError(pattern)
    idx = torch.cat([d["indices"] for d in shards]); order = idx.argsort()
    d = {k: torch.cat([s[k] for s in shards])[order] for k in ("q_mean", "sem")}
    d["input"] = shards[0]["input"]
    recs = [json.loads(l) for l in open(d["input"])]
    recs = [recs[i] for i in idx[order].tolist()]
    return d, recs


def eff_rank(X: np.ndarray) -> dict:
    Xc = X - X.mean(0)
    ev = np.linalg.eigvalsh(np.cov(Xc, rowvar=False))
    ev = np.clip(ev, 0, None)
    p = ev / ev.sum()
    return {"participation": float(ev.sum() ** 2 / (ev**2).sum()), "entropy": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))}


def cos2(X: np.ndarray, n: int, rng) -> float:
    i = rng.integers(0, len(X), n); j = rng.integers(0, len(X), n)
    keep = i != j
    a, b = X[i[keep]], X[j[keep]]
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return float((c**2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="outputs/context_features/q3_4b/train/shard0.pt")
    ap.add_argument("--test", default="outputs/context_features/q3_4b_indist6_ph/ood/shard*.pt")
    ap.add_argument("--stream_name", default="in-distribution test stream")
    ap.add_argument("--out", default="outputs/address/tsne_q3_4b_indist")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--n_plot", type=int, default=0, help="0 = every event")
    ap.add_argument("--phi_layer", type=int, default=1, help="index into the stored [L/3, 2L/3, L] mean-pooled layers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    tr, tr_recs = load(args.train)
    te, te_recs = load(args.test)
    H = tr["q_mean"].shape[1] // 3
    print(f"train {len(tr_recs)} ood {len(te_recs)} hidden {H} ({time.time()-t0:.0f}s)", flush=True)

    # --- the paper's phi: task centroids of the mean-pooled question vector (cap 200 per task), whitening, softmax
    sl = slice(args.phi_layer * H, (args.phi_layer + 1) * H)
    qtr = tr["q_mean"][:, sl].float()
    by_task = {}
    for h, r in zip(qtr, tr_recs):
        by_task.setdefault(r["task_type"], [])
        if len(by_task[r["task_type"]]) < 200:
            by_task[r["task_type"]].append(h)
    tasks = sorted(by_task)
    allh = torch.cat([torch.stack(by_task[t]) for t in tasks])
    mean, std = allh.mean(0), allh.std(0) + 1e-6
    C = torch.stack([torch.stack(by_task[t]).mean(0) for t in tasks]); C = (C - mean) / std
    q = (te["q_mean"][:, sl].float() - mean) / std
    sim = (q @ C.T) / (q.norm(dim=1, keepdim=True) * C.norm(dim=1)[None] + 1e-9)
    p = torch.softmax(sim / 0.1, dim=1)
    g = torch.Generator().manual_seed(args.seed)
    W = torch.randn(len(tasks), 16, generator=g) / len(tasks) ** 0.5
    phi = p @ W; phi = phi / (phi.norm(dim=1, keepdim=True) + 1e-8)
    phi = phi.numpy()
    print(f"phi from train tasks {tasks}; mean max-prob {p.max(1).values.mean():.3f}", flush=True)

    # --- the new psi: Projection on the candidate-judge aggregate, fitted on train only
    proj = Projection(tr["sem"].float(), dim=args.dim)
    psi = proj(te["sem"].float()).double()
    psi = torch.cat([psi, torch.ones(len(psi), 1, dtype=psi.dtype)], 1)          # [N, dim+1]
    A = args.lam * torch.eye(psi.shape[1], dtype=psi.dtype) + psi.T @ psi         # coverage after the whole stream
    ev, U = torch.linalg.eigh(A)
    W_half = U @ torch.diag(ev.rsqrt()) @ U.T
    psi_w = (psi @ W_half)
    psi, psi_w = psi.float().numpy(), psi_w.float().numpy()

    geo = {}
    for name, X in (("phi", phi), ("psi_raw", psi), ("psi_whitened", psi_w)):
        geo[name] = {"dim": int(X.shape[1]), **eff_rank(X), "cos2_random_pairs": cos2(X, 100_000, rng)}
        print(name, geo[name], flush=True)
    # --- how much of the peers' correctness is readable from each address: leave-one-out k-NN (Euclidean in the space,
    #     which for psi_w is the A^{-1} metric), against the per-task mean as the "task-only" baseline
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import roc_auc_score
    labels = np.array([[int(v) for _, v in sorted(te_recs[i]["correctness_by_peer"].items())] for i in range(len(te_recs))], dtype=float)  # [N, P]
    frac_all = labels.mean(1)
    task_all = np.array([r["task_type"] for r in te_recs])
    task_mean = np.array([frac_all[task_all == t].mean() for t in task_all])
    knn = {"task_mean_baseline": {"corr": float(np.corrcoef(task_mean, frac_all)[0, 1]),
                                   "mae": float(np.abs(task_mean - frac_all).mean())}}
    for k in (10, 30):
        for name, X in (("phi", phi), ("psi_raw", psi), ("psi_whitened", psi_w)):
            nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
            _, nb = nn.kneighbors(X)
            nb = nb[:, 1:]                                   # drop the point itself
            pred = frac_all[nb].mean(1)
            per_peer = labels[nb].mean(1)                     # [N, P] neighbour vote per peer
            aucs = [roc_auc_score(labels[:, j], per_peer[:, j]) for j in range(labels.shape[1]) if 0 < labels[:, j].mean() < 1]
            within = {}
            for t in sorted(set(task_all)):
                m = task_all == t
                within[t] = float(np.corrcoef(pred[m], frac_all[m])[0, 1])
            knn[f"{name}_k{k}"] = {"corr": float(np.corrcoef(pred, frac_all)[0, 1]), "mae": float(np.abs(pred - frac_all).mean()),
                                   "per_peer_auc_mean": float(np.mean(aucs)), "within_task_corr": within}
            print(name, k, knn[f"{name}_k{k}"], flush=True)
    print("task-mean baseline", knn["task_mean_baseline"], flush=True)
    # --- the readouts the memories actually use, run prequentially along the stream (read before write, stream order):
    #     raw-sum  r_p = x^T b_p            (Sigma-Mem's accumulation without decay, unnormalised)
    #     Kalman   r_p = x^T Lambda^{-1} b_p (the record; Lambda = lam I + sum x x^T)
    def prequential(X, lam):
        N, D = X.shape; P = labels.shape[1]
        Pm = np.eye(D) / lam; b = np.zeros((D, P)); S = labels * 2 - 1
        raw = np.zeros((N, P)); kal = np.zeros((N, P))
        for t in range(N):
            x = X[t]
            raw[t] = x @ b
            kal[t] = x @ (Pm @ b)
            Px = Pm @ x; Pm -= np.outer(Px, Px) / (1.0 + x @ Px)
            b += np.outer(x, S[t])
        return raw, kal
    def score(pred):
        f = pred.mean(1)
        aucs = [roc_auc_score(labels[:, j], pred[:, j]) for j in range(labels.shape[1]) if 0 < labels[:, j].mean() < 1]
        fav = pred.argmax(1); disagree = (labels.min(1) != labels.max(1))
        return {"corr": float(np.corrcoef(f, frac_all)[0, 1]), "per_peer_auc_mean": float(np.mean(aucs)),
                "favourite_right_on_disagreements": float(labels[np.arange(len(labels)), fav][disagree].mean())}
    preq = {}
    for name, X in (("phi", np.c_[phi, np.ones(len(phi))].astype(np.float64)), ("psi", psi.astype(np.float64))):
        raw, kal = prequential(X, args.lam)
        preq[f"{name}_rawsum"] = score(raw); preq[f"{name}_kalman"] = score(kal)
        print(name, "rawsum", preq[f"{name}_rawsum"], "kalman", preq[f"{name}_kalman"], flush=True)
    json.dump({"tasks_train": tasks, "lam": args.lam, "geometry": geo, "knn": knn, "prequential": preq}, open(out / "geometry.json", "w"), indent=1)

    # --- t-SNE
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_plot = args.n_plot or len(te_recs)
    idx = np.sort(rng.choice(len(te_recs), size=min(n_plot, len(te_recs)), replace=False))
    src = np.array([te_recs[i]["source"] for i in idx]); tt = np.array([te_recs[i]["task_type"] for i in idx])
    sources = sorted(set(src), key=lambda s: (tt[src == s][0], s))
    cmap = plt.get_cmap("tab10" if len(sources) <= 10 else "tab20")
    colours = {s: cmap(k % cmap.N) for k, s in enumerate(sources)}
    panels = [("(a) paper's φ: soft task prototype (16-d)", phi), ("(b) new ψ: question + responses (513-d)", psi),
              ("(c) ψ in the metric of A: A^{-1/2} ψ", psi_w)]
    frac = np.array([np.mean(list(te_recs[i]["correctness_by_peer"].values())) for i in idx])   # share of the peers that were right
    fig, axes = plt.subplots(2, 3, figsize=(22, 15))
    sc = None
    for col, (title, X) in enumerate(panels):
        t1 = time.time()
        Y = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=args.seed, n_jobs=8).fit_transform(X[idx])
        np.save(out / f"tsne_{title[1]}.npy", Y)
        ax = axes[0, col]
        for s in sources:
            m = src == s
            ax.scatter(Y[m, 0], Y[m, 1], s=10, alpha=0.6, c=[colours[s]], marker=TASK_MARK.get(tt[m][0], "o"), linewidths=0, label=f"{s} ({tt[m][0]})")
        ax.set_title(title, fontsize=15); ax.set_xticks([]); ax.set_yticks([])
        g_ = geo[{"a": "phi", "b": "psi_raw", "c": "psi_whitened"}[title[1]]]
        ax.text(0.01, 0.01, f"random-pair cos² {g_['cos2_random_pairs']:.3f}",
                transform=ax.transAxes, fontsize=12, ha="left", va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8))
        ax = axes[1, col]
        order = np.argsort(np.abs(frac - 0.5))[::-1]          # draw the decided events last so they stay visible
        sc = ax.scatter(Y[order, 0], Y[order, 1], s=10, alpha=0.7, c=frac[order], cmap="RdYlGn", vmin=0, vmax=1, linewidths=0)
        ax.set_title("same map, coloured by the peers' correctness labels", fontsize=13); ax.set_xticks([]); ax.set_yticks([])
        print(f"{title}: t-SNE {time.time()-t1:.0f}s", flush=True)
    axes[0, 1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=min(len(sources), 6), fontsize=13, markerscale=3, frameon=False)
    cax = fig.add_axes([0.3, 0.045, 0.4, 0.012])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("share of the six peers that answered correctly (red: all wrong, green: all right)", fontsize=13)
    fig.suptitle(f"Qwen3-4B, {args.stream_name} ({len(idx):,} of {len(te_recs):,} events; top: colour = source dataset, marker = task type; "
                 f"bottom: colour = correctness; training tasks were {', '.join(tasks)})", fontsize=14)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.10, hspace=0.16, wspace=0.06)
    fig.savefig(out / "tsne_addresses.png", dpi=160)
    print(f"wrote {out/'tsne_addresses.png'} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
