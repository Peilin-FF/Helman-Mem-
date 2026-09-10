"""Figures for the attention-mass measurement (scripts/attention_mass.py).

  A  layers x peer-rank heatmaps: mean share of peer attention on the peer of each reliability rank
     (rank 1 = the record's favourite), one panel per model and gamma. Shows where in depth the
     model prefers reliable peers, and what the tilt does to that.
  B  share on the favourite peer by layer, one line per model, solid gamma = 0, dashed gamma = 3.
  C  one example prompt: layers x peers heatmaps (peers ordered by the record), correctness marked,
     for every model at gamma 0 and 3.
  D  token-resolution map of one example if the run dumped it: layers x prompt tokens, the attention
     of the last prompt token, with the peer blocks outlined.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["frozen", "ours", "control", "question-only"]


def load(d: Path) -> dict:
    data = {}
    for f in sorted(glob.glob(str(d / "*_*.json"))):
        if f.endswith("summary.json"):
            continue
        j = json.load(open(f))
        data[(j["model"], j["stream"])] = j
    return data


def rank_shares(res: list[dict], gamma: str, q: str) -> np.ndarray:
    """mean over prompts of [layers, rank] share of peer mass, peers sorted by the record (desc)."""
    acc, n = None, 0
    for r in res:
        m = np.asarray(r["mass"][gamma][q])                      # [L, P]
        order = np.argsort(-np.asarray(r["probs"]))
        m = m[:, order]
        s = m / np.clip(m.sum(1, keepdims=True), 1e-12, None)
        acc = s if acc is None else acc + s
        n += 1
    return acc / max(n, 1)


def fig_a(data, stream, q, out):
    models = [m for m in ORDER if (m, stream) in data]
    gammas = ["0.0", "3.0"]
    fig, axes = plt.subplots(len(models), 2, figsize=(9, 2.6 * len(models)), squeeze=False)
    P = len(data[(models[0], stream)]["results"][0]["probs"])
    vmax = 0
    mats = {}
    for i, m in enumerate(models):
        for j, g in enumerate(gammas):
            mats[(m, g)] = rank_shares(data[(m, stream)]["results"], g, q)
            vmax = max(vmax, mats[(m, g)].max())
    for i, m in enumerate(models):
        for j, g in enumerate(gammas):
            ax = axes[i, j]
            im = ax.imshow(mats[(m, g)].T, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_yticks(range(P)); ax.set_yticklabels([f"rank {k+1}" if k else "favourite" for k in range(P)], fontsize=8)
            ax.set_title(f"{m}, γ = {float(g):.0f}", fontsize=10)
            if i == len(models) - 1:
                ax.set_xlabel("layer")
            else:
                ax.set_xticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="share of peer attention")
    fig.suptitle(f"Where the attention goes among the peers, by reliability rank ({stream}, query = {'last prompt token' if q == 'last' else 'tokens after the peers'}; uniform = {1/P:.3f})", fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_b(data, stream, q, out):
    models = [m for m in ORDER if (m, stream) in data]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colours = dict(zip(ORDER, plt.get_cmap("tab10").colors))
    P = len(data[(models[0], stream)]["results"][0]["probs"])
    for m in models:
        for g, ls in (("0.0", "-"), ("3.0", "--")):
            s = rank_shares(data[(m, stream)]["results"], g, q)[:, 0]
            ax.plot(range(len(s)), s, ls, color=colours[m], label=f"{m}, γ = {float(g):.0f}", lw=1.8)
    ax.axhline(1 / P, color="grey", lw=0.8, ls=":", label="uniform")
    ax.set_xlabel("layer"); ax.set_ylabel("share of peer attention on the record's favourite")
    ax.set_title(f"Does the model prefer the peer the record trusts? ({stream}, query = {'last prompt token' if q == 'last' else 'tokens after the peers'})", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_c(data, stream, q, out, which: int = 0):
    models = [m for m in ORDER if (m, stream) in data]
    ids = set.intersection(*[{r["id"] for r in data[(m, stream)]["results"]} for m in models])
    # choose a prompt where the record is decisive and the favourite is right, unless asked otherwise
    pool = sorted(ids)
    by = {m: {r["id"]: r for r in data[(m, stream)]["results"]} for m in models}
    pick = None
    for i in pool:
        r = by[models[0]][i]
        pr, co = np.asarray(r["probs"]), np.asarray(r["correct"])
        if pr.max() - pr.min() > 0.3 and co[int(np.argmax(pr))] == 1 and co.sum() <= 3:
            if which == 0:
                pick = i; break
            which -= 1
    pick = pick or pool[0]
    fig, axes = plt.subplots(len(models), 2, figsize=(9, 2.6 * len(models)), squeeze=False)
    r0 = by[models[0]][pick]; order = np.argsort(-np.asarray(r0["probs"]))
    labels = [f"p={r0['probs'][k]:.2f} {'✓' if r0['correct'][k] else '✗'}" for k in order]
    vmax = max(np.asarray(by[m][pick]["mass"][g][q])[:, order].max() for m in models for g in ("0.0", "3.0"))
    for i, m in enumerate(models):
        for j, g in enumerate(("0.0", "3.0")):
            ax = axes[i, j]
            mat = np.asarray(by[m][pick]["mass"][g][q])[:, order]
            im = ax.imshow(mat.T, aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(f"{m}, γ = {float(g):.0f}", fontsize=10)
            if i == len(models) - 1:
                ax.set_xlabel("layer")
            else:
                ax.set_xticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="attention mass on the block")
    fig.suptitle(f"One event ({pick}, {stream}): attention of the last prompt token on each peer block, peers ordered by the record", fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return pick


def fig_d(dump_files, out):
    """Token-resolution map from attention_mass.py --dump_ids output."""
    dumps = [json.load(open(f)) for f in dump_files]
    if not dumps:
        return
    fig, axes = plt.subplots(len(dumps), 1, figsize=(14, 2.4 * len(dumps)), squeeze=False)
    for ax, d in zip(axes[:, 0], dumps):
        rows = np.asarray(d["row_last"])            # [L, K]
        im = ax.imshow(rows, aspect="auto", cmap="inferno", vmin=0, vmax=np.percentile(rows, 99.5), interpolation="nearest")
        for k, (s, e) in enumerate(d["spans"]):
            ax.axvline(s - 0.5, color="cyan", lw=0.6); ax.axvline(e - 0.5, color="cyan", lw=0.6)
            ax.text((s + e) / 2, -1.5, f"peer {k+1}\np={d['probs'][k]:.2f} {'✓' if d['correct'][k] else '✗'}", ha="center", va="bottom", fontsize=7, color="black")
        ax.set_ylabel("layer"); ax.set_title(f"{d['model']}, γ = {d['gamma']:.0f}: attention of the last prompt token over every prompt token", fontsize=9, pad=28)
    axes[-1, 0].set_xlabel("prompt token")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label="attention")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/address/attention_mass")
    ap.add_argument("--q", default="last", choices=["last", "post"])
    args = ap.parse_args()
    d = Path(args.dir); data = load(d)
    for stream in sorted({s for _, s in data}):
        fig_a(data, stream, args.q, d / f"figA_rank_heatmap_{stream}_{args.q}.png")
        fig_b(data, stream, args.q, d / f"figB_favourite_by_layer_{stream}_{args.q}.png")
        pick = fig_c(data, stream, args.q, d / f"figC_example_{stream}_{args.q}.png")
        print(f"{stream}: figures A, B, C written (example {pick})")
    dumps = sorted(glob.glob(str(d / "dump_*.json")))
    if dumps:
        fig_d(dumps, d / "figD_token_map.png"); print(f"figure D written from {len(dumps)} dumps")


if __name__ == "__main__":
    main()
