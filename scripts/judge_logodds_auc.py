"""Baseline for any trained judgement: how well does the frozen judge's own label-free Yes/No log-odds
separate right from wrong peer answers, with no memory and no labels?  Reads the stored context
features (``margins`` = unsteered Yes-No log-odds per peer, [N, P]) and the records' correctness."""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="glob of shard*.pt")
    ap.add_argument("--records", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    shards = [torch.load(p, map_location="cpu", weights_only=False) for p in sorted(glob.glob(args.features))]
    idx = torch.cat([d["indices"] for d in shards]); order = idx.argsort()
    marg = torch.cat([d["margins"] for d in shards])[order].float().numpy()          # [N, P]
    recs = [json.loads(l) for l in open(args.records)]
    recs = [recs[i] for i in idx[order].tolist()]
    P = marg.shape[1]
    def labels(r):
        d = r.get("correctness_by_peer") or {k: int(float(v) >= 0.5) for k, v in r["peer_correct"].items()}
        return [int(v) for _, v in sorted(d.items())][:P]
    lab = np.array([labels(r) for r in recs], float)
    keep = (lab.min(1) != lab.max(1))                                                # events where peers disagree
    y, s = lab[keep].ravel(), marg[keep].ravel()
    auc = roc_auc_score(y, s)
    fav = np.mean([lab[i][int(np.argmax(marg[i]))] for i in np.nonzero(keep)[0]])
    # per-event rank correlation with correctness, the same statistic the trained verdicts will be scored by
    from scipy.stats import spearmanr
    rho = np.nanmean([spearmanr(marg[i], lab[i]).correlation for i in np.nonzero(keep)[0]])
    print(f"{args.name}: events={len(recs)} disagreeing={int(keep.sum())}  AUC={auc:.3f}  favourite right={100*fav:.1f}%  mean Spearman(score, correct)={rho:+.3f}")


if __name__ == "__main__":
    main()
