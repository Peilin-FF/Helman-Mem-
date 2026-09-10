"""Summarise scripts/attention_mass.py outputs into one table per stream (see that file for the metrics)."""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def nanmean(xs) -> float:
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def metrics(res: list[dict], gamma: str, q: str, layer_sel, density: bool = False) -> dict:
    """density=True divides each block's mass by its token count, removing the block-length confound."""
    cr, cc, fav, cshare, pshare, ent, n = [], [], [], [], [], [], 0
    for r in res:
        m = np.asarray(r["mass"][gamma][q])            # [layers, peers]
        tot = np.asarray(r["mass"][gamma]["total_" + q])
        m, tot = m[layer_sel], tot[layer_sel]
        peer = m.mean(0)                                # mean over the chosen layers -> [peers]
        if density:
            lens = np.asarray([max(e - s, 1) for s, e in r["spans"]], float)
            peer = peer / lens
        if peer.sum() <= 0:
            continue
        probs, correct = np.asarray(r["probs"]), np.asarray(r["correct"], float)
        share = peer / peer.sum()
        cr.append(spearman(peer, probs)); cc.append(spearman(peer, correct))
        fav.append(share[int(np.argmax(probs))] * len(probs))
        cshare.append((share * correct).sum() / max(correct.mean(), 1e-9))
        pshare.append(float(peer.sum() / max(tot.mean(), 1e-9)))
        p = share[share > 0]
        ent.append(float(-(p * np.log(p)).sum() / math.log(len(share))))
        n += 1
    return {"n": n, "corr_record": nanmean(cr), "corr_correct": nanmean(cc), "favourite": nanmean(fav),
            "correct_share": nanmean(cshare), "peer_share": nanmean(pshare), "entropy": nanmean(ent),
            "se_corr_record": float(np.nanstd(cr) / math.sqrt(max(len(cr), 1))),
            "se_corr_correct": float(np.nanstd(cc) / math.sqrt(max(len(cc), 1)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/address/attention_mass")
    ap.add_argument("--order", default="frozen,ours,control,question-only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(str(Path(args.dir) / "*_*.json")))
    data = {}
    for f in files:
        d = json.load(open(f))
        data[(d["model"], d["stream"])] = d
    order = [m for m in args.order.split(",")]
    streams = sorted({s for _, s in data})
    summary = {}
    for stream in streams:
        for q, qname in (("last", "query = last prompt token"), ("post", "query = tokens after the peers")):
            for lname, lsel, dens in (("all layers", slice(None), False), ("late third", None, False),
                                      ("all layers, PER-TOKEN density", slice(None), True)):
                print(f"\n=== {stream} | {qname} | {lname} ===")
                print(f"{'model':<14}{'gamma':>6}{'n':>5}{'corr(record)':>15}{'corr(correct)':>15}{'favourite':>11}{'correct share':>15}{'peer share':>12}{'entropy':>9}")
                for m in order:
                    d = data.get((m, stream))
                    if not d:
                        continue
                    nl = d["n_layers"]
                    sel = lsel if lsel is not None else slice(2 * nl // 3, nl)
                    for g in d["results"][0]["mass"].keys() if d["results"] else []:
                        r = metrics(d["results"], g, q, sel, density=dens)
                        summary[f"{stream}|{q}|{lname}|{m}|gamma{g}"] = r
                        print(f"{m:<14}{float(g):>6.0f}{r['n']:>5}{r['corr_record']:>+10.3f} ±{r['se_corr_record']:.3f}"
                              f"{r['corr_correct']:>+10.3f} ±{r['se_corr_correct']:.3f}{r['favourite']:>11.2f}{r['correct_share']:>15.2f}"
                              f"{r['peer_share']:>12.2f}{r['entropy']:>9.2f}")
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
