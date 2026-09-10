"""Did training under the tilt teach the central model to use the tilt better?

The aggregate comparison (ours vs the no-memory control, deployed) is a wash, but it mixes together
events where the memory said something with events where it said nothing.  This separates them.

The tilt is exactly zero when the record is flat (spread = max_j p_j - min_j p_j <= 0.1), so on those
events every model sees a bit-identical prompt and computation: any accuracy difference there is the
models' general training difference, not the tilt.  On events where the tilt is active the difference
is that same baseline plus whatever training under the tilt bought.  So the quantity to look at is

    lift(model)  =  acc(peers + tilt)  -  acc(peers, no tilt)      on the same events

which is what each model converts the same record into, and then whether ours converts more than the
control does, bin by bin in the record's spread.  The partition uses only the record, never a label.

A second view splits by whether the record's favourite peer was actually right.  That one does use
labels, so it is a diagnostic of how much each model *leans* on the memory, not a headline number:
a model that follows the memory harder gains more when the record is right and loses more when wrong.

Reads the stored generations.jsonl of the four models in the tilt and peers conditions; no GPU.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MODELS = [
    ("frozen",        "outputs/gen/q3_4b/base6_nothink/{s}6_{c}"),
    ("ours",          "outputs/rl/run3b_tilt/hf/global_step_276/eval_{s}6_{c}"),
    ("control",       "outputs/rl/ctrl3b_peers/hf/global_step_276/eval_{s}6_{c}"),
    ("question-only", "outputs/rl/solo3b_q/hf/global_step_276/eval_{s}6_{c}"),
]


def load(root: Path, tmpl: str, stream: str, cond: str) -> dict:
    p = root / tmpl.format(s=stream, c=cond) / "generations.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[r["id"]] = r
    return out


def spread(mp) -> float:
    return (max(mp) - min(mp)) if mp else 0.0


def _se(xs) -> float:
    """Standard error of the mean of a paired per-event difference."""
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def wilson_se(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    p = k / n
    return math.sqrt(max(p * (1 - p), 1e-12) / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--stream", default="oodfull", choices=["oodfull", "indist"])
    ap.add_argument("--flat", type=float, default=0.1, help="spread at or below which the tilt is identically zero")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)

    tilt = {n: load(root, t, args.stream, "tilt") for n, t in MODELS}
    peers = {n: load(root, t, args.stream, "peers") for n, t in MODELS}
    have = [n for n, _ in MODELS if tilt.get(n) and peers.get(n)]
    if not have:
        raise SystemExit(f"no generations found for stream {args.stream}")
    ids = set.intersection(*[set(tilt[n]) & set(peers[n]) for n in have])
    print(f"stream={args.stream}  models={have}  events matched in both conditions: {len(ids)}\n", flush=True)

    ref = tilt[have[0]]
    bins = [("flat (tilt = 0)", lambda s: s <= args.flat),
            ("spread 0.10-0.20", lambda s: args.flat < s <= 0.20),
            ("spread 0.20-0.30", lambda s: 0.20 < s <= 0.30),
            ("spread > 0.30", lambda s: s > 0.30)]
    groups = {name: [i for i in ids if pred(spread(ref[i]["memory_prob"]))] for name, pred in bins}
    groups["ALL"] = sorted(ids)
    groups["tilt active (spread > %.2f)" % args.flat] = [i for i in ids if spread(ref[i]["memory_prob"]) > args.flat]

    report = {"stream": args.stream, "n_matched": len(ids), "bins": {}}
    order = ["flat (tilt = 0)", "spread 0.10-0.20", "spread 0.20-0.30", "spread > 0.30",
             "tilt active (spread > %.2f)" % args.flat, "ALL"]
    print(f"{'bin':<28} {'n':>6}  " + "  ".join(f"{n:>22}" for n in have))
    print(f"{'':<28} {'':>6}  " + "  ".join(f"{'tilt  peers   lift':>22}" for _ in have))
    for name in order:
        g = groups[name]
        if not g:
            continue
        cells, rec = [], {}
        for n in have:
            at = sum(tilt[n][i]["correct"] for i in g) / len(g)
            ap_ = sum(peers[n][i]["correct"] for i in g) / len(g)
            lift = at - ap_
            # paired: the two conditions are the SAME events, so the error is on the per-event
            # difference, not the sum of two independent binomial errors (which is far too wide)
            d = [tilt[n][i]["correct"] - peers[n][i]["correct"] for i in g]
            se = _se(d)
            cells.append(f"{100*at:5.1f} {100*ap_:6.1f} {100*lift:+6.2f}")
            rec[n] = {"acc_tilt": at, "acc_peers": ap_, "lift": lift, "lift_se": se, "n": len(g)}
        report["bins"][name] = rec
        print(f"{name:<28} {len(g):>6}  " + "  ".join(f"{c:>22}" for c in cells))

    if "ours" in have and "control" in have:
        print("\nours minus control, per bin (positive = training under the tilt converts the same record into more):")
        for name in order:
            r = report["bins"].get(name)
            if not r:
                continue
            # paired difference-in-differences, event by event
            e = [(tilt["ours"][i]["correct"] - peers["ours"][i]["correct"])
                 - (tilt["control"][i]["correct"] - peers["control"][i]["correct"]) for i in groups[name]]
            d, se = (sum(e) / len(e)), _se(e)
            flag = "  <-- " + ("above noise" if abs(d) > 2 * se else "within noise")
            print(f"   {name:<28} delta lift = {100*d:+6.2f}  (se {100*se:4.2f}){flag}")

    print("\ndiagnostic: does each model lean on the record? accuracy under the tilt, split by whether")
    print("the record's favourite peer was in fact right (labels used, so this is a lean measure only):")
    act = groups["tilt active (spread > %.2f)" % args.flat]
    print(f"{'model':<16} {'fav right: n':>13} {'acc':>7}   {'fav wrong: n':>13} {'acc':>7}   {'gap':>7}")
    for n in have:
        right = [i for i in act if tilt[n][i]["peer_correct"][max(range(len(tilt[n][i]["memory_prob"])), key=lambda j: tilt[n][i]["memory_prob"][j])] == 1]
        wrong = [i for i in act if i not in set(right)]
        ar = sum(tilt[n][i]["correct"] for i in right) / max(len(right), 1)
        aw = sum(tilt[n][i]["correct"] for i in wrong) / max(len(wrong), 1)
        report.setdefault("lean", {})[n] = {"n_fav_right": len(right), "acc_fav_right": ar,
                                            "n_fav_wrong": len(wrong), "acc_fav_wrong": aw, "gap": ar - aw}
        print(f"{n:<16} {len(right):>13} {100*ar:6.1f}   {len(wrong):>13} {100*aw:6.1f}   {100*(ar-aw):+6.1f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
