"""Did the tilt make the *training signal* better, not just the test-time answer?

The hypothesis: by damping the unreliable peers' tokens, the tilt makes the good peers' solutions more
salient in the prompt, so the sampled answers are better and GRPO learns from better material.

If that is right, the tilt arm's reward should be higher than the control's at matched steps, and the
gap should be visible early rather than only after the model has adapted. Both runs use the same data,
the same order (seed 2) and the same schedule, so steps are comparable.

Confound to keep in mind and reported alongside: the tilt arm's rollouts are generated WITH the tilt,
so a reward gap mixes "the tilt makes better answers right now" with "the tilt makes learning better".
The slope, not the level, is what speaks to the second.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def load(root: Path, run: str) -> dict:
    out = {}
    p = root / "outputs/rl" / run / "metrics.jsonl"
    for line in open(p):
        d = json.loads(line)
        s = d.get("training/global_step", d.get("step"))
        if s is None:
            continue
        out[int(s)] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--a", default="run3b_tilt", help="the tilt arm")
    ap.add_argument("--b", default="ctrl3b_peers", help="the control")
    ap.add_argument("--key", default="critic/score/mean")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    A, B = load(root, args.a), load(root, args.b)
    common = sorted(set(A) & set(B))
    print(f"{args.a} vs {args.b}: {len(common)} matched steps, {common[0]}-{common[-1]}\n")

    def mean(d, ks, key):
        v = [d[k][key] for k in ks if key in d[k]]
        return st.mean(v) if v else float("nan")

    rows = []
    print(f"{'steps':<12}{'n':>5}{'tilt arm':>11}{'control':>10}{'diff':>9}")
    for lo, hi in [(1, 40), (41, 80), (81, 120), (121, 160), (161, 200), (201, 240), (241, 300)]:
        ks = [k for k in common if lo <= k <= hi]
        if not ks:
            continue
        a, b = mean(A, ks, args.key), mean(B, ks, args.key)
        rows.append({"lo": lo, "hi": hi, "n": len(ks), "a": a, "b": b, "diff": a - b})
        print(f"{f'{lo}-{hi}':<12}{len(ks):>5}{a:>11.3f}{b:>10.3f}{a-b:>+9.3f}")
    ks = common
    a, b = mean(A, ks, args.key), mean(B, ks, args.key)
    print(f"{'ALL':<12}{len(ks):>5}{a:>11.3f}{b:>10.3f}{a-b:>+9.3f}")

    # slope: does the gap shrink as the control catches up? a persistent gap that does not close is
    # the tilt's generation advantage; a gap that opens early and closes is a learning-speed effect
    if len(rows) >= 3:
        first, last = rows[0]["diff"], rows[-1]["diff"]
        print(f"\ngap in the first band {100*first:+.1f} pts, in the last {100*last:+.1f} pts "
              f"({'closing' if abs(last) < abs(first) else 'persistent or widening'})")

    for key in ("actor/entropy_loss", "response_length/mean", "actor/kl_loss"):
        a, b = mean(A, common, key), mean(B, common, key)
        if a == a and b == b:
            print(f"{key:<26} tilt arm {a:.3f}   control {b:.3f}   diff {a-b:+.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"a": args.a, "b": args.b, "key": args.key, "bands": rows}, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
