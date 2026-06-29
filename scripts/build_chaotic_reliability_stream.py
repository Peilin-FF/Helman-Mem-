#!/usr/bin/env python3
"""Build the Chaotic Reliability Stream (time-varying peer reliability diagnostic).

Unlike the Balanced Single-Correct split (exactly-one-correct, prior-free), this
stream keeps EVERY example with at least one correct peer (y_G+y_P+y_Q >= 1) and
uses all 7 label patterns {100,010,001,110,101,011,111}. Peer reliability CHANGES
over time: the stream is a sequence of regimes; each regime starts with a CHAOTIC
phase (every window re-samples a fresh random target marginal vector) and then
settles into a STABLE phase (the regime's fixed target). Per (window, task type)
we sample target marginals [p_G,p_P,p_Q] from the template set below (randomly
permuted) and draw examples from the 7 label-pattern buckets so the window
approximately matches the targets — pattern probabilities are the independent-
Bernoulli likelihood prod p_i^y_i (1-p_i)^(1-y_i) renormalized over the 7 valid
patterns (000 excluded).

Source: data/v3/graded/*.jsonl, ruler peer_target_value>0.5 (same as the BSC split).
peer_0=Gemma, peer_1=Phi, peer_2=Qwen. Buckets with thin supply (code multi-correct
patterns have single-digit counts) are recycled via reshuffled cycles; reuse is
reported.

Each record carries: label_pattern, task_type, regime_id, window_id, phase,
window_target_marginals, window_realized_marginals, correctness_by_peer,
correct_peers. Eval counts a prediction correct iff the selected peer is one of
the correct peers (already the eval_joint_selector rule).

Output:
  data/v3_diagnostic/chaotic_reliability_stream.jsonl
  data/v3_diagnostic/chaotic_stream_report.json
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

from feedback_state.tasks import peer_target_value, task_type_of

PEERS = ["peer_0", "peer_1", "peer_2"]
PNAME = ["Gemma", "Phi", "Qwen"]
TASKS = ["math", "code", "rag"]
PATTERNS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
TEMPLATES = {
    "one_strong":     [0.80, 0.45, 0.35],
    "two_strong":     [0.75, 0.70, 0.35],
    "mild_bias":      [0.65, 0.55, 0.45],
    "near_equal":     [0.55, 0.53, 0.50],
    "low_confidence": [0.45, 0.42, 0.40],
}


def pattern_probs(p):
    """Pattern distribution over the 7 >=1-correct patterns whose CONDITIONED marginals
    match the target p. Conditioning on excluding 000 inflates every marginal by
    1/(1-prod(1-p)), so first solve (fixed point) for adjusted Bernoulli params p'
    with E[y_i | not 000] = p_i, then renormalize the independent likelihood."""
    adj = list(p)
    for _ in range(60):
        q0 = (1 - adj[0]) * (1 - adj[1]) * (1 - adj[2])
        new = [min(0.999, max(0.001, pi * (1 - q0))) for pi in p]
        if max(abs(a - b) for a, b in zip(adj, new)) < 1e-9:
            adj = new
            break
        adj = new
    w = []
    for y in PATTERNS:
        v = 1.0
        for pi, yi in zip(adj, y):
            v *= pi if yi else (1.0 - pi)
        w.append(v)
    z = sum(w)
    return [x / z for x in w]


class BucketCycle:
    """Shuffled no-replacement cycle over a bucket; reshuffles when exhausted."""

    def __init__(self, items, rng):
        self.items, self.rng, self.idx, self.epochs = list(items), rng, 0, 0
        self.rng.shuffle(self.items)

    def draw(self):
        if not self.items:
            return None
        if self.idx >= len(self.items):
            self.rng.shuffle(self.items)
            self.idx, self.epochs = 0, self.epochs + 1
        r = self.items[self.idx]
        self.idx += 1
        return r


def sample_target(rng):
    name = rng.choice(list(TEMPLATES))
    vals = list(TEMPLATES[name])
    rng.shuffle(vals)  # random permutation across the 3 peers
    return name, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", type=int, default=6)
    ap.add_argument("--windows_per_regime", type=int, default=6)
    ap.add_argument("--chaotic_windows", type=int, default=2,
                    help="leading windows of each regime with per-window random targets")
    ap.add_argument("--per_task_per_window", type=int, default=10,
                    help="examples per task type per window (window size = 3x this)")
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--outdir", default="data/v3_diagnostic")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # 1) pool: every graded example with >=1 correct peer, bucketed by (task, pattern)
    buckets = defaultdict(list)
    for f in sorted(glob.glob("data/v3/graded/*.jsonl")):
        ds = os.path.basename(f).replace(".jsonl", "")
        for line in open(f):
            r = json.loads(line)
            tt = task_type_of(r)
            if tt not in TASKS:
                continue
            y = tuple(1 if peer_target_value(r, k, str(r["peer_responses"][k])) > 0.5 else 0
                      for k in PEERS)
            if sum(y) < 1:
                continue
            r = dict(r)
            r["_dataset"] = ds
            r["label_pattern"] = "".join(map(str, y))
            r["correctness_by_peer"] = {PEERS[j]: y[j] for j in range(3)}
            r["correct_peers"] = [PNAME[j] for j in range(3) if y[j]]
            buckets[(tt, y)].append(r)
    cycles = {k: BucketCycle(v, rng) for k, v in buckets.items()}
    supply = {f"{tt}-{''.join(map(str, y))}": len(buckets[(tt, y)]) for tt, y in buckets}

    # 2) regimes -> windows -> per-task sampling
    stream, window_log = [], []
    wid = 0
    for reg in range(args.regimes):
        stable_target = {tt: sample_target(rng) for tt in TASKS}  # the regime's resting point
        for w in range(args.windows_per_regime):
            phase = "chaotic" if w < args.chaotic_windows else "stable"
            window, tgt_by_task = [], {}
            for tt in TASKS:
                if phase == "chaotic":
                    tname, p = sample_target(rng)  # fresh random target every window
                else:
                    tname, p = stable_target[tt]
                tgt_by_task[tt] = {"template": tname,
                                   "p": {PNAME[i]: round(p[i], 3) for i in range(3)}}
                q = pattern_probs(p)
                # largest-remainder apportionment of the k slots to the 7 patterns:
                # deterministic counts track the target marginals much tighter than
                # multinomial sampling at small per-window k
                k = args.per_task_per_window
                exact = [qi * k for qi in q]
                counts = [int(e) for e in exact]
                rem = k - sum(counts)
                for j in sorted(range(len(q)), key=lambda j: exact[j] - counts[j], reverse=True)[:rem]:
                    counts[j] += 1
                draw_patterns = [y for y, c in zip(PATTERNS, counts) for _ in range(c)]
                rng.shuffle(draw_patterns)
                for y in draw_patterns:
                    rec = cycles[(tt, y)].draw() if (tt, y) in cycles else None
                    if rec is None:  # bucket empty in the pool: resample a valid pattern
                        alt = [pp for pp in PATTERNS if (tt, pp) in cycles]
                        y = rng.choices(alt, weights=[q[PATTERNS.index(pp)] for pp in alt], k=1)[0]
                        rec = cycles[(tt, y)].draw()
                    window.append(dict(rec))
            rng.shuffle(window)  # mix tasks inside the window
            # realized empirical marginals (window-level and per task)
            def marg(rows):
                n = max(1, len(rows))
                return {PNAME[i]: round(sum(r["correctness_by_peer"][PEERS[i]] for r in rows) / n, 3)
                        for i in range(3)}
            realized = marg(window)
            realized_by_task = {tt: marg([r for r in window if task_type_of(r) == tt]) for tt in TASKS}
            for r in window:
                tt = task_type_of(r)
                r.update(regime_id=reg, window_id=wid, phase=phase,
                         window_target_marginals=tgt_by_task[tt]["p"],
                         window_target_template=tgt_by_task[tt]["template"],
                         window_realized_marginals=realized_by_task[tt])
            stream.extend(window)
            window_log.append({"window_id": wid, "regime_id": reg, "phase": phase,
                               "targets": tgt_by_task, "realized": realized,
                               "realized_by_task": realized_by_task, "size": len(window)})
            wid += 1

    # 3) report
    pat_dist = defaultdict(int)
    for r in stream:
        pat_dist[r["label_pattern"]] += 1
    reuse = {f"{tt}-{''.join(map(str, y))}": c.epochs for (tt, y), c in cycles.items() if c.epochs}
    n_unique = len({r["id"] for r in stream})
    report = {
        "seed": args.seed, "ruler": "peer_target_value>0.5",
        "peer_mapping": dict(zip(PEERS, PNAME)),
        "regimes": args.regimes, "windows_per_regime": args.windows_per_regime,
        "chaotic_windows_per_regime": args.chaotic_windows,
        "window_size": 3 * args.per_task_per_window, "total": len(stream),
        "unique_examples": n_unique,
        "pattern_distribution": dict(sorted(pat_dist.items())),
        "bucket_supply": supply, "bucket_reuse_epochs": reuse,
        "windows": window_log,
    }
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "chaotic_reliability_stream.jsonl")
    with open(out, "w") as fh:
        for r in stream:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(args.outdir, "chaotic_stream_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"stream: {len(stream)} examples ({n_unique} unique), "
          f"{args.regimes}x{args.windows_per_regime} windows of {3*args.per_task_per_window}")
    print(f"pattern distribution: {dict(sorted(pat_dist.items()))}")
    print(f"buckets recycled (epochs>0): {reuse}")
    mean_err = sum(abs(wl['realized'][p] - sum(wl['targets'][tt]['p'][p] for tt in TASKS) / 3)
                   for wl in window_log for p in PNAME) / (len(window_log) * 3)
    print(f"mean |realized - target| marginal error per window: {mean_err:.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
