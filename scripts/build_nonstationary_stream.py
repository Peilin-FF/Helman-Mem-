#!/usr/bin/env python3
"""Build the Non-Stationary Reliability Stream (final spec).

Three fixed peers (Gemma=peer_0, Phi=peer_1, Qwen-Coder=peer_2), three task types
(math, code, rag). Keep every example with >=1 correct peer (all 7 label patterns
100/010/001/110/101/011/111 — NOT the exactly-one BSC rule).

The stream is a sequence of regimes. For each (regime, task type) a target marginal
reliability vector [p_G,p_P,p_Q] is sampled from the templates below with peer
identities randomly permuted. Each regime has two phases:
  1. TRANSITION: window targets interpolate linearly from the previous regime's
     vector to the new vector, plus Gaussian noise (sigma=0.06, clipped).
  2. STABLE: window targets stay fixed at the new vector.
Windows match their target marginals via (a) a fixed-point correction for the
>=1-correct conditioning (otherwise every marginal inflates ~+6pt) and (b)
largest-remainder apportionment of window slots to the 7 patterns.

Run with --seed S for each of the 3 seeds; every method must then be evaluated on
the SAME file (same example order). Output:
  data/chaotic_data/nonstat_stream_seed{S}.jsonl
  data/chaotic_data/nonstat_stream_seed{S}_report.json

Each record carries: stream_seed, regime_id, window_id, phase (transition|stable),
label_pattern, task_type, window_target_marginals (the generative truth for its
window+task), window_realized_marginals, correctness_by_peer, correct_peers.
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
NOISE_SIGMA = 0.06


def pattern_probs(p):
    """Distribution over the 7 >=1-correct patterns whose CONDITIONED marginals match
    p: fixed-point for adjusted Bernoulli params, then renormalized likelihood."""
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


def apportion(q, k):
    """Largest-remainder apportionment of k slots to pattern probabilities q."""
    exact = [qi * k for qi in q]
    counts = [int(e) for e in exact]
    for j in sorted(range(len(q)), key=lambda j: exact[j] - counts[j],
                    reverse=True)[: k - sum(counts)]:
        counts[j] += 1
    return counts


class BucketCycle:
    """Shuffled no-replacement cycle; reshuffles when exhausted (reuse counted)."""

    def __init__(self, items, rng):
        self.items, self.rng, self.idx, self.epochs = list(items), rng, 0, 0
        self.rng.shuffle(self.items)

    def draw(self):
        if self.idx >= len(self.items):
            self.rng.shuffle(self.items)
            self.idx, self.epochs = 0, self.epochs + 1
        r = self.items[self.idx]
        self.idx += 1
        return r


def sample_target(rng):
    name = rng.choice(list(TEMPLATES))
    vals = list(TEMPLATES[name])
    rng.shuffle(vals)
    return name, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", type=int, default=6)
    ap.add_argument("--windows_per_regime", type=int, default=6)
    ap.add_argument("--transition_windows", type=int, default=2)
    ap.add_argument("--per_task_per_window", type=int, default=10)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", default="data/chaotic_data")
    args = ap.parse_args()
    rng = random.Random(args.seed)

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

    stream, window_log = [], []
    wid = 0
    prev_target = {tt: [0.5, 0.5, 0.5] for tt in TASKS}  # neutral start before regime 0
    T = args.transition_windows
    for reg in range(args.regimes):
        new_target = {}
        for tt in TASKS:
            tname, p = sample_target(rng)
            new_target[tt] = (tname, p)
        for w in range(args.windows_per_regime):
            phase = "transition" if w < T else "stable"
            window, tgt_by_task = [], {}
            for tt in TASKS:
                tname, p_new = new_target[tt]
                if phase == "transition":
                    # interpolate prev -> new, plus Gaussian noise
                    lam = (w + 1) / (T + 1)
                    p = [min(0.95, max(0.05,
                         (1 - lam) * prev_target[tt][i] + lam * p_new[i]
                         + rng.gauss(0.0, NOISE_SIGMA))) for i in range(3)]
                else:
                    p = list(p_new)
                tgt_by_task[tt] = {"template": tname,
                                   "p": {PNAME[i]: round(p[i], 3) for i in range(3)}}
                counts = apportion(pattern_probs(p), args.per_task_per_window)
                draws = [y for y, c in zip(PATTERNS, counts) for _ in range(c)]
                rng.shuffle(draws)
                for y in draws:
                    window.append(dict(cycles[(tt, y)].draw()))
            rng.shuffle(window)

            def marg(rows):
                n = max(1, len(rows))
                return {PNAME[i]: round(sum(r["correctness_by_peer"][PEERS[i]] for r in rows) / n, 3)
                        for i in range(3)}
            realized_by_task = {tt: marg([r for r in window if task_type_of(r) == tt])
                                for tt in TASKS}
            for r in window:
                tt = task_type_of(r)
                r.update(stream_seed=args.seed, regime_id=reg, window_id=wid, phase=phase,
                         window_target_marginals=tgt_by_task[tt]["p"],
                         window_target_template=tgt_by_task[tt]["template"],
                         window_realized_marginals=realized_by_task[tt])
            stream.extend(window)
            window_log.append({"window_id": wid, "regime_id": reg, "phase": phase,
                               "targets": tgt_by_task, "realized_by_task": realized_by_task,
                               "size": len(window)})
            wid += 1
        prev_target = {tt: list(new_target[tt][1]) for tt in TASKS}

    pat_dist = defaultdict(int)
    for r in stream:
        pat_dist[r["label_pattern"]] += 1
    reuse = {f"{tt}-{''.join(map(str, y))}": c.epochs for (tt, y), c in cycles.items() if c.epochs}
    err = [abs(wl["realized_by_task"][tt][p] - wl["targets"][tt]["p"][p])
           for wl in window_log for tt in TASKS for p in PNAME]
    report = {"seed": args.seed, "ruler": "peer_target_value>0.5",
              "peer_mapping": dict(zip(PEERS, PNAME)),
              "regimes": args.regimes, "windows_per_regime": args.windows_per_regime,
              "transition_windows": T, "noise_sigma": NOISE_SIGMA,
              "window_size": 3 * args.per_task_per_window, "total": len(stream),
              "unique_examples": len({r["id"] for r in stream}),
              "pattern_distribution": dict(sorted(pat_dist.items())),
              "bucket_reuse_epochs": reuse,
              "mean_abs_marginal_error": sum(err) / len(err),
              "windows": window_log}
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"nonstat_stream_seed{args.seed}.jsonl")
    with open(out, "w") as fh:
        for r in stream:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(args.outdir, f"nonstat_stream_seed{args.seed}_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"seed {args.seed}: {len(stream)} examples, "
          f"mean |realized-target| = {report['mean_abs_marginal_error']:.3f}, "
          f"patterns {dict(sorted(pat_dist.items()))}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
