"""Precompute generation prompts with the online memory state for a stream (decide-then-update).

The Kalman memory is run along the stream exactly as at test time (cold start,
written with the peers' verified labels after each event).  For every event the
prompt carries the memory's predictive probability and evidence count for each
peer answer as they were *before* that event's labels were seen.  Peer order in
the prompt is randomly permuted per event (the memory is identity-indexed).

  PYTHONPATH=. python scripts/build_generation_prompts.py --model q3_4b --stream train --order fixed \
      --out outputs/gen/q3_4b/prompts_train_fixed.jsonl [--proj-ckpt outputs/address/q3_4b/d64.pt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feedback_state.addresses import Projection
from feedback_state.feature_streams import load_stream, load_stream_from
from feedback_state.memory_generator import build_messages, prob_from_logit, target_text
from feedback_state.memory_runtime import MemoryRuntime
from feedback_state.permutations import random_order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q3_4b")
    ap.add_argument("--stream", default="train")
    ap.add_argument("--fit-stream", default="train")
    ap.add_argument("--order", default="fixed")
    ap.add_argument("--design", default="qc")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--proj-ckpt", type=Path, default=None)
    ap.add_argument("--permute", choices=["on", "off"], default="on")
    ap.add_argument("--target", choices=["peer", "final", "self", "onpolicy", "onpolicy_peer", "onpolicy_hinted"], default="peer",
                    help="self = the central model's own generation when it was graded correct (from --self-generations), else a correct peer's, else the gold answer")
    ap.add_argument("--self-generations", type=Path, default=None, help="generations.jsonl of a solo evaluation on this stream (for --target self / onpolicy)")
    ap.add_argument("--peer-generations", type=Path, default=None, help="generations.jsonl of a with-peers (memory mode) evaluation on this stream (for --target onpolicy)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    fit = load_stream(args.fit_stream, args.model)
    if args.proj_ckpt is not None:
        ck = torch.load(args.proj_ckpt, map_location="cpu", weights_only=False)
        proj_q, proj_c = Projection(state=ck["proj_q"], device=device), Projection(state=ck["proj_c"], device=device)
    else:
        proj_q = Projection(fit.sem, args.dim, device)
        proj_c = Projection(fit.peer_hidden.reshape(-1, fit.peer_hidden.shape[-1]), args.dim, device)
    fs = fit if args.stream == args.fit_stream else load_stream(args.stream, args.model)
    runtime = MemoryRuntime(design=args.design, proj_q=proj_q, proj_c=proj_c, num_peers=fs.num_peers, lam=args.lam, device=device)
    runtime.attach(fs)
    runtime.reset()
    N = len(fs)
    order = np.arange(N) if args.order == "fixed" else np.random.default_rng(int(args.order.replace("shuffled", ""))).permutation(N)
    if args.limit:
        order = order[: args.limit]
    rng = np.random.default_rng(args.seed)
    labels = fs.labels.numpy()
    own: dict = {}
    withpeers: dict = {}
    if args.target in ("onpolicy", "onpolicy_peer", "onpolicy_hinted"):
        if args.self_generations is None or args.peer_generations is None:
            raise ValueError("--target onpolicy needs --self-generations and --peer-generations")
        for path, table in ((args.self_generations, own), (args.peer_generations, withpeers)):
            for line in path.open():
                g = json.loads(line)
                if int(g.get("correct", 0)) == 1 and g.get("generation"):
                    table[str(g["id"])] = g["generation"]
        print(f"[gen-prompts] on-policy targets: with-peers correct {len(withpeers)}, solo correct {len(own)}", flush=True)
    if args.target == "self":
        if args.self_generations is None:
            raise ValueError("--target self needs --self-generations")
        for line in args.self_generations.open():
            g = json.loads(line)
            if int(g.get("correct", 0)) == 1 and g.get("generation"):
                own[str(g["id"])] = g["generation"]
        print(f"[gen-prompts] own correct generations available for {len(own)} events", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.out.open("w") as f:
        for pos, t in enumerate(order.tolist()):
            r = int(fs.real[t])
            if r < 1:
                continue
            y = labels[t, :r]
            ell, n_eff, _, X = runtime.read(t)
            perm = random_order(r, int(rng.integers(1 << 30))) if args.permute == "on" else list(range(r))
            texts = [fs.texts[t][p] for p in perm]
            probs = [prob_from_logit(float(ell[p])) for p in perm]
            evid = [float(n_eff[p]) for p in perm]
            rec = fs.records[t]
            rid = str(fs.ids[t]); source = None; target = None
            if args.target == "onpolicy_hinted":      # own solo first (zero drift), then the own-words hinted solution
                if rid in own: target, source = own[rid], "own_solo"
                elif rid in withpeers: target, source = withpeers[rid], "own_hinted"
            elif args.target in ("onpolicy", "onpolicy_peer"):
                if rid in withpeers: target, source = withpeers[rid], "own_with_peers"
                elif rid in own: target, source = own[rid], "own_solo"
                elif args.target == "onpolicy_peer":
                    correct_slots = [s for s in range(r) if int(y[perm[s]]) == 1]
                    if correct_slots:
                        best = max(correct_slots, key=lambda s: probs[s])   # most reliable verified-correct teacher
                        target = target_text(rec, [texts[best]], [1], mode="peer"); source = f"teacher_slot{best}"
            elif args.target == "self" and rid in own:
                target, source = own[rid], "own_solo"
            else:
                target, source = target_text(rec, texts, [int(y[p]) for p in perm], mode=args.target if args.target in ("peer", "final") else "peer"), "peer_or_final"
            row = {
                "pos": pos, "id": fs.ids[t], "task_type": fs.task[t], "source": fs.source[t], "target_source": source,
                "peer_order": [int(p) for p in perm], "peer_correct": [int(y[p]) for p in perm],
                "memory_prob": [round(p, 4) for p in probs], "memory_evidence": [round(e, 1) for e in evid],
                "messages_memory": build_messages(rec, texts, mode="memory", probs=probs, evidence=evid),
                "messages_peers": build_messages(rec, texts, mode="peers"),
                "messages_solo": build_messages(rec, texts, mode="solo"),
                "target": target,
            }
            f.write(json.dumps(row) + "\n")
            n_written += 1
            runtime.write(t, X, y)
    print(f"[gen-prompts] wrote {n_written} rows to {args.out}")
    import collections
    counts = collections.Counter(json.loads(l).get("target_source") for l in args.out.open())
    print(f"[gen-prompts] target sources: {dict(counts)}")


if __name__ == "__main__":
    main()
