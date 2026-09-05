"""The central model as one more voter: memory-weighted vote over {own answer, peer answers}.

Uses the frozen model's solo generations on a stream (correctness per event, final answer text),
the peers' answers and labels, and the Kalman memory on the question address with one head per
source (3 peers + self).  Decision per event (decide-then-update, cold start): Nitzan-Paroush
weighted majority over answer groups with weights logit P(source correct | history).  No
fine-tuning, so the model's own reasoning is untouched; peers only override it when the memory
says they are more reliable on questions like this one and they agree.

  PYTHONPATH=. python scripts/self_vote_sim.py --model q3_4b --stream indist --order shuffled0 \
      --prompts outputs/gen/q3_4b/prompts_indist_shuffled0.jsonl --solo outputs/gen/q3_4b/frozen/eval_indist_shuffled0_solo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feedback_state.addresses import Projection
from feedback_state.feature_streams import load_stream
from feedback_state.kalman_memory import KalmanMemory
from feedback_state.tasks import _normalize_qa, qa_extract_answer
from feedback_state.utils import extract_final_answer, math_equal


def final_of(task: str, text: str) -> str:
    if task == "math":
        return extract_final_answer(text)
    if task == "rag":
        return _normalize_qa(qa_extract_answer(text))
    return ""


def groups_of(task: str, finals: list[str]) -> list[int]:
    ids = [-1] * len(finals); nxt = 0
    for i in range(len(finals)):
        if ids[i] >= 0:
            continue
        ids[i] = nxt
        if finals[i]:
            for j in range(i + 1, len(finals)):
                if ids[j] < 0 and finals[j] and (math_equal(finals[i], finals[j]) if task == "math" else finals[i] == finals[j]):
                    ids[j] = nxt
        nxt += 1
    return ids


def curve(h, w=10):
    h = np.array(h); n = len(h); e = np.linspace(0, n, w + 1).astype(int)
    return f"{100*h.mean():6.2f}  {100*h[:n//2].mean():5.1f}->{100*h[n//2:].mean():5.1f} | " + " ".join(f"{100*h[a:b].mean():4.1f}" for a, b in zip(e[:-1], e[1:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q3_4b")
    ap.add_argument("--stream", default="indist")
    ap.add_argument("--order", default="shuffled0")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--solo", type=Path, required=True, help="directory with the solo generations.jsonl")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    fit = load_stream("train", args.model)
    proj_q = Projection(fit.sem, args.dim, device)
    fs = load_stream(args.stream, args.model)
    psi = proj_q(fs.sem).to(torch.float64)
    index = {str(i): k for k, i in enumerate(fs.ids)}
    solo = {str(g["id"]): g for g in (json.loads(l) for l in (args.solo / "generations.jsonl").open())}
    rows = [json.loads(l) for l in args.prompts.open()]
    P = fs.num_peers
    mem = KalmanMemory(args.dim + 1, P + 1, lam=args.lam, device=device)
    hits = {k: [] for k in ("solo", "memory_select", "vote_self", "vote_self_no_mem", "oracle_all", "oracle_peers")}
    by_task = {}
    for r in rows:
        rid = str(r["id"]); t = index[rid]; task = fs.task[t]
        g = solo.get(rid)
        if g is None:
            continue
        keys = sorted(fs.records[t].get("peer_responses", {}))
        peer_texts = [str(fs.records[t]["peer_responses"][k]) for k in keys]
        peer_y = [int(v) for v in fs.labels[t, : len(keys)].tolist()]
        own_y = int(g["correct"])
        x = torch.cat([psi[t], torch.ones(1, dtype=torch.float64, device=psi.device)]).unsqueeze(0)
        mu, var = mem.read(x)
        p = mem.prob(mu[0], var.expand(P + 1))
        ell = mem.logit(p).cpu().numpy()          # heads: peers 0..P-1, self = P
        finals = [final_of(task, tx) for tx in peer_texts] + [final_of(task, g["generation"])]
        y_all = peer_y + [own_y]
        grp = groups_of(task, finals) if task in ("math", "rag") else list(range(P + 1))
        def vote(weights):
            tot = {}
            for i, gi in enumerate(grp):
                tot[gi] = tot.get(gi, 0.0) + float(weights[i])
            best = max(tot.values())
            members = [i for i, gi in enumerate(grp) if tot[gi] == best]
            return max(members, key=lambda i: weights[i])
        sel_vote = vote(ell)
        sel_vote0 = vote(np.array([1.0] * P + [1.0 + 1e-6]))   # plain majority, ties to self
        sel_mem = int(np.argmax(ell[:P]))
        hits["solo"].append(own_y)
        hits["memory_select"].append(peer_y[sel_mem])
        hits["vote_self"].append(y_all[sel_vote])
        hits["vote_self_no_mem"].append(y_all[sel_vote0])
        hits["oracle_all"].append(max(y_all))
        hits["oracle_peers"].append(max(peer_y))
        by_task.setdefault(task, {k: [] for k in hits})
        for k in hits:
            by_task[task][k].append(hits[k][-1])
        mem.write(x[0], torch.tensor([1.0 if v else -1.0 for v in y_all], dtype=torch.float64))
    print(f"[self-vote] {args.model} {args.stream} {args.order} events={len(hits['solo'])}")
    for k, v in hits.items():
        print(f"  {k:18s} {curve(v)}")
    for task, d in by_task.items():
        print(f"  -- {task}: " + "  ".join(f"{k}={100*np.mean(v):.1f}" for k, v in d.items()))


if __name__ == "__main__":
    main()
