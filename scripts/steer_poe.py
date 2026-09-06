"""Steering at the decision, not in the prompt: memory-guided reranking of the final answer.

The central model reads the peers (any prompt) and writes its reasoning; at the point where it commits
("Final answer:") the candidates are the distinct answers on the table (each peer's, plus the model's own),
and the choice is made by

    score(a) = log p_model(a | prompt, own reasoning) / len(a)  +  lambda * sum_{peers i giving a} logit(reliability_i)  +  beta * [a is the model's own answer]

so the record enters as a prior that cannot be ignored, weighted by lambda; lambda = 0 is the model's own
preference among the candidates, lambda -> inf is the reliability-weighted vote.  One vLLM scoring pass
(prompt log-probs of each candidate continuation) yields the whole (lambda, beta) grid offline.

  PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 python scripts/steer_poe.py --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B \
      --records data/ood/test.jsonl --prompts outputs/gen/q3_4b/prompts_ood_shuffled0.jsonl --pos_start 4000 --pos_count 1500 \
      --generations outputs/gen/q3_4b/frozen/eval_ood_shuffled0_memory_vllm/generations.jsonl --output outputs/gen/q3_4b/steer/ood/poe_number
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
from pathlib import Path

from feedback_state.answer_groups import answer_groups
from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import grade, render_prompt, strip_thinking
from feedback_state.memory_rl import peer_texts_in_prompt_order
from feedback_state.tasks import task_type_of
from scripts.steer_prompts import final_of

MARK = "Final answer:"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--central_model", required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True, help="prompt rows whose messages_memory the generations answered")
    p.add_argument("--generations", type=Path, required=True, help="generations.jsonl of the model on those prompts (its reasoning is reused)")
    p.add_argument("--mode", default="memory", help="which messages_<mode> the generations answered")
    p.add_argument("--pos_start", type=int, default=None)
    p.add_argument("--pos_count", type=int, default=1500)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192)
    return p.parse_args()


def surface(task: str, text: str) -> str | None:
    a = final_of(task, text)
    if not a:
        return None
    a = a.strip().strip(".")
    if task == "mcqa":
        m = re.search(r"\(([A-Za-z0-9])\)", a)
        return f"({m.group(1).upper()})" if m else (f"({a[0].upper()})" if len(a) == 1 else a)
    if task == "boolqa":
        return a.lower()
    return a[:80]


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    rows = {str(r["id"]): r for r in map(json.loads, args.prompts.open())}
    gens = [g for g in map(json.loads, args.generations.open()) if args.pos_start is None or args.pos_start <= int(g["pos"]) < args.pos_start + args.pos_count]
    tok = AutoTokenizer.from_pretrained(args.central_model, local_files_only=True)
    events, requests = [], []   # requests: (event index, candidate index, token ids, span start)
    for g in gens:
        rec = records[str(g["id"])]; row = rows[str(g["id"])]; task = task_type_of(rec)
        if task == "code":
            continue
        texts = peer_texts_in_prompt_order(rec, row["peer_order"])
        own = strip_thinking(g["generation"])
        cand_texts = texts + [own]
        groups = answer_groups(rec, cand_texts)
        cands: dict[int, dict] = {}
        for i, t in enumerate(cand_texts):
            s = surface(task, t)
            if s is None:
                continue
            c = cands.setdefault(groups[i], {"surface": s, "peers": [], "own": False})
            if i < len(texts):
                c["peers"].append(i)
            else:
                c["own"] = True
        if not cands:
            continue
        prompt = render_prompt(tok, row[f"messages_{args.mode}"], thinking=False)
        if MARK in own:
            prefix = prompt + own[: own.rfind(MARK) + len(MARK)]
        else:
            prefix = prompt + own.rstrip() + "\n" + MARK
        pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        ev = {"id": g["id"], "pos": g["pos"], "task_type": task, "peer_correct": row["peer_correct"], "memory_prob": row["memory_prob"], "own_correct": int(g["correct"]),
              "candidates": []}
        for gi, c in cands.items():
            full_ids = tok(prefix + " " + c["surface"], add_special_tokens=False)["input_ids"]
            n = 0
            while n < min(len(pre_ids), len(full_ids)) and pre_ids[n] == full_ids[n]:
                n += 1
            if n >= len(full_ids):
                continue
            c["correct"] = int(grade(rec, MARK + " " + c["surface"]))
            c["n_tokens"] = len(full_ids) - n
            ev["candidates"].append(c)
            requests.append((len(events), len(ev["candidates"]) - 1, full_ids, n))
        if ev["candidates"]:
            events.append(ev)
    print(f"[poe] {len(events)} events, {len(requests)} candidate scorings", flush=True)
    llm = LLM(model=args.central_model, tokenizer=args.central_model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enable_prefix_caching=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outs = llm.generate([{"prompt_token_ids": ids[: args.max_model_len - 1]} for _, _, ids, _ in requests], params, use_tqdm=True)
    for (ei, ci, ids, n), o in zip(requests, outs):
        lp = o.prompt_logprobs
        total = 0.0; cnt = 0
        for pos in range(n, min(len(ids), len(lp))):
            d = lp[pos]
            if d is None:
                continue
            entry = d.get(ids[pos])
            if entry is not None:
                total += float(entry.logprob); cnt += 1
        events[ei]["candidates"][ci]["logp_sum"] = total
        events[ei]["candidates"][ci]["logp_mean"] = total / max(1, cnt)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "poe_scores.jsonl").open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    summary = sweep(events)
    (args.output / "poe_summary.json").write_text(json.dumps(summary, indent=1))
    best = max(summary["grid"], key=lambda r: r["accuracy"])
    print(f"[poe] own answer {summary['own_accuracy']:.2f} | model prior only {summary['grid'][0]['accuracy']:.2f} | best lambda={best['lambda']} beta={best['beta']}: {best['accuracy']:.2f} "
          f"(selection events {best['selection_accuracy']:.1f} vs favourite {summary['favourite_selection_accuracy']:.1f}, own {summary['own_selection_accuracy']:.1f})")


def sweep(events: list[dict]) -> dict:
    def logit(p): return math.log(max(p, 1e-3) / max(1 - p, 1e-3))
    def informative(ev):
        pc = ev["peer_correct"]; pr = ev["memory_prob"]
        groups = {tuple(c["peers"]) for c in ev["candidates"] if c["peers"]}
        return len(groups) > 1 and (max(pr) - min(pr) > 0.1)
    sel = [ev for ev in events if informative(ev) and any(c["correct"] for c in ev["candidates"] if c["peers"])]
    def fav_correct(ev):
        top = max(range(len(ev["memory_prob"])), key=lambda s: ev["memory_prob"][s])
        return int(ev["peer_correct"][top])
    grid = []
    for gate in (False, True):
      for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e6):
        for beta in (0.0, 0.5, 1.0, 2.0):
            def pick(ev):
                if gate and not informative(ev):
                    return {"correct": ev["own_correct"]}
                best, bs = None, -1e18
                for c in ev["candidates"]:
                    s = c["logp_mean"] + lam * sum(logit(ev["memory_prob"][i]) for i in c["peers"]) + beta * (1.0 if c["own"] else 0.0)
                    if s > bs:
                        best, bs = c, s
                return best
            acc = sum(pick(ev)["correct"] for ev in events) / len(events)
            sacc = sum(pick(ev)["correct"] for ev in sel) / max(1, len(sel))
            grid.append({"gate": gate, "lambda": lam if lam < 1e5 else "inf", "beta": beta, "accuracy": 100 * acc, "selection_accuracy": 100 * sacc})
    return {"n_events": len(events), "n_selection": len(sel), "own_accuracy": 100 * sum(ev["own_correct"] for ev in events) / len(events),
            "own_selection_accuracy": 100 * sum(ev["own_correct"] for ev in sel) / max(1, len(sel)),
            "favourite_selection_accuracy": 100 * sum(fav_correct(ev) for ev in sel) / max(1, len(sel)),
            "by_task": dict(collections.Counter(ev["task_type"] for ev in events)), "grid": grid}


if __name__ == "__main__":
    main()
