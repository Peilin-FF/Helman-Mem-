"""Method 1: the record steers the decision, not the model (information-form evidence fusion).

The central model answers alone (no peers, no notes in its prompt).  Where the peers' answers differ and the
record is not flat, the memory chooses the answer with the largest summed log-odds of its supporters:

    score(a) = sum_{i: v_i = a} [ logit(p_i) + log(K - 1) ]        (Nitzan-Paroush; K = distinct answers on the table)

The model's own answer is one supporter of its group, with weight logit(q): q = its own running accuracy on the
task along the stream (the self-record, read-before-write), or a constant.  Elsewhere the model's answer stands.
Under a Beta posterior the expected weight is psi(a) - psi(b) with a = p n + 1, b = (1 - p) n + 1 (``--beta_weights``).
Output = the standard evaluator's format (generations.jsonl + eval_metrics.json), so the probe analysis and the
pages read it like any other run; each row records where the answer came from.

  PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 python scripts/fusion_decision.py --records data/ood/test.jsonl \
      --prompts outputs/gen/q3_4b/prompts_ood_shuffled0.jsonl --generations outputs/gen/q3_4b/frozen/eval_ood_shuffled0_solo_vllm/generations.jsonl \
      --pos_start 4000 --pos_count 1500 --output outputs/gen/q3_4b/steer/ood/fusion_self
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

from feedback_state.answer_groups import answer_groups
from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import strip_thinking
from feedback_state.memory_rl import peer_texts_in_prompt_order
from feedback_state.tasks import task_type_of
from tests.experiments.common.evaluate_memory_generator import write_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True, help="stream prompt rows (memory_prob, memory_evidence, peer_order per event)")
    p.add_argument("--generations", type=Path, required=True, help="the central model's alone generations over the stream")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pos_start", type=int, default=None)
    p.add_argument("--pos_count", type=int, default=1500)
    p.add_argument("--weight", choices=["self", "constant"], default="self", help="the model's own vote weight: logit of its self-record, or --w")
    p.add_argument("--w", type=float, default=0.0)
    p.add_argument("--no_k_term", action="store_true", help="drop the log(K-1) term (plain logit weights)")
    p.add_argument("--beta_weights", action="store_true", help="psi(a) - psi(b) weights instead of logit(p)")
    p.add_argument("--min_spread", type=float, default=0.1, help="the record is flat below this spread of estimates: keep the model's answer")
    p.add_argument("--fuse_code", action="store_true", help="also fuse on code events (agreement is not measurable there; default keeps the model's program)")
    p.add_argument("--swap_record", action="store_true", help="control: permute the record by rank before fusing")
    p.add_argument("--windows", type=int, default=10)
    return p.parse_args()


def digamma(x: float) -> float:
    r = 0.0
    while x < 6:
        r -= 1 / x
        x += 1
    f = 1 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))


def logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def swap_by_rank(probs, evid):
    ranked = sorted(range(len(probs)), key=lambda s: probs[s])
    new_p, new_e = list(probs), list(evid)
    for i, s in enumerate(ranked):
        new_p[s], new_e[s] = probs[ranked[-1 - i]], evid[ranked[-1 - i]]
    return new_p, new_e


def main() -> None:
    args = parse_args()
    t0 = time.time()
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    rows = {str(r["id"]): r for r in map(json.loads, args.prompts.open())}
    gens = sorted(map(json.loads, args.generations.open()), key=lambda g: int(g["pos"]))
    # self-record: the model's running accuracy per task along the whole stream, read before each event is written
    tally: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    self_q: dict[str, float] = {}
    for g in gens:
        t = g["task_type"]
        self_q[str(g["id"])] = (tally[t][1] + 1) / (tally[t][0] + 2)
        tally[t][0] += 1
        tally[t][1] += int(g["correct"])
    if args.pos_start is not None:
        gens = [g for g in gens if args.pos_start <= int(g["pos"]) < args.pos_start + args.pos_count]
    out_rows, texts_out, counts = [], [], collections.Counter()
    for g in gens:
        rec = records[str(g["id"])]
        row = dict(rows[str(g["id"])])
        task = task_type_of(rec)
        texts = peer_texts_in_prompt_order(rec, row["peer_order"])
        probs, evid = [float(p) for p in row["memory_prob"]], [float(e) for e in row.get("memory_evidence", [0.0] * len(texts))]
        if args.swap_record:
            probs, evid = swap_by_rank(probs, evid)
            row["memory_prob"], row["memory_evidence"] = probs, evid
        own = g["generation"]
        source = "model"
        chosen = own
        fuse = (max(probs) - min(probs) > args.min_spread) and (task != "code" or args.fuse_code)
        if fuse:
            groups = answer_groups(rec, texts + [strip_thinking(own)])
            peers, mine = groups[:-1], groups[-1]
            if len(set(peers)) > 1:
                K = len(set(peers) | {mine})
                bonus = 0.0 if args.no_k_term else math.log(K - 1)
                scores: dict[int, float] = collections.defaultdict(float)
                for s, gid in enumerate(peers):
                    w = (digamma(probs[s] * evid[s] + 1.0) - digamma((1 - probs[s]) * evid[s] + 1.0)) if args.beta_weights else logit(probs[s])
                    scores[gid] += w + bonus
                wm = logit(self_q[str(g["id"])]) if args.weight == "self" else args.w
                scores[mine] += wm + bonus
                best = max(scores, key=scores.get)
                if best != mine:
                    s = peers.index(best)
                    chosen, source = texts[s], f"peer{s + 1}"
        counts[source] += 1
        row["fused_from"] = source
        row["self_q"] = self_q[str(g["id"])]
        out_rows.append(row)
        texts_out.append(chosen)
    ns = SimpleNamespace(output=args.output, mode="alone+fusion", thinking="off", max_new_tokens=0, engine="fusion", checkpoint=None,
                         central_model=str(args.generations), prompts=args.prompts, windows=args.windows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_results(ns, out_rows, records, texts_out, t0)
    gpath = args.output / "generations.jsonl"
    lines = [json.loads(l) for l in gpath.open()]
    with gpath.open("w") as f:
        for g, r in zip(lines, out_rows):
            g["fused_from"], g["self_q"] = r["fused_from"], r["self_q"]
            f.write(json.dumps(g) + "\n")
    (args.output / "fusion_config.json").write_text(json.dumps({"weight": args.weight, "w": args.w, "k_term": not args.no_k_term, "beta_weights": args.beta_weights,
                                                                 "min_spread": args.min_spread, "fuse_code": args.fuse_code, "swap_record": args.swap_record, "sources": dict(counts)}, indent=1))
    print(f"[fusion] answers taken from: {dict(counts)}")


if __name__ == "__main__":
    main()
