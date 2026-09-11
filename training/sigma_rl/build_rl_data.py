"""Memory-annotated prompt stream  ->  verl parquet, one file per regime.

  PYTHONPATH=. python -m training.sigma_rl.build_rl_data \
      --prompts outputs/gen/q3_4b/prompts_train_fixed.jsonl --records data/mixed_train_big/train.jsonl \
      --out outputs/rl/data/q3_4b/train_memory.parquet --guided memory

``--guided`` decides what the central model sees next to the question when it answers at stream time
(its answer is graded only afterwards):

  memory       every peer solution, each annotated with the memory's reliability estimate       ours (no labels before the answer)
  peers        every peer solution, no annotation                                                 labels-after ablation, no memory
  hint_memory  one solution: the peer the memory trusts most                                      ours, single hint (no labels)
  hint_label   one solution: the most reliable *verified-correct* peer                            labels BEFORE the answer = classical baseline
  hint_random  one solution: a random verified-correct peer                                       labels before, no memory
  none         the question only                                                                  plain RLVR

Rows are written in stream order: the memory state of event t was computed from the feedback of
events < t only, so training must also visit them in this order (data.shuffle: False).

Row layout (verl conventions):
  data_source   task type (math / rag / code / ...) -> per-task validation metrics
  prompt        question-only chat messages (the policy's own prompt)
  guided_prompt the guided chat messages above, or null
  reward_model  {"style": "rule", "ground_truth": gold answer}
  extra_info    memory state at this stream position (memory_prob / memory_evidence per slot), peer_correct,
                peer_order, verified flag, guided kind / slot / correctness, the protocol tag (peer_outcome_v1 =
                labels only after the answer; labels_before_v1 = the classical baseline), and the full record (JSON)
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import random
from pathlib import Path

import pandas as pd

from feedback_state.attn_bias import peer_char_spans
from feedback_state.verdict import add_verdict_instruction
from feedback_state.evidence import add_evidence_block
from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import build_messages, domain_note
from feedback_state.memory_rl import choose_hint_slot, hint_messages, labeled_peer_messages, peer_texts_in_prompt_order
from training.sigma_rl.outcome_protocol import PROTOCOL   # "peer_outcome_v1": the answer is graded only after it is given

LABELS_BEFORE_PROTOCOL = "labels_before_v1"
GUIDED = ("memory", "peers", "hint_memory", "hint_label", "hint_random", "peers_labeled", "none")
LABELS_BEFORE = {"hint_label", "hint_random", "peers_labeled"}
HINT_CHOICE = {"hint_memory": "memory", "hint_label": "label_memory", "hint_random": "label_random"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True, help="prompt stream from scripts/build_generation_prompts.py")
    ap.add_argument("--records", type=Path, required=True, help="the stream JSONL (records with peer_responses, answers, tests)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--guided", choices=GUIDED, default="memory")
    ap.add_argument("--verified_fraction", type=float, default=1.0, help="fraction of prompts whose verifier is available after the answer (the rest use the memory pseudo-reward)")
    ap.add_argument("--task_quota", default=None, help="e.g. math:0.4,rag:0.4,code:0.2 with --n_prompts (train_rlvr's sampling); default: keep every row")
    ap.add_argument("--n_prompts", type=int, default=None)
    ap.add_argument("--every", type=int, default=1, help="keep every k-th row (even subsample along the stream)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt_source", choices=["solo", "memory", "peers"], default="solo",
                    help="the policy's own prompt: question only (solo), or every peer solution with the history (memory) / without it (peers)")
    ap.add_argument("--swap_history", action="store_true", help="control: the history permuted by rank (highest reliability on the least trusted peer)")
    ap.add_argument("--evidence", type=Path, default=None, help="method B stage 0: JSON from scripts/evidence_checks.py; adds an Evidence block to peers/memory prompts")
    ap.add_argument("--verdict", action="store_true", help="method A: ask for a 'Trust: a > b > ...' line before the answer (peers/memory prompts)")
    ap.add_argument("--solo_fraction", type=float, default=0.0, help="fraction of events whose whole group is answered under the question-only prompt (keeps the model's own ability)")
    return ap.parse_args()


def guided_messages(r: dict, rec: dict, peer_texts: list[str], mode: str, rng: random.Random):
    """Returns (messages or None, slot or -1, correct flag or -1)."""
    if mode == "none":
        return None, -1, -1
    any_correct = int(max(int(x) for x in r["peer_correct"])) if r["peer_correct"] else 0
    if mode == "memory":
        return r["messages_memory"], -1, any_correct
    if mode == "peers":
        return r["messages_peers"], -1, any_correct
    if mode == "peers_labeled":
        return labeled_peer_messages(rec, peer_texts, r["peer_correct"]), -1, any_correct
    valid = [bool(str(t).strip()) for t in peer_texts]
    slot = choose_hint_slot(r["peer_correct"], r["memory_prob"], choice=HINT_CHOICE[mode], rng=rng, valid=valid)
    if slot is None:
        return None, -1, -1
    return hint_messages(rec, peer_texts[slot], float(r["memory_prob"][slot]), float(r["memory_evidence"][slot])), int(slot), int(r["peer_correct"][slot])


def main() -> None:
    args = parse_args()
    evidence = json.load(open(args.evidence))["lines"] if args.evidence else None
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in args.prompts.open()]
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    if args.task_quota:
        by_task: dict[str, list] = {}
        for r in rows:
            by_task.setdefault(r["task_type"], []).append(r)
        chosen = []
        for part in args.task_quota.split(","):
            task, frac = part.split(":")
            pool = by_task.get(task, [])
            rng.shuffle(pool)
            chosen += pool[: int((args.n_prompts or len(rows)) * float(frac))]
        rows = sorted(chosen, key=lambda r: int(r.get("pos", 0)))   # back to stream order
    rows = rows[:: max(1, args.every)]
    if args.limit:
        rows = rows[: args.limit]
    out_rows, n_guided, n_guided_correct, n_verified = [], 0, 0, 0
    for i, r in enumerate(rows):
        rec = records[str(r["id"])]
        peer_texts = peer_texts_in_prompt_order(rec, r["peer_order"])
        verified = rng.random() < args.verified_fraction
        guided, slot, correct = guided_messages(r, rec, peer_texts, args.guided, rng)
        if args.guided in LABELS_BEFORE and not verified:
            guided, slot, correct = None, -1, -1   # labels-before needs the labels
        probs, evid = [float(x) for x in r["memory_prob"]], [float(x) for x in r["memory_evidence"]]
        domain_counts = r.get("memory_domain")
        if args.swap_history:
            ranked = sorted(range(len(probs)), key=lambda s_: probs[s_])
            new_p, new_e, new_d = list(probs), list(evid), (list(domain_counts) if domain_counts else None)
            for i_, s_ in enumerate(ranked):
                new_p[s_], new_e[s_] = probs[ranked[-1 - i_]], evid[ranked[-1 - i_]]
                if new_d is not None:
                    new_d[s_] = domain_counts[ranked[-1 - i_]]
            probs, evid, domain_counts = new_p, new_e, new_d
        source = "solo" if (args.prompt_source == "solo" or rng.random() < args.solo_fraction) else args.prompt_source
        if source == "solo":
            main_prompt = r["messages_solo"]
        elif source == "peers":
            main_prompt = r["messages_peers"]
        else:
            domain = [domain_note(r.get("task_type_note", r["task_type"]), *d) for d in domain_counts] if domain_counts else None
            main_prompt = build_messages(rec, peer_texts, mode="memory", probs=probs, evidence=evid, domain=domain)
        if evidence is not None and source != "solo":
            ev = evidence.get(str(r["id"]), {})
            keys = sorted(ev)
            lines = [ev[keys[int(p)]] for p in r["peer_order"]] if keys else []
            # the stored lines are numbered in canonical order; renumber to prompt order
            lines = [re.sub(r"^Peer \d+:", f"Peer {j + 1}:", ln) for j, ln in enumerate(lines)]
            main_prompt = add_evidence_block(main_prompt, lines)
        if args.verdict and source != "solo":
            main_prompt = add_verdict_instruction(main_prompt)
        n_guided += guided is not None
        n_guided_correct += int(correct == 1)
        n_verified += verified
        # character spans of the peer blocks inside the user turn: the attention tilt (data.attn_gamma) is placed on them
        spans = peer_char_spans(peer_texts, main_prompt[-1]["content"]) if source != "solo" else []
        out_rows.append({
            "data_source": str(r["task_type"]),
            "prompt": main_prompt,
            "guided_prompt": guided,
            "reward_model": {"style": "rule", "ground_truth": str(rec.get("answer", ""))},
            "extra_info": {
                "index": i, "id": str(r["id"]), "pos": int(r.get("pos", i)), "task_type": str(r["task_type"]), "source": str(r.get("source", "")),
                "verified": bool(verified), "guided": args.guided, "labels_before": args.guided in LABELS_BEFORE,
                "protocol": LABELS_BEFORE_PROTOCOL if args.guided in LABELS_BEFORE else PROTOCOL,
                "guided_slot": int(slot), "guided_correct": int(correct),
                "memory_prob": probs, "memory_evidence": evid, "prompt_source": source, "swapped_history": bool(args.swap_history),
                "peer_spans": [[int(a), int(b)] for a, b in spans],
                "peer_correct": [int(x) for x in r["peer_correct"]], "peer_order": [int(x) for x in r["peer_order"]],
                "record": json.dumps(rec),
            },
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(args.out, index=False)
    tasks = collections.Counter(x["data_source"] for x in out_rows)
    sources = collections.Counter(x["extra_info"]["prompt_source"] for x in out_rows)
    print(f"[build-rl-data] wrote {len(out_rows)} rows to {args.out}: tasks={dict(tasks)} prompt={dict(sources)}{' (swapped history)' if args.swap_history else ''} guided={args.guided} rows_with_guidance={n_guided} "
          f"guidance_correct={n_guided_correct} verified={n_verified}")


if __name__ == "__main__":
    main()
