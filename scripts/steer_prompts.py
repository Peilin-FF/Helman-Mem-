"""Training-free steering: different ways of putting the memory's reliability record in front of the central model.

Each variant rewrites ``messages_memory`` of the probe rows, so the standard evaluator runs unchanged
(``evaluate_memory_generator --mode memory``).  For every variant a swapped-note control is written as well
(the record permuted by rank, so the highest reliability is printed on the least trusted peer): a model that
is steered by the record follows the newly favoured peer under the swap, a model that reads the content does not.

  PYTHONPATH=. python scripts/steer_prompts.py --records data/ood/test.jsonl --prompts outputs/gen/q3_4b/prompts_ood_shuffled0.jsonl \
      --solo outputs/gen/q3_4b/frozen/eval_ood_shuffled0_solo_vllm/generations.jsonl --start 4000 --count 1500 --out_dir outputs/gen/q3_4b/steer/ood

Variants (all use the same question / options / context block as the original prompt):
  number      the original prompt: a probability in each peer's header (reference)
  ordinal     peers without numbers, then a reliability paragraph in words (right on ~9 of 10 similar cases), ranked, with a directive
  sorted      peers re-ordered most reliable first, the rank in each header (renamed Peer 1..k in that order)
  favourite   only the most reliable peer's solution is shown (the others withheld, their records mentioned)
  filtered    peers with an estimate below 0.35 are withheld; the rest as in ordinal
  vote        the peers' final answers grouped, each group with its combined reliability weight, the leading answer named
  self_note   ordinal plus the central model's own track record on this task (answering alone, running mean)
  verify      ordinal plus a two-step instruction: check the most reliable peer's solution first, adopt it unless a definite error is found
  defer       two-pass: the model's own alone answer is shown next to the most reliable peer's when they disagree and the record is
              confident; otherwise the alone prompt is used unchanged
When the record is flat (estimates within 0.1 of each other, or no evidence) every variant says so and falls back to
judging on content, so the directive never points at an arbitrary peer.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Sequence

from feedback_state.answer_groups import answer_groups
from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import INSTRUCTIONS, SYSTEM_PEERS, _clip, build_messages
from feedback_state.memory_rl import peer_texts_in_prompt_order
from feedback_state.tasks import _choice_labels, _rag_context_text, boolqa_extract_answer, mcqa_extract_answer, qa_extract_answer, shortqa_extract_answer, task_type_of
from feedback_state.utils import extract_final_answer

VARIANTS = ("number", "ordinal", "sorted", "favourite", "filtered", "vote", "self_note", "verify", "defer")
CHAR_LIMIT = 3000
FLAT_SPREAD = 0.1

SYSTEM_STEER = (
    "You are the central model of a multi-agent system. Several peer models answered the same question. A reliability "
    "memory has tracked, from verified outcomes on earlier similar questions, how often each peer was right. Its record is "
    "your primary guide to which peer to trust: a peer that was right on most similar cases is usually right again, a peer "
    "that was wrong on most similar cases is usually wrong again, and agreement between unreliable peers is not evidence. "
    "Verify what you can, then produce your own final answer."
)


# ----------------------------------------------------------------------------------------------- record helpers
def counts(p: float, e: float) -> tuple[int, int] | None:
    n = int(round(float(e)))
    if n < 1:
        return None
    return int(round(float(p) * n)), n


def describe(name: str, p: float, e: float) -> str:
    c = counts(p, e)
    return f"{name}: right on about {c[0]} of {c[1]} similar past cases" if c else f"{name}: no similar past cases yet"


def is_flat(probs: Sequence[float], evid: Sequence[float]) -> bool:
    return (max(probs) - min(probs) <= FLAT_SPREAD) or all(int(round(float(e))) < 1 for e in evid)


def ranked_slots(probs: Sequence[float]) -> list[int]:
    return sorted(range(len(probs)), key=lambda s: -float(probs[s]))


def memory_paragraph(names: Sequence[str], probs: Sequence[float], evid: Sequence[float]) -> str:
    if is_flat(probs, evid):
        return ("Reliability memory: no useful difference between the peers' records on this kind of question; "
                "judge the answers on their content.")
    ranked = ranked_slots(probs)
    lines = ["Reliability memory (verified outcomes on earlier, similar questions):"]
    for r, s in enumerate(ranked):
        tag = " (most reliable)" if r == 0 else (" (least reliable)" if r == len(ranked) - 1 else "")
        lines.append(f"- {describe(names[s] + tag, probs[s], evid[s])}")
    top, low = names[ranked[0]], names[ranked[-1]]
    lines.append(f"Unless you can verify an answer yourself, prefer {top}'s answer. Do not follow the majority when it "
                 f"contradicts {top}, and do not adopt an answer that only {low} gives.")
    return "\n".join(lines)


def swap_by_rank(probs: Sequence[float], evid: Sequence[float]) -> tuple[list[float], list[float]]:
    ranked = sorted(range(len(probs)), key=lambda s: probs[s])
    new_p, new_e = list(probs), list(evid)
    for i, s in enumerate(ranked):
        new_p[s], new_e[s] = probs[ranked[-1 - i]], evid[ranked[-1 - i]]
    return new_p, new_e


# ----------------------------------------------------------------------------------------------- prompt pieces
def question_parts(rec: dict, task: str) -> list[str]:
    parts = [f"Question:\n{str(rec.get('problem', rec.get('question', ''))).strip()}"]
    if task == "mcqa":
        labels = _choice_labels(rec)
        choices = rec.get("choices") or []
        if choices:
            parts.append("Options:\n" + "\n".join(f"({labels[i] if i < len(labels) else chr(65 + i)}) {c}" for i, c in enumerate(choices)))
    if task == "rag":
        ctx = _rag_context_text(rec)
        if ctx:
            parts.append(f"Context / Evidence:\n{ctx}")
    return parts


def peer_blocks(names: Sequence[str], texts: Sequence[str], heads: Sequence[str] | None = None) -> str:
    return "Peer answers:\n\n" + "\n\n".join(f"{(heads[i] if heads else names[i])}:\n{_clip(t, CHAR_LIMIT)}" for i, t in enumerate(texts))


def final_of(task: str, text: str) -> str:
    if task == "mcqa":
        return mcqa_extract_answer(text)
    if task == "boolqa":
        return boolqa_extract_answer(text)
    if task == "shortqa":
        return shortqa_extract_answer(text)
    if task == "rag":
        return qa_extract_answer(text)
    if task == "math":
        return extract_final_answer(text) or ""
    return ""


def messages(system: str, parts: Sequence[str]) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


# ----------------------------------------------------------------------------------------------- variants
def build_variant(variant: str, rec: dict, row: dict, probs: list[float], evid: list[float], *, solo_text: str | None, self_record: tuple[int, int] | None) -> tuple[list[dict], dict]:
    """Returns (messages, extra row fields).  ``probs``/``evid`` are the record as shown (possibly swapped)."""
    task = task_type_of(rec)
    texts = peer_texts_in_prompt_order(rec, row["peer_order"])
    k = len(texts)
    names = [f"Peer {i + 1}" for i in range(k)]
    instr = "Instruction: " + INSTRUCTIONS.get(task, INSTRUCTIONS["shortqa"])
    q = question_parts(rec, task)
    flat = is_flat(probs, evid)
    ranked = ranked_slots(probs)
    fav = ranked[0]
    extra: dict[str, Any] = {"steer_variant": variant, "steer_flat": flat}

    if variant == "number":
        return build_messages(rec, texts, mode="memory", probs=probs, evidence=evid), extra
    if variant == "ordinal":
        return messages(SYSTEM_STEER, q + [peer_blocks(names, texts), memory_paragraph(names, probs, evid), instr]), extra
    if variant == "sorted":
        order = ranked if not flat else list(range(k))
        new_texts = [texts[s] for s in order]
        heads = []
        for r, s in enumerate(order):
            tag = "" if flat else (" (most reliable on similar questions: " if r == 0 else (" (least reliable on similar questions: " if r == k - 1 else " ("))
            c = counts(probs[s], evid[s])
            heads.append(f"Peer {r + 1}" + ("" if flat else tag + (f"right on about {c[0]} of {c[1]} past cases)" if c else "no similar past cases yet)")))
        note = ("Reliability memory: no useful difference between the peers' records on this kind of question; judge the answers on their content."
                if flat else "The peers are listed from most to least reliable on similar questions according to the reliability memory. Unless you can verify an answer yourself, prefer Peer 1's answer over the others, and do not follow a majority that contradicts Peer 1.")
        extra.update(peer_order=[row["peer_order"][s] for s in order], peer_correct=[row["peer_correct"][s] for s in order],
                     memory_prob=[probs[s] for s in order], memory_evidence=[evid[s] for s in order])
        return messages(SYSTEM_STEER, q + [peer_blocks([f"Peer {r + 1}" for r in range(k)], new_texts, heads), note, instr]), extra
    if variant == "favourite":
        if flat:
            return messages(SYSTEM_PEERS, q + [peer_blocks(names, texts), instr]), extra
        others = ", ".join(describe(names[s], probs[s], evid[s]) for s in ranked[1:])
        return messages(SYSTEM_STEER, q + [f"Answer of the peer that the reliability memory rates most reliable on similar questions ({describe(names[fav], probs[fav], evid[fav])}):\n{_clip(texts[fav], CHAR_LIMIT)}",
                                           f"The other peers' answers were withheld because their records on similar questions are weaker ({others}).",
                                           "Treat the shown answer as strong evidence: verify it, adopt its final answer unless you find a definite error, and answer in the required format. " + instr]), extra
    if variant == "filtered":
        keep = [s for s in range(k) if probs[s] >= 0.35] if not flat else list(range(k))
        if not keep:
            keep = [fav]
        withheld = [s for s in range(k) if s not in keep]
        parts = q + [peer_blocks([names[s] for s in keep], [texts[s] for s in keep])]
        if withheld:
            parts.append("Withheld by the reliability memory (records too weak on similar questions): " + "; ".join(describe(names[s], probs[s], evid[s]) for s in withheld) + ".")
        parts += [memory_paragraph([names[s] for s in keep], [probs[s] for s in keep], [evid[s] for s in keep]) if len(keep) > 1 else f"Reliability memory: {describe(names[keep[0]], probs[keep[0]], evid[keep[0]])}.", instr]
        return messages(SYSTEM_STEER, parts), extra
    if variant == "vote":
        if flat or task == "code":
            return build_variant("ordinal", rec, row, probs, evid, solo_text=solo_text, self_record=self_record)
        groups = answer_groups(rec, texts)
        by_group: dict[int, list[int]] = collections.defaultdict(list)
        for s, g in enumerate(groups):
            by_group[g].append(s)
        def weight(slots):
            lo = sum(math.log(max(probs[s], 1e-3) / max(1 - probs[s], 1e-3)) for s in slots)
            return 1 / (1 + math.exp(-lo))
        lines = ["Reliability-weighted vote over the peers' final answers (weight = combined record of the supporting peers):"]
        scored = sorted(by_group.items(), key=lambda kv: -weight(kv[1]))
        for g, slots in scored:
            surface = final_of(task, texts[slots[0]]) or "(no parsable final answer)"
            lines.append(f"- Answer {_clip(surface, 120)}: weight {weight(slots):.2f}, supported by " + ", ".join(describe(names[s], probs[s], evid[s]) for s in slots))
        lead = final_of(task, texts[scored[0][1][0]]) or "(unparsable)"
        lines.append(f"The memory's leading answer is: {_clip(lead, 120)}. Prefer it unless you can verify that it is wrong.")
        return messages(SYSTEM_STEER, q + [peer_blocks(names, texts), "\n".join(lines), instr]), extra
    if variant == "self_note":
        para = memory_paragraph(names, probs, evid)
        if self_record and self_record[1] >= 5:
            r, n = self_record
            para += f"\nYour own record: answering alone, you were right on about {r} of {n} earlier {task} questions."
            c = counts(probs[fav], evid[fav])
            if c and not flat and c[0] / c[1] > r / max(1, n):
                para += f" {names[fav]}'s record on similar questions is better than yours; when you disagree with {names[fav]} and cannot show its error, its answer is the better bet."
        return messages(SYSTEM_STEER, q + [peer_blocks(names, texts), para, instr]), extra
    if variant == "verify":
        if flat:
            return messages(SYSTEM_STEER, q + [peer_blocks(names, texts), memory_paragraph(names, probs, evid), instr]), extra
        step = (f"Procedure: first check {names[fav]}'s solution against the question step by step ({names[fav]} is the peer with the best record on similar questions). "
                f"If it holds, adopt its final answer. Only if you find a definite error in it, solve the question yourself. Do not switch to another peer's answer merely because more peers give it.")
        return messages(SYSTEM_STEER, q + [peer_blocks(names, texts), memory_paragraph(names, probs, evid), step, instr]), extra
    if variant == "defer":
        extra["steer_triggered"] = False
        if solo_text is None or flat or probs[fav] < 0.6:
            return row["messages_solo"], extra
        groups = answer_groups(rec, texts + [solo_text])
        if groups[-1] == groups[fav]:
            return row["messages_solo"], extra
        extra["steer_triggered"] = True
        return messages(SYSTEM_STEER, q + [f"Your own earlier answer (working alone):\n{_clip(solo_text, 1500)}",
                                           f"{names[fav]}, the peer that the reliability memory rates most reliable on similar questions ({describe(names[fav], probs[fav], evid[fav])}), answered differently:\n{_clip(texts[fav], CHAR_LIMIT)}",
                                           f"Your earlier answer disagrees with the most reliable peer. Re-examine the question. If you cannot show a definite error in {names[fav]}'s solution, adopt its final answer; otherwise keep yours.", instr]), extra
    raise ValueError(variant)


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--solo", type=Path, default=None, help="frozen model's alone generations on the whole stream (self_note, defer)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=1500)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    rows = sorted((json.loads(l) for l in args.prompts.open()), key=lambda r: r["pos"])
    solo: dict[str, dict] = {}
    if args.solo:
        solo = {str(g["id"]): g for g in map(json.loads, args.solo.open())}
    # the central model's own running record per task (read-before-write), from its alone answers along the stream
    self_rec: dict[str, tuple[int, int]] = {}
    tally: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        t = r["task_type"]
        self_rec[str(r["id"])] = (tally[t][1], tally[t][0])
        g = solo.get(str(r["id"]))
        if g is not None:
            tally[t][0] += 1
            tally[t][1] += int(g["correct"])
    variants = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slice_rows = rows[args.start: args.start + args.count]
    stats: dict[str, collections.Counter] = {}
    for v in variants:
        stats[v] = collections.Counter()
        for swapped in (False, True):
            out = args.out_dir / f"prompts_{v}{'_swapped' if swapped else ''}.jsonl"
            with out.open("w") as f:
                for r in slice_rows:
                    rec = records[str(r["id"])]
                    probs, evid = list(r["memory_prob"]), list(r["memory_evidence"])
                    if swapped:
                        probs, evid = swap_by_rank(probs, evid)
                    g = solo.get(str(r["id"]))
                    msgs, extra = build_variant(v, rec, r, probs, evid, solo_text=(g["generation"] if g else None), self_record=self_rec.get(str(r["id"])))
                    row = dict(r)
                    row["memory_prob"], row["memory_evidence"] = probs, evid
                    if swapped:
                        row["memory_prob_original"], row["memory_evidence_original"] = r["memory_prob"], r["memory_evidence"]
                    row.update(extra)
                    row["messages_memory"] = msgs
                    f.write(json.dumps(row) + "\n")
                    if not swapped:
                        stats[v]["rows"] += 1
                        stats[v]["flat"] += int(extra.get("steer_flat", False))
                        stats[v]["triggered"] += int(extra.get("steer_triggered", False))
        print(f"[steer] {v}: {dict(stats[v])} -> {args.out_dir}/prompts_{v}[_swapped].jsonl")


if __name__ == "__main__":
    main()
