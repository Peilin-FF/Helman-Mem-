"""Does the central model use the memory's reliability notes?  Two probes on already-generated answers plus a
counterfactual prompt builder.

  analyze   For each evaluated generation (memory mode: notes shown; peers mode: same peers, no notes) decide whether the
            model's answer equals the answer of the peer the memory trusts most ("follows the top peer"), the majority,
            or none of them.  The informative events are those where the peers disagree and the notes are not flat.
            If the notes are used, the follow-top rate is higher with notes than without, above all when the trusted peer
            is in the minority; if the notes are ignored, the two modes coincide.
  swap      Counterfactual prompt file: the same events with the notes permuted so that the highest reliability is shown on
            the peer the memory trusts least (and vice versa).  A model that reads the notes follows the newly favoured
            peer; a model that reads the content does not move.

  PYTHONPATH=. python scripts/memory_use_probe.py analyze --records data/ood/test.jsonl --prompts outputs/gen/q3_4b/prompts_ood_shuffled0.jsonl \
      --eval memory=outputs/gen/q3_4b/frozen/eval_ood_shuffled0_memory_vllm --eval peers=outputs/gen/q3_4b/frozen/eval_ood_shuffled0_peers_vllm
  PYTHONPATH=. python scripts/memory_use_probe.py swap --records data/ood/test.jsonl --prompts outputs/gen/q3_4b/prompts_ood_shuffled0.jsonl \
      --start 4000 --count 800 --out outputs/gen/q3_4b/probe/prompts_ood_probe.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from feedback_state.answer_groups import answer_groups
from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import build_messages, strip_thinking
from feedback_state.memory_rl import peer_texts_in_prompt_order

NOTE_WORDS = re.compile(r"reliab|probability correct|estimated probability|past case|track record|trust|more likely to be correct|weight", re.I)


def load_records(path: Path) -> dict:
    return {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(path).records}


def prompt_rows(path: Path) -> dict:
    return {str(r["id"]): r for r in map(json.loads, path.open())}


def swap_notes(row: dict, rec: dict) -> dict:
    """Notes permuted by rank: the top reliability goes to the least trusted peer and vice versa (middle unchanged)."""
    probs, evid = list(row["memory_prob"]), list(row["memory_evidence"])
    ranked = sorted(range(len(probs)), key=lambda s: probs[s])
    new_p, new_e = list(probs), list(evid)
    for i, s in enumerate(ranked):
        new_p[s], new_e[s] = probs[ranked[-1 - i]], evid[ranked[-1 - i]]
    texts = peer_texts_in_prompt_order(rec, row["peer_order"])
    out = dict(row)
    out["memory_prob_original"], out["memory_evidence_original"] = probs, evid
    out["memory_prob"], out["memory_evidence"] = new_p, new_e
    out["messages_memory"] = build_messages(rec, texts, mode="memory", probs=new_p, evidence=new_e)
    return out


def cmd_swap(args) -> None:
    records = load_records(args.records)
    rows = [json.loads(l) for l in args.prompts.open()][args.start : args.start + args.count]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    orig = args.out.with_name(args.out.stem.replace("_swapped", "") + ".jsonl") if "_swapped" in args.out.stem else args.out
    swapped = args.out if "_swapped" in args.out.stem else args.out.with_name(args.out.stem + "_swapped.jsonl")
    n_inf = 0
    with orig.open("w") as fo, swapped.open("w") as fs:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
            fs.write(json.dumps(swap_notes(r, records[str(r["id"])])) + "\n")
            n_inf += int(max(r["memory_prob"]) - min(r["memory_prob"]) > args.min_spread)
    print(f"[probe] wrote {len(rows)} rows (positions {args.start}..{args.start + len(rows) - 1}) to {orig} and {swapped}; notes with spread > {args.min_spread}: {n_inf}")


def event_stats(rec: dict, row: dict, gen: dict, min_spread: float) -> dict | None:
    if gen["task_type"] == "code":
        return None
    texts = peer_texts_in_prompt_order(rec, row["peer_order"])
    probs = list(gen.get("memory_prob") or row["memory_prob"])
    text = gen["generation"]
    answer = strip_thinking(text)
    groups = answer_groups(rec, texts + [answer])
    peers, mine = groups[:-1], groups[-1]
    counts = collections.Counter(peers)
    best = counts.most_common(2)
    plurality = best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else None
    top, low = max(range(len(probs)), key=lambda s: probs[s]), min(range(len(probs)), key=lambda s: probs[s])
    think = text.rsplit("</think>", 1)[0] if "</think>" in text else ""
    plurality_correct = int(gen["peer_correct"][peers.index(plurality)]) if plurality is not None else 0
    any_correct = int(max(gen["peer_correct"]))
    return {"plurality_correct": plurality_correct, "any_peer_correct": any_correct, "plurality_exists": plurality is not None,"task": gen["task_type"], "correct": int(gen["correct"]), "split": len(set(peers)) > 1, "spread": max(probs) - min(probs) > min_spread,
            "follow_top": mine == peers[top], "follow_low": mine == peers[low], "follow_majority": plurality is not None and mine == plurality,
            "novel": mine not in peers, "top_minority": plurality is not None and peers[top] != plurality, "top_correct": int(gen["peer_correct"][top]),
            "top_prob": probs[top], "slot_follow": [(probs[s], mine == peers[s], int(gen["peer_correct"][s])) for s in range(len(probs))],
            "has_think": bool(think), "think_mentions_notes": bool(NOTE_WORDS.search(think)) if think else False, "think_len": len(think)}


def rate(xs, key, cond=lambda e: True):
    sel = [e for e in xs if cond(e)]
    return (100.0 * sum(bool(e[key]) for e in sel) / len(sel) if sel else float("nan")), len(sel)


def summarize(events: list[dict]) -> dict:
    inf = [e for e in events if e["split"] and e["spread"]]
    minority = [e for e in inf if e["top_minority"]]
    out = {"n": len(events), "accuracy": rate(events, "correct")[0], "n_informative": len(inf),
           "follow_top_all_split": rate(events, "follow_top", lambda e: e["split"])[0],
           "follow_top_informative": rate(inf, "follow_top")[0], "follow_low_informative": rate(inf, "follow_low")[0],
           "follow_majority_informative": rate(inf, "follow_majority")[0], "novel_informative": rate(inf, "novel")[0],
           "accuracy_informative": rate(inf, "correct")[0],
           "n_top_minority": len(minority), "follow_top_when_minority": rate(minority, "follow_top")[0],
           "follow_majority_when_minority": rate(minority, "follow_majority")[0], "accuracy_when_minority": rate(minority, "correct")[0],
           "accuracy_when_minority_top_correct": rate(minority, "correct", lambda e: e["top_correct"] == 1)[0],
           "n_minority_top_correct": rate(minority, "correct", lambda e: e["top_correct"] == 1)[1],
           "follow_top_when_minority_top_correct": rate(minority, "follow_top", lambda e: e["top_correct"] == 1)[0],
           "follow_top_when_minority_top_wrong": rate(minority, "follow_top", lambda e: e["top_correct"] == 0)[0]}
    bins = [(0.0, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)]
    cal = {}
    for lo, hi in bins:
        sel = [(f, c) for e in inf for p, f, c in e["slot_follow"] if lo <= p < hi]
        cal[f"{lo:.2f}-{min(hi, 1.0):.2f}"] = {"n": len(sel), "follow": 100.0 * sum(f for f, _ in sel) / len(sel) if sel else float("nan"),
                                              "peer_correct": 100.0 * sum(c for _, c in sel) / len(sel) if sel else float("nan")}
    out["follow_by_shown_prob"] = cal
    # who is right where it matters: the memory's favourite vs the peers' plurality vs the model
    out["informative_top_correct"] = rate(inf, "top_correct")[0]
    out["informative_plurality_correct"] = rate(inf, "plurality_correct")[0]
    out["informative_any_peer_correct"] = rate(inf, "any_peer_correct")[0]
    out["minority_plurality_correct"] = rate(minority, "plurality_correct")[0]
    out["minority_top_correct"] = rate(minority, "top_correct")[0]
    out["n_unanimous"] = len([e for e in events if not e["split"]])
    out["unanimous_accuracy"] = rate(events, "correct", lambda e: not e["split"])[0]
    out["unanimous_peer_correct"] = rate(events, "any_peer_correct", lambda e: not e["split"])[0]
    out["n_split_flat"] = len([e for e in events if e["split"] and not e["spread"]])
    # a reader that follows the favourite only when its estimate is high, and otherwise keeps the model's own answer
    for thr in (0.65, 0.8, 0.9):
        gated = [e for e in inf if e["top_prob"] >= thr]
        pol = sum((e["top_correct"] if e["top_prob"] >= thr else e["correct"]) for e in inf) / max(1, len(inf))
        out[f"gate_{thr}"] = {"n": len(gated), "policy_accuracy_informative": 100 * pol, "favourite_correct_in_gate": rate(gated, "top_correct")[0], "model_correct_in_gate": rate(gated, "correct")[0],
                              "stream_gain_points": (100 * pol - out["accuracy_informative"]) * len(inf) / max(1, len(events))}
    thinks = [e for e in events if e["has_think"]]
    if thinks:
        out["thinking"] = {"n": len(thinks), "mentions_notes": rate(thinks, "think_mentions_notes")[0], "mean_think_chars": sum(e["think_len"] for e in thinks) / len(thinks)}
    out["by_task"] = {t: {"n": len([e for e in inf if e["task"] == t]), "follow_top": rate(inf, "follow_top", lambda e, t=t: e["task"] == t)[0],
                          "follow_top_when_minority": rate(minority, "follow_top", lambda e, t=t: e["task"] == t)[0]} for t in sorted({e["task"] for e in inf})}
    return out


def cmd_analyze(args) -> None:
    records = load_records(args.records)
    rows = prompt_rows(args.prompts)
    result = {}
    for spec in args.eval:
        name, path = spec.split("=", 1)
        gens = [json.loads(l) for l in (Path(path) / "generations.jsonl").open()]
        if args.max_examples:
            gens = gens[: args.max_examples]
        if args.pos_start is not None:
            gens = [g for g in gens if args.pos_start <= int(g["pos"]) < args.pos_start + args.pos_count]
        events = [s for g in gens if (s := event_stats(records[str(g["id"])], rows[str(g["id"])], g, args.min_spread))]
        result[name] = summarize(events)
        if args.examples and any(e["has_think"] for e in events):
            shown = 0
            for g in gens:
                text = g["generation"]
                if "</think>" in text and NOTE_WORDS.search(text.rsplit("</think>", 1)[0]) and shown < args.examples:
                    m = NOTE_WORDS.search(text)
                    print(f"[{name}] {g['id']} ({g['task_type']}, correct={g['correct']}): ...{text[max(0, m.start() - 200): m.start() + 300].replace(chr(10), ' ')}...")
                    shown += 1
    for name, s in result.items():
        print(f"\n== {name}: n={s['n']} acc={s['accuracy']:.2f} | informative (peers split, notes not flat) n={s['n_informative']}: follow top {s['follow_top_informative']:.1f} / "
              f"low {s['follow_low_informative']:.1f} / majority {s['follow_majority_informative']:.1f} / novel {s['novel_informative']:.1f}, acc {s['accuracy_informative']:.2f}")
        print(f"   top peer in minority n={s['n_top_minority']}: follow top {s['follow_top_when_minority']:.1f} (top correct: {s['follow_top_when_minority_top_correct']:.1f}, top wrong: {s['follow_top_when_minority_top_wrong']:.1f}), "
              f"follow majority {s['follow_majority_when_minority']:.1f}, acc {s['accuracy_when_minority']:.2f}")
        print(f"   who is right on informative events: memory's favourite {s['informative_top_correct']:.1f}%, plurality {s['informative_plurality_correct']:.1f}%, any peer {s['informative_any_peer_correct']:.1f}%, model {s['accuracy_informative']:.1f}% | in minority cases: favourite {s['minority_top_correct']:.1f}%, plurality {s['minority_plurality_correct']:.1f}%, model {s['accuracy_when_minority']:.1f}% | unanimous n={s['n_unanimous']}: peers {s['unanimous_peer_correct']:.1f}%, model {s['unanimous_accuracy']:.1f}%; split but flat notes n={s['n_split_flat']}")
        print("   gated reader (follow the favourite only when its estimate >= t): " + ", ".join(f"t={t}: n={s[f'gate_{t}']['n']}, favourite {s[f'gate_{t}']['favourite_correct_in_gate']:.1f}% vs model {s[f'gate_{t}']['model_correct_in_gate']:.1f}% there, informative acc {s[f'gate_{t}']['policy_accuracy_informative']:.1f} ({s[f'gate_{t}']['stream_gain_points']:+.2f} pts on the slice)" for t in (0.65, 0.8, 0.9)))
        print("   follow rate by shown probability: " + ", ".join(f"{k}: {v['follow']:.1f}% (n={v['n']}, peer acc {v['peer_correct']:.0f}%)" for k, v in s["follow_by_shown_prob"].items()))
        if "thinking" in s:
            print(f"   thinking: {s['thinking']['n']} traces, {s['thinking']['mentions_notes']:.1f}% mention reliability/notes, mean {s['thinking']['mean_think_chars']:.0f} chars")
        print("   by task: " + ", ".join(f"{t}: follow top {v['follow_top']:.1f} (minority {v['follow_top_when_minority']:.1f}, n={v['n']})" for t, v in s["by_task"].items()))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(f"[probe] wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--records", type=Path, required=True); a.add_argument("--prompts", type=Path, required=True)
    a.add_argument("--eval", action="append", required=True, help="name=dir (dir holds generations.jsonl)")
    a.add_argument("--max_examples", type=int, default=None); a.add_argument("--min_spread", type=float, default=0.1)
    a.add_argument("--pos_start", type=int, default=None, help="restrict to stream positions [pos_start, pos_start+pos_count) (whole-stream runs vs probe slices)")
    a.add_argument("--pos_count", type=int, default=1500)
    a.add_argument("--examples", type=int, default=0, help="print this many thinking excerpts that mention the notes")
    a.add_argument("--out", type=Path, default=None)
    s = sub.add_parser("swap")
    s.add_argument("--records", type=Path, required=True); s.add_argument("--prompts", type=Path, required=True)
    s.add_argument("--start", type=int, default=0); s.add_argument("--count", type=int, default=800); s.add_argument("--min_spread", type=float, default=0.1)
    s.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    (cmd_swap if args.cmd == "swap" else cmd_analyze)(args)


if __name__ == "__main__":
    main()
