"""Decide: per event, the central model's final answer is its reading of the peers or its own answer, by its reading line.

    python -m pipeline.decide --record outputs/record/q3_4b/indist6_misleading_p050+own/shuffled0.fit-self.jsonl \
        --reading outputs/eval/q3_4b/indist6_misleading_p050+own/tilt --own outputs/eval/q3_4b/indist6/solo \
        --output outputs/eval/q3_4b/indist6_misleading_p050+own/decide

The record estimates the peers and the central model's own answer alike (pipeline.record --own-slot): per event T is the
top estimate among the peers and kappa the estimate of the own answer. Walking the stream in the record's order, read
before write: decide with the reading line of the event's task type (feedback_state.reading_line), then write the verified
reading outcome into it. The reading outcome feeds nothing else, neither the record nor the tilt, so this pass gives
exactly what an online loop would. Writes generations.jsonl (the final answer and the decision per event) and
eval_metrics.json (accuracy, share read, the reading line per task type, and the same by trust band).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from pipeline.config import shown
from pipeline.evaluate import summarise

BANDS = 5


def load_generations(directory: Path) -> dict[str, dict]:
    return {str(g["id"]): g for g in map(json.loads, (directory / "generations.jsonl").open())}


def decide(rows: list[dict], reading: dict[str, dict], own: dict[str, dict], prior: tuple[float, float], lam: float) -> tuple[list[dict], dict]:
    from feedback_state.reading_line import ReadingLine

    lines: dict[str, ReadingLine] = {}
    out, mismatched = [], 0
    for r in sorted(rows, key=lambda r: int(r["pos"])):
        rid = str(r["id"])
        if rid not in reading or rid not in own:
            raise SystemExit(f"event {rid} has no {'reading' if rid not in reading else 'own'} answer")
        trust, kappa = max(float(p) for p in r["memory_prob"]), float(r["own_prob"])
        line = lines.setdefault(r["task_type"], ReadingLine(prior, lam))
        rho, delta = line.estimate()
        value = line.value(trust, kappa)
        read = value >= kappa
        read_ok, own_ok = int(reading[rid]["correct"]), int(own[rid]["correct"])
        mismatched += own_ok != int(r["own_correct"])
        chosen = reading[rid] if read else own[rid]
        out.append({"pos": r["pos"], "id": rid, "task_type": r["task_type"], "source": r.get("source"), "correct": int(chosen["correct"]),
                    "decision": "read" if read else "own", "trust": round(trust, 4), "own_prob": round(kappa, 4),
                    "reading_value": round(value, 4), "rho": round(rho, 4), "delta": round(delta, 4),
                    "read_correct": read_ok, "own_correct": own_ok, "peer_correct": r["peer_correct"], "generation": chosen["generation"]})
        line.write(trust, kappa, read_ok)
    by_task = collections.defaultdict(list)
    for o in out:
        by_task[o["task_type"]].append(o)
    summary = {"own_labels_mismatched": mismatched, "reading_line": {}}
    for task, line in sorted(lines.items()):
        rho, delta = line.estimate()
        mean_own = float(np.mean([o["own_prob"] for o in by_task[task]]))
        summary["reading_line"][task] = {"events": line.events, "rho": rho, "delta": delta, "mean_own_prob": mean_own,
                                         "switch_at_mean_own_prob": line.switch(mean_own), "reads_at_mean_own_prob": line.reads_when(mean_own),
                                         "read_share": float(np.mean([o["decision"] == "read" for o in by_task[task]]))}
    return out, summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", type=Path, required=True, help="the record with the own answer (pipeline.record --own-slot)")
    ap.add_argument("--reading", type=Path, required=True, help="the central model's evaluation reading the peers")
    ap.add_argument("--own", type=Path, required=True, help="its question-only evaluation")
    ap.add_argument("--prior", default="0.5,0.0", help="rho, delta before any event")
    ap.add_argument("--lam", type=float, default=1.0, help="the prior's precision")
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    prior = tuple(float(x) for x in args.prior.split(","))
    rows = [json.loads(line) for line in args.record.open()]
    out, summary = decide(rows, load_generations(args.reading), load_generations(args.own), prior, args.lam)
    mean = lambda key, sel=out: float(np.mean([o[key] for o in sel])) if sel else None
    bands = collections.defaultdict(list)
    for o in out:
        bands[min(BANDS - 1, int(o["trust"] * BANDS))].append(o)
    metrics = {"condition": "decide", "record": shown(args.record), "reading": shown(args.reading), "own": shown(args.own),
               "prior": list(prior), "lam": args.lam}
    metrics.update(summarise(out, args.windows))
    metrics.update({"read_share": float(np.mean([o["decision"] == "read" for o in out])), "always_read": mean("read_correct"),
                    "own_answer": mean("own_correct"), "either_right": float(np.mean([max(o["read_correct"], o["own_correct"]) for o in out])),
                    **summary,
                    "by_trust": {f"{b / BANDS:.1f}-{(b + 1) / BANDS:.1f}": {"events": len(sel), "read_share": float(np.mean([o["decision"] == "read" for o in sel])),
                                                                          "always_read": mean("read_correct", sel), "own_answer": mean("own_correct", sel),
                                                                          "decided": mean("correct", sel)}
                                 for b, sel in sorted(bands.items())}})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "generations.jsonl").open("w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    (args.output / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    lines = "; ".join(f"{t} rho {v['rho']:.2f} delta {v['delta']:.2f}" for t, v in metrics["reading_line"].items())
    print(f"[decide] {args.output}: accuracy {100 * metrics['accuracy']:.2f} (always read {100 * metrics['always_read']:.2f}, "
          f"own answer {100 * metrics['own_answer']:.2f}), read on {100 * metrics['read_share']:.1f}% of events; {lines}", flush=True)


if __name__ == "__main__":
    main()
