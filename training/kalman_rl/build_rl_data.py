"""A record file -> the verl parquet the GRPO trainer reads, one row per event, in stream order.

    PYTHONPATH=. python -m training.kalman_rl.build_rl_data --record outputs/record/q3_4b/train6/fixed.fit-train6.jsonl \
        --stream data/mixed_train_big6/train.jsonl --prompt peers --solo-fraction 0.25 --out outputs/train/data/q3_4b_train6_peers.parquet

--prompt decides what the policy answers from:
  peers        the question and the six peers' answers (the record's estimates travel in extra_info, for the tilt)
  solo         the question alone
  combination  both prompts (prompt, prompt_solo): the trainer's combination picks one per question before the rollouts and
               sets the tilt from its own online record (feedback_state.combination), so no estimates are stored
--solo-fraction makes that share of the events question-only anyway, so a peers-trained model keeps its own ability.

Row layout (verl conventions): data_source = task type; prompt = chat messages; reward_model = {style: rule,
ground_truth}; extra_info = the event (index, id, pos, task_type, source), the record's estimates at this position
(memory_prob, memory_evidence per slot), the peer blocks' character spans in the user turn (peer_spans, where the tilt
is placed), peer_correct, peer_order, prompt_source, and the full stream record as JSON (for grading).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import pandas as pd

from feedback_state.attn_bias import peer_char_spans
from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import peer_texts_in_prompt_order

PROTOCOL = "peer_outcome_v1"   # the answer is graded only after it is given; recorded in every row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", type=Path, required=True, help="pipeline.record output for the stream")
    ap.add_argument("--stream", type=Path, required=True, help="the stream JSONL (answers, tests)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt", choices=["peers", "solo", "combination"], required=True)
    ap.add_argument("--solo-fraction", type=float, default=0.0)
    ap.add_argument("--every", type=int, default=1, help="keep every k-th event (an even subsample along the stream)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in args.record.open()][:: max(1, args.every)]
    if args.limit:
        rows = rows[: args.limit]
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.stream).records}
    out_rows = []
    for i, r in enumerate(rows):
        rec = records[str(r["id"])]
        rng.random()   # one draw per event that the earlier builder spent on its verifier flag: keeps the solo split of the released runs
        if args.prompt == "combination":
            source = "combination"
        else:
            source = "solo" if (args.prompt == "solo" or rng.random() < args.solo_fraction) else "peers"
        prompt = r["messages_solo"] if source == "solo" else r["messages_peers"]
        spans = peer_char_spans(peer_texts_in_prompt_order(rec, r["peer_order"]), prompt[-1]["content"]) if source != "solo" else []
        extra = {"index": i, "id": str(r["id"]), "pos": int(r.get("pos", i)), "task_type": str(r["task_type"]), "source": str(r.get("source", "")),
                 "protocol": PROTOCOL, "prompt_source": source}
        if source != "combination":   # combination reads its estimates online, at training time
            extra.update(memory_prob=[float(x) for x in r["memory_prob"]], memory_evidence=[float(x) for x in r["memory_evidence"]])
        extra.update(peer_spans=[[int(a), int(b)] for a, b in spans], peer_correct=[int(x) for x in r["peer_correct"]],
                     peer_order=[int(x) for x in r["peer_order"]], record=json.dumps(rec))
        row = {"data_source": str(r["task_type"]), "prompt": prompt,
               "reward_model": {"style": "rule", "ground_truth": str(rec.get("answer", ""))}, "extra_info": extra}
        if source == "combination":
            row["prompt_solo"] = r["messages_solo"]
        out_rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(args.out, index=False)
    print(f"[build-rl-data] {len(out_rows)} rows -> {args.out}: tasks {dict(collections.Counter(x['data_source'] for x in out_rows))}, "
          f"prompt {dict(collections.Counter(x['extra_info']['prompt_source'] for x in out_rows))}")


if __name__ == "__main__":
    main()
