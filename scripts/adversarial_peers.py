"""Misleading but relevant answers from a peer model: generated, graded, and re-generated until they are usable.

The honest peers of the six-peer streams answer as well as they can.  A robustness experiment needs the opposite: a
peer whose answer is confident, on topic, in the usual format, and *wrong*.  Asking a model once for that is not
enough -- roughly half of its answers come out correct anyway, some announce the trick, some refuse, and some code
answers degenerate into a table of hard-coded outputs.  This script therefore treats every event as a small search:

  attempt 0   the misleading prompt (feedback_state.adversarial.misleading_prompt), temperature 0.2, as the honest
              peers were sampled
  check       grade the answer with the pipeline's own rule (token-F1 for reading, exact match for math, the hidden
              tests for code) and run feedback_state.adversarial.accept: wrong, no meta-commentary, no refusal, the
              gold answer not named, the task's answer format, grounded in the passage, a real program
  attempt k   only the events that failed, told what was wrong with the previous attempt, at a higher temperature
  force       for math / multiple-choice / yes-no, an answer that is still correct after the last attempt has its
              conclusion rewritten to a wrong one (feedback_state.adversarial.force_wrong) and is re-graded; those
              rows are marked ``forced`` so they can be counted or dropped later

Output: ``<output>/<split>.jsonl`` with one row per event ({id, source, task_type, response, target, correct,
misled, accepted, forced, attempts, reasons}) and ``summary_<split>.json`` with the acceptance statistics -- the
numbers that say whether the adversarial stream is really adversarial.  Answers are never mixed into a stream here;
``scripts/build_adversarial_stream.py`` does that under a regime.

  PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 python scripts/adversarial_peers.py \
      --model /mnt/data/peilin/HF_MODEL/gemma-3-4b-it --records data/indist6/test.jsonl \
      --output outputs/peer_adv/indist6/gemma-3-4b-it [--reasoning] [--num_shards 8 --shard_index 0]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from feedback_state.adversarial import accept, force_wrong, misleading_prompt, plausible_wrong_number
from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import strip_thinking
from feedback_state.tasks import task_type_of
from scripts.peer_answers import MAX_TOKENS, graded_value, is_misled

FORCEABLE = {"math", "mcqa", "boolqa"}   # tasks whose conclusion can be rewritten without rewriting the argument


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reasoning", action="store_true", help="4096-token budget, grade the text after the last </think>")
    p.add_argument("--max_attempts", type=int, default=3, help="generations per event before the forcing pass")
    p.add_argument("--temperatures", default="0.2,0.7,1.0", help="one per attempt (the last is reused if there are more attempts)")
    p.add_argument("--no_force", action="store_true", help="do not rewrite the conclusion of answers that stay correct")
    p.add_argument("--grade_workers", type=int, default=8, help="threads for grading (code and math grade in subprocesses)")
    p.add_argument("--mislead_rate", type=float, default=1.0, help="fraction of events to answer adversarially (1.0 = all, so any regime can be applied later)")
    p.add_argument("--mislead_seed", type=int, default=0)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--sources", default=None, help="comma-separated source datasets to keep")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--no_prefix_caching", action="store_true",
                   help="vLLM without prefix caching (DeepSeek's MLA attention crashes with it in vLLM 0.8.5's V1 engine)")
    return p.parse_args()


def render(tok, content: str) -> str:
    try:
        return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except (TypeError, ValueError):
        return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)


def grade_all(items: list[tuple[dict, str]], workers: int) -> list[float]:
    """The pipeline's grading rule on every (record, answer); code and math run in subprocesses, so threads help."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(lambda it: graded_value(*it), items))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args.output.mkdir(parents=True, exist_ok=True)
    records = [r for i, r in enumerate(JsonlDataset(args.records).records) if i % args.num_shards == args.shard_index]
    if args.sources:
        keep = set(args.sources.split(","))
        records = [r for r in records if r.get("source") in keep]
    if args.max_examples:
        records = records[: args.max_examples]
    chosen = [i for i, r in enumerate(records) if is_misled(r, args.model, args.mislead_rate, args.mislead_seed)]
    temps = [float(x) for x in args.temperatures.split(",")] or [0.2]
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=args.trust_remote_code)
    budget = lambda r: 4096 if args.reasoning else MAX_TOKENS.get(task_type_of(r), 512)
    answer_of = (lambda t: strip_thinking(t)) if args.reasoning else (lambda t: t)

    print(f"[adv-peers] {args.model}: {len(records)} records (shard {args.shard_index}/{args.num_shards}), "
          f"{len(chosen)} to answer adversarially, up to {args.max_attempts} attempts at temperatures {temps}", flush=True)
    t0 = time.time()
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enable_prefix_caching=not args.no_prefix_caching,
              trust_remote_code=args.trust_remote_code, enforce_eager=args.enforce_eager, seed=0)

    # per event: the accepted answer, or the best attempt so far and why it was not accepted
    done: dict[int, dict] = {}
    # "reasons" = the faults of the latest attempt (what the next prompt complains about);
    # "best" / "best_reasons" = the attempt kept for the forcing pass and its own faults
    state: dict[int, dict] = {i: {"reasons": [], "attempts": 0, "best": None, "best_value": None, "best_reasons": []}
                              for i in chosen}
    pending = list(chosen)
    attempt_stats: list[dict] = []
    attempts_total = max(1, args.max_attempts)
    for attempt in range(attempts_total):
        if not pending:
            break
        last = attempt == attempts_total - 1     # on the last attempt a soft fault no longer costs the answer
        temp = temps[min(attempt, len(temps) - 1)]
        # a peer that keeps solving the problem correctly is given the wrong value to arrive at -- a derivation that
        # reaches it is misleading, where a conclusion rewritten afterwards contradicts the lines above it
        targets = {i: plausible_wrong_number(records[i], answer_of(state[i]["best"]))
                   for i in pending
                   if last and task_type_of(records[i]) == "math" and "still_correct" in state[i]["best_reasons"] and state[i]["best"]}
        prompts = [render(tok, misleading_prompt(records[i], attempt=attempt, complaints=state[i]["reasons"],
                                                 target_answer=targets.get(i))) for i in pending]
        params = [SamplingParams(temperature=temp, top_p=0.95, max_tokens=budget(records[i]), seed=attempt) for i in pending]
        outs = llm.generate(prompts, params, use_tqdm=True)
        texts = [o.outputs[0].text for o in outs]
        values = grade_all([(records[i], answer_of(t)) for i, t in zip(pending, texts)], args.grade_workers)
        nxt, accepted_now = [], 0
        reasons_now: collections.Counter = collections.Counter()
        for i, text, value in zip(pending, texts, values):
            st = state[i]
            st["attempts"] = attempt + 1
            verdict = accept(records[i], answer_of(text), value=value, strict=not last)
            if verdict.ok:
                done[i] = {"response": text, "target": value, "attempts": attempt + 1, "reasons": [],
                           "soft": verdict.soft, "forced": None}
                accepted_now += 1
                continue
            reasons_now.update(verdict.reasons)
            # keep the attempt that is closest to usable: one that fails only by being correct can still be forced
            only_correct = lambda rs: set(rs) <= {"still_correct"}   # fails only by being right: can still be turned around
            better = st["best"] is None or (only_correct(verdict.reasons) and not only_correct(st["best_reasons"]))
            if better:
                st["best"], st["best_value"], st["best_reasons"] = text, value, list(verdict.reasons)
            st["reasons"] = verdict.reasons
            nxt.append(i)
        attempt_stats.append({"attempt": attempt, "temperature": temp, "generated": len(pending),
                              "accepted": accepted_now, "given_a_target": len(targets), "reasons": dict(reasons_now)})
        print(f"[adv-peers] attempt {attempt} (T={temp}): {accepted_now}/{len(pending)} accepted, "
              f"remaining {len(nxt)}, reasons {dict(reasons_now)}", flush=True)
        pending = nxt
    del llm

    # true last resort: rewrite the conclusion of answers that are still correct, then re-grade and re-check.  Such an
    # answer argues for one value and concludes another, so it is counted apart and can be dropped when the stream is built
    forced_ok = 0
    if pending and not args.no_force:
        cand = [i for i in pending if task_type_of(records[i]) in FORCEABLE and state[i]["best"] is not None]
        rewritten = {}
        for i in cand:
            text, what = force_wrong(records[i], state[i]["best"])
            if what is not None:
                rewritten[i] = text
        values = grade_all([(records[i], answer_of(rewritten[i])) for i in rewritten], args.grade_workers)
        for (i, text), value in zip(rewritten.items(), values):
            verdict = accept(records[i], answer_of(text), value=value, strict=False)
            if verdict.ok:
                done[i] = {"response": text, "target": value, "attempts": state[i]["attempts"], "reasons": [],
                           "soft": verdict.soft, "forced": True}
                forced_ok += 1
        pending = [i for i in pending if i not in done]
        print(f"[adv-peers] forced conclusions: {forced_ok} of {len(rewritten)} rewritten answers became usable", flush=True)

    rows, by_task = [], collections.defaultdict(lambda: [0, 0, 0])   # task -> [n, accepted, forced]
    unusable: collections.Counter = collections.Counter()
    for idx, r in enumerate(records):
        if idx not in state:
            continue
        task = task_type_of(r)
        got = done.get(idx)
        if got is None:
            st = state[idx]
            got = {"response": st["best"] or "", "target": st["best_value"] if st["best_value"] is not None else 0.0,
                   "attempts": st["attempts"], "reasons": st["best_reasons"], "soft": [], "forced": None}
            unusable.update(st["best_reasons"] or ["empty"])
        value = float(got["target"])
        rows.append({"id": r.get("id"), "source": r.get("source"), "task_type": task, "response": got["response"],
                     "target": value, "correct": int(round(value)), "misled": True, "accepted": idx in done,
                     "forced": bool(got["forced"]), "attempts": got["attempts"], "reasons": got["reasons"],
                     "soft": got.get("soft", [])})
        by_task[task][0] += 1
        by_task[task][1] += int(idx in done)
        by_task[task][2] += int(bool(got["forced"]))

    stem = args.records.stem
    tag = ("." + args.sources.replace(",", "_")) if args.sources else ""
    name = f"{stem}{tag}" if args.num_shards == 1 else f"{stem}{tag}.shard{args.shard_index}of{args.num_shards}"
    with (args.output / f"{name}.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = max(1, len(rows))
    summary = {
        "model": args.model, "records": str(args.records), "n": len(rows), "reasoning": args.reasoning,
        "max_attempts": args.max_attempts, "temperatures": temps, "forced_allowed": not args.no_force,
        "accepted": sum(r["accepted"] for r in rows), "accepted_pct": 100 * sum(r["accepted"] for r in rows) / n,
        "forced": sum(r["forced"] for r in rows), "still_correct_pct": 100 * sum(r["correct"] for r in rows) / n,
        "mean_attempts": sum(r["attempts"] for r in rows) / n,
        "accepted_with_soft_fault": sum(bool(r["soft"]) for r in rows if r["accepted"]),
        "by_task": {t: {"n": v[0], "accepted_pct": 100 * v[1] / max(1, v[0]), "forced": v[2]} for t, v in sorted(by_task.items())},
        "attempts": attempt_stats, "unusable_reasons": dict(unusable), "seconds": time.time() - t0,
    }
    (args.output / f"summary_{name}.json").write_text(json.dumps(summary, indent=1))
    print(f"[adv-peers] {summary['accepted']}/{len(rows)} usable ({summary['accepted_pct']:.1f}%), "
          f"{summary['forced']} forced, still correct {summary['still_correct_pct']:.1f}%, "
          f"{summary['mean_attempts']:.2f} attempts/event, {summary['seconds']:.0f}s -> {args.output / f'{name}.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
