"""Streams: derive a stream from an existing one, labels recomputed from the answers actually in it.

    add       append new peers (their pipeline.peers answers) as peer_<k>; events missing an answer are dropped
    replace   put a peer's misleading answers in place of its honest ones, on the events a regime selects

    PYTHONPATH=. python -m pipeline.streams add --base data/indist6/test.jsonl \
        --answers outputs/peers/indist6/honest --peer Mistral-7B-Instruct-v0.3 --out data/indist7/test.jsonl
    PYTHONPATH=. python -m pipeline.streams replace --base data/indist6/test.jsonl \
        --answers outputs/peers/indist6/misleading --regime p050 --out data/indist6_adv_p050/test.jsonl

A regime is a name the ad-hoc forms cover (pNNN = that share of every peer's answers, exactly; kN = N misleading peers on
every event) or a JSON spec (--regime-spec, see feedback_state.adversarial.Regime). Besides the stream, manifest.json
records what was built: requested and realised ratio per peer, forced and unavailable counts, accuracy before and after,
answer lengths.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
from pathlib import Path


def peer_text(response: str) -> str:
    """The peer-block text of a response: the answer after a think block; an unfinished think block is cut at 3,000 chars."""
    text = str(response or "")
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text or len(text) > 3000:
        return text[:3000].rstrip() + "\n[... reasoning cut off, no final answer]"
    return text


def load_answers(directory: Path) -> tuple[dict[str, dict], dict]:
    """{event id -> answer row} and the generation summary of one peer's pipeline.peers output."""
    rows = {}
    for f in sorted(glob.glob(str(directory / "shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            rows[str(r["id"])] = r
    summaries = sorted(glob.glob(str(directory / "summary.shard*.json")))
    summary = json.load(open(summaries[0])) if summaries else {}
    return rows, summary


def cmd_add(args) -> None:
    peers = []
    for name in args.peer:
        rows, summary = load_answers(args.answers / name)
        if not rows:
            raise SystemExit(f"no answers under {args.answers / name}")
        peers.append((name, rows, summary))
        print(f"[streams] {name}: {len(rows)} answers")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    dropped = collections.Counter()
    with args.out.open("w") as f:
        for line in args.base.open():
            rec = json.loads(line)
            n_in += 1
            rid = str(rec["id"])
            missing = [name for name, rows, _ in peers if rid not in rows]
            if missing:
                dropped.update(missing)
                continue
            keys = sorted(rec["peer_responses"], key=lambda k: int(k.split("_")[1]))
            for name, rows, summary in peers:
                k = f"peer_{len(keys)}"
                r = rows[rid]
                gp = summary.get("generation_params", {})
                rec["peer_responses"][k] = peer_text(r["response"])
                rec.setdefault("peer_metadata", {})[k] = {"model": summary.get("model", name), "received_context": bool(gp.get("context", True)),
                                                          "num_samples": 1, "generation_params": gp}
                rec.setdefault("peer_correct", {})[k] = float(r.get("target", r["correct"]))
                if "correctness_by_peer" in rec:
                    rec["correctness_by_peer"][k] = int(r["correct"])
                keys.append(k)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    manifest = {"kind": "add", "base": str(args.base), "out": str(args.out), "peers": [p[0] for p in peers],
                "events_in": n_in, "events_out": n_out, "dropped_for_missing_answers": dict(dropped)}
    (args.out.parent / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[streams] {n_in} events in, {n_out} out, dropped {dict(dropped)} -> {args.out}")


def cmd_replace(args) -> None:
    from feedback_state.adversarial import Regime, adhoc_spec, load_regimes, record_positions

    spec = json.loads(args.regime_spec) if args.regime_spec else adhoc_spec(args.regime)
    if spec is None:
        raise SystemExit(f"regime {args.regime!r} is neither pNNN / kN nor given with --regime-spec")
    regime: Regime = load_regimes({args.regime: spec})[args.regime]
    events, keys, names = [], [], []
    for i, line in enumerate(args.base.open()):
        if args.limit is not None and i >= args.limit:
            break
        rec = json.loads(line)
        if not keys:
            keys = sorted(rec.get("peer_responses", {}), key=lambda k: int(k.split("_")[1]))
            names = [str(rec.get("peer_metadata", {}).get(k, {}).get("model", k)).split("/")[-1] for k in keys]
        soft, hard = rec.get("peer_correct") or {}, rec.get("correctness_by_peer") or {}
        events.append({"id": str(rec["id"]), "honest": [int(round(float(soft.get(k, hard.get(k, 0))))) for k in keys]})
    n = len(events)
    positions = record_positions(n, args.order)
    answers = {m: load_answers(args.answers / m)[0] for m in names}
    usable = {m: [bool(r and r.get("accepted") and (not args.drop_forced or not r.get("forced"))) for r in (answers[m].get(e["id"]) for e in events)]
              for m in names}
    print(f"[streams] {args.base} ({n} events), regime {regime.name}: {regime.describe()}")
    missing = [m for i, m in enumerate(names) if regime.covers(i) and not answers[m]]
    if missing:
        print(f"[streams] WARNING no misleading answers for {missing}: those peers stay honest")
    if regime.kind == "count":
        joint = regime.joint_masks(events, [usable[m] for m in names])
        masks = {m: joint[p] for p, m in enumerate(names)}
    else:
        masks = {m: regime.mask(p, events, positions=positions, honest=[e["honest"][p] for e in events], available=usable[m])
                 for p, m in enumerate(names)}
    stat = {m: collections.Counter() for m in names}
    acc = {m: [0, 0] for m in names}
    lens = {m: ([], []) for m in names}
    per_event = collections.Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f, args.base.open() as src:
        for idx, line in enumerate(src):
            if idx >= n:
                break
            rec = json.loads(line)
            for p, (k, m) in enumerate(zip(keys, names)):
                meta = rec.setdefault("peer_metadata", {}).setdefault(k, {})
                meta.update({"regime": regime.name, "misled": False, "adversarial_forced": False})
                acc[m][0] += events[idx]["honest"][p]
                lens[m][0].append(len(str(rec["peer_responses"].get(k, ""))))
                if masks[m][idx]:
                    stat[m]["selected"] += 1
                    if not usable[m][idx]:
                        stat[m]["unavailable"] += 1
                    else:
                        row = answers[m][str(rec["id"])]
                        text, value = peer_text(row["response"]), float(row.get("target", row.get("correct", 0)))
                        rec["peer_responses"][k] = text
                        rec.setdefault("peer_correct", {})[k] = value
                        if "correctness_by_peer" in rec:
                            rec["correctness_by_peer"][k] = int(round(value))
                        meta.update({"misled": True, "adversarial_forced": bool(row.get("forced")), "adversarial_attempts": int(row.get("attempts", 0))})
                        stat[m]["misled"] += 1
                        stat[m]["forced"] += int(bool(row.get("forced")))
                        lens[m][1].append(len(text))
                acc[m][1] += int(round(float(rec.get("peer_correct", {}).get(k, events[idx]["honest"][p]))))
            per_event[sum(bool(rec["peer_metadata"][k].get("misled")) for k in keys)] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    mean = lambda xs: float(statistics.fmean(xs)) if xs else None
    covered = sum(regime.covers(q) for q in range(len(names)))
    peers = {m: {"index": i, "covered": regime.covers(i), "selected": stat[m]["selected"], "misled": stat[m]["misled"],
                 "forced": stat[m]["forced"], "unavailable": stat[m]["unavailable"], "usable_answers": int(sum(usable[m])),
                 "requested_ratio": 100 * (0.0 if not regime.covers(i) else float(regime.rate) if regime.kind == "fraction"
                                           else min(1.0, regime.count / max(1, covered)) if regime.kind == "count" else stat[m]["selected"] / max(1, n)),
                 "realised_ratio": 100 * stat[m]["misled"] / max(1, n),
                 "accuracy_honest": 100 * acc[m][0] / max(1, n), "accuracy_in_stream": 100 * acc[m][1] / max(1, n),
                 "mean_chars_honest": mean(lens[m][0]), "mean_chars_misleading": mean(lens[m][1])} for i, m in enumerate(names)}
    manifest = {"kind": "replace", "base": str(args.base), "out": str(args.out), "answers": str(args.answers), "events": n,
                "order": args.order, "drop_forced": args.drop_forced,
                "regime": {"name": regime.name, "kind": regime.kind, "rate": regime.rate, "exact": regime.exact, "count": regime.count,
                           "at": regime.at, "peers": list(regime.peers) if regime.peers is not None else "all",
                           "description": regime.describe(), "note": regime.note},
                "poisoned_answers": sum(s["misled"] for s in stat.values()),
                "poison_ratio": 100 * sum(s["misled"] for s in stat.values()) / max(1, n * len(names)),
                "events_by_misleading_peers": {str(k): per_event[k] for k in range(len(names) + 1)}, "peers": peers}
    (args.out.parent / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[streams] {manifest['poisoned_answers']}/{n * len(names)} answers replaced ({manifest['poison_ratio']:.1f}%), "
          f"events by misleading peers {manifest['events_by_misleading_peers']} -> {args.out}")
    for m, v in peers.items():
        print(f"   peer_{v['index']} {m:34s} ratio {v['requested_ratio']:5.1f}% asked -> {v['realised_ratio']:5.1f}% reached, "
              f"accuracy {v['accuracy_honest']:5.1f}% -> {v['accuracy_in_stream']:5.1f}%")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--base", type=Path, required=True)
    a.add_argument("--answers", type=Path, required=True, help="the directory holding one pipeline.peers output per peer")
    a.add_argument("--peer", action="append", required=True, help="a peer directory name under --answers (repeatable)")
    a.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("replace")
    r.add_argument("--base", type=Path, required=True)
    r.add_argument("--answers", type=Path, required=True, help="the directory holding each peer's misleading answers, by model name")
    r.add_argument("--regime", required=True)
    r.add_argument("--regime-spec", default=None, help="JSON, e.g. '{\"kind\": \"fraction\", \"rate\": 1.0, \"peers\": [1, 4]}'")
    r.add_argument("--order", default="shuffled0", help="the order the record walks the stream (the flip regime depends on it)")
    r.add_argument("--drop-forced", action="store_true", help="never use an answer whose conclusion was rewritten")
    r.add_argument("--limit", type=int, default=None, help="only the first N events")
    r.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    (cmd_add if args.cmd == "add" else cmd_replace)(args)


if __name__ == "__main__":
    main()
