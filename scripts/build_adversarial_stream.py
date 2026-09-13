"""Build a six-peer stream whose peers answer adversarially, under one regime.

The honest stream (``data/indist6/test.jsonl``) and the adversarial answers (``scripts/adversarial_peers.py``, one
directory per peer model) go in; a stream of the same events with the same six peers comes out, with the answers of the
peers the regime selects replaced by their misleading-but-relevant counterparts and every label recomputed from the
answer that is actually in the stream.  Nothing else changes: the events, their order, the peer identities and the
number of peers are those of the honest stream, so the two runs differ only in what the peers said.

    PYTHONPATH=. python scripts/build_adversarial_stream.py --base data/indist6/test.jsonl \
        --adv outputs/peer_adv/indist6 --config training/configs/adversarial.yaml --regime p050 \
        --out data/indist6_adv_p050/test.jsonl

The **poison ratio is what is asked for**: an exact fraction regime takes events in a fixed per-peer order from those
that have a usable adversarial answer until the requested share of the stream is poisoned, so 0%, 25%, 50% and 100%
mean 0%, 25%, 50% and as much as the peers could be made to get wrong.  The realised ratio is reported per peer.

Every record's ``peer_metadata[peer_k]`` gains ``misled`` (this answer is adversarial), ``adversarial_forced`` (its
conclusion was rewritten) and ``regime``; ``manifest.json`` beside the output holds the regime, the requested and
realised ratios, the per-peer accuracy before and after, and the answer-length comparison that says whether the
adversarial answers still look like answers.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
from pathlib import Path

import yaml

from feedback_state.adversarial import Regime, record_positions, regimes_from_config


def peer_text(response: str) -> str:
    """The peer-block text of a response, exactly as scripts/merge_peers.py writes it (reasoning models included)."""
    text = str(response or "")
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text or len(text) > 3000:
        return text[:3000].rstrip() + "\n[... reasoning cut off, no final answer]"
    return text


def load_adversarial(root: Path, split: str) -> dict[str, dict[str, dict]]:
    """model name -> {event id -> row}, from <root>/<model>/<split>*.jsonl (shards included)."""
    out: dict[str, dict[str, dict]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        rows: dict[str, dict] = {}
        for f in sorted(glob.glob(str(d / f"{split}.jsonl"))) + sorted(glob.glob(str(d / f"{split}.shard*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                rows[str(r["id"])] = r
        if rows:
            out[d.name] = rows
    return out


def scan(base: Path, limit: int | None = None) -> tuple[list[dict], list[str], list[str]]:
    """One pass over the stream: the small per-event facts (id, task, honest labels), never the answers themselves."""
    events, keys, names = [], [], []
    for i, line in enumerate(base.open()):
        if limit is not None and i >= limit:
            break
        rec = json.loads(line)
        if not keys:
            keys = sorted(rec.get("peer_responses", {}))
            names = [str(rec.get("peer_metadata", {}).get(k, {}).get("model", k)).split("/")[-1] for k in keys]
        soft = rec.get("peer_correct") or {}
        hard = rec.get("correctness_by_peer") or {}
        events.append({"id": str(rec["id"]), "task_type": str(rec.get("task_type") or ""),
                       "honest": [int(round(float(soft.get(k, hard.get(k, 0))))) for k in keys]})
    return events, keys, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="the honest six-peer stream")
    ap.add_argument("--adv", type=Path, required=True, help="directory with one sub-directory of adversarial answers per peer model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("training/configs/adversarial.yaml"))
    ap.add_argument("--regime", required=True, help="a regime named in the config (regimes: or sweep:)")
    ap.add_argument("--order", default="shuffled0", help="the stream order the record will walk (only the flip regime depends on it)")
    ap.add_argument("--drop_forced", action="store_true", help="do not use answers whose conclusion was rewritten by force_wrong")
    ap.add_argument("--split", default=None, help="the answer files' stem (default: the base file's stem)")
    ap.add_argument("--limit", type=int, default=None, help="keep only the first N events (the smoke runs, where only that many were answered)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) if args.config.exists() else {}
    regimes = regimes_from_config(cfg, [args.regime])   # p030 / k2 work without being in the config
    if args.regime not in regimes:
        raise SystemExit(f"unknown regime {args.regime!r}; the config knows {sorted(regimes)}")
    regime: Regime = regimes[args.regime]
    split = args.split or args.base.stem
    adv = load_adversarial(args.adv, split)
    if not adv:
        raise SystemExit(f"no adversarial answers under {args.adv} (expected <model>/{split}*.jsonl)")

    events, keys, names = scan(args.base, args.limit)
    n = len(events)
    positions = record_positions(n, args.order)
    print(f"[adv-stream] {args.base} ({n} events), peers " + ", ".join(f"{i}:{m}" for i, m in enumerate(names)))
    print(f"[adv-stream] regime {regime.name}: {regime.describe()}" + (f" -- {regime.note}" if regime.note else ""))
    missing = [m for i, m in enumerate(names) if regime.covers(i) and m not in adv]
    if missing:
        print(f"[adv-stream] WARNING no adversarial answers for {missing}: those peers stay honest")

    usable_row = {}
    masks = {}
    for p, m in enumerate(names):
        rows = adv.get(m, {})
        usable = [bool(r and r.get("accepted") and (not args.drop_forced or not r.get("forced")))
                  for r in (rows.get(e["id"]) for e in events)]
        usable_row[m] = usable
        if regime.kind != "count":
            masks[m] = regime.mask(p, events, positions=positions, honest=[e["honest"][p] for e in events], available=usable)
    if regime.kind == "count":   # one joint choice per event: exactly `count` misleading peers
        joint = regime.joint_masks(events, [usable_row[m] for m in names])
        masks = {m: joint[p] for p, m in enumerate(names)}

    stat = {m: {"misled": 0, "forced": 0, "unavailable": 0, "selected": 0} for m in names}
    acc = {m: [0, 0, 0] for m in names}       # [events, honest right, right in the stream]
    lens = {m: ([], []) for m in names}       # honest / adversarial answer lengths
    by_task: dict = collections.defaultdict(lambda: collections.Counter())
    per_event = collections.Counter()          # how many misleading peers each event ended up with
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f, args.base.open() as src:
        for idx, line in enumerate(src):
            if args.limit is not None and idx >= args.limit:
                break
            rec = json.loads(line)
            rid, task = str(rec["id"]), str(rec.get("task_type") or "")
            for p, (k, m) in enumerate(zip(keys, names)):
                honest_right = events[idx]["honest"][p]
                acc[m][0] += 1
                acc[m][1] += honest_right
                lens[m][0].append(len(str(rec["peer_responses"].get(k, ""))))
                meta = rec.setdefault("peer_metadata", {}).setdefault(k, {})
                meta.update({"regime": regime.name, "misled": False, "adversarial_forced": False})
                want = masks[m][idx]
                stat[m]["selected"] += int(want)
                row = adv.get(m, {}).get(rid) if want else None
                if want and not usable_row[m][idx]:
                    stat[m]["unavailable"] += 1
                if want and usable_row[m][idx] and row is not None:
                    text = peer_text(row["response"])
                    value = float(row.get("target", row.get("correct", 0)))
                    rec["peer_responses"][k] = text
                    rec.setdefault("peer_correct", {})[k] = value
                    if "correctness_by_peer" in rec:
                        rec["correctness_by_peer"][k] = int(round(value))
                    meta["misled"] = True
                    meta["adversarial_forced"] = bool(row.get("forced"))
                    meta["adversarial_attempts"] = int(row.get("attempts", 0))
                    stat[m]["misled"] += 1
                    stat[m]["forced"] += int(bool(row.get("forced")))
                    lens[m][1].append(len(text))
                    by_task[task]["misled"] += 1
                acc[m][2] += int(round(float(rec.get("peer_correct", {}).get(k, honest_right))))
                by_task[task]["peer_events"] += 1
            per_event[sum(bool(rec["peer_metadata"][k].get("misled")) for k in keys)] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    mean = lambda xs: float(statistics.fmean(xs)) if xs else None
    peers_manifest = {}
    for i, m in enumerate(names):
        peers_manifest[m] = {
            "index": i, "covered": regime.covers(i), "selected": stat[m]["selected"], "misled": stat[m]["misled"],
            "forced": stat[m]["forced"], "unavailable": stat[m]["unavailable"],
            "usable_answers": int(sum(usable_row[m])),
            "requested_ratio": 100 * (0.0 if not regime.covers(i) else
                                      float(regime.rate) if regime.kind == "fraction" else
                                      min(1.0, regime.count / max(1, sum(regime.covers(q) for q in range(len(names)))))
                                      if regime.kind == "count" else stat[m]["selected"] / max(1, n)),
            "realised_ratio": 100 * stat[m]["misled"] / max(1, n),
            "accuracy_honest": 100 * acc[m][1] / max(1, acc[m][0]),
            "accuracy_in_stream": 100 * acc[m][2] / max(1, acc[m][0]),
            "mean_chars_honest": mean(lens[m][0]), "mean_chars_adversarial": mean(lens[m][1]),
        }
    manifest = {
        "base": str(args.base), "out": str(args.out), "adversarial_answers": str(args.adv), "events": n,
        "order": args.order, "drop_forced": args.drop_forced,
        "regime": {"name": regime.name, "kind": regime.kind, "rate": regime.rate, "exact": regime.exact, "count": regime.count,
                   "at": regime.at, "peers": list(regime.peers) if regime.peers is not None else "all",
                   "description": regime.describe(), "note": regime.note},
        "poisoned_answers": sum(v["misled"] for v in stat.values()),
        "poison_ratio": 100 * sum(v["misled"] for v in stat.values()) / max(1, n * len(names)),
        "events_by_misleading_peers": {str(k): per_event[k] for k in range(len(names) + 1)},
        "peers": peers_manifest, "by_task": {t: dict(c) for t, c in sorted(by_task.items())},
    }
    (args.out.parent / "manifest.json").write_text(json.dumps(manifest, indent=1))
    short = sum(v["unavailable"] for v in stat.values())
    print(f"[adv-stream] {manifest['poisoned_answers']}/{n * len(names)} peer answers replaced "
          f"({manifest['poison_ratio']:.1f}% of the stream, {sum(v['forced'] for v in stat.values())} forced, "
          f"{short} selected without a usable answer) -> {args.out}")
    print("[adv-stream] events by number of misleading peers: "
          + ", ".join(f"{k}: {per_event[k]}" for k in range(len(names) + 1)))
    for i, m in enumerate(names):
        v = peers_manifest[m]
        print(f"   peer_{i} {m:34s} ratio {v['requested_ratio']:5.1f}% asked -> {v['realised_ratio']:5.1f}% reached  "
              f"accuracy {v['accuracy_honest']:5.1f}% -> {v['accuracy_in_stream']:5.1f}%  "
              f"chars {int(v['mean_chars_honest'] or 0):5d} -> {int(v['mean_chars_adversarial'] or 0):5d}")


if __name__ == "__main__":
    main()
