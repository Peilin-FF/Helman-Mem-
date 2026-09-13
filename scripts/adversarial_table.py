"""The robustness table: what the memory is worth when the peers are misleading (README_adversarial.md).

    python scripts/adversarial_table.py                     every regime in training/configs/adversarial.yaml's run list
    python scripts/adversarial_table.py --regimes all100    one regime
    python scripts/adversarial_table.py --no-probe          skip the (slow) pass over the prompt files

Three tables, written to outputs/gen/adversarial/table.md and printed:

  1. accuracy    per regime and stream, the central model's accuracy with peers + memory (the tilt), with the peers
                 alone, with the question alone, and with the record permuted by rank (the control), against the same
                 model's honest rows.  ``tilt - peers`` is the memory's contribution, ``peers - solo`` the (now
                 negative) value of reading the peers at all.
  2. record      how well the record still separates right from wrong peer answers (AUC, favourite right on mixed
                 events), what probability it gives a misleading answer against an honest one, and how often its
                 favourite is a misleading answer -- the memory's side of the experiment.
  3. peers       per peer: accuracy before and after, how many answers were replaced, how many of the adversarial
                 attempts were usable, and the answer lengths, which say whether the misleading answers still look
                 like answers.
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import yaml

from feedback_state.adversarial import regimes_from_config

CONDS = ("tilt", "peers", "solo", "swap")
STREAM_DIR = {"indist6": "indist", "ood6": "oodfull"}


def auc(pairs: list[tuple[float, int]]) -> float:
    """P(score of a positive > score of a negative), ties at half (the same estimator as record_quality.py)."""
    pos = sorted(p for p, y in pairs if y == 1)
    neg = sorted(p for p, y in pairs if y == 0)
    if not pos or not neg:
        return float("nan")
    return sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p))
               for p in pos) / (len(pos) * len(neg))


def load(path: Path) -> dict | None:
    f = path / "eval_metrics.json"
    return json.load(open(f)) if f.exists() else None


def pct(m: dict | None, key: str = "accuracy") -> str:
    return "  -  " if m is None else f"{100 * m[key]:5.1f}"


def delta(a: dict | None, b: dict | None) -> str:
    return "-" if not (a and b) else f"{100 * (a['accuracy'] - b['accuracy']):+.1f}"


def record_quality(gen_dir: Path, stream: str) -> tuple[str, dict | None]:
    f = gen_dir / f"record_{stream}.json"
    if not f.exists():
        return "-", None
    res = json.load(open(f))
    k = "memory" if "memory" in res else next((k for k, v in res.items() if isinstance(v, dict) and "auc" in v), None)
    if k is None:
        return "-", res
    return f"{res[k]['auc']:.2f} / {res[k]['favourite_acc_mixed']:.0f}%", res


def misled_probe(prompts: Path, stream_jsonl: Path, cache: Path) -> dict | None:
    """What the record thinks of the misleading answers: mean estimate, AUC against 'is misleading', favourite misled.

    Read once and cached: the prompt files are hundreds of megabytes.
    """
    if cache.exists():
        return json.load(open(cache))
    if not (prompts.exists() and stream_jsonl.exists()):
        return None
    misled: dict[str, list[bool]] = {}
    for line in stream_jsonl.open():
        rec = json.loads(line)
        meta = rec.get("peer_metadata", {})
        keys = sorted(rec.get("peer_responses", {}))
        misled[str(rec["id"])] = [bool(meta.get(k, {}).get("misled")) for k in keys]
    pairs: list[tuple[float, int]] = []
    sums = [0.0, 0], [0.0, 0]        # [sum, n] for honest / misleading answers
    fav_misled = [0, 0]
    n_rows = 0
    for line in prompts.open():
        row = json.loads(line)
        flags = misled.get(str(row["id"]))
        if flags is None:
            continue
        n_rows += 1
        probs = [float(p) for p in row["memory_prob"]]
        slot_misled = [bool(flags[int(p)]) if int(p) < len(flags) else False for p in row["peer_order"]]
        for p, m in zip(probs, slot_misled):
            pairs.append((p, int(m)))
            sums[int(m)][0] += p
            sums[int(m)][1] += 1
        if any(slot_misled):
            best = max(range(len(probs)), key=lambda s: probs[s])
            fav_misled[0] += int(slot_misled[best])
            fav_misled[1] += 1
    if not n_rows:
        return None
    res = {"events": n_rows,
           "mean_prob_honest": sums[0][0] / max(1, sums[0][1]), "n_honest": sums[0][1],
           "mean_prob_misled": sums[1][0] / max(1, sums[1][1]), "n_misled": sums[1][1],
           "auc_misled": 1 - auc(pairs),          # P(an honest answer scores above a misleading one)
           "favourite_misled_pct": 100 * fav_misled[0] / max(1, fav_misled[1]), "n_events_with_misled": fav_misled[1]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/configs/adversarial.yaml")
    ap.add_argument("--tag", default=None, help="central model tag (default: the config's central.tag)")
    ap.add_argument("--regimes", nargs="*", help="regime directory names (default: the config's run list)")
    ap.add_argument("--no-probe", action="store_true", help="skip the pass over the prompt files")
    ap.add_argument("--peers-root", default=None, help="where the adversarial answers live (default: the config's generation.root)")
    ap.add_argument("--smoke", action="store_true", help="label the table as a smoke run")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out, mem = cfg["outputs"], cfg["memory"]
    tag = args.tag or str(cfg["central"]["tag"])
    known = regimes_from_config(cfg)
    regimes = args.regimes or [r for entry in cfg["run"] for r in (sorted(k for k in known if k.startswith(str(cfg.get("sweep", {}).get("prefix", "p")))) if entry == "sweep" else [entry])]
    tests = [s for s in cfg["streams"]["test"]]
    evals, gen_dir = Path(out["evals"]) / tag, Path(out["prompts"]) / tag
    honest = Path(out["honest"]) if out.get("honest") and not args.smoke else None   # a 48-event smoke row next to the real honest rows would only mislead
    data_root = Path(out.get("streams", "data"))
    conds = [c for c in CONDS if c in cfg["evaluation"]["conditions"]]

    head = ["| stream · peers | " + " | ".join(f"{c}" for c in conds) + " | tilt − peers | tilt − solo | peers − solo |",
            "|---|" + "---:|" * (len(conds) + 3)]
    rows = list(head)
    for st in tests:
        h = {c: load(honest / f"{STREAM_DIR[st]}6_{c}") for c in conds} if honest else {c: None for c in conds}
        if any(h.values()):
            rows.append(f"| {st} · honest peers | " + " | ".join(pct(h[c]) for c in conds) +
                        f" | {delta(h.get('tilt'), h.get('peers'))} | {delta(h.get('tilt'), h.get('solo'))} | {delta(h.get('peers'), h.get('solo'))} |")
        for r in regimes:
            m = {c: load(evals / r / f"{STREAM_DIR[st]}6_{c}") for c in conds}
            if not any(m.values()):
                continue
            rows.append(f"| {st} · {r} | " + " | ".join(pct(m[c]) for c in conds) +
                        f" | {delta(m.get('tilt'), m.get('peers'))} | {delta(m.get('tilt'), m.get('solo'))} | {delta(m.get('peers'), m.get('solo'))} |")
    accuracy = "\n".join(rows)

    rows = ["| stream · peers | record AUC / favourite right | P(correct) on honest answers | on misleading answers | AUC honest vs misleading | favourite is misleading |",
            "|---|---:|---:|---:|---:|---:|"]
    for st in tests:
        for r in regimes:
            name = f"{st}_adv_{r}"
            rq, _ = record_quality(gen_dir, name)
            probe = None
            if not args.no_probe:
                probe = misled_probe(gen_dir / f"prompts_{name}_probe.jsonl",
                                     data_root / f"{name}/{'train' if 'train' in st else 'test'}.jsonl",
                                     gen_dir / f"misled_{name}.json")
            if rq == "-" and probe is None:
                continue
            cells = ["-", "-", "-", "-"] if probe is None else [
                f"{probe['mean_prob_honest']:.2f}", f"{probe['mean_prob_misled']:.2f}",
                f"{probe['auc_misled']:.2f}", f"{probe['favourite_misled_pct']:.0f}%"]
            rows.append(f"| {st} · {r} | {rq} | " + " | ".join(cells) + " |")
    record = "\n".join(rows)

    rows = ["| stream · peers | peer | accuracy honest | in the stream | answers replaced | forced | usable attempts | chars honest → adversarial |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for st in tests:
        for r in regimes:
            man = data_root / f"{st}_adv_{r}" / "manifest.json"
            if not man.exists():
                continue
            m = json.load(open(man))
            for name, v in sorted(m["peers"].items(), key=lambda kv: kv[1]["index"]):
                s = Path(args.peers_root or cfg["generation"].get("root", "outputs/peer_adv")) / st / name
                summaries = sorted(s.glob("summary_*.json"))
                acc = [json.load(open(f)) for f in summaries]
                usable = f"{sum(a['accepted'] for a in acc) / max(1, sum(a['n'] for a in acc)) * 100:.0f}%" if acc else "-"
                rows.append(f"| {st} · {r} | peer_{v['index']} {name} | {v['accuracy_honest']:.1f} | {v['accuracy_in_stream']:.1f} | "
                            f"{v['misled']} | {v['forced']} | {usable} | "
                            f"{int(v['mean_chars_honest'] or 0)} → {int(v['mean_chars_adversarial'] or 0)} |")
    peers = "\n".join(rows)

    title = f"# Misleading peers: the memory under adversarial peers ({tag}{', smoke' if args.smoke else ''})\n"
    notes = "\n".join(f"- **{r}**: {known[r.replace('_smoke', '')].note or known[r.replace('_smoke', '')].describe()}"
                      for r in regimes if r.replace("_smoke", "") in known)
    text = (f"{title}\n{notes}\n\n## Accuracy\n\n{accuracy}\n\n"
            f"tilt = peers + memory, peers = the same prompt without the memory, solo = the question alone, "
            f"swap = the tilt with the record permuted by rank (control).\n\n"
            f"## What the record makes of the misleading answers\n\n{record}\n\n"
            f"AUC honest vs misleading is the probability that the record scores an honest answer above a misleading "
            f"one on the same event; 0.50 is blind, 1.00 is perfect.\n\n## The peers\n\n{peers}\n")
    dest = args.out or (Path(out["evals"]) / ("table_smoke.md" if args.smoke else "table.md"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    print(text)
    print(f"[adv-table] written to {dest}")


if __name__ == "__main__":
    main()
