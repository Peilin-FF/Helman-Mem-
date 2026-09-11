"""The comparison table for other central-model families (docs section 23).

    python scripts/families_table.py            full runs: outputs/gen/families/<tag>/full_<stream>6_<cond>[_q3addr]/
    python scripts/families_table.py --smoke    smoke runs: outputs/gen/families/<tag>/smoke_indist6_<cond>[_q3addr]/ + sanity checks

Each family is tested with its OWN memory (the record's address from that family's model; launch_family_memory.sh);
directories with the suffix _q3addr hold the comparison where the same central model reads the Qwen3-4B-addressed
record instead.  Reference row: the frozen Qwen3-4B with its own record (outputs/gen/q3_4b/base6_nothink/<stream>6_<cond>/).
Conditions: tilt = peers + memory, peers = the same prompt without the memory, solo = the question only (no swapped-record
control for the families: user, 2026-09-11).  The record's own quality (AUC against the peers' verified labels, favourite right on mixed
events) comes from outputs/gen/<tag>/record_<stream>.json when scripts/record_quality.py has been run.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NAMES = {"q3_4b": "Qwen3-4B (frozen)", "llama3": "Meta-Llama-3-8B (base, plain layout)", "llama31": "Meta-Llama-3.1-8B-Instruct",
         "ministral": "Ministral-8B-Instruct-2410", "qwen25": "Qwen2.5-7B-Instruct", "phi4": "phi-4 (14B)"}
CONDS = ("tilt", "peers", "solo")
ADDRS = (("", "own record"), ("_q3addr", "Qwen3-4B record"))
FAM = Path("outputs/gen/families")
REF = Path("outputs/gen/q3_4b/base6_nothink")


def load(path: Path) -> dict | None:
    f = path / "eval_metrics.json"
    return json.load(open(f)) if f.exists() else None


def pct(m: dict | None, key: str = "accuracy") -> str:
    return "  -  " if m is None else f"{100 * m[key]:5.1f}"


def name(tag: str, sfx: str) -> str:
    base = tag[:-6] if tag.endswith("_smoke") else tag
    n = NAMES.get(base, base) + (" [smoke]" if tag.endswith("_smoke") else "")
    return n + ("" if tag == "q3_4b" else f" · {dict(ADDRS)[sfx]}")


def record_quality(tag: str, stream: str) -> str:
    """AUC / favourite-right of the family's own record on a stream, from record_quality.py's JSON."""
    base = "q3_4b" if tag == "q3_4b" else tag
    f = Path("outputs/gen") / base / f"record_{stream}.json"
    if not f.exists():
        return "-"
    res = json.load(open(f))
    k = "memory" if "memory" in res else next((k for k, v in res.items() if isinstance(v, dict) and "auc" in v), None)
    return "-" if k is None else f"{res[k]['auc']:.2f} / {res[k]['favourite_acc_mixed']:.0f}%"


def metrics(tag: str, what: str, st: str, cond: str, sfx: str) -> dict | None:
    if tag == "q3_4b":
        return load(REF / f"{st}6_{cond}") if sfx == "" else None
    return load(FAM / tag / f"{what}_{st}6_{cond}{sfx}")


def full_table(tags: list[str]) -> str:
    lines = ["| central model · record | own record AUC / fav (in, OOD) | in-dist: peers + memory | peers | question only | OOD: peers + memory | peers | question only | tilt − peers (in / OOD) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tag in tags:
        for sfx, _ in ADDRS:
            row = {(st, c): metrics(tag, "full", st, c, sfx) for st in ("indist", "oodfull") for c in CONDS}
            if all(v is None for v in row.values()):
                continue
            d = []
            for st in ("indist", "oodfull"):
                a, b = row[(st, "tilt")], row[(st, "peers")]
                d.append(f"{100 * (a['accuracy'] - b['accuracy']):+.1f}" if a and b else "-")
            rq = "-" if sfx else f"{record_quality(tag, 'indist6')}, {record_quality(tag, 'ood6')}"
            cells = [pct(row[(st, c)]) for st in ("indist", "oodfull") for c in CONDS]
            lines.append(f"| {name(tag, sfx)} | {rq} | " + " | ".join(cells) + f" | {d[0]} / {d[1]} |")
    return "\n".join(lines)


def by_task_table(tags: list[str]) -> str:
    out = []
    for tag in tags:
        for sfx, _ in ADDRS:
            for st in ("indist", "oodfull"):
                ms = {c: metrics(tag, "full", st, c, sfx) for c in CONDS}
                if all(v is None for v in ms.values()):
                    continue
                tasks = sorted({t for m in ms.values() if m for t in m["by_task"]})
                out.append(f"{name(tag, sfx)}, {st}: " + "; ".join(
                    f"{t}: " + "/".join(f"{100 * ms[c]['by_task'][t]:.1f}" if ms[c] and t in ms[c]["by_task"] else "-" for c in CONDS) for t in tasks)
                    + "   (tilt/peers/solo)")
    return "\n".join(out)


def smoke_table(tags: list[str]) -> str:
    """Per model and record: accuracy in the three conditions on the smoke slice, and the checks that the tilt reached the
    engine: tilted prompts (from the evaluator's log), the bias (Triton) backend, generations differing tilt vs peers,
    and the record's quality on the slice when it is the family's own."""
    lines = ["| model · record | n | peers + memory | peers | question only | prompts tilted | bias backend | gens differ tilt vs peers | own record AUC / fav (slice) | tokenizer / template |",
             "|---|---:|---:|---:|---:|---:|---|---:|---|---|"]
    for tag in tags:
        for sfx, _ in ADDRS:
            ms = {c: metrics(tag, "smoke", "indist", c, sfx) for c in CONDS}
            if all(v is None for v in ms.values()):
                continue
            n = next((m["num_samples"] for m in ms.values() if m), 0)
            log = Path("logs") / f"fam_{tag}_smoke_indist_tilt{sfx}.out"
            tilted, backend, note = "-", "-", ""
            if log.exists():
                txt = log.read_text(errors="replace")
                m = re.search(r"(\d+)/(\d+) prompts tilted", txt)
                tilted = f"{m.group(1)}/{m.group(2)}" if m else "not logged"
                backend = "yes" if re.search(r"TRITON_ATTN_VLLM_V1|Triton", txt) else "NO"
                m2 = re.search(r"\[gen-eval\] central model .*?: (.*?);", txt)
                note = m2.group(1) if m2 else ""

            def differ(a: str, b: str) -> str:
                fa = FAM / tag / f"smoke_indist6_{a}{sfx}" / "generations.jsonl"; fb = FAM / tag / f"smoke_indist6_{b}{sfx}" / "generations.jsonl"
                if not (fa.exists() and fb.exists()):
                    return "-"
                ga = [json.loads(l)["generation"] for l in open(fa)]; gb = [json.loads(l)["generation"] for l in open(fb)]
                k = min(len(ga), len(gb))
                return f"{sum(x != y for x, y in zip(ga[:k], gb[:k]))}/{k}"
            rq = "-" if sfx else record_quality(tag, "indist6")
            lines.append(f"| {name(tag, sfx)} | {n} | {pct(ms['tilt'])} | {pct(ms['peers'])} | {pct(ms['solo'])} | {tilted} | {backend} | "
                         f"{differ('tilt', 'peers')} | {rq} | {note} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--by_task", action="store_true")
    ap.add_argument("--out", default=None, help="write the table here as well (default outputs/gen/families/table[_smoke].md)")
    args = ap.parse_args()
    tags = ["q3_4b"] + (sorted(p.name for p in FAM.iterdir() if p.is_dir()) if FAM.exists() else [])
    text = smoke_table([t for t in tags if t != "q3_4b"]) if args.smoke else full_table(tags)
    if args.by_task and not args.smoke:
        text += "\n\n" + by_task_table(tags)
    print(text)
    out = Path(args.out) if args.out else FAM / ("table_smoke.md" if args.smoke else "table.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
