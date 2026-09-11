"""The comparison table for other central-model families under the same memory (docs section 23).

    python scripts/families_table.py            full runs: outputs/gen/families/<tag>/full_<stream>6_<cond>/
    python scripts/families_table.py --smoke    smoke runs: outputs/gen/families/<tag>/smoke_indist6_<cond>/ + sanity checks

Reference row: the frozen Qwen3-4B (outputs/gen/q3_4b/base6_nothink/<stream>6_<cond>/).  Conditions: tilt = peers + memory,
peers = the same prompt without the memory, solo = the question only, tilt_swapped = the record permuted by rank.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NAMES = {"q3_4b": "Qwen3-4B (frozen; the model the record was built with)", "llama3": "Meta-Llama-3-8B (base, plain layout)",
         "llama31": "Meta-Llama-3.1-8B-Instruct", "ministral": "Ministral-8B-Instruct-2410", "qwen25": "Qwen2.5-7B-Instruct", "phi4": "phi-4 (14B)"}
CONDS = ("tilt", "peers", "solo", "tilt_swapped")
FAM = Path("outputs/gen/families")
REF = Path("outputs/gen/q3_4b/base6_nothink")


def load(path: Path) -> dict | None:
    f = path / "eval_metrics.json"
    return json.load(open(f)) if f.exists() else None


def pct(m: dict | None, key: str = "accuracy") -> str:
    return "  -  " if m is None else f"{100 * m[key]:5.1f}"


def full_table(tags: list[str]) -> str:
    lines = ["| central model | in-dist: peers + memory | peers | question only | swapped | OOD: peers + memory | peers | question only | swapped | tilt − peers (in / OOD) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag in tags:
        row = {}
        for st in ("indist", "oodfull"):
            for c in CONDS:
                row[(st, c)] = load(REF / f"{st}6_{c}") if tag == "q3_4b" else load(FAM / tag / f"full_{st}6_{c}")
        if all(v is None for v in row.values()):
            continue
        d = []
        for st in ("indist", "oodfull"):
            a, b = row[(st, "tilt")], row[(st, "peers")]
            d.append(f"{100 * (a['accuracy'] - b['accuracy']):+.1f}" if a and b else "-")
        cells = [pct(row[(st, c)]) for st in ("indist", "oodfull") for c in CONDS]
        lines.append(f"| {NAMES.get(tag, tag)} | " + " | ".join(cells) + f" | {d[0]} / {d[1]} |")
    return "\n".join(lines)


def by_task_table(tags: list[str]) -> str:
    out = []
    for tag in tags:
        for st in ("indist", "oodfull"):
            ms = {c: (load(REF / f"{st}6_{c}") if tag == "q3_4b" else load(FAM / tag / f"full_{st}6_{c}")) for c in CONDS}
            if all(v is None for v in ms.values()):
                continue
            tasks = sorted({t for m in ms.values() if m for t in m["by_task"]})
            out.append(f"{NAMES.get(tag, tag)}, {st}: " + "; ".join(
                f"{t}: " + "/".join(f"{100 * ms[c]['by_task'][t]:.1f}" if ms[c] and t in ms[c]["by_task"] else "-" for c in CONDS) for t in tasks)
                + "   (tilt/peers/solo/swapped)")
    return "\n".join(out)


def smoke_table(tags: list[str]) -> str:
    """Per model: accuracy in the four conditions on the smoke slice, and the checks that the tilt actually reached the engine:
    the number of tilted prompts (from the evaluator's log), whether vLLM took the bias (Triton) backend, and how many
    generations differ between tilt and peers, and between tilt and the swapped record."""
    lines = ["| model | n | tilt | peers | solo | swapped | prompts tilted | bias backend | gens differ tilt vs peers | tilt vs swapped | note |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    for tag in tags:
        ms = {c: load(FAM / tag / f"smoke_indist6_{c}") for c in CONDS}
        if all(v is None for v in ms.values()):
            continue
        n = next((m["num_samples"] for m in ms.values() if m), 0)
        log = Path("logs") / f"fam_{tag}_smoke_indist_tilt.out"
        tilted, backend, note = "-", "-", ""
        if log.exists():
            txt = log.read_text(errors="replace")
            m = re.search(r"(\d+)/(\d+) prompts tilted", txt)
            tilted = f"{m.group(1)}/{m.group(2)}" if m else "not logged"
            backend = "yes" if re.search(r"TRITON_ATTN_VLLM_V1|Triton", txt) else "NO"
            m2 = re.search(r"\[gen-eval\] central model .*?: (.*?);", txt)
            note = m2.group(1) if m2 else ""
        def differ(a: str, b: str) -> str:
            fa, fb = FAM / tag / f"smoke_indist6_{a}" / "generations.jsonl", FAM / tag / f"smoke_indist6_{b}" / "generations.jsonl"
            if not (fa.exists() and fb.exists()):
                return "-"
            ga = [json.loads(l)["generation"] for l in open(fa)]; gb = [json.loads(l)["generation"] for l in open(fb)]
            k = min(len(ga), len(gb))
            return f"{sum(x != y for x, y in zip(ga[:k], gb[:k]))}/{k}"
        lines.append(f"| {NAMES.get(tag, tag)} | {n} | {pct(ms['tilt'])} | {pct(ms['peers'])} | {pct(ms['solo'])} | {pct(ms['tilt_swapped'])} | {tilted} | {backend} | "
                     f"{differ('tilt', 'peers')} | {differ('tilt', 'tilt_swapped')} | {note} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--by_task", action="store_true")
    ap.add_argument("--out", default=None, help="write the table here as well (default outputs/gen/families/table[_smoke].md)")
    args = ap.parse_args()
    tags = ["q3_4b"] + sorted(p.name for p in FAM.iterdir() if p.is_dir()) if FAM.exists() else ["q3_4b"]
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
