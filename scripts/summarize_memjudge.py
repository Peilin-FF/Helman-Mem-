"""Collect memory-judge evaluations and memory simulations into one table.

  PYTHONPATH=. python scripts/summarize_memjudge.py [--root outputs/memjudge] [--memsim outputs/memsim] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt_curve(c: dict) -> str:
    cum = c.get("cumulative", {})
    cs = " ".join(f"{k}:{100 * v:.1f}" for k, v in cum.items() if k in ("250", "1000", "4000"))
    return (f"{100 * c['total']:6.2f}  {100 * c['first_half']:5.1f}->{100 * c['second_half']:5.1f} | "
            + " ".join(f"{100 * w:4.1f}" for w in c["windows"]) + (f" | cum {cs}" if cs else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs/memjudge"))
    ap.add_argument("--memsim", type=Path, default=Path("outputs/memsim"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    for p in sorted(args.root.rglob("eval_metrics.json")):
        m = json.loads(p.read_text())
        rel = str(p.parent.relative_to(args.root))
        if "_smoke" in rel:
            continue
        rows.append({"run": rel, "n": m["num_samples"], "order": m.get("order"), "memory": m.get("memory"), "online_lr": m.get("online_lr", 0.0),
                     "kappa": m.get("kappa"), "fused": m["fused"], "judge_only": m["judge_only"], "memory_only": m["memory_only"]})
    print("=== memory-judge evaluations (fused / judge-only / memory-only from the same pass)")
    for r in rows:
        print(f"{r['run']}  n={r['n']} kappa={r['kappa']:.3f} online_lr={r['online_lr']}")
        for k in ("fused", "judge_only", "memory_only"):
            print(f"    {k:12s} {fmt_curve(r[k])}")
    sims = []
    for p in sorted(args.memsim.rglob("*.json")):
        if "smoke" in p.name:
            continue
        d = json.loads(p.read_text())
        for stream, block in d.get("streams", {}).items():
            for order, res in block.items():
                if order == "probes":
                    sims.append({"file": str(p), "stream": stream, "probes": res})
                    continue
                for name, c in res["reference"].items():
                    sims.append({"file": str(p), "stream": stream, "order": order, "arm": f"ref:{name}", "route": c})
                for name, c in res["memory"].items():
                    sims.append({"file": str(p), "stream": stream, "order": order, "arm": name, "route": c["route"], "vote": c["vote"]})
    print("\n=== memory simulations (route)")
    for s in sims:
        if "probes" in s:
            print(f"{Path(s['file']).parent.name}/{Path(s['file']).stem} {s['stream']:7s} probes " + "  ".join(f"{d}: " + " ".join(f"{k}={100 * v:.1f}" for k, v in pr.items()) for d, pr in s["probes"].items()))
        else:
            print(f"{Path(s['file']).parent.name}/{Path(s['file']).stem} {s['stream']:7s} {s['order']:9s} {s['arm']:20s} {fmt_curve(s['route'])}")
    if args.json:
        args.json.write_text(json.dumps({"evals": rows, "sims": sims}, indent=1))


if __name__ == "__main__":
    main()
