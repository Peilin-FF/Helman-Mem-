"""Join the shard directories of a sharded evaluation (evaluate_memory_generator.py --shard k/N) into one result.

    PYTHONPATH=. python scripts/merge_eval_shards.py --out outputs/gen/families/q35_9b/full_oodfull6_tilt

Reads <out>/shard*/generations.jsonl, restores the stream order, recomputes eval_metrics.json exactly as the evaluator
does (accuracy, per-task accuracy, the accuracy curve along the stream, the oracle and majority references, the verdict
scores), and writes <out>/generations.jsonl and <out>/eval_metrics.json.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def curve(hits: np.ndarray, windows: int) -> dict:
    n = len(hits); edges = np.linspace(0, n, windows + 1).astype(int)
    return {"total": float(hits.mean()), "n": int(n),
            "windows": [float(hits[a:b].mean()) for a, b in zip(edges[:-1], edges[1:]) if b > a],
            "first_half": float(hits[: n // 2].mean()), "second_half": float(hits[n // 2 :].mean()),
            "cumulative": {str(k): float(hits[:k].mean()) for k in (250, 500, 1000, 2000, 4000, 8000, 16000) if k <= n}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=10)
    args = ap.parse_args()
    shards = sorted(glob.glob(str(args.out / "shard*" / "generations.jsonl")))
    if not shards:
        raise SystemExit(f"no shard*/generations.jsonl under {args.out}")
    rows = [json.loads(l) for f in shards for l in open(f)]
    rows.sort(key=lambda r: int(r["pos"]))
    meta = json.load(open(Path(shards[0]).parent / "eval_metrics.json"))
    hits = np.array([int(r["correct"]) for r in rows])
    oracle = np.array([int(max(r["peer_correct"])) for r in rows])
    majority = np.array([int(sum(r["peer_correct"]) * 2 > len(r["peer_correct"])) for r in rows])
    metrics = {k: meta.get(k) for k in ("mode", "thinking", "max_new_tokens", "engine", "checkpoint", "central_model", "prompts")}
    metrics.update({"accuracy": float(hits.mean()), "num_samples": int(len(hits)), "shards": len(shards),
                    "generated": curve(hits, args.windows), "oracle_any_peer": curve(oracle, args.windows),
                    "peer_majority_correct": curve(majority, args.windows),
                    "by_task": {t: float(np.mean([int(r["correct"]) for r in rows if r["task_type"] == t])) for t in sorted({r["task_type"] for r in rows})}})
    if any(r.get("verdict") is not None for r in rows) or "verdict" in meta:
        from tests.experiments.common.evaluate_memory_generator import verdict_metrics
        metrics["verdict"] = verdict_metrics([(r.get("verdict"), [float(x) for x in (r.get("memory_prob") or [])], [int(x) for x in r["peer_correct"]]) for r in rows])
    with (args.out / "generations.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (args.out / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[merge] {args.out}: {len(shards)} shards, {len(rows)} events, accuracy {100 * metrics['accuracy']:.2f}, by_task "
          f"{ {k: round(100 * v, 1) for k, v in metrics['by_task'].items()} }" + (f", verdict {metrics['verdict']}" if "verdict" in metrics else ""))


if __name__ == "__main__":
    main()
