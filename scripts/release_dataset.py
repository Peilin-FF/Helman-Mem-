"""Package the six-peer streams for the GitHub repository (datasets/, gzip; no Hugging Face involved).

    data/mixed_train_big6/train.jsonl   17,709 events  GSM8K + SQuAD + APPS, six peers      (training stream: fits the addresses)
    data/indist6/test.jsonl              4,319 events  GSM8K + SQuAD + APPS test sets        (in-distribution stream)
    data/ood6/test.jsonl                17,403 events  PIQA, MMLU, OpenBookQA, SciQ, BBH, SuperGLUE (the OOD stream)

    PYTHONPATH=. python scripts/release_dataset.py            # writes datasets/<stream>.jsonl.gz + datasets/manifest.json
    bash datasets/unpack.sh                                   # on another machine: restores data/<dir>/<file>.jsonl

The records are copied verbatim except that peer_metadata[*].model, which holds local paths in some streams, is rewritten
to the Hugging Face model id.  gzip keeps every file under GitHub's 100 MB limit (26 + 90 + 5 MB).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import time
from pathlib import Path

STREAMS = [   # (path under data/, archive name, description)
    ("mixed_train_big6/train.jsonl", "mixed_train_big6.train.jsonl.gz", "training stream: GSM8K, SQuAD and APPS, six peers (fits the record's addresses)"),
    ("indist6/test.jsonl", "indist6.test.jsonl.gz", "in-distribution test stream: GSM8K test, SQuAD dev and APPS test (with the hidden tests), six peers"),
    ("ood6/test.jsonl", "ood6.test.jsonl.gz", "OOD test stream: PIQA, MMLU, OpenBookQA, SciQ, BBH and SuperGLUE, six peers"),
]
HF_IDS = {"gemma-3-4b-it": "google/gemma-3-4b-it", "Phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
          "Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct", "Meta-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
          "DeepSeek-Coder-V2-Lite-Instruct": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}


def hf_id(model: str) -> str:
    name = model.rstrip("/").split("/")[-1]
    return HF_IDS.get(name, model if ("/" in model and not model.startswith("/")) else name)


def package(src: Path, dst: Path) -> dict:
    rows, sources, tasks, peers = 0, collections.Counter(), collections.Counter(), {}
    with src.open() as fin, gzip.open(dst, "wt", compresslevel=6) as fout:
        for line in fin:
            r = json.loads(line)
            for k, meta in (r.get("peer_metadata") or {}).items():
                if isinstance(meta, dict) and meta.get("model"):
                    meta["model"] = hf_id(str(meta["model"]))
                    peers.setdefault(k, meta["model"])
            rows += 1
            sources[str(r.get("source"))] += 1
            tasks[str(r.get("task_type"))] += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256()
    with dst.open("rb") as h:
        for chunk in iter(lambda: h.read(1 << 20), b""):
            digest.update(chunk)
    return {"archive": dst.name, "unpacks_to": f"data/{src.relative_to(src.parents[1])}", "bytes_gz": dst.stat().st_size, "sha256_gz": digest.hexdigest(),
            "rows": rows, "sources": dict(sorted(sources.items())), "task_types": dict(sorted(tasks.items())), "peers": dict(sorted(peers.items()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("datasets"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel, name, desc in STREAMS:
        t0 = time.time()
        e = package(args.data / rel, args.out / name)
        e["description"] = desc
        entries.append(e)
        print(f"[datasets] {name}: {e['rows']:,} rows, {e['bytes_gz'] / 1e6:.0f} MB gz, sources {e['sources']} ({time.time() - t0:.0f}s)", flush=True)
    (args.out / "manifest.json").write_text(json.dumps({"version": 3, "date": time.strftime("%Y-%m-%d"), "files": entries}, indent=1) + "\n")
    print(f"[datasets] wrote {args.out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
