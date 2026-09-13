"""Publish the registered datasets to the Hugging Face Hub and pin the release in datasets/manifest.json.

Every dataset in configs/datasets/ with a `release:` file name is packed (gzip, reproducible) into a staging folder:
    stream, misleading   the stream file; peer_metadata[*].model is rewritten to the Hugging Face model id
    answers              the peers' generated answers, one row per (event, peer), with `peer` (the model directory) added
together with manifest.json (sha256, rows, peers' generation summaries, each misleading stream's manifest and content
digest) and the dataset card README.md. The folder is uploaded to the dataset repo, and datasets/manifest.json records
the repo, the revision it created and the files, so that `python datasets/download.py` fetches exactly this release.

    PYTHONPATH=. python datasets/release_dataset.py --repo Sssunset/kalman-mem-peers              # pack and upload
    PYTHONPATH=. python datasets/release_dataset.py --repo Sssunset/kalman-mem-peers --no-upload  # pack only
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import io
import json
import time
from pathlib import Path

from pipeline.config import REPO, load, shown
from pipeline.layout import Layout
from pipeline.streams import digest

VERSION = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gz_writer(dst: Path):
    """A text writer to a gzip file without a timestamp, so that the same content always gives the same bytes."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = dst.open("wb")
    return raw, io.TextIOWrapper(gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6), encoding="utf-8")


def pack_stream(L: Layout, name: str, dst: Path, stage: Path) -> dict:
    spec = L.stream(name)
    hf = {m["name"]: m.get("hf_id", m["name"]) for m in L.peer_models(spec["peer_names"])}
    rows, sources, tasks, peers = 0, collections.Counter(), collections.Counter(), {}
    raw, fout = gz_writer(dst)
    with spec["path"].open() as fin:
        for line in fin:
            r = json.loads(line)
            for k, meta in (r.get("peer_metadata") or {}).items():
                if isinstance(meta, dict) and meta.get("model"):
                    meta["model"] = hf.get(str(meta["model"]).rstrip("/").split("/")[-1], meta["model"])
                    peers.setdefault(k, meta["model"])
            rows += 1
            sources[str(r.get("source"))] += 1
            tasks[str(r.get("task_type"))] += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    fout.close()
    raw.close()
    entry = {"dataset": name, "kind": spec["kind"], "file": str(dst.relative_to(stage)), "unpacks_to": shown(spec["path"]),
             "rows": rows, "bytes": dst.stat().st_size, "sha256": sha256(dst), "sources": dict(sorted(sources.items())),
             "task_types": dict(sorted(tasks.items())), "peers": dict(sorted(peers.items(), key=lambda kv: int(kv[0].split("_")[1])))}
    if spec["kind"] == "misleading":
        man = json.load(open(spec["path"].parent / "manifest.json"))
        entry.update({"base": spec["base"], "answers": spec["answers"], "regime": spec["regime"], "poison_ratio": man["poison_ratio"],
                      "digest": digest(spec["path"]), "stream_manifest": {k: v for k, v in man.items() if k not in ("base", "out", "answers")}})
    return entry


def pack_answers(L: Layout, name: str, dst: Path, stage: Path) -> dict:
    spec = L.stream(name)
    rows, summaries = 0, {}
    raw, fout = gz_writer(dst)
    for p in L.peer_models(L.stream(spec["base"])["peer_names"]):
        d = spec["path"] / p["name"]
        if not (d / "complete.json").exists():
            raise SystemExit(f"{d}: not complete (run the peers step first)")
        n = 0
        for f in sorted(glob.glob(str(d / "shard*.jsonl"))):
            for line in open(f):
                fout.write(json.dumps({"peer": p["name"], **json.loads(line)}, ensure_ascii=False) + "\n")
                n += 1
        parts = [json.load(open(f)) for f in sorted(glob.glob(str(d / "summary.shard*.json")))]
        total = max(1, sum(s["n"] for s in parts))
        summaries[p["name"]] = {
            "model": p.get("hf_id", p["name"]), "n": n, "mode": parts[0]["mode"], "generation_params": parts[0].get("generation_params"),
            "max_attempts": parts[0].get("max_attempts"), "forced_allowed": parts[0].get("forced_allowed"),
            "accepted": sum(s.get("accepted", 0) for s in parts), "accepted_pct": 100 * sum(s.get("accepted", 0) for s in parts) / total,
            "forced": sum(s.get("forced", 0) for s in parts), "mean_attempts": sum(s.get("mean_attempts", 0) * s["n"] for s in parts) / total,
            "unusable_reasons": dict(sum((collections.Counter(s.get("unusable_reasons", {})) for s in parts), collections.Counter()))}
        rows += n
    fout.close()
    raw.close()
    return {"dataset": name, "kind": "answers", "file": str(dst.relative_to(stage)), "unpacks_to": shown(spec["path"]),
            "rows": rows, "bytes": dst.stat().st_size, "sha256": sha256(dst), "base": spec["base"], "mode": spec["mode"], "peers": summaries}


def card(entries: list[dict]) -> str:
    streams = [e for e in entries if e["kind"] == "stream"]
    mis = [e for e in entries if e["kind"] == "misleading"]
    answers = [e for e in entries if e["kind"] == "answers"]
    configs = "\n".join(f"- config_name: {e['dataset']}\n  data_files: {e['file']}" for e in streams + mis + answers)
    peers = streams[0]["peers"] if streams else {}
    rows = "\n".join(f"| `{e['dataset']}` | {e['rows']:,} | {', '.join(sorted(e['sources']))} |" for e in streams)
    mrows = "\n".join(f"| `{e['dataset']}` | `{e['regime']}` | {e['poison_ratio']:.1f}% | {e['rows']:,} |" for e in mis)
    arows = "\n".join(f"| `{e['dataset']}` | {e['rows']:,} | " + ", ".join(f"{p} {v['accepted_pct']:.0f}%" for p, v in e["peers"].items()) + " |"
                      for e in answers)
    ptable = "\n".join(f"| `{k}` | {v} |" for k, v in peers.items())
    return f"""---
license: other
task_categories:
- question-answering
language:
- en
pretty_name: Kalman Mem six-peer streams with misleading peers
configs:
{configs}
---

# Kalman Mem: six-peer streams with misleading peers

Event streams for studying a central model that reads several peer models' answers and learns, over the stream, which
peers to trust. Each event is a question with its gold answer, the answers of six peer models and each answer's verified
correctness. The misleading datasets replace a controlled share of the peers' answers with verified-wrong, on-topic
answers. Code, pipeline and documentation: https://github.com/Peilin-FF/Helman-Mem-

## Streams

| dataset | events | sources |
|---|---:|---|
{rows}

One JSON object per line, in stream order (the order matters: a memory is run along it). Fields: `id`, `source`,
`task_type` (math / rag / code; mcqa / boolqa / shortqa), `problem`, `answer` (+ `answer_aliases`, `context`, `choices` /
`choice_labels`, or the code tests), `peer_responses` (`peer_0` ... `peer_5`), `peer_metadata` (model and settings; in a
misleading stream also `regime`, `misled`, `adversarial_forced`), `peer_correct` / `correctness_by_peer`.

| peer | model |
|---|---|
{ptable}

## Misleading datasets

Each replaces that share of every peer's answers with the peer's own misleading answer, on a different set of events per
peer (nested across rates). A peer keeps its honest answer where it has no usable misleading answer, so the top rates
reach less than asked.

| dataset | regime | misleading share reached | events |
|---|---|---:|---:|
{mrows}

## Misleading answers

One row per (event, peer): `peer`, `id`, `source`, `task_type`, `response`, `correct`, `accepted` (graded wrong and
passing every check: format, no refusal or leak, on topic), `forced` (conclusion rewritten), `attempts`, `reasons`.

| dataset | rows | usable share per peer |
|---|---:|---|
{arows}

## Use

`manifest.json` lists every file with its sha256. With the code repository, `python datasets/download.py` downloads a
pinned revision into the layout the pipeline reads. Each record keeps its source benchmark in `source`; use is subject
to the licences of the source benchmarks and of the models that generated the answers.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Sssunset/kalman-mem-peers", help="the Hugging Face dataset repo")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--stage", type=Path, default=None, help="staging folder (default: outputs/release/<repo name>)")
    ap.add_argument("--manifest", type=Path, default=REPO / "datasets" / "manifest.json", help="where to write the pinned manifest")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--private", action="store_true", help="create the repo private (only when it does not exist yet)")
    args = ap.parse_args()
    L = Layout(load(args.config))
    stage = args.stage or L.outputs / "release" / args.repo.split("/")[-1]
    stage.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, d in L.registry.datasets.items():
        if not d.get("release"):
            continue
        t0 = time.time()
        dst = stage / d["release"]
        entry = (pack_answers if d["kind"] == "answers" else pack_stream)(L, name, dst, stage)
        entries.append(entry)
        print(f"[release] {name} ({d['kind']}): {entry['rows']:,} rows, {entry['bytes'] / 1e6:.0f} MB -> {d['release']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    manifest = {"version": VERSION, "date": time.strftime("%Y-%m-%d"), "repo": args.repo, "files": entries}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (stage / "README.md").write_text(card(entries))
    if args.no_upload:
        print(f"[release] packed into {stage} (not uploaded)")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    commit = api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(stage),
                               commit_message=f"release {manifest['date']}: {len(entries)} datasets")
    manifest["revision"] = commit.oid
    args.manifest.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[release] uploaded to https://huggingface.co/datasets/{args.repo} at {commit.oid}; pinned in {args.manifest}")


if __name__ == "__main__":
    main()
