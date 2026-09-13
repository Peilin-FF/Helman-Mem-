"""Package the registered datasets for the GitHub repository (datasets/, gzip; no Hugging Face involved).

Every dataset in configs/datasets/ with a `release:` archive name is packed, by kind:
    stream       the stream file, records verbatim except peer_metadata[*].model, rewritten to the Hugging Face id
    answers      every peer's generated answers, one row per (event, peer) with `peer` (the model directory) added,
                 and each peer's generation summary in the manifest
and every misleading dataset gets a manifest entry (no archive): it is rebuilt from its stream and its answers, and the
entry holds the content digest of the stream built here, which bash datasets/unpack.sh checks after rebuilding.

    PYTHONPATH=. python datasets/release_dataset.py            # writes datasets/<archive> + datasets/manifest.json
    bash datasets/unpack.sh                                   # on another machine: restores and rebuilds data/

gzip keeps every archive under GitHub's 100 MB limit.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import time
from pathlib import Path

from pipeline.config import load, shown
from pipeline.layout import Layout
from pipeline.streams import digest

VERSION = 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_stream(L: Layout, name: str, dst: Path) -> dict:
    spec = L.stream(name)
    hf = {m["name"]: m.get("hf_id", m["name"]) for m in L.peer_models(spec["peer_set"])}
    rows, sources, tasks, peers = 0, collections.Counter(), collections.Counter(), {}
    with spec["path"].open() as fin, gzip.open(dst, "wt", compresslevel=6) as fout:
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
    return {"dataset": name, "kind": "stream", "archive": dst.name, "unpacks_to": shown(spec["path"]), "bytes_gz": dst.stat().st_size,
            "sha256_gz": sha256(dst), "rows": rows, "sources": dict(sorted(sources.items())), "task_types": dict(sorted(tasks.items())),
            "peers": dict(sorted(peers.items(), key=lambda kv: int(kv[0].split("_")[1])))}


def pack_answers(L: Layout, name: str, dst: Path) -> dict:
    spec = L.stream(name)
    rows, summaries = 0, {}
    with gzip.open(dst, "wt", compresslevel=6) as fout:
        for p in L.peer_models(L.stream(spec["base"])["peer_set"]):
            d = spec["path"] / p["name"]
            shards = sorted(glob.glob(str(d / "shard*.jsonl")))
            if not (d / "complete.json").exists():
                raise SystemExit(f"{d}: not complete (run the peers step first)")
            n = 0
            for f in shards:
                for line in open(f):
                    r = json.loads(line)
                    fout.write(json.dumps({"peer": p["name"], **r}, ensure_ascii=False) + "\n")
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
    return {"dataset": name, "kind": "answers", "archive": dst.name, "unpacks_to": shown(spec["path"]), "bytes_gz": dst.stat().st_size,
            "sha256_gz": sha256(dst), "rows": rows, "base": spec["base"], "mode": spec["mode"], "peers": summaries}


def describe_misleading(L: Layout, name: str) -> dict | None:
    spec = L.stream(name)
    if not spec["path"].exists():
        print(f"[datasets] {name}: not built, left out of the manifest (run the streams step)")
        return None
    man = json.load(open(spec["path"].parent / "manifest.json"))
    return {"dataset": name, "kind": "misleading", "builds_to": shown(spec["path"]), "base": spec["base"], "answers": spec["answers"],
            "regime": spec["regime"], "events": man["events"], "poison_ratio": man["poison_ratio"], "digest": digest(spec["path"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", type=Path, default=Path("datasets"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="pack only these datasets or groups and keep the manifest's other entries (gzip stamps the time, so re-packing an "
                         "unchanged stream still changes its archive)")
    args = ap.parse_args()
    L = Layout(load(args.config))
    args.out.mkdir(parents=True, exist_ok=True)
    old = {}
    if (args.out / "manifest.json").exists():
        by_archive = {d["release"]: n for n, d in L.registry.datasets.items() if d.get("release")}
        for e in json.load(open(args.out / "manifest.json")).get("files", []):
            name = e.get("dataset") or by_archive.get(e.get("archive"))
            if name:   # entries of an older manifest carry no dataset name and no kind
                old[name] = {"dataset": name, "kind": e.get("kind", "stream"), **e}
    only = set(L.registry.expand(args.only)) if args.only is not None else None
    entries = []
    for name, d in L.registry.datasets.items():
        if only is not None and name not in only:
            if name in old:
                entries.append(old[name])
            continue
        t0 = time.time()
        if d["kind"] == "misleading":
            e = describe_misleading(L, name)
        elif d.get("release"):
            e = (pack_stream if d["kind"] == "stream" else pack_answers)(L, name, args.out / d["release"])
        else:
            continue
        if e is None:
            continue
        entries.append(e)
        size = f", {e['bytes_gz'] / 1e6:.0f} MB gz" if "bytes_gz" in e else ""
        print(f"[datasets] {name} ({e['kind']}): {e.get('rows', e.get('events')):,} rows{size} ({time.time() - t0:.0f}s)", flush=True)
    (args.out / "manifest.json").write_text(json.dumps({"version": VERSION, "date": time.strftime("%Y-%m-%d"), "files": entries}, indent=1) + "\n")
    print(f"[datasets] wrote {args.out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
