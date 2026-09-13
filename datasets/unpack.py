"""Restore the released datasets into data/ from datasets/manifest.json (Python standard library only).

    python3 datasets/unpack.py            streams -> data/<dir>/<file>.jsonl; answers -> data/<dataset>/<peer>/
    python3 datasets/unpack.py --verify   after the misleading streams are built: compare their content digests

bash datasets/unpack.sh runs both, with the build in between. An existing file is kept, never overwritten.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def restore(entry: dict) -> bool:
    archive = ROOT / "datasets" / entry["archive"]
    ok = sha256(archive) == entry["sha256_gz"]
    print(("ok     " if ok else "MISMATCH ") + f"{entry['archive']} ({entry['rows']:,} rows)")
    if not ok:
        return False
    target = ROOT / entry["unpacks_to"]
    if entry["kind"] == "stream":
        if target.exists() and target.stat().st_size:
            print(f"keep   {entry['unpacks_to']}")
            return True
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(archive, "rt") as fin, target.open("w") as fout:
            for line in fin:
                fout.write(line)
        print(f"wrote  {entry['unpacks_to']}")
        return True
    # answers: one directory per peer, as the peers step writes them (one shard, its summary, complete.json)
    todo = {p for p in entry["peers"] if not (target / p / "complete.json").exists()}
    if not todo:
        print(f"keep   {entry['unpacks_to']}/ ({len(entry['peers'])} peers)")
        return True
    handles, counts = {}, collections.Counter()
    try:
        with gzip.open(archive, "rt") as fin:
            for line in fin:
                peer = json.loads(line)["peer"]
                if peer not in todo:
                    continue
                if peer not in handles:
                    (target / peer).mkdir(parents=True, exist_ok=True)
                    handles[peer] = (target / peer / "shard0of1.jsonl").open("w")
                row = json.loads(line)
                del row["peer"]
                handles[peer].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[peer] += 1
    finally:
        for h in handles.values():
            h.close()
    for peer in sorted(todo):
        summary = dict(entry["peers"][peer])
        (target / peer / "summary.shard0of1.json").write_text(json.dumps(summary, indent=1))
        (target / peer / "complete.json").write_text(json.dumps({"shards": 1, "restored_from": entry["archive"]}))
        print(f"wrote  {entry['unpacks_to']}/{peer}/ ({counts[peer]:,} answers)")
    return True


def verify(entries: list[dict]) -> bool:
    sys.path.insert(0, str(ROOT))
    from pipeline.streams import digest

    ok = True
    for e in entries:
        path = ROOT / e["builds_to"]
        if not path.exists():
            print(f"MISSING {e['builds_to']}")
            ok = False
            continue
        same = digest(path) == e["digest"]
        ok &= same
        print(("ok     " if same else "MISMATCH ") + f"{e['builds_to']} ({e['events']:,} events, {e['poison_ratio']:.1f}% misleading)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    files = json.load(open(ROOT / "datasets" / "manifest.json"))["files"]
    if args.verify:
        ok = verify([e for e in files if e["kind"] == "misleading"])
    else:
        ok = all([restore(e) for e in files if e["kind"] in ("stream", "answers")])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
