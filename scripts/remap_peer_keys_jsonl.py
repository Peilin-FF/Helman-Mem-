#!/usr/bin/env python3
"""Remap peer_N keys in a JSONL while keeping all peer-indexed fields aligned."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PEER_MAP_FIELDS = (
    "peer_responses",
    "peer_metadata",
    "peer_correct",
    "correctness_by_peer",
    "peer_samples",
)
PEER_STRING_FIELDS = ("strong_peer", "swapped_with")
PEER_LIST_FIELDS = ("weak_peers", "target_peers", "correct_peers", "peer_keys")


def parse_remap(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --remap entry {item!r}; expected OLD=NEW")
        old, new = item.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def remap_value(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [mapping.get(v, v) if isinstance(v, str) else v for v in value]
    return value


def remap_record(record: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    out = json.loads(json.dumps(record))
    for field in PEER_MAP_FIELDS:
        table = out.get(field)
        if isinstance(table, dict):
            out[field] = {mapping.get(k, k): v for k, v in table.items()}
    for field in PEER_STRING_FIELDS:
        if field in out:
            out[field] = remap_value(out[field], mapping)
    for field in PEER_LIST_FIELDS:
        if field in out:
            out[field] = remap_value(out[field], mapping)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--remap", action="append", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapping = parse_remap(args.remap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            out = remap_record(json.loads(line), mapping)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"[remap_peer_keys] {args.input} -> {args.output}: {n}")


if __name__ == "__main__":
    main()
