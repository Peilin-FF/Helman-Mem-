#!/usr/bin/env python3
"""Merge selected peer fields from a reference JSONL into a target JSONL by uid/id."""
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


def _parse_remap(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --remap entry {item!r}; expected OLD=NEW")
        old, new = item.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def _record_key(record: dict[str, Any], fallback_index: int | None = None) -> str:
    if record.get("uid") is not None:
        return str(record["uid"])
    if record.get("id") is not None:
        return str(record["id"])
    return str(fallback_index)


def _remap_value(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [mapping.get(v, v) if isinstance(v, str) else v for v in value]
    return value


def remap_record(record: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    if not mapping:
        return dict(record)
    out = json.loads(json.dumps(record))
    for field in PEER_MAP_FIELDS:
        table = out.get(field)
        if isinstance(table, dict):
            out[field] = {mapping.get(k, k): v for k, v in table.items()}
    for field in PEER_STRING_FIELDS:
        if field in out:
            out[field] = _remap_value(out[field], mapping)
    for field in PEER_LIST_FIELDS:
        if field in out:
            out[field] = _remap_value(out[field], mapping)
    return out


def load_reference(path: Path, peer_key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            key = _record_key(record, i)
            payload: dict[str, Any] = {}
            for field in PEER_MAP_FIELDS:
                table = record.get(field)
                if isinstance(table, dict) and peer_key in table:
                    payload[field] = table[peer_key]
            if "peer_responses" not in payload:
                raise ValueError(f"Reference record {key} missing {peer_key} response")
            out[key] = payload
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--peer_key", default=None, help="Backward-compatible alias for both keys.")
    p.add_argument("--reference_peer_key", default=None)
    p.add_argument("--target_peer_key", default=None)
    p.add_argument("--remap", action="append", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapping = _parse_remap(args.remap)
    ref_key = args.reference_peer_key or args.peer_key or "peer_4"
    target_key = args.target_peer_key or args.peer_key or ref_key
    ref = load_reference(args.reference, ref_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.target.open() as fin, args.output.open("w") as fout:
        for i, line in enumerate(fin):
            if not line.strip():
                continue
            record = remap_record(json.loads(line), mapping)
            key = _record_key(record, i)
            if key not in ref:
                raise KeyError(f"{key} missing from reference {args.reference}")
            for field, value in ref[key].items():
                record.setdefault(field, {})[target_key] = value
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    print(
        f"[merge_peer] {args.target} + {target_key} from "
        f"{args.reference}:{ref_key} -> {args.output}: {n}"
    )


if __name__ == "__main__":
    main()
