#!/usr/bin/env python3
"""Fill missing peer_correct entries without recomputing existing labels."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from feedback_state.code_exec import score_code_record
from feedback_state.tasks import code_extract_answer, get_task, task_type_of
from feedback_state.utils import append_jsonl


def _record_key(record: dict[str, Any], fallback_index: int | None = None) -> str:
    if record.get("uid") is not None:
        return str(record["uid"])
    if record.get("id") is not None:
        return str(record["id"])
    return str(fallback_index)


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                done.add(_record_key(json.loads(line), i))
            except json.JSONDecodeError:
                continue
    return done


def _grade_non_code(args: tuple[dict[str, Any], str, str]) -> float:
    record, _key, text = args
    task = get_task(task_type_of(record))
    return float(task.target_fn(text, record))


def _worker(args: tuple[dict[str, Any], str, str], q: mp.Queue) -> None:
    try:
        q.put(_grade_non_code(args))
    except Exception:
        q.put(0.0)


def _grade_with_timeout(record: dict[str, Any], key: str, text: str, timeout: float) -> float:
    if task_type_of(record) == "code":
        code = code_extract_answer(text)
        result = score_code_record(record, code, timeout=timeout)
        return 1.0 if result.passed else 0.0
    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker, args=((record, key, text), q))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return 0.0
    try:
        return float(q.get_nowait())
    except Exception:
        return 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--peer_key", action="append", default=[])
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    done = _completed_keys(args.output)
    requested = set(args.peer_key)
    pending: list[tuple[int, dict[str, Any]]] = []
    with args.input.open() as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if _record_key(record, i) in done:
                continue
            pending.append((i, record))
            if args.max_samples is not None and len(pending) >= int(args.max_samples):
                break

    buffer: list[dict[str, Any]] = []
    changed = 0
    for _, record in tqdm(pending, desc="fill_missing_peer_correct"):
        responses = dict(record.get("peer_responses", {}))
        pc = dict(record.get("peer_correct") or {})
        keys = sorted(requested or responses.keys())
        for key in keys:
            if key not in responses:
                continue
            if not args.overwrite and key in pc:
                continue
            pc[key] = _grade_with_timeout(record, key, str(responses[key]), float(args.timeout))
            changed += 1
        record["peer_correct"] = pc
        if isinstance(record.get("correctness_by_peer"), dict):
            cbp = dict(record["correctness_by_peer"])
            for key, value in pc.items():
                cbp[key] = int(round(float(value)))
            record["correctness_by_peer"] = cbp
        buffer.append(record)
        if len(buffer) >= 64:
            append_jsonl(args.output, buffer)
            buffer = []
    if buffer:
        append_jsonl(args.output, buffer)
    print(f"[fill_missing_peer_correct] wrote {args.output}; filled {changed} labels")


if __name__ == "__main__":
    main()
