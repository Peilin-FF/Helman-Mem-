#!/usr/bin/env python3
"""Add one generated peer response to an existing Setting-A JSONL.

This is for peer-count generalization runs where the original peers have already
answered the test stream. It can also remap peer keys before adding the new peer,
for example old 4-peer data with Qwen/Ministral swapped:

  old peer_0 -> new peer_0
  old peer_1 -> new peer_1
  old peer_3 -> new peer_2
  old peer_2 -> new peer_3
  new model  -> new peer_4
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from feedback_state.generation import GenerationConfig, TextGenerator, render_instruction_prompt
from feedback_state.tasks import build_peer_prompt, task_type_of
from feedback_state.utils import append_jsonl


PEER_MAP_FIELDS = (
    "peer_responses",
    "peer_metadata",
    "peer_correct",
    "correctness_by_peer",
    "peer_samples",
)
PEER_STRING_FIELDS = ("strong_peer", "swapped_with")
PEER_LIST_FIELDS = ("weak_peers", "target_peers", "correct_peers", "peer_keys")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _parse_remap(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --remap entry {item!r}; expected OLD=NEW")
        old, new = item.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def _parse_tokens_by_task(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    if not text:
        return out
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Bad --max-new-tokens-by-task entry {part!r}; expected task=N")
        key, value = part.split("=", 1)
        out[key.strip().lower()] = int(value)
    return out


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--peer_key", default="peer_4")
    p.add_argument("--remap", action="append", default=[], help="Peer key remap OLD=NEW")
    p.add_argument("--deprive_context_model", action="append", default=[])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--write_chunk_size", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument(
        "--max_new_tokens_by_task",
        default="math=2048,code=1024,rag=256,boolqa=48",
        help="Comma-separated task=N overrides.",
    )
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--use_vllm", type=_as_bool, default=True)
    p.add_argument("--local_files_only", type=_as_bool, default=True)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapping = _parse_remap(args.remap)
    tokens_by_task = _parse_tokens_by_task(args.max_new_tokens_by_task)
    done = _completed_keys(args.output)
    deprive_tokens = [str(x).lower() for x in args.deprive_context_model]
    model_l = str(args.model).lower()
    with_context = not any(token in model_l for token in deprive_tokens)

    gen_cfg = GenerationConfig(
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        dtype=str(args.dtype),
        device=str(args.device),
        use_vllm=bool(args.use_vllm),
        local_files_only=bool(args.local_files_only),
    )
    generator = TextGenerator(str(args.model), gen_cfg)
    try:
        pending: list[tuple[int, dict[str, Any]]] = []
        with args.input.open() as handle:
            for i, line in enumerate(handle):
                if not line.strip():
                    continue
                if i % int(args.num_shards) != int(args.shard_index):
                    continue
                record = remap_record(json.loads(line), mapping)
                if _record_key(record, i) in done:
                    continue
                if args.peer_key in dict(record.get("peer_responses", {})):
                    raise ValueError(f"{args.peer_key} already exists in {_record_key(record, i)}")
                pending.append((i, record))
                if args.max_samples is not None and len(pending) >= int(args.max_samples):
                    break

        for start in range(0, len(pending), int(args.write_chunk_size)):
            chunk_pairs = pending[start : start + int(args.write_chunk_size)]
            groups: dict[str, list[tuple[int, dict[str, Any], str]]] = defaultdict(list)
            for local_idx, (line_index, record) in enumerate(chunk_pairs):
                prompt_text = build_peer_prompt(record, with_context=with_context)
                prompt = render_instruction_prompt(generator.tokenizer, prompt_text)
                groups[task_type_of(record)].append((local_idx, record, prompt))

            generated_by_local = [""] * len(chunk_pairs)
            gen_tokens_by_local = [int(args.max_new_tokens)] * len(chunk_pairs)
            for task_type, items in groups.items():
                generator.config.max_new_tokens = int(tokens_by_task.get(task_type, args.max_new_tokens))
                for b in tqdm(
                    range(0, len(items), int(args.batch_size)),
                    desc=f"{Path(args.input).name}:{task_type}",
                ):
                    sub = items[b : b + int(args.batch_size)]
                    texts = generator.generate([prompt for _, _, prompt in sub])
                    for (local_idx, _, _), text in zip(sub, texts):
                        generated_by_local[local_idx] = text
                        gen_tokens_by_local[local_idx] = int(generator.config.max_new_tokens)

            output_records: list[dict[str, Any]] = []
            for local_idx, (_, record) in enumerate(chunk_pairs):
                key = str(args.peer_key)
                record.setdefault("peer_responses", {})[key] = generated_by_local[local_idx]
                record.setdefault("peer_metadata", {})[key] = {
                    "model": str(args.model),
                    "is_adversarial": False,
                    "known_incorrect": False,
                    "received_context": bool(task_type_of(record) != "rag" or with_context),
                    "num_samples": 1,
                    "generation_params": {
                        "max_new_tokens": gen_tokens_by_local[local_idx],
                        "temperature": gen_cfg.temperature,
                        "top_p": gen_cfg.top_p,
                        "backend": generator.backend,
                    },
                }
                output_records.append(record)
            append_jsonl(args.output, output_records)
    finally:
        generator.close()


if __name__ == "__main__":
    main()
