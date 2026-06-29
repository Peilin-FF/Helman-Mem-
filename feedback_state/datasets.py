"""Unified loader for math benchmarks and the DeepMath training set.

Each entry in REGISTRY maps a short name to the HuggingFace path + the field
names we expect for ``problem`` and ``answer``. Use ``load_math_dataset`` to
get a list of standardized records:

    [{"id": "...", "problem": "...", "answer": "...", "source": "..."}, ...]

Field names on HF mirrors drift; pass ``field_overrides`` (or override in YAML)
if a mirror uses different names. Pass ``hf_path_override`` to point at a fork.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from feedback_state.utils import extract_final_answer


@dataclass
class DatasetSpec:
    hf_path: str
    default_split: str = "test"
    default_config: str | None = None
    problem_field: str = "problem"
    answer_field: str = "answer"
    id_field: str | None = None
    # Special-case post-processing for benchmark answers that are not a plain string.
    answer_kind: str = "string"  # "string" | "list_first" | "boxed_extract" | "gsm8k_hash"


REGISTRY: dict[str, DatasetSpec] = {
    "deepmath": DatasetSpec(
        hf_path="zwhe99/DeepMath-103K",
        default_split="train",
        # DeepMath's column names vary across mirrors — fall back to the
        # heuristic via utils.infer_deepmath_fields when these aren't found.
        problem_field="question",
        answer_field="final_answer",
        answer_kind="boxed_extract",
    ),
    "math500": DatasetSpec(
        hf_path="HuggingFaceH4/MATH-500",
        default_split="test",
        problem_field="problem",
        answer_field="answer",
    ),
    "aime2024": DatasetSpec(
        hf_path="Maxwell-Jia/AIME_2024",
        default_split="train",
        problem_field="Problem",
        answer_field="Answer",
        id_field="ID",
    ),
    "amc": DatasetSpec(
        hf_path="AI-MO/aimo-validation-amc",
        default_split="train",
        problem_field="problem",
        answer_field="answer",
    ),
    "olympiadbench": DatasetSpec(
        hf_path="Hothan/OlympiadBench",
        default_split="train",
        default_config="OE_TO_maths_en_COMP",
        problem_field="question",
        answer_field="final_answer",
        answer_kind="list_first",
    ),
    "minerva": DatasetSpec(
        hf_path="math-ai/minervamath",
        default_split="test",
        problem_field="question",
        answer_field="answer",
    ),
    "college_math": DatasetSpec(
        hf_path="math-ai/college-math",
        default_split="test",
        problem_field="question",
        answer_field="answer",
    ),
    "gsm8k": DatasetSpec(
        # local HF_DATA/gsm8k/main parquet dir (offline); train split.
        hf_path="/workspace/cloud_android/fengpeilin/HF_DATA/gsm8k/main",
        default_split="train",
        default_config="main",
        problem_field="question",
        answer_field="answer",
        answer_kind="gsm8k_hash",  # gold is the bare number after the trailing "#### N"
    ),
}


def _extract_answer(raw: Any, kind: str) -> str:
    if raw is None:
        return ""
    if kind == "list_first":
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if item:
                    return extract_final_answer(str(item))
            return ""
        return extract_final_answer(str(raw))
    if kind == "boxed_extract":
        return extract_final_answer(str(raw))
    if kind == "gsm8k_hash":
        # GSM8K gold answer ends with "#### <final>"; keep the bare final answer.
        return str(raw).split("####")[-1].strip()
    return str(raw).strip()


def _read_field(example: dict[str, Any], primary: str, fallbacks: tuple[str, ...] = ()) -> Any:
    if primary in example and example[primary] not in (None, ""):
        return example[primary]
    for key in fallbacks:
        if key in example and example[key] not in (None, ""):
            return example[key]
    return None


def standardize_example(
    example: dict[str, Any],
    spec: DatasetSpec,
    *,
    source: str,
    index: int,
) -> dict[str, str] | None:
    problem = _read_field(
        example, spec.problem_field, fallbacks=("problem", "question", "Problem", "input", "prompt")
    )
    if not problem:
        return None
    raw_answer = _read_field(
        example, spec.answer_field,
        fallbacks=("final_answer", "answer", "Answer", "solution", "target", "label"),
    )
    answer = _extract_answer(raw_answer, spec.answer_kind)
    record_id = (
        str(example.get(spec.id_field)) if spec.id_field and example.get(spec.id_field) is not None
        else str(example.get("id", example.get("idx", index)))
    )
    return {
        "id": f"{source}:{record_id}",
        "problem": str(problem).strip(),
        "answer": answer,
        "source": source,
    }


def load_math_dataset(
    name: str,
    *,
    split: str | None = None,
    config: str | None = None,
    start_index: int = 0,
    max_samples: int | None = None,
    hf_path_override: str | None = None,
    field_overrides: dict[str, str] | None = None,
    cache_dir: str | None = None,
) -> list[dict[str, str]]:
    """Return a standardized list of records for any registered benchmark."""
    from datasets import DatasetDict, load_dataset

    key = name.lower().replace("-", "_")
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Registered: {sorted(REGISTRY)}. "
            "Pass hf_path_override and field_overrides to load a custom one."
        )
    spec = REGISTRY[key]
    if field_overrides:
        spec = DatasetSpec(
            hf_path=hf_path_override or spec.hf_path,
            default_split=spec.default_split,
            default_config=spec.default_config,
            problem_field=field_overrides.get("problem", spec.problem_field),
            answer_field=field_overrides.get("answer", spec.answer_field),
            id_field=field_overrides.get("id", spec.id_field),
            answer_kind=field_overrides.get("answer_kind", spec.answer_kind),
        )
    hf_path = hf_path_override or spec.hf_path
    cfg = config or spec.default_config
    sp = split or spec.default_split
    local_path = Path(hf_path).expanduser()
    if local_path.exists() and local_path.is_file():
        loaded = load_dataset("json", data_files={sp: str(local_path)}, cache_dir=cache_dir)
    elif local_path.exists() and local_path.is_dir():
        # Local HF dataset dir of parquet/arrow shards (offline). The config name does
        # not apply to a bare directory load, so pass the split directly.
        loaded = load_dataset(str(local_path), split=sp, cache_dir=cache_dir)
    elif cfg is not None:
        loaded = load_dataset(hf_path, cfg, cache_dir=cache_dir)
    else:
        loaded = load_dataset(hf_path, cache_dir=cache_dir)
    ds = loaded[sp] if isinstance(loaded, DatasetDict) else loaded
    start = max(0, int(start_index))
    stop = len(ds) if max_samples is None else min(start + int(max_samples), len(ds))
    ds = ds.select(range(start, stop))
    out: list[dict[str, str]] = []
    for index, example in enumerate(ds):
        # Fallback ids must use the GLOBAL dataset index: with a slice-local index,
        # a start_index>0 slice would reuse ids 0..N and collide with earlier slices.
        record = standardize_example(dict(example), spec, source=key, index=start + index)
        if record is not None:
            out.append(record)
    return out


def list_datasets() -> list[str]:
    return sorted(REGISTRY)
