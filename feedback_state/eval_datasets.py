"""Unified loaders for RAG-QA and code datasets, standardised for the task registry.

Every loader returns records shaped so the trust-state pipeline is task-agnostic:

    {
        "id": str,
        "task_type": "rag" | "code",      # math goes through datasets.load_math_dataset
        "problem": str,                    # the question / function prompt (the encoder context)
        "answer": str,                     # canonical gold (string)
        "source": str,
        # rag extras:
        "answer_aliases": [str, ...],
        "context": [str | [title, [sent...]], ...],   # dataset-provided "retrieved" docs
        # code extras:
        "code_format": "asserts" | "unittest" | "io" | "functional",
        "test": str, "test_setup": str, "entry_point": str | None, "test_cases": [...],
    }

These feed scripts/generate_setting_a_peers.py (peer generation) and, for code,
scripts/score_code_peers.py (offline pass@1 -> record["peer_correct"]).

Add a dataset = add one entry to REGISTRY. Add a task mixture = concatenate the
output JSONLs; nothing else changes.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

# Map HuggingFace dataset names to local directories under HF_DATA for offline use.
# Loaders call load_dataset(<hub_name>, ...); when HF_DATA/<dir> exists we point at
# the local copy so everything runs with HF_DATASETS_OFFLINE=1.
_HF_DATA_DIR = os.environ.get("HF_DATA_DIR", "/workspace/cloud_android/fengpeilin/HF_DATA")
_LOCAL_DIRS = {
    "hotpot_qa": "hotpot_qa",
    "trivia_qa": "trivia_qa",
    "squad": "squad/plain_text",
    "openai_humaneval": "openai_humaneval",
    "mbpp": "mbpp",
    "bigcode/bigcodebench": "bigcodebench",
    "livecodebench/code_generation_lite": "code_generation_lite",
    "codeparrot/apps": "apps",
}


def _dataset_path(hub_name: str) -> str:
    """Return a local HF_DATA dir if present, else the hub name (online fallback)."""
    sub = _LOCAL_DIRS.get(hub_name)
    if sub:
        local = os.path.join(_HF_DATA_DIR, sub)
        if os.path.isdir(local):
            return local
    return hub_name


# task_type for each registered dataset name.
RAG_DATASETS = {"hotpotqa", "triviaqa", "squad"}
CODE_DATASETS = {"humaneval", "mbpp", "bigcodebench", "livecodebench", "apps"}


def task_type_for(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key in {d.replace("-", "_") for d in RAG_DATASETS}:
        return "rag"
    if key in {d.replace("-", "_") for d in CODE_DATASETS}:
        return "code"
    return "math"


def _slice(ds, start_index: int, max_samples: int | None):
    start = max(0, int(start_index))
    stop = len(ds) if max_samples is None else min(start + int(max_samples), len(ds))
    if isinstance(ds, list):  # plain jsonl-loaded list (e.g. livecodebench)
        return ds[start:stop]
    return ds.select(range(start, stop))


# ---------------------------------------------------------------------------
# RAG-QA loaders
# ---------------------------------------------------------------------------

def load_hotpotqa(split: str, start_index: int, max_samples: int | None, cache_dir: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(_dataset_path("hotpot_qa"), "distractor", split=split or "validation", cache_dir=cache_dir)
    ds = _slice(ds, start_index, max_samples)
    out = []
    for ex in ds:
        ctx = ex.get("context", {})
        titles = ctx.get("title", []) if isinstance(ctx, dict) else []
        sentences = ctx.get("sentences", []) if isinstance(ctx, dict) else []
        context = [[t, s] for t, s in zip(titles, sentences)]
        out.append({
            "id": f"hotpotqa:{ex.get('id', ex.get('_id', len(out)))}",
            "task_type": "rag",
            "problem": str(ex.get("question", "")).strip(),
            "answer": str(ex.get("answer", "")).strip(),
            "answer_aliases": [],
            "context": context,
            "source": "hotpotqa",
        })
    return out


def load_triviaqa(split: str, start_index: int, max_samples: int | None, cache_dir: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(_dataset_path("trivia_qa"), "rc", split=split or "validation", cache_dir=cache_dir)
    ds = _slice(ds, start_index, max_samples)
    out = []
    for ex in ds:
        answer = ex.get("answer", {}) or {}
        value = str(answer.get("value", "")).strip()
        aliases = list(answer.get("aliases", []) or answer.get("normalized_aliases", []) or [])
        # Build "retrieved" context from the provided evidence documents.
        context: list[str] = []
        for key in ("entity_pages", "search_results"):
            block = ex.get(key, {}) or {}
            for field in ("wiki_context", "search_context", "description"):
                vals = block.get(field, []) if isinstance(block, dict) else []
                for v in (vals if isinstance(vals, list) else [vals]):
                    if v:
                        context.append(str(v))
        out.append({
            "id": f"triviaqa:{ex.get('question_id', len(out))}",
            "task_type": "rag",
            "problem": str(ex.get("question", "")).strip(),
            "answer": value,
            "answer_aliases": aliases,
            "context": context[:5],  # cap to keep prompts bounded
            "source": "triviaqa",
        })
    return out


def load_squad(split: str, start_index: int, max_samples: int | None, cache_dir: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    # local HF_DATA/squad/plain_text (offline); train split for the training set.
    ds = load_dataset(_dataset_path("squad"), split=split or "train", cache_dir=cache_dir)
    ds = _slice(ds, start_index, max_samples)
    out = []
    for ex in ds:
        ans = ex.get("answers", {}) or {}
        texts = list(ans.get("text", []) or [])
        gold = str(texts[0]).strip() if texts else ""
        aliases = [str(a).strip() for a in texts[1:]]
        context = str(ex.get("context", "")).strip()
        out.append({
            "id": f"squad:{ex.get('id', len(out))}",
            "task_type": "rag",
            "problem": str(ex.get("question", "")).strip(),
            "answer": gold,
            "answer_aliases": aliases,
            "context": [context] if context else [],
            "source": "squad",
        })
    return out


# ---------------------------------------------------------------------------
# Code loaders
# ---------------------------------------------------------------------------

def load_humaneval(split: str, start_index: int, max_samples: int | None, cache_dir: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(_dataset_path("openai_humaneval"), split=split or "test", cache_dir=cache_dir)
    ds = _slice(ds, start_index, max_samples)
    out = []
    for ex in ds:
        out.append({
            "id": f"humaneval:{ex.get('task_id', len(out))}",
            "task_type": "code",
            "problem": str(ex.get("prompt", "")),
            "answer": str(ex.get("canonical_solution", "")),
            "code_format": "asserts",
            "test": str(ex.get("test", "")),
            "test_setup": "",
            "entry_point": str(ex.get("entry_point", "")) or None,
            "source": "humaneval",
        })
    return out


def load_mbpp(split: str, start_index: int, max_samples: int | None, cache_dir: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(_dataset_path("mbpp"), split=split or "test", cache_dir=cache_dir)
    ds = _slice(ds, start_index, max_samples)
    out = []
    for ex in ds:
        tests = list(ex.get("test_list", []) or [])
        # Reveal the function name/signature to the peer via the example tests.
        prompt = str(ex.get("text", "")).strip()
        if tests:
            prompt += "\nYour code should satisfy these tests:\n" + "\n".join(tests)
        out.append({
            "id": f"mbpp:{ex.get('task_id', len(out))}",
            "task_type": "code",
            "problem": prompt,
            "answer": str(ex.get("code", "")),
            "code_format": "asserts",
            "test": "\n".join(tests),
            "test_setup": str(ex.get("test_setup_code", "")),
            "entry_point": None,
            "source": "mbpp",
        })
    return out


def load_bigcodebench(split: str, start_index: int, max_samples: int | None, cache_dir: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    # Versioned configs change over time; the default split holds the tasks.
    ds = load_dataset(_dataset_path("bigcode/bigcodebench"), split=split or "v0.1.0_hf", cache_dir=cache_dir)
    ds = _slice(ds, start_index, max_samples)
    out = []
    for ex in ds:
        out.append({
            "id": f"bigcodebench:{ex.get('task_id', len(out))}",
            "task_type": "code",
            "problem": str(ex.get("complete_prompt", ex.get("instruct_prompt", ""))),
            "answer": str(ex.get("canonical_solution", "")),
            "code_format": "unittest",
            "test": str(ex.get("test", "")),
            "test_setup": "",
            "entry_point": str(ex.get("entry_point", "")) or None,
            "source": "bigcodebench",
        })
    return out


def load_livecodebench(split: str, start_index: int, max_samples: int | None, cache_dir: str | None) -> list[dict[str, Any]]:
    # The HF copy ships a loader SCRIPT (code_generation_lite.py) which new datasets
    # versions refuse to run. The data itself is plain jsonl (test.jsonl), so read it
    # directly when the local dir is present; fall back to the hub otherwise.
    local_dir = _dataset_path("livecodebench/code_generation_lite")
    jsonl = os.path.join(local_dir, "test.jsonl")
    if os.path.isfile(jsonl):
        ds = []
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    ds.append(json.loads(line))
        ds = _slice(ds, start_index, max_samples)
    else:
        from datasets import load_dataset

        ds = load_dataset(local_dir, split=split or "test", cache_dir=cache_dir)
        ds = _slice(ds, start_index, max_samples)
    out = []
    for ex in ds:
        meta = ex.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        func_name = (meta or {}).get("func_name")
        raw_cases = ex.get("public_test_cases", "[]")
        try:
            cases = json.loads(raw_cases) if isinstance(raw_cases, str) else list(raw_cases or [])
        except Exception:
            cases = []
        # Normalise to {"input":..., "output":...}
        norm_cases = [{"input": c.get("input"), "output": c.get("output")} for c in cases if isinstance(c, dict)]
        out.append({
            "id": f"livecodebench:{ex.get('question_id', len(out))}",
            "task_type": "code",
            "problem": str(ex.get("question_content", "")) + ("\n\n" + str(ex.get("starter_code", "")) if ex.get("starter_code") else ""),
            "answer": "",
            "code_format": "functional" if func_name else "io",
            "test": "",
            "test_setup": "",
            "entry_point": func_name,
            "test_cases": norm_cases,
            "source": "livecodebench",
        })
    return out


# Easiness filter for the APPS code training source. APPS problems range from
# introductory to competition; we keep only the EASY, gradeable slice so the peers
# actually spread (vs BigCodeBench where every peer scored ~0%):
#   * stdin/stdout style only (no "fn_name") -> graded out-of-the-box by check_io;
#     LeetCode-style class-method problems need a bespoke harness, so we drop them.
#   * shortest reference solution <= APPS_MAX_SOL_CHARS -> short answer == easier.
APPS_MAX_SOL_CHARS = int(os.environ.get("APPS_MAX_SOL_CHARS", "400"))

# Optional allowlist of APPS ids whose own reference solution passes the exact-match
# check_io grader (built once by scripts/build_apps_solvable_ids.py). Many APPS
# problems accept multiple valid outputs, so exact-match rejects a correct answer --
# those are false-negatives for EVERY peer and flatten the trust gradient. When this
# file exists we keep only verified-solvable problems. Path is overridable for tests.
APPS_SOLVABLE_IDS = os.environ.get("APPS_SOLVABLE_IDS", "data/apps_solvable_ids.json")


def _io_text(value: Any) -> str:
    # An APPS test case's input/output is either a single string or a list of
    # lines. Join lists with newlines (so the program reads real stdin lines),
    # never str() the list -- that would feed it the literal "['2', '3 3', ...]".
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value)


def load_apps(split: str, start_index: int, max_samples: int | None, cache_dir: str | None) -> list[dict[str, Any]]:
    # APPS ships as plain train.jsonl/test.jsonl; read directly for offline use.
    local_dir = _dataset_path("codeparrot/apps")
    jsonl = os.path.join(local_dir, f"{split or 'train'}.jsonl")
    rows = []
    if os.path.isfile(jsonl):
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        from datasets import load_dataset

        rows = list(load_dataset(local_dir, split=split or "train", cache_dir=cache_dir))

    # Filter to the easy, stdin-style slice BEFORE slicing so shards stay aligned.
    solvable: set[str] | None = None
    if APPS_SOLVABLE_IDS and os.path.isfile(APPS_SOLVABLE_IDS):
        with open(APPS_SOLVABLE_IDS) as f:
            solvable = {str(x) for x in json.load(f)}
    kept = []
    for ex in rows:
        if solvable is not None and str(ex.get("id")) not in solvable:
            continue
        try:
            io = json.loads(ex["input_output"]) if isinstance(ex.get("input_output"), str) else (ex.get("input_output") or {})
        except Exception:
            continue
        inputs, outputs = io.get("inputs") or [], io.get("outputs") or []
        if not inputs or "fn_name" in io:  # stdin-style only
            continue
        try:
            sols = json.loads(ex["solutions"]) if isinstance(ex.get("solutions"), str) else (ex.get("solutions") or [])
        except Exception:
            sols = []
        if not sols or min(len(s) for s in sols) > APPS_MAX_SOL_CHARS:
            continue
        kept.append((ex, inputs, outputs))

    kept = _slice(kept, start_index, max_samples)
    out = []
    for ex, inputs, outputs in kept:
        # stdin/stdout cases. A single case's input/output may itself be a list of
        # lines (APPS stores some cases that way); join with newlines instead of
        # str()-ing the Python list literal, which would otherwise feed the program
        # the text "['2', '3 3', ...]" and make every such record unsolvable.
        cases = [{"input": _io_text(i), "output": _io_text(o)} for i, o in zip(inputs, outputs)]
        out.append({
            "id": f"apps:{ex.get('id', len(out))}",
            "task_type": "code",
            "problem": str(ex.get("question", "")),
            "answer": "",
            "code_format": "io",
            "test": "",
            "test_setup": "",
            "entry_point": None,
            "test_cases": cases,
            "source": "apps",
        })
    return out


REGISTRY: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "hotpotqa": load_hotpotqa,
    "triviaqa": load_triviaqa,
    "squad": load_squad,
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
    "bigcodebench": load_bigcodebench,
    "livecodebench": load_livecodebench,
    "apps": load_apps,
}


def load_eval_dataset(
    name: str,
    *,
    split: str | None = None,
    start_index: int = 0,
    max_samples: int | None = None,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Load a registered RAG/code dataset into standardised records.

    Math datasets are not here — use feedback_state.datasets.load_math_dataset and
    tag the records with task_type="math".
    """
    key = name.lower().replace("-", "_")
    aliases = {"hotpot_qa": "hotpotqa", "trivia_qa": "triviaqa", "openai_humaneval": "humaneval"}
    key = aliases.get(key, key)
    if key not in REGISTRY:
        raise KeyError(f"Unknown eval dataset {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[key](split, start_index, max_samples, cache_dir)


def list_eval_datasets() -> list[str]:
    return sorted(REGISTRY)
