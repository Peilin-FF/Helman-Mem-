"""Task registry: a modular abstraction over heterogeneous task types.

The trust-state pipeline only ever needs a per-peer correctness label
``c_j ∈ [0, 1]`` per record. Math equivalence, open-domain QA (EM/F1) and code
pass@1 all reduce to that scalar. This module is the single place that knows how
to turn a (record, peer_response) into:

  * a *soft* target ``target(...) -> [0,1]`` used as the BCE label and for the
    selection floors/ceiling, and
  * a *binary* ``is_correct(...) -> bool`` used for reported accuracy, and
  * ``extract_answer(...)`` for human-readable predictions, and
  * ``build_peer_prompt(...)`` for the offline peer-generation step.

Every task standardises its question under ``record["problem"]`` and its gold
under ``record["answer"]`` so the encoder + selection model are task-agnostic;
only correctness scoring dispatches on ``record["task_type"]``.

Adding a new task = register one ``TaskSpec``. Adding a new task *mixture* = just
mix records with different ``task_type`` values in one JSONL; nothing else changes.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from feedback_state.utils import extract_final_answer, math_equal

DEFAULT_TASK_TYPE = "math"


def task_type_of(record: dict[str, Any]) -> str:
    return str(record.get("task_type") or DEFAULT_TASK_TYPE).lower()


# ---------------------------------------------------------------------------
# Open-domain QA metrics (SQuAD / HotpotQA / TriviaQA style)
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)


def _normalize_qa(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD normalisation)."""
    text = str(text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _qa_gold_answers(record: dict[str, Any]) -> list[str]:
    golds = [str(record.get("answer", ""))]
    aliases = record.get("answer_aliases") or record.get("aliases") or []
    if isinstance(aliases, (list, tuple)):
        golds.extend(str(a) for a in aliases)
    return [g for g in golds if g.strip()]


def qa_extract_answer(text: str) -> str:
    """Pull a short answer span out of a peer's free-form QA response."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    # Prefer an explicit "Answer: X" tail if present.
    match = list(re.finditer(r"(?:final\s+answer|answer)\s*(?:is|:)\s*([^\n\r]+)", raw, flags=re.IGNORECASE))
    if match:
        return match[-1].group(1).strip().strip(".")
    # Else the last non-empty line (models often end with the short answer).
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return (lines[-1] if lines else raw).strip().strip(".")


def qa_f1(pred: str, golds: list[str]) -> float:
    pred_tokens = _normalize_qa(pred).split()
    best = 0.0
    for gold in golds:
        gold_tokens = _normalize_qa(gold).split()
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def qa_exact_match(pred: str, golds: list[str]) -> bool:
    npred = _normalize_qa(pred)
    return any(npred == _normalize_qa(g) for g in golds)


# ---------------------------------------------------------------------------
# Code extraction (used for reporting + by the offline scorer)
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def code_extract_answer(text: str) -> str:
    """Extract a code block from a peer response (fenced block, else raw text)."""
    raw = str(text or "")
    blocks = _CODE_FENCE_RE.findall(raw)
    if blocks:
        return blocks[-1].strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# TaskSpec + registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskSpec:
    name: str
    # (peer_text, record) -> soft correctness in [0, 1] (BCE target)
    target_fn: Callable[[str, dict[str, Any]], float]
    # (peer_text, record) -> binary correctness (reported accuracy)
    correct_fn: Callable[[str, dict[str, Any]], bool]
    # peer_text -> human-readable extracted answer
    extract_fn: Callable[[str], str]
    # (record, with_context) -> prompt string for offline peer generation
    prompt_fn: Callable[[dict[str, Any], bool], str]
    # True if correctness is precomputed per-peer (read from record["peer_correct"])
    precomputed: bool = False


def _math_target(text: str, record: dict[str, Any]) -> float:
    pred = extract_final_answer(str(text))
    return 1.0 if (pred and math_equal(pred, str(record.get("answer", "")))) else 0.0


def _math_correct(text: str, record: dict[str, Any]) -> bool:
    return _math_target(text, record) >= 0.5


def _math_prompt(record: dict[str, Any], with_context: bool) -> str:
    return (
        "Solve this math problem. Give a concise solution and end with the final answer.\n\n"
        f"Problem:\n{record.get('problem', '')}"
    )


def _rag_context_text(record: dict[str, Any]) -> str:
    # Cap retrieved context to keep the peer prompt within model length. TriviaQA
    # ships whole wiki/web docs (median ~54k chars, max ~514k) which blow past the
    # context window; HotpotQA passages are short so this never affects them.
    max_chars = 8000
    ctx = record.get("context", "")
    if isinstance(ctx, (list, tuple)):
        parts = []
        for i, passage in enumerate(ctx, start=1):
            if isinstance(passage, (list, tuple)):  # HotpotQA [title, [sentences]]
                title = str(passage[0]) if passage else ""
                body = " ".join(str(s) for s in passage[1]) if len(passage) > 1 else ""
                parts.append(f"[{i}] {title}: {body}".strip())
            else:
                parts.append(f"[{i}] {str(passage)}")
        text = "\n".join(parts)
    else:
        text = str(ctx)
    return text[:max_chars]


def _rag_target(text: str, record: dict[str, Any]) -> float:
    # Soft graded trust via token-F1 against gold (and aliases).
    return qa_f1(qa_extract_answer(text), _qa_gold_answers(record))


def _rag_correct(text: str, record: dict[str, Any]) -> bool:
    return qa_exact_match(qa_extract_answer(text), _qa_gold_answers(record))


def _rag_prompt(record: dict[str, Any], with_context: bool) -> str:
    question = record.get("problem", "")
    if with_context:
        context = _rag_context_text(record)
        return (
            "Answer the question using the retrieved context. Respond with a short answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
    # Deprived peer (e.g. Gemma for RAG): no retrieved documents.
    return (
        "Answer the question with a short answer.\n\n"
        f"Question: {question}\nAnswer:"
    )


def _code_precomputed(record: dict[str, Any], peer_key: str | None) -> float | None:
    table = record.get("peer_correct")
    if isinstance(table, dict) and peer_key is not None and peer_key in table:
        return float(table[peer_key])
    return None


def _code_target(text: str, record: dict[str, Any]) -> float:
    # Code correctness is precomputed offline (see scripts/score_code_peers.py)
    # and looked up by peer_key in peer_target_value; this text-only fallback
    # cannot execute, so it returns 0.0 (unknown == untrusted).
    return 0.0


def _code_correct(text: str, record: dict[str, Any]) -> bool:
    return False


def _code_prompt(record: dict[str, Any], with_context: bool) -> str:
    # APPS-style problems are graded by running the program against stdin/stdout,
    # so the peer must read input() and print the answer -- NOT define a function.
    if str(record.get("code_format", "")) == "io":
        return (
            "Write a complete Python program that reads from standard input and "
            "writes the answer to standard output. Use input()/sys.stdin to read and "
            "print() to write. Return only the program inside a ```python code block.\n\n"
            f"{record.get('problem', '')}"
        )
    return (
        "Complete the following Python function. Return only the full function "
        "implementation inside a ```python code block.\n\n"
        f"{record.get('problem', '')}"
    )


REGISTRY: dict[str, TaskSpec] = {
    "math": TaskSpec("math", _math_target, _math_correct, extract_final_answer, _math_prompt),
    "rag": TaskSpec("rag", _rag_target, _rag_correct, qa_extract_answer, _rag_prompt),
    "code": TaskSpec(
        "code", _code_target, _code_correct, code_extract_answer, _code_prompt, precomputed=True
    ),
}


def get_task(name: str) -> TaskSpec:
    key = str(name or DEFAULT_TASK_TYPE).lower()
    if key not in REGISTRY:
        raise KeyError(f"Unknown task_type {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[key]


def register_task(spec: TaskSpec) -> None:
    REGISTRY[spec.name.lower()] = spec


# ---------------------------------------------------------------------------
# Dispatching helpers used by the data collator and the evaluator
# ---------------------------------------------------------------------------

def peer_target_value(record: dict[str, Any], peer_key: str | None, peer_text: str) -> float:
    """Soft per-peer correctness in [0,1] (the BCE training target / coverage)."""
    # Precomputed labels (any task): if the record carries a peer_correct map, trust
    # it. Lets us pre-grade slow math (sympy) ONCE offline and have eval read it
    # instead of re-running the grader per eval job (which can hang on pathological
    # sympy exprs). Code always uses this path (offline pass@1).
    if peer_key is not None:
        pre = _code_precomputed(record, peer_key)
        if pre is not None:
            return max(0.0, min(1.0, pre))
    task = get_task(task_type_of(record))
    if task.precomputed:
        pre = _code_precomputed(record, peer_key)
        if pre is not None:
            return max(0.0, min(1.0, pre))
    return float(task.target_fn(peer_text, record))


def peer_is_correct(record: dict[str, Any], peer_key: str | None, peer_text: str) -> bool:
    """Binary per-peer correctness used for reported accuracy."""
    task = get_task(task_type_of(record))
    if task.precomputed:
        pre = _code_precomputed(record, peer_key)
        if pre is not None:
            return pre >= 0.5
    return bool(task.correct_fn(peer_text, record))


def extract_answer(record: dict[str, Any], peer_text: str) -> str:
    return get_task(task_type_of(record)).extract_fn(peer_text)


def build_peer_prompt(record: dict[str, Any], *, with_context: bool = True) -> str:
    return get_task(task_type_of(record)).prompt_fn(record, with_context)
