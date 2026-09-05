"""Group the candidates of one event by the answer they give.

Reliability-weighted voting (Nitzan-Paroush) needs to know which candidates
agree.  Agreement is defined per task type with the repository's own graders so
that a vote never uses information the evaluator does not have:

* mcqa / boolqa / shortqa: the frozen OOD canonicalisation of ``ood_routing``;
* math: pairwise ``math_equal`` on the extracted final answers;
* rag: SQuAD-normalised extracted short answers;
* code: no agreement is measurable (every program is its own group).

Correctness labels are never consulted here.
"""
from __future__ import annotations

from typing import Any, Sequence

from feedback_state.ood_routing import INVALID_ANSWER, canonical_answer
from feedback_state.tasks import _normalize_qa, qa_extract_answer, task_type_of
from feedback_state.utils import extract_final_answer, math_equal


def answer_groups(record: dict[str, Any], texts: Sequence[str]) -> list[int]:
    """Return group ids (0..k-1) per candidate; equal ids = same answer."""
    n = len(texts)
    task = task_type_of(record)
    if task in ("mcqa", "boolqa", "shortqa"):
        keys = [canonical_answer(record, t) for t in texts]
        return _ids([None if k == INVALID_ANSWER else k for k in keys])
    if task == "math":
        finals = [extract_final_answer(t) for t in texts]
        ids = [-1] * n
        nxt = 0
        for i in range(n):
            if ids[i] >= 0:
                continue
            ids[i] = nxt
            if finals[i]:
                for j in range(i + 1, n):
                    if ids[j] < 0 and finals[j] and math_equal(finals[i], finals[j]):
                        ids[j] = nxt
            nxt += 1
        return ids
    if task == "rag":
        keys = [_normalize_qa(qa_extract_answer(t)) for t in texts]
        return _ids([k or None for k in keys])
    return list(range(n))  # code and unknown task types: no measurable agreement


def _ids(keys: list) -> list[int]:
    """Dense group ids; ``None`` (unparsable) answers never share a group."""
    marked = [("__unique__", i) if k is None else k for i, k in enumerate(keys)]
    dense: dict = {}
    return [dense.setdefault(x, len(dense)) for x in marked]
