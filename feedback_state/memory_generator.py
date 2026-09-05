"""Generation with the competence memory: the central model answers, peers are weighted evidence.

Selection is bounded by the per-event oracle (some peer must be right).  Here the
central model writes its own answer, reading the peers' answers together with the
memory's estimate of each peer's reliability on questions like this one.  The
memory (KalmanMemory on frozen-judge addresses) is written only with the peers'
verified labels, so its trajectory along a stream is independent of what the
central model generates: prompts with the exact decide-then-update memory state
can be precomputed for a whole stream, training is ordinary supervised
fine-tuning on (prompt, correct target) pairs, and evaluation is batched decoding.

Prompt modes:
  memory   peers' answers annotated with P(correct) and the evidence count
  peers    peers' answers without any reliability information (ablation)
  solo     the question only (the central model's own ability)
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from feedback_state.tasks import (
    _choice_labels,
    _rag_context_text,
    code_extract_answer,
    peer_is_correct,
    task_type_of,
)

SYSTEM = (
    "You are the central model of a multi-agent system. Several peer models answered the same question. "
    "A reliability memory has tracked, from verified feedback on earlier questions, how often each peer was "
    "correct on similar questions; its estimate for each peer answer is given together with the number of "
    "similar past cases it rests on. Treat the peer answers as evidence weighted by their reliability, verify "
    "them yourself, and produce your own final answer."
)
SYSTEM_PEERS = (
    "You are the central model of a multi-agent system. Several peer models answered the same question. "
    "Treat their answers as evidence, verify them yourself, and produce your own final answer."
)
SYSTEM_SOLO = "Answer the question."

INSTRUCTIONS = {
    "math": "Solve the problem. Reason briefly, then end with a line of the form 'Final answer: <number>'.",
    "rag": "Answer the question using the evidence. End with a line of the form 'Final answer: <short answer>'.",
    "mcqa": "Choose the correct option. End with a line of the form 'Final answer: (<letter>)'.",
    "boolqa": "Decide. End with a line of the form 'Final answer: <yes or no>'.",
    "shortqa": "Answer briefly. End with a line of the form 'Final answer: <answer>'.",
    "code": "Write a complete Python program that reads from standard input and writes the answer to standard "
            "output (use input()/sys.stdin and print()). Return the program inside a single ```python code block.",
}


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def peer_block(index: int, text: str, prob: float | None, evidence: float | None, *, char_limit: int) -> str:
    head = f"Peer {index + 1}"
    if prob is not None:
        n = int(round(evidence or 0.0))
        cases = "no similar past cases yet" if n < 1 else f"based on {n} similar past case{'s' if n != 1 else ''}"
        head += f" (estimated probability correct: {prob:.2f}, {cases})"
    return f"{head}:\n{_clip(text, char_limit)}"


def build_messages(record: dict[str, Any], texts: Sequence[str], *, mode: str, probs: Sequence[float] | None = None,
                   evidence: Sequence[float] | None = None, include_context: bool = True, char_limit: int = 3000) -> list[dict]:
    task = task_type_of(record)
    parts = [f"Question:\n{str(record.get('problem', record.get('question', ''))).strip()}"]
    if task == "mcqa":
        labels = _choice_labels(record)
        choices = record.get("choices") or []
        if choices:
            parts.append("Options:\n" + "\n".join(f"({labels[i] if i < len(labels) else chr(65 + i)}) {c}" for i, c in enumerate(choices)))
    if include_context and task == "rag":
        ctx = _rag_context_text(record)
        if ctx:
            parts.append(f"Context / Evidence:\n{ctx}")
    if mode != "solo":
        blocks = []
        for i, t in enumerate(texts):
            p = float(probs[i]) if (mode == "memory" and probs is not None) else None
            e = float(evidence[i]) if (mode == "memory" and evidence is not None) else None
            blocks.append(peer_block(i, t, p, e, char_limit=char_limit))
        parts.append("Peer answers:\n\n" + "\n\n".join(blocks))
    parts.append("Instruction: " + INSTRUCTIONS.get(task, INSTRUCTIONS["shortqa"]))
    system = {"memory": SYSTEM, "peers": SYSTEM_PEERS, "solo": SYSTEM_SOLO}[mode]
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


def render_prompt(tokenizer, messages: list[dict], *, thinking: bool = False) -> str:
    """Chat template with the generation prompt; Qwen3's thinking mode is off unless ``thinking`` is set."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(thinking))
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


THINK_END = "</think>"


def strip_thinking(text: str) -> str:
    """The answer part of a generation: everything after the last closing think tag (the whole text when there is none)."""
    if THINK_END in text:
        return text.rsplit(THINK_END, 1)[1]
    return text


def target_text(record: dict[str, Any], texts: Sequence[str], correct: Sequence[int], *, mode: str = "peer", char_limit: int = 4000) -> str | None:
    """Supervised target: a correct peer's response (shortest) or, failing that, the gold final answer."""
    task = task_type_of(record)
    if mode == "peer":
        cands = [str(t) for t, c in zip(texts, correct) if int(c) == 1 and str(t).strip()]
        if cands:
            best = min(cands, key=len)
            if task == "code":
                code = code_extract_answer(best)
                return f"```python\n{code.strip()}\n```" if code.strip() else None
            return _clip(best, char_limit)
    if task == "code":
        return None
    gold = str(record.get("answer", "")).strip()
    if not gold:
        return None
    if task == "mcqa":
        return f"Final answer: ({gold})"
    return f"Final answer: {gold}"


def grade(record: dict[str, Any], text: str, *, code_timeout: float = 10.0) -> bool:
    task = task_type_of(record)
    text = strip_thinking(text)   # never grade the reasoning trace, only the answer after it
    if task == "code":
        program = code_extract_answer(text)
        if not program.strip():
            return False
        from data.builders.common.code_grading import score_code_record  # sandboxed subprocess execution
        return bool(score_code_record(record, program, timeout=code_timeout).passed)
    return bool(peer_is_correct(record, None, text))


def prob_from_logit(ell: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(ell)))
