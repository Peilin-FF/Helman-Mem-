"""Generating and grading a peer model's answers: the parts every peer-generation run shares.

The prompt is the task's peer prompt (feedback_state.tasks.build_peer_prompt), rendered with the peer's own chat template
as one user turn with thinking off. Honest answers are sampled at temperature 0.2, top-p 0.95, as the six peers of the
released streams were. Grading is the streams' rule: token-F1 >= 0.5 for reading, exact equality for math (in a
subprocess with a hard timeout, since sympy can hang), the hidden tests executed for code.
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from feedback_state.tasks import get_task, task_type_of

DEFAULT_MAX_TOKENS = {"math": 512, "rag": 256, "code": 768, "boolqa": 96, "mcqa": 96, "shortqa": 96}
# A misleading answer argues for its conclusion: 96 tokens cut most arguments off before the final line (the
# 2026-09-13 run had 37% of one peer's OOD answers rewritten for that reason), so the short-answer tasks get 256.
MISLEADING_MAX_TOKENS = {**DEFAULT_MAX_TOKENS, "boolqa": 256, "mcqa": 256, "shortqa": 256}
REASONING_MAX_TOKENS = 4096


def render(tokenizer, content: str) -> str:
    """One user turn through the peer's chat template, thinking off where the template knows the switch."""
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt_ids(tokenizer, content: str) -> dict:
    """The engine input for one user turn: the rendered chat template as token ids, with no special tokens added.

    A chat template already starts with the model's BOS token where the model needs one; handing the rendered text to
    the engine lets its tokenizer add a second one. Gemma-3, Llama-3.1 and both DeepSeek peers then see a doubled BOS,
    and DeepSeek-Coder-V2-Lite answers with unrelated text or symbol runs.
    """
    return {"prompt_token_ids": tokenizer(render(tokenizer, content), add_special_tokens=False)["input_ids"]}


def max_tokens(record: dict, budgets: dict | None = None, reasoning: bool = False) -> int:
    if reasoning:
        return REASONING_MAX_TOKENS
    return int((budgets or DEFAULT_MAX_TOKENS).get(task_type_of(record), 512))


def _target(record: dict, text: str) -> float:
    return float(get_task(task_type_of(record)).target_fn(text, record))


def _worker(record, text, q):
    try:
        q.put(_target(record, text))
    except Exception:
        q.put(0.0)


def graded_value(record: dict, text: str, timeout: float = 4.0) -> float:
    """The verified correctness in [0, 1] of one answer (token-F1 for reading, 0/1 otherwise)."""
    task = task_type_of(record)
    if task == "code":
        from feedback_state.memory_generator import grade

        return 1.0 if grade(record, text) else 0.0
    if task != "math":
        try:
            return _target(record, text)
        except Exception:
            return 0.0
    q = mp.Queue()
    proc = mp.Process(target=_worker, args=(record, text, q))
    proc.start(); proc.join(timeout)
    if proc.is_alive():
        proc.terminate(); proc.join()
        return 0.0
    try:
        return q.get_nowait()
    except Exception:
        return 0.0


def grade_all(items: Iterable[tuple[dict, str]], workers: int = 8) -> list[float]:
    """Grade many answers; code and math grade in subprocesses, so threads are enough to parallelise."""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(lambda it: graded_value(*it), items))
