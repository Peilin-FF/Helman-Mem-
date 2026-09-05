"""Memory hooks for reinforcement learning of the central model.

The memory (the Kalman/RLS posterior over peer reliability on the question address)
is a label of the peers' historical performance.  During RL it enters in exactly two
places, both pure functions of (record, prompt row, generated text):

  hint      when the policy fails a verified prompt (every sample wrong), the solution of
            the peer the memory trusts most among the verified-correct ones is shown as a
            hint; the model re-answers in its own words and, if that answer verifies, the
            answer is trained under the question-only prompt (internalisation, never
            imitation of the peer's text);
  pseudo    when a prompt has no verifier, the reliability-weighted vote over the peers'
            answers replaces the verifier reward.

Shared by the single-GPU trainer (feedback_state.train_rlvr) and the multi-GPU stack
(training/sigma_rl).
"""
from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np

from feedback_state.answer_groups import answer_groups
from feedback_state.memory_generator import INSTRUCTIONS, SYSTEM_PEERS, _clip
from feedback_state.tasks import _choice_labels, _rag_context_text, task_type_of

HINT_SYSTEM = "You are a careful problem solver."
HINT_CHOICES = ("memory", "label_memory", "label_random", "none")


def hint_messages(record: dict[str, Any], hint_text: str, prob: float, evidence: float, *, char_limit: int = 3000) -> list[dict]:
    """Chat messages of a hinted attempt: the question (+ options / context), one peer solution
    labelled with the memory's reliability estimate, and the task instruction."""
    task = task_type_of(record)
    n = int(round(float(evidence)))
    parts = [f"Question:\n{str(record.get('problem', record.get('question', ''))).strip()}"]
    if task == "mcqa" and record.get("choices"):
        parts.append("Options:\n" + "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(record["choices"])))
    if task == "rag":
        ctx = _rag_context_text(record)
        if ctx:
            parts.append(f"Context / Evidence:\n{ctx}")
    parts.append(f"A peer whose reliability on questions like this is estimated at {float(prob):.2f} (from {n} similar past cases) answered:\n{str(hint_text).strip()[:char_limit]}")
    parts.append("Instruction: Use the peer's answer only as a hint. Solve the problem yourself, in your own words. " + INSTRUCTIONS.get(task, INSTRUCTIONS["shortqa"]))
    return [{"role": "system", "content": HINT_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def peer_texts_in_prompt_order(record: dict[str, Any], peer_order: Sequence[int]) -> list[str]:
    """The peers' responses in the order they appear in the prompt (slot -> canonical peer id via peer_order)."""
    keys = sorted(record.get("peer_responses", {}))
    return [str(record["peer_responses"][keys[int(p)]]) for p in peer_order]


def choose_hint_slot(peer_correct: Sequence[int], memory_prob: Sequence[float], *, choice: str = "memory", rng: random.Random | None = None,
                     valid: Sequence[bool] | None = None) -> int | None:
    """Slot of the peer whose solution is shown to the central model.

    memory         the peer the memory trusts most, labels unknown (our regime: the label comes after the answer)
    label_memory   the most reliable *verified-correct* peer (labels before the answer: the classical baseline)
    label_random   a random verified-correct peer (labels before, no memory)
    none           no hint
    ``valid`` marks slots with a usable (non-empty) solution.
    """
    if choice not in HINT_CHOICES:
        raise ValueError(f"hint choice {choice!r} not in {HINT_CHOICES}")
    slots = [s for s in range(len(memory_prob)) if valid is None or bool(valid[s])]
    if choice == "none" or not slots:
        return None
    if choice == "memory":
        return max(slots, key=lambda s: float(memory_prob[s]))
    correct = [s for s in slots if int(peer_correct[s]) == 1]
    if not correct:
        return None
    if choice == "label_memory":
        return max(correct, key=lambda s: float(memory_prob[s]))
    return (rng or random).choice(correct)


def memory_pseudo_reward(record: dict[str, Any], peer_texts: Sequence[str], memory_prob: Sequence[float], text: str) -> float | None:
    """1 if the sample's answer falls in the peers' answer group with the largest reliability-weighted
    vote (log-odds weights) and that group is trusted (total log-odds > 0), 0 if it falls elsewhere,
    None when agreement is not measurable (code, or no trusted group)."""
    groups = answer_groups(record, list(peer_texts) + [str(text)])
    if task_type_of(record) == "code" and len(set(groups[:-1])) == len(groups[:-1]):
        return None
    weights = [float(np.log(max(float(pr), 1e-4) / max(1.0 - float(pr), 1e-4))) for pr in memory_prob]
    totals: dict[int, float] = {}
    for g, w in zip(groups[:-1], weights):
        totals[g] = totals.get(g, 0.0) + w
    if not totals:
        return None
    best = max(totals, key=totals.get)
    if totals[best] <= 0.0:
        return None
    return 1.0 if groups[-1] == best else 0.0


LABELED_SYSTEM = (
    "You are the central model of a multi-agent system. Several peer models answered the same question, and each "
    "answer has been verified: it is marked correct or incorrect. Use the verified answers as you see fit and produce "
    "your own final answer."
)


def labeled_peer_messages(record: dict[str, Any], texts: Sequence[str], correct: Sequence[int], *, include_context: bool = True, char_limit: int = 3000) -> list[dict]:
    """The peers' solutions with their verified correctness (the classical regime: labels before the answer, no memory).
    Same layout as feedback_state.memory_generator.build_messages(mode="peers"), with a verdict in each peer's header."""
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
    blocks = [f"Peer {i + 1} (verified: {'correct' if int(c) == 1 else 'incorrect'}):\n{_clip(str(t), char_limit)}" for i, (t, c) in enumerate(zip(texts, correct))]
    parts.append("Peer answers:\n\n" + "\n\n".join(blocks))
    parts.append("Instruction: " + INSTRUCTIONS.get(task, INSTRUCTIONS["shortqa"]))
    return [{"role": "system", "content": LABELED_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
