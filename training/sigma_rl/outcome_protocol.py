"""Label-private, answer-then-feedback protocol for the new generation experiment.

This is an orchestration layer around the ORIGINAL MemoryRuntime, not a new
memory implementation or a replacement for ActivationSteerer. In particular it
does not choose how multiple peer evidence vectors enter one generation. That
model/rollout integration must be explicit; the legacy text-guided trainer is
not a valid consumer of this protocol.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch

from feedback_state.memory_generator import INSTRUCTIONS, render_prompt
from feedback_state.memory_runtime import MemoryRuntime
from feedback_state.tasks import task_type_of

PROTOCOL = "peer_outcome_v1"
MAX_PROMPT_LENGTH = 8192
SYSTEM = (
    "Solve the question yourself. Other models' responses are provided as material "
    "to examine, not as verified solutions. Compare their reasoning, check it, and "
    "produce your own solution and final answer. You may use useful parts of any "
    "response or reject all of them."
)


def _context_text(context: Any) -> str:
    """The existing passage layout, without its hidden 8,000-character cut-off."""
    if not isinstance(context, (list, tuple)):
        return str(context or "")
    parts = []
    for i, passage in enumerate(context, 1):
        if isinstance(passage, (list, tuple)):
            title = str(passage[0]) if passage else ""
            body = " ".join(str(s) for s in passage[1]) if len(passage) > 1 else ""
            parts.append(f"[{i}] {title}: {body}".strip())
        else:
            parts.append(f"[{i}] {passage}")
    return "\n".join(parts)


def public_messages(record: Mapping[str, Any], peer_texts: Sequence[str]) -> list[dict[str, str]]:
    """Build from public fields only; never inspect gold, scores or peer labels.

    All peers are retained, including empty/incorrect responses. There is no
    per-peer character limit and no reliability-to-text conversion. The final
    tokenizer budget is enforced separately and never silently falls back to a
    solo prompt. Peer slot order is chosen by the caller without seeing labels.
    """
    if not peer_texts:
        raise ValueError("the peer-conditioned experiment requires at least one peer")
    task = task_type_of(record)
    question = str(record.get("problem", record.get("question", ""))).strip()
    if not question:
        raise ValueError("missing public question")
    parts = [f"Question:\n{question}"]
    if task == "mcqa" and record.get("choices"):
        choices = record["choices"]
        labels = record.get("choice_labels") or [chr(65 + i) for i in range(len(choices))]
        if len(labels) != len(choices):
            raise ValueError("choice labels must align with choices")
        parts.append("Options:\n" + "\n".join(f"({label}) {choice}" for label, choice in zip(labels, choices)))
    if task == "rag":
        context = _context_text(record.get("context"))
        if context:
            parts.append("Context / Evidence:\n" + context)
    parts.append("Peer responses:\n\n" + "\n\n".join(
        f"Peer {i + 1}:\n{text}" for i, text in enumerate(peer_texts)
    ))
    parts.append("Instruction: " + INSTRUCTIONS.get(task, INSTRUCTIONS["shortqa"]))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def checked_prompt_ids(tokenizer, messages: Sequence[dict], *, max_length: int = MAX_PROMPT_LENGTH) -> list[int]:
    """Reject an over-budget event, without dropping any peer or consuming feedback."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    ids = tokenizer.encode(render_prompt(tokenizer, list(messages)), add_special_tokens=False)
    if not ids:
        raise ValueError("empty tokenized prompt")
    if len(ids) > max_length:
        raise ValueError(
            f"complete question + peers prompt has {len(ids)} tokens, exceeds {max_length}; "
            "no truncation, peer selection or solo fallback was performed"
        )
    return [int(i) for i in ids]


@dataclass(frozen=True)
class PolicyObservation:
    """Only public text and the original pre-feedback evidence [peers, 4].

    No reference to the record/FeatureStream/runtime/labels is sent to a worker.
    The evidence is a detached snapshot, kept unchanged for old log-probability
    and policy-loss recomputation after the live matrix advances.
    """
    position: int
    messages: tuple[tuple[str, str], ...]
    peer_order: tuple[int, ...]             # slot -> canonical identity
    evidence: torch.Tensor                 # original MemoryRuntime.read()[2]

    def chat_messages(self) -> list[dict[str, str]]:
        return [{"role": role, "content": content} for role, content in self.messages]

    def copy(self) -> "PolicyObservation":
        return PolicyObservation(self.position, self.messages, self.peer_order, self.evidence.detach().clone())


@dataclass(frozen=True)
class CompletedEvent:
    observation: PolicyObservation
    responses: tuple[str, ...]
    rewards: tuple[float, ...]              # center outcomes only, no peer labels


def binary_outcome(value: Any, *, name: str) -> float:
    if torch.is_tensor(value) and value.numel() == 1:
        value = value.item()
    if not isinstance(value, Real) or value not in (0, 1):
        raise ValueError(f"{name} must be a verified binary outcome, got {value!r}")
    return float(value)


class CausalPeerEpisode:
    """One ordered stream; original read -> center answers -> grade -> original write.

    GPU sampling for the same question can be parallel (a GRPO group). Reading
    the next question waits for feedback on this one. A GPU optimizer batch may
    contain several completed questions, each with its own frozen observation.
    Labels are fetched lazily only after ALL center responses are graded.

    This wrapper does not reset or change the matrix, fit projections, permute
    identities in storage, or insert the center's reward into peer heads.
    """
    def __init__(
        self,
        runtime: MemoryRuntime,
        *,
        feedback: Callable[[int], Sequence[int]],
        rollouts_per_question: int,
        start_position: int = 0,
    ) -> None:
        if rollouts_per_question < 2:
            raise ValueError("GRPO requires at least two own-policy samples per question")
        if start_position < 0:
            raise ValueError("start_position must be nonnegative")
        self.runtime = runtime
        self.feedback = feedback
        self.rollouts_per_question = int(rollouts_per_question)
        self.next_position = int(start_position)
        self._pending: tuple[PolicyObservation, torch.Tensor, int] | None = None

    def begin(
        self, position: int, record: Mapping[str, Any], peer_texts: Sequence[str],
        *, peer_order: Sequence[int] | None = None,
    ) -> PolicyObservation:
        if self._pending is not None:
            raise RuntimeError("the previous question must finish before reading another question")
        if position != self.next_position:
            raise ValueError(f"expected stream position {self.next_position}, got {position}")
        r = self.runtime.real[position]
        if len(peer_texts) != r:
            raise ValueError("peer texts must match the real peers in the original memory addresses")
        order = tuple(range(r)) if peer_order is None else tuple(peer_order)
        if sorted(order) != list(range(r)):
            raise ValueError("peer_order must contain every real peer exactly once")
        messages = public_messages(record, [peer_texts[p] for p in order])
        _, _, evidence, rows = self.runtime.read(position)
        if evidence.shape != (r, 4) or not torch.isfinite(evidence).all():
            raise ValueError("expected original finite [peers, 4] memory evidence")
        observation = PolicyObservation(
            position, tuple((m["role"], m["content"]) for m in messages), order,
            evidence[list(order)].detach().clone(),
        )
        self._pending = (observation, rows.detach().clone(), self.runtime.mem.writes)
        return observation.copy()

    def complete(
        self, responses: Sequence[str], *, score: Callable[[str], float],
    ) -> CompletedEvent:
        if self._pending is None:
            raise RuntimeError("begin a question and generate its responses before feedback")
        observation, rows, writes_before = self._pending
        if isinstance(responses, str) or len(responses) != self.rollouts_per_question:
            raise ValueError(f"expected {self.rollouts_per_question} center responses, not a selected best answer")
        if not all(isinstance(text, str) for text in responses):
            raise TypeError("center responses must be strings")
        if self.runtime.mem.writes != writes_before:
            raise RuntimeError("memory was written before the center's responses were scored")
        # A verifier failure leaves the event pending and the matrix unchanged.
        replies = tuple(responses)
        rewards = tuple(binary_outcome(score(text), name="center reward") for text in replies)
        labels = list(self.feedback(observation.position))
        r = len(observation.peer_order)
        if len(labels) != r:
            raise ValueError("post-answer feedback must cover every real peer in canonical order")
        labels = [int(binary_outcome(y, name="peer feedback")) for y in labels]
        if self.runtime.mem.writes != writes_before:
            raise RuntimeError("a scorer or feedback provider mutated the memory")
        snapshot = self.runtime.mem.snapshot()
        try:
            self.runtime.write(observation.position, rows, labels)
        except Exception:
            self.runtime.mem.restore(snapshot)
            raise
        self._pending = None
        self.next_position += 1
        return CompletedEvent(observation.copy(), replies, rewards)

    @property
    def awaiting_response(self) -> bool:
        return self._pending is not None
