"""Pure prompt helpers for Sigma-Mem candidate judging and feedback records."""
from __future__ import annotations

from typing import Sequence

_CANDIDATE_JUDGE_INSTRUCTION = (
    "Should the candidate response under review be selected as the best answer "
    "for the task? Answer only Yes or No."
)


def build_candidate_judge_prompt(
    question: str,
    slot_names: Sequence[str],
    slot_texts: Sequence[str],
    candidate_slot: int,
    *,
    context: str | None = None,
    include_identity: bool = True,
    real: int | None = None,
    slot_notes: Sequence[str] | None = None,
) -> str:
    """Joint comparison prompt for peer-count-invariant candidate scoring.

    The model still sees all available responses, but the continuation is a shared
    Yes/No decision rather than a peer-name token such as "Peer 3". This makes the
    scoring interface reusable when the number of peers changes at evaluation time.
    """
    n = len(slot_names) if real is None else min(real, len(slot_names))
    if not (0 <= int(candidate_slot) < n):
        raise ValueError(f"candidate_slot={candidate_slot} out of range for real={n}")
    parts = [f"Question:\n{str(question).strip()}"]
    if context:
        parts.append(f"Context / Evidence:\n{str(context).strip()}")
    parts.append("Peer responses:")
    for slot in range(n):
        head = f"Response {slot}"
        if slot == int(candidate_slot):
            head += " [candidate under review]"
        if include_identity:
            head += f" [{slot_names[slot]}]"
        if slot_notes is not None and slot < len(slot_notes) and slot_notes[slot]:
            head += f" ({slot_notes[slot]})"
        parts.append(f"{head}:\n{str(slot_texts[slot]).strip()}")
    parts.append(f"Candidate under review: Response {int(candidate_slot)}")
    if slot_notes is not None and any(slot_notes):
        parts.append("Reliability memory: the note after each response is the estimated probability that its "
                     "author is correct on questions like this one, from verified feedback on earlier questions, "
                     "with the number of similar past cases it rests on.")
    parts.append(f"Instruction:\n{_CANDIDATE_JUDGE_INSTRUCTION}")
    parts.append("Answer:")
    return "\n\n".join(parts)
