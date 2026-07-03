"""Pure helpers for the JOINT-input multi-agent selectors (Variants A & B).

Unlike the per-peer-independent path (each peer encoded separately), here ALL peers
are written into ONE prompt so the central model can compare responses jointly. The
shared online state S steers the central model's attention (Delta-Mem), not the text.

These helpers are torch-free (string/list construction only) so they're unit-testable
without a GPU. They reuse the canonical peer view + the existing slot/peer convention:
peers appear in SLOT order; the AR target and BCE labels are SLOT-indexed; eval maps
the selected slot back to a stable peer_id exactly like the rest of the repo.
"""
from __future__ import annotations

from typing import Any, Sequence

PEER_SEP = "<PEER_SEP>"
_INSTRUCTION = (
    "Select the peer or peers whose response should be trusted. "
    f"Answer using only peer IDs separated by {PEER_SEP}."
)
_CANDIDATE_JUDGE_INSTRUCTION = (
    "Should the candidate response under review be selected as the best answer "
    "for the task? Answer only Yes or No."
)


def peer_id_string(slot: int) -> str:
    return f"Peer {slot}"


def candidate_peer_strings(num_peers: int) -> list[str]:
    """The candidate answer strings scored at eval time: ['Peer 0', 'Peer 1', ...]."""
    return [peer_id_string(s) for s in range(num_peers)]


def build_joint_prompt(
    question: str,
    slot_names: Sequence[str],
    slot_texts: Sequence[str],
    *,
    context: str | None = None,
    include_identity: bool = True,
    real: int | None = None,
) -> str:
    """One joint prompt with every peer's id/name + response (slot order).

    ``real`` (if given) limits the peer blocks to the real peers (pads omitted).
    Identity (the model name) moves with the peer, in ``id`` mode.
    """
    n = len(slot_names) if real is None else min(real, len(slot_names))
    parts = [f"Question:\n{str(question).strip()}"]
    if context:
        parts.append(f"Context / Evidence:\n{str(context).strip()}")
    for slot in range(n):
        head = f"Peer {slot}"
        if include_identity:
            head += f" [{slot_names[slot]}]"
        parts.append(f"{head}:\n{str(slot_texts[slot]).strip()}")
    parts.append(f"Instruction:\n{_INSTRUCTION}")
    parts.append("Answer:")
    return "\n\n".join(parts)


def build_candidate_judge_prompt(
    question: str,
    slot_names: Sequence[str],
    slot_texts: Sequence[str],
    candidate_slot: int,
    *,
    context: str | None = None,
    include_identity: bool = True,
    real: int | None = None,
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
        parts.append(f"{head}:\n{str(slot_texts[slot]).strip()}")
    parts.append(f"Candidate under review: Response {int(candidate_slot)}")
    parts.append(f"Instruction:\n{_CANDIDATE_JUDGE_INSTRUCTION}")
    parts.append("Answer:")
    return "\n\n".join(parts)


def _label(correct: bool | None) -> str:
    return "correct" if correct else "incorrect"


def build_feedback_prompt(
    question: str,
    slot_names: Sequence[str],
    slot_texts: Sequence[str],
    correctness_by_slot: Sequence[bool],
    *,
    context: str | None = None,
    include_identity: bool = True,
    real: int | None = None,
    selected_slot: int | None = None,
    selected_correct: bool | None = None,
    include_selected_peer: bool = True,
    labels_only: bool = False,
    answer_labels: bool = False,
    short_answers: Sequence[str] | None = None,
) -> str:
    """The WRITE-pass prompt with EXPLICIT per-peer correctness labels.

    Unlike the selection prompt (which only reveals the *target* peer id), this
    grounds the Delta-Mem write in correctness for EVERY peer:

        Peer 0 [Gemma]:
        {response_0}
        Feedback: incorrect
        ...
        Selected peer:
        Peer 1
        Selected peer correctness:
        correct

    The selected-peer lines are optional (``include_selected_peer``); the all-peer
    ``Feedback: correct/incorrect`` lines are always present. Guaranteed to contain
    the substring "Feedback: correct" or "Feedback: incorrect" (write-safety check).

    ``labels_only`` produces a MINIMAL write: it drops the question, context, and the
    peer response texts, keeping only ``Peer j [identity]: Feedback: correct/incorrect``
    per peer. A per-peer (real=1) labels-only write is ~23 tokens, so the whole write
    fits inside the state-decay survival window and the trust update isn't wiped by the
    re-read of long response text (see plan / MEMORY.md).
    """
    n = len(slot_names) if real is None else min(real, len(slot_names))
    if labels_only:
        parts = []
        for slot in range(n):
            head = f"Peer {slot}" + (f" [{slot_names[slot]}]" if include_identity else "")
            c = bool(correctness_by_slot[slot]) if slot < len(correctness_by_slot) else False
            parts.append(f"{head}: Feedback: {_label(c)}")
        if include_selected_peer and selected_slot is not None:
            sc = selected_correct
            if sc is None and 0 <= int(selected_slot) < n:
                sc = bool(correctness_by_slot[int(selected_slot)])
            parts.append(f"Selected peer: Peer {int(selected_slot)} ({_label(sc)})")
        return "\n".join(parts)
    if answer_labels:
        # Like labels_only, but each peer's SHORT ANSWER precedes its label so the
        # state records WHAT the peer said (e.g. \boxed{42} / "3/5 tests passed"),
        # not just the bare correct/incorrect oracle bit. Still short (~16 tok/peer)
        # to survive the state-decay window. short_answers is slot-indexed.
        sa = list(short_answers or [])
        parts = []
        for slot in range(n):
            head = f"Peer {slot}" + (f" [{slot_names[slot]}]" if include_identity else "")
            c = bool(correctness_by_slot[slot]) if slot < len(correctness_by_slot) else False
            ans = str(sa[slot]).strip() if slot < len(sa) and sa[slot] is not None else ""
            ans = " ".join(ans.split())  # collapse whitespace/newlines into one line
            if ans:
                parts.append(f"{head}: {ans} Feedback: {_label(c)}")
            else:
                parts.append(f"{head}: Feedback: {_label(c)}")
        if include_selected_peer and selected_slot is not None:
            sc = selected_correct
            if sc is None and 0 <= int(selected_slot) < n:
                sc = bool(correctness_by_slot[int(selected_slot)])
            parts.append(f"Selected peer: Peer {int(selected_slot)} ({_label(sc)})")
        return "\n".join(parts)
    parts = [f"Question:\n{str(question).strip()}"]
    if context:
        parts.append(f"Context / Evidence:\n{str(context).strip()}")
    for slot in range(n):
        head = f"Peer {slot}" + (f" [{slot_names[slot]}]" if include_identity else "")
        c = bool(correctness_by_slot[slot]) if slot < len(correctness_by_slot) else False
        parts.append(f"{head}:\n{str(slot_texts[slot]).strip()}\nFeedback: {_label(c)}")
    if include_selected_peer and selected_slot is not None:
        parts.append(f"Selected peer:\nPeer {int(selected_slot)}")
        sc = selected_correct
        if sc is None and 0 <= int(selected_slot) < n:
            sc = bool(correctness_by_slot[int(selected_slot)])
        parts.append(f"Selected peer correctness:\n{_label(sc)}")
    return "\n\n".join(parts)


def peer_response_char_spans(
    question: str,
    slot_names: Sequence[str],
    slot_texts: Sequence[str],
    *,
    context: str | None = None,
    include_identity: bool = True,
    real: int | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    """Build the joint prompt AND return each peer response's [char_start, char_end).

    The collator maps these char spans to token spans via the tokenizer's
    offset_mapping (Variant B pools the joint hidden states over the response span).
    """
    prompt = build_joint_prompt(
        question, slot_names, slot_texts, context=context,
        include_identity=include_identity, real=real,
    )
    n = len(slot_names) if real is None else min(real, len(slot_names))
    spans: list[tuple[int, int]] = []
    cursor = 0
    for slot in range(n):
        resp = str(slot_texts[slot]).strip()
        head = f"Peer {slot}" + (f" [{slot_names[slot]}]" if include_identity else "") + ":"
        h = prompt.find(head, cursor)
        start = prompt.find(resp, h + len(head)) if (h >= 0 and resp) else -1
        if start >= 0:
            spans.append((start, start + len(resp)))
            cursor = start + len(resp)
        else:
            spans.append((-1, -1))
    return prompt, spans


def canonical_target_peer_id(
    correctness_by_peer_id: Sequence[bool],
    *,
    target_peer_ids: Sequence[int] | None = None,
    reliability_by_peer_id: Sequence[float] | None = None,
) -> int | None:
    """Deterministic single target among the correct peers (multi_correct_target_mode='canonical').

    Prefer the highest-trust correct peer when a reliability estimate is available;
    otherwise the lowest-index correct peer. Returns None when no peer is correct.
    """
    if target_peer_ids:
        correct = list(target_peer_ids)
    else:
        correct = [i for i, c in enumerate(correctness_by_peer_id) if c]
    if not correct:
        return None
    if reliability_by_peer_id is not None:
        return max(correct, key=lambda i: (reliability_by_peer_id[i], -i))
    return min(correct)


def ar_target_text(target_slots: Sequence[int]) -> str:
    """The autoregressive target string for a (possibly multi-) peer target.

    Canonical mode passes a single slot; multi-target passes several -> joined by
    the separator, e.g. 'Peer 0 <PEER_SEP> Peer 2'.
    """
    return f" {PEER_SEP} ".join(peer_id_string(s) for s in target_slots)


def peer_label_vector(correctness_by_slot: Sequence[bool], num_peers: int) -> list[float]:
    """BCE label vector z_j in {0,1} per SLOT (Variant B)."""
    return [1.0 if (s < len(correctness_by_slot) and correctness_by_slot[s]) else 0.0 for s in range(num_peers)]


def needs_peer_sep(num_correct: int) -> bool:
    """Whether the <PEER_SEP> special token is actually used (multi-target targets)."""
    return num_correct > 1
