"""Modular, safe Delta-Mem WRITE protocol for the joint shared-state selector.

Separates two concepts cleanly:
  * write_enabled  — whether Delta-Mem may update the shared online state at all.
  * the WRITE prompt — what tokens the controlled write pass reads:
      - "none"             : no write pass (state unchanged);
      - "selection_tokens" : the ordinary selection prompt (no correctness labels);
      - "feedback"         : an explicit feedback prompt with per-peer correct/incorrect.

Candidate scoring is ALWAYS read-only (handled in JointDeltaMemSelector.score_candidates);
this module only runs the optional, separate write pass AFTER selection. Used by both
train and eval so the protocol is identical and configurable.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from feedback_state.joint_prompt import build_feedback_prompt, build_joint_prompt

WRITE_POLICIES = ("none", "selection_tokens", "feedback")


def build_short_answers(
    record: dict,
    slot_texts: Sequence[str],
    slot_peer_keys: Sequence[str] | None = None,
    real: int | None = None,
) -> list[str]:
    """Per-slot SHORT answer for the answer_labels write style.

      * math -> _locate_final_answer(resp): raw final answer incl. full LaTeX
                \\boxed{...} (untruncated).
      * rag  -> qa_extract_answer(resp): short final answer.
      * code -> "<passed>/<total> tests passed" from record["peer_test_counts"]
                (granular re-grading); falls back to "tests passed"/"tests failed"
                from peer_correct if counts are absent.

    slot_peer_keys maps slot -> the peer_responses key (peer_0/1/2) so code counts
    line up with the (possibly permuted) slot order. If None, assumes slot order ==
    sorted peer_responses order.
    """
    # imported lazily to keep this module torch-only at import time
    from feedback_state.tasks import qa_extract_answer, task_type_of
    from feedback_state.utils import _locate_final_answer

    tt = task_type_of(record)
    n = len(slot_texts) if real is None else min(int(real), len(slot_texts))
    keys = list(slot_peer_keys) if slot_peer_keys is not None else sorted(
        dict(record.get("peer_responses", {})).keys()
    )
    counts = record.get("peer_test_counts") or {}
    pc = record.get("peer_correct") or {}
    out: list[str] = []
    for slot in range(n):
        resp = str(slot_texts[slot]) if slot < len(slot_texts) else ""
        if tt == "code":
            key = keys[slot] if slot < len(keys) else None
            if key is not None and key in counts:
                p, t = counts[key]
                out.append(f"{int(p)}/{int(t)} tests passed")
            elif key is not None and key in pc:
                out.append("tests passed" if float(pc[key]) > 0.5 else "tests failed")
            else:
                out.append("")
        elif tt == "rag":
            ans = qa_extract_answer(resp)
            # rag answers are genuinely short; when qa_extract_answer falls back to
            # the last LINE (no "Answer:" marker) it can grab a whole sentence. Cap
            # at 24 words so a rambling fallback can't blow the state-decay window.
            # (math is intentionally NOT capped — full \boxed{} content is kept.)
            words = ans.split()
            out.append(" ".join(words[:24]) if len(words) > 24 else ans)
        else:  # math (default)
            out.append(_locate_final_answer(resp))
    return out


def resolve_write_policy(policy: str, use_feedback: bool) -> str:
    """Effective policy. ``use_feedback_in_write=False`` downgrades feedback->selection_tokens
    (write enabled, but without correctness labels)."""
    p = str(policy).lower()
    if p not in WRITE_POLICIES:
        raise ValueError(f"write_policy must be one of {WRITE_POLICIES}, got {policy!r}")
    if p == "feedback" and not use_feedback:
        return "selection_tokens"
    return p


def run_write_policy(
    model,
    tokenizer,
    *,
    policy: str,
    use_feedback: bool,
    question: str,
    context: str | None,
    slot_names: Sequence[str],
    slot_texts: Sequence[str],
    correctness_by_slot: Sequence[bool],
    real: int,
    include_identity: bool,
    selected_slot: int | None = None,
    selected_correct: bool | None = None,
    include_selected_peer: bool = True,
    write_prompt_style: str = "full",
    short_answers: Sequence[str] | None = None,
    max_length: int = 1536,
    device: torch.device | None = None,
    debug: bool = False,
    grad: bool = False,
) -> dict[str, Any]:
    """Run the controlled write pass for one example. Returns a debug/log dict.

    Safety guarantees:
      * "none"            -> no state change (no write pass).
      * "selection_tokens"-> write prompt has NO "Feedback:" labels (asserted).
      * "feedback"        -> write prompt HAS "Feedback: correct/incorrect" (asserted).
      * state L1 norm is captured before/after (logged); writes run under no_grad +
        the state is detached afterwards (carry values, not the graph).
    """
    eff = resolve_write_policy(policy, use_feedback)
    if eff == "none":
        return {"write_policy": "none", "wrote": False}

    if eff == "selection_tokens":
        prompt = build_joint_prompt(
            question, slot_names, slot_texts, context=context,
            include_identity=include_identity, real=real,
        )
        assert "Feedback:" not in prompt, "selection_tokens write must NOT contain feedback labels"
    else:  # feedback
        style = str(write_prompt_style).lower()
        prompt = build_feedback_prompt(
            question, slot_names, slot_texts, correctness_by_slot,
            context=context, include_identity=include_identity, real=real,
            selected_slot=selected_slot, selected_correct=selected_correct,
            include_selected_peer=include_selected_peer,
            labels_only=(style == "labels_only"),
            answer_labels=(style == "answer_labels"),
            short_answers=short_answers,
        )
        assert ("Feedback: correct" in prompt) or ("Feedback: incorrect" in prompt), \
            "feedback write must contain explicit correctness labels"

    device = device or next(model.parameters()).device
    enc = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
    ids = torch.tensor([enc["input_ids"]], device=device)
    mask = torch.ones_like(ids)

    norm_before = model.state_norm() if debug else None
    if grad:
        model.write_pass_grad(ids, mask)  # differentiable write (training; bounded BPTT by caller)
    else:
        model.write_pass(ids, mask)  # write_enabled True -> read -> write_enabled False -> detach
    norm_after = model.state_norm() if debug else None
    info = {"write_policy": eff, "wrote": True, "prompt_has_feedback": "Feedback:" in prompt}
    if debug:
        info.update(state_norm_before=norm_before, state_norm_after=norm_after,
                    state_norm_delta=(norm_after - norm_before))
    return info


def assert_scoring_readonly(model, before_norm: float, after_norm: float, *, atol: float = 1e-3) -> None:
    """Candidate scoring must not change S. Raise if the norm drifted (write leak)."""
    if abs(after_norm - before_norm) > atol:
        raise AssertionError(
            f"shared state changed during candidate scoring ({before_norm:.6g} -> {after_norm:.6g}); "
            "writes must be disabled while scoring"
        )
