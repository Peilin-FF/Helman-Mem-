"""Auditable prompt-protocol choices for Sigma candidate judging."""
from __future__ import annotations

from typing import Any

from feedback_state.tasks import _rag_context_text


PROMPT_PROTOCOL_CURRENT = "current"
PROMPT_PROTOCOL_LEGACY_2685 = "legacy_2685"
LEGACY_PROMPT_MAX_LENGTH = 8192


def prompt_protocol_name(legacy_prompt_protocol: bool) -> str:
    return (
        PROMPT_PROTOCOL_LEGACY_2685
        if legacy_prompt_protocol
        else PROMPT_PROTOCOL_CURRENT
    )


def prompt_context_format(legacy_prompt_protocol: bool) -> str:
    return (
        "legacy_python_str_context"
        if legacy_prompt_protocol
        else "rag_context_text_8000"
    )


def candidate_tokenization_format(legacy_prompt_protocol: bool) -> str:
    return (
        "tokenizer_truncation_max_length_8192"
        if legacy_prompt_protocol
        else "complete_prompt_fail_on_overflow"
    )


def validate_prompt_protocol(*, legacy_prompt_protocol: bool, max_length: int) -> None:
    """Keep the historical protocol tied to its original context window."""
    if legacy_prompt_protocol and int(max_length) != LEGACY_PROMPT_MAX_LENGTH:
        raise ValueError(
            "legacy_prompt_protocol=on requires max_length=8192; "
            f"got {int(max_length)}"
        )


def candidate_context_text(
    record: dict[str, Any],
    *,
    include_context: bool,
    legacy_prompt_protocol: bool,
) -> str:
    """Return context text using either the current or exact historical format."""
    if not include_context:
        return ""
    if legacy_prompt_protocol:
        return str(record.get("retrieved_context", record.get("context", "")))
    return _rag_context_text(record)
