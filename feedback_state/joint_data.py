"""Prompt tokenization helpers for Sigma-Mem Yes/No candidate scoring."""
from __future__ import annotations

import torch

from feedback_state.joint_prompt import build_candidate_judge_prompt
from feedback_state.prompt_protocol import validate_prompt_protocol

def yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Token ids for the shared Yes/No candidate-utility continuations."""
    return (
        tokenizer.encode(" Yes", add_special_tokens=False),
        tokenizer.encode(" No", add_special_tokens=False),
    )


def batch_candidate_judge_inputs(
    tokenizer,
    question: str,
    slot_names: list[str],
    slot_texts: list[str],
    *,
    context: str | None = None,
    include_identity: bool = False,
    real: int,
    max_length: int,
    device: torch.device | None = None,
    legacy_prompt_protocol: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one Yes/No judge prompt per candidate into a right-padded batch."""
    prompts = [
        build_candidate_judge_prompt(
            question,
            slot_names,
            slot_texts,
            slot,
            context=context,
            include_identity=include_identity,
            real=real,
        )
        for slot in range(int(real))
    ]
    if not prompts:
        raise ValueError("real must be at least one for candidate scoring")
    validate_prompt_protocol(
        legacy_prompt_protocol=legacy_prompt_protocol,
        max_length=max_length,
    )

    encoded: list[list[int]] = []
    for slot, prompt in enumerate(prompts):
        if legacy_prompt_protocol:
            ids = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=int(max_length),
            )["input_ids"]
        else:
            ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
            if len(ids) > int(max_length):
                raise ValueError(
                    "Candidate judge prompt exceeds max_length: "
                    f"slot={slot}, length={len(ids)}, max_length={int(max_length)}"
                )
        encoded.append(ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    width = max(len(ids) for ids in encoded)
    input_ids = torch.full((len(encoded), width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask
