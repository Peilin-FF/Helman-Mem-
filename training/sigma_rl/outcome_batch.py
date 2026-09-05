"""Strict DataProto transport for outcome training on the vendored parallel stack.

These are data/trajectory guards, not a model injection implementation. A rollout
backend must explicitly consume and return peer_evidence and peer_mask. The old
HF/vLLM workers drop these fields and therefore fail this contract, rather than
silently running a different experiment.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.utils.model import compute_position_id_with_mask

from training.sigma_rl.outcome_protocol import MAX_PROMPT_LENGTH, PolicyObservation, checked_prompt_ids

POLICY_TENSORS = ("input_ids", "attention_mask", "position_ids", "peer_evidence", "peer_mask")


def policy_batch(observations: Sequence[PolicyObservation], tokenizer, *, max_length: int = MAX_PROMPT_LENGTH) -> DataProto:
    if not observations:
        raise ValueError("cannot build an empty policy batch")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must have an explicit pad token")
    encoded = [checked_prompt_ids(tokenizer, obs.chat_messages(), max_length=max_length) for obs in observations]
    ids = torch.full((len(encoded), max_length), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    peers = max(len(obs.peer_order) for obs in observations)
    evidence = torch.zeros((len(encoded), peers, 4), dtype=torch.float64)
    peer_mask = torch.zeros((len(encoded), peers), dtype=torch.bool)
    raw = np.empty(len(encoded), dtype=object)
    for i, (tokens, obs) in enumerate(zip(encoded, observations)):
        n = len(obs.peer_order)
        if n < 1 or obs.evidence.shape != (n, 4) or not torch.isfinite(obs.evidence).all():
            raise ValueError("invalid original per-peer evidence snapshot")
        ids[i, -len(tokens):] = torch.tensor(tokens)
        attention[i, -len(tokens):] = 1
        evidence[i, :n] = obs.evidence.detach().cpu()
        peer_mask[i, :n] = True
        raw[i] = tokens
    return DataProto.from_dict(
        tensors={"input_ids": ids, "attention_mask": attention, "position_ids": compute_position_id_with_mask(attention),
                 "peer_evidence": evidence, "peer_mask": peer_mask},
        non_tensors={"raw_prompt_ids": raw},
    )


def actor_inputs(batch: DataProto) -> DataProto:
    """Allowlist rather than forwarding a full training record to generation."""
    missing = set(POLICY_TENSORS) - set(batch.batch.keys())
    if missing:
        raise ValueError(f"missing policy condition tensors: {sorted(missing)}")
    return batch.select(batch_keys=list(POLICY_TENSORS), non_tensor_batch_keys=["raw_prompt_ids"], meta_info_keys=[])


def checked_generation(prompt: DataProto, generated: DataProto, *, samples_per_question: int) -> DataProto:
    """Verify sampling and training use EXACTLY the same question/peers/history.

    Parallel scheduling padding must have been removed already. No copying a
    guided response onto a solo prompt, or replacing its snapshot by a newly
    updated matrix, is allowed.
    """
    if samples_per_question < 2 or len(generated) != len(prompt) * samples_per_question:
        raise ValueError("wrong GRPO group size; unpad scheduling rows before constructing trajectories")
    expected = prompt.repeat(repeat_times=samples_per_question, interleave=True)
    for key in ("peer_evidence", "peer_mask"):
        if key not in generated.batch.keys():
            raise ValueError(f"rollout backend did not return {key}; tensor-memory integration is not implemented")
        if not torch.equal(generated.batch[key].cpu(), expected.batch[key].cpu()):
            raise ValueError(f"rollout changed the policy condition: {key}")
    if not torch.equal(generated.batch["prompts"].cpu(), expected.batch["input_ids"].cpu()):
        raise ValueError("generated response was reassigned to a different prompt")
    width = prompt.batch["input_ids"].shape[-1]
    sequence = generated.batch["input_ids"]
    if not torch.equal(sequence[:, :width], generated.batch["prompts"]) or not torch.equal(sequence[:, width:], generated.batch["responses"]):
        raise ValueError("trajectory token sequence does not match its prompt and response")
    if not torch.equal(generated.batch["attention_mask"][:, :width].cpu(), expected.batch["attention_mask"].cpu()):
        raise ValueError("generation changed the prompt attention mask")
    positions = compute_position_id_with_mask(generated.batch["attention_mask"])
    active = generated.batch["attention_mask"].bool()
    if not torch.equal(positions[active], generated.batch["position_ids"][active]):
        raise ValueError("generation returned inconsistent active-token positions")
    return generated


def add_outcome_advantages(batch: DataProto, rewards: torch.Tensor, group_ids: Sequence[str]) -> DataProto:
    """Use the repository's GRPO implementation; all-equal groups remain zero signal."""
    if rewards.shape != batch.batch["responses"].shape or len(group_ids) != len(batch):
        raise ValueError("outcome rewards/group ids must align with center responses")
    width = batch.batch["prompts"].shape[-1]
    mask = batch.batch["attention_mask"][:, width:]
    if mask.shape != rewards.shape or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("invalid response mask")
    lengths = mask.sum(-1).long()
    if (lengths < 1).any() or (mask[:, 1:] > mask[:, :-1]).any():
        raise ValueError("responses must be nonempty and right-padded")
    if not torch.isfinite(rewards).all() or not torch.all((rewards == 0) | (rewards == 1)):
        raise ValueError("training rewards must be binary center outcomes")
    outside_last = rewards.clone()
    outside_last[torch.arange(len(batch), device=rewards.device), lengths - 1] = 0
    if outside_last.any():
        raise ValueError("only final outcome reward is allowed, not per-step supervision")
    indices = np.asarray(group_ids, dtype=object)
    for group in dict.fromkeys(group_ids):
        rows = np.flatnonzero(indices == group).tolist()
        if len(rows) < 2:
            raise ValueError("GRPO needs multiple own-policy responses in every group")
        for key in ("prompts", "peer_evidence", "peer_mask"):
            values = batch.batch[key][rows]
            if not torch.equal(values, values[:1].expand_as(values)):
                raise ValueError(f"GRPO group mixes different policy conditions: {key}")
        prompt_masks = batch.batch["attention_mask"][rows, :width]
        if not torch.equal(prompt_masks, prompt_masks[:1].expand_as(prompt_masks)):
            raise ValueError("GRPO group mixes different prompt attention masks")
    batch.batch["response_mask"] = mask
    batch.batch["token_level_scores"] = rewards
    batch.batch["token_level_rewards"] = rewards
    batch.non_tensor_batch["uid"] = indices
    advantages, returns = compute_grpo_outcome_advantage(rewards, mask, indices)
    batch.batch["advantages"], batch.batch["returns"] = advantages, returns
    return batch
