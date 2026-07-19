"""Frozen-center Yes/No candidate scoring used by Sigma-Mem.

Historical evidence enters the frozen center model through the residual steering
hooks installed by :class:`ActivationSteerer`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

class CandidateUtilityScorer(nn.Module):
    """Read-only scoring facade around a frozen causal language model."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        if freeze_backbone:
            for parameter in self.base_model.parameters():
                parameter.requires_grad_(False)

    def _score_continuation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        continuation: list[int],
    ) -> torch.Tensor:
        """Return the summed log probability of a multi-token Yes/No continuation."""
        batch_size = input_ids.size(0)
        continuation_ids = torch.tensor(
            continuation, device=input_ids.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
        extended_ids = torch.cat([input_ids, continuation_ids], dim=1)
        extended_mask = torch.cat(
            [attention_mask, attention_mask.new_ones(batch_size, continuation_ids.size(1))],
            dim=1,
        )
        output = self.base_model(
            input_ids=extended_ids,
            attention_mask=extended_mask,
            use_cache=False,
            return_dict=True,
        )
        prompt_length = input_ids.size(1)
        target = extended_ids[:, prompt_length:]
        prediction = output.logits[:, prompt_length - 1 : -1, :]
        token_logprob = torch.log_softmax(prediction.float(), dim=-1).gather(
            -1, target.unsqueeze(-1)
        ).squeeze(-1)
        return token_logprob.sum(dim=1)

    def score_candidate_utility(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positive: list[int],
        negative: list[int],
    ) -> torch.Tensor:
        """Return log P(positive) minus log P(negative) for each prompt row."""
        if len(positive) == 1 and len(negative) == 1:
            output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            last = attention_mask.long().sum(dim=1).clamp_min(1) - 1
            rows = torch.arange(input_ids.size(0), device=input_ids.device)
            logprob = torch.log_softmax(output.logits[rows, last, :].float(), dim=-1)
            return logprob[:, int(positive[0])] - logprob[:, int(negative[0])]
        return self._score_continuation(
            input_ids, attention_mask, positive
        ) - self._score_continuation(input_ids, attention_mask, negative)
