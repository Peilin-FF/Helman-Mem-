"""Memory-augmented judge: the center model reads the competence memory while scoring candidates.

Score of candidate p (one Yes/No judge prompt per candidate, all peers visible):

    score_p = z_p(theta; memory evidence in the residual stream) + kappa * ell_p

* ``z_p`` is the LoRA-adapted center model's log P(Yes) - log P(No) computed with
  the memory's evidence vector for that candidate injected into the upper decoder
  layers (the paper's ActivationSteerer interface, rank = evidence dim);
* ``ell_p`` is the memory's predictive log-odds that the candidate is correct and
  ``kappa`` a learned scalar: with conditionally independent sources the Bayes
  posterior log-odds is the sum, so the judge is trained to supply the residual
  the memory cannot explain.

The memory itself (``KalmanMemory`` on frozen-feature addresses) is not trained;
only the interaction is: LoRA adapters, the steering projection/gain, kappa.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from feedback_state.joint_models import CandidateUtilityScorer
from feedback_state.lora import DEFAULT_TARGETS, apply_lora, load_lora_state_dict, lora_parameters, lora_state_dict
from feedback_state.symmetric_memory import ActivationSteerer


class MemoryJudge(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        *,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_targets=DEFAULT_TARGETS,
        evidence_dim: int = 4,
        use_steer: bool = True,
        use_prior: bool = True,
        kappa_init: float = 1.0,
        steer_layer_frac: float = 0.5,
    ) -> None:
        super().__init__()
        self.scorer = CandidateUtilityScorer(base, freeze_backbone=True)   # freezes every base weight
        self.lora_modules = apply_lora(base, rank=lora_rank, alpha=lora_alpha, targets=lora_targets) if lora_rank > 0 else []
        self.steerer = ActivationSteerer(base, rank=evidence_dim, layer_frac=steer_layer_frac, gain_init=1.0, proj_std=1e-2) if use_steer else None
        self.use_prior = bool(use_prior)
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
        self.evidence_dim = int(evidence_dim)

    @property
    def base(self) -> nn.Module:
        return self.scorer.base_model

    def trainable_groups(self, *, lora_lr: float, steer_lr: float, weight_decay: float = 0.0) -> list[dict]:
        groups = []
        lp = lora_parameters(self.base)
        if lp:
            groups.append({"params": lp, "lr": lora_lr, "weight_decay": weight_decay})
        small = [self.kappa] + ([p for p in self.steerer.parameters() if p.requires_grad] if self.steerer is not None else [])
        groups.append({"params": small, "lr": steer_lr, "weight_decay": 0.0})
        return groups

    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, yes_ids, no_ids, *, evidence: torch.Tensor | None, mem_logit: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (fused scores [r], judge-only log-odds z [r])."""
        if self.steerer is not None:
            self.steerer.steer_vec = evidence.to(torch.float32) if evidence is not None else None
        try:
            z = self.scorer.score_candidate_utility(input_ids, attention_mask, yes_ids, no_ids)
        finally:
            if self.steerer is not None:
                self.steerer.steer_vec = None
        z = z.float()
        if self.use_prior and mem_logit is not None:
            return z + self.kappa * mem_logit.to(z.device, z.dtype), z
        return z, z

    # ---- persistence ----
    def state_payload(self) -> dict:
        return {
            "lora": lora_state_dict(self.base),
            "lora_modules": self.lora_modules,
            "steerer": self.steerer.state_dict() if self.steerer is not None else None,
            "kappa": float(self.kappa.detach()),
            "use_prior": self.use_prior,
            "evidence_dim": self.evidence_dim,
        }

    def load_payload(self, payload: dict) -> None:
        if payload.get("lora"):
            load_lora_state_dict(self.base, payload["lora"])
        if self.steerer is not None and payload.get("steerer") is not None:
            self.steerer.load_state_dict(payload["steerer"])
        with torch.no_grad():
            self.kappa.fill_(float(payload.get("kappa", 1.0)))
        self.use_prior = bool(payload.get("use_prior", self.use_prior))


def selection_loss(scores: torch.Tensor, correct: torch.Tensor) -> torch.Tensor | None:
    """-log P(selected candidate is correct) under softmax(scores); None when the event carries no signal."""
    correct = correct.to(scores.device).bool().reshape(-1)
    if correct.sum() == 0 or correct.sum() == correct.numel():
        return None
    logp = torch.log_softmax(scores.reshape(-1), dim=0)
    return -torch.logsumexp(logp[correct], dim=0)
