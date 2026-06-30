"""Joint-input multi-agent selectors with a SHARED online state S that steers the
central model's ATTENTION via the real Delta-Mem mechanism (deltamem.core.delta),
NOT prefix tuning.

The selector (``ar_shared_state_selector``) reads ALL peers jointly and
AUTOREGRESSIVELY generates the selected peer id ("Peer j"). Trained with LM loss on
the target id; evaluated by scoring P("Peer j") per peer.

Shared state S: Delta-Mem attaches an online delta-rule memory inside the attention
of target layers. It is ONE shared memory over the whole sequence (hence shared
across peers), and peer identity is part of the input tokens, so its read/write keys
naturally encode peer- and context-specific reliability. Set ``use_shared_state=False``
for the no-memory joint control (LoRA fine-tune, no S).

Reuse: the central model is config-driven (``central_model`` — e.g. Qwen3-0.6B or
Qwen3-4B just work); state lifecycle uses the Delta-Mem online-state API; the
SelectionOutput contract matches the existing trainer/eval.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANT_AR = "ar_shared_state_selector"
VARIANT_BCE = "joint_bce_shared_state_selector"


@dataclass
class JointSelectionOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor | None = None          # [B, P] for BCE
    candidate_logprobs: torch.Tensor | None = None  # [B, P] (AR eval)


def _default_delta_config(base_model, cfg: dict[str, Any]):
    """Build an HFDeltaMemConfig from the experiment config (sensible defaults)."""
    from deltamem.core.delta import HFDeltaMemConfig

    n_layers = int(getattr(base_model.config, "num_hidden_layers", 0))
    target_layers = cfg.get("delta_target_layers")
    if not target_layers:
        # Steer the upper half of the stack by default (cheap + effective).
        target_layers = tuple(range(max(0, n_layers // 2), n_layers)) if n_layers else ()
    return HFDeltaMemConfig(
        rank=int(cfg.get("delta_rank", 8)),
        alpha=float(cfg.get("alpha", cfg.get("delta_alpha", 16.0))),
        num_state_heads=int(cfg.get("delta_num_state_heads", 1)),
        state_update_mode=str(cfg.get("delta_state_update_mode", "standard")),
        beta_bias_init=float(cfg.get("delta_beta_bias_init", -1.5)),
        target_modules=tuple(cfg.get("delta_target_modules", ("self_attn",))),
        target_layers=tuple(target_layers),
        memory_readout_mode="delta",
        # output_init: 'zero' makes step-0 == frozen base BUT zero delta projections give
        # a zero read->output Jacobian, which DEADLOCKS training the WRITE projections
        # (d loss/d reads == 0 blocks grad to memory_k/v/beta). Use 'random' (scaled by
        # online_gain, small) to break that deadlock so write projections can train.
        output_init=str(cfg.get("delta_output_init", "zero")),
        online_gain=float(cfg.get("delta_online_gain", 0.05)),
    )


class JointDeltaMemSelector(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        num_peers: int,
        model_variant: str = VARIANT_AR,
        use_shared_state: bool = True,
        delta_cfg: dict[str, Any] | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_peers = int(num_peers)
        self.model_variant = str(model_variant)
        self.use_shared_state = bool(use_shared_state)
        hidden_size = int(base_model.config.hidden_size)
        self.peer_head = nn.Linear(hidden_size, 1)
        # Identity trust-readout head: maps a peer's per-peer state S_s (content-free,
        # identity-keyed) to a scalar trust bias added to that candidate's logp. This is
        # the "selection stays anonymous, identity only feeds the feedback channel" path
        # — trust is read from the accumulated state via a CONSTANT query (not the current
        # response hidden), so CF content swaps cannot scramble retrieval. Lazily sized to
        # the flattened per-peer state on first use; trained end-to-end (no GT signal).
        self.trust_head = None
        self._trust_head_dim = None
        self.use_trust_head = False
        # Dual-memory gate: M = lambda*M_R + (1-lambda)*M_I, lambda=sigmoid(gate_logit).
        # M_R = response-only memory (anonymous write); M_I = response+identity memory
        # (write prompt carries the real name). Learnable scalar gate trained end-to-end;
        # it can drive lambda->1 to ignore identity (e.g. CF p0) or ->0 to lean on it
        # (stable reliability). Created lazily when dual memory is enabled.
        self.gate_logit = None
        self.use_dual_memory = False
        self._delta_attached = False
        self._delta_config = None
        if self.use_shared_state:
            # The Delta-Mem shared-state path was removed in the repo cleanup (the deltamem
            # package is gone). The current method runs with use_shared_state=False (trust
            # enters via SymmetricTrustMemory + ActivationSteerer, not token-level Delta-Mem).
            raise RuntimeError(
                "use_shared_state=True requires the 'deltamem' package, which was removed. "
                "The current symmetric-memory selector uses use_shared_state=False."
            )
        elif freeze_backbone:
            # No-memory control. If the base is a PEFT/LoRA model, get_peft_model
            # already froze the backbone and left the LoRA adapters trainable —
            # don't blanket-freeze (that would kill the LoRA grads). Otherwise this
            # is the pure frozen baseline: freeze everything.
            if hasattr(self.base_model, "peft_config"):
                pass
            else:
                for p in self.base_model.parameters():
                    p.requires_grad_(False)

    # ---- shared-state lifecycle (delegates to the Delta-Mem online-state API) ----
    def reset_state(self, *args, **kwargs) -> None:
        if self._delta_attached:
            from deltamem.core.delta import reset_delta_mem_states

            reset_delta_mem_states(self.base_model)

    def set_write_enabled(self, enabled: bool) -> None:
        if self._delta_attached:
            from deltamem.core.delta import set_delta_mem_write_enabled

            set_delta_mem_write_enabled(self.base_model, bool(enabled))

    def get_online_state(self):
        if not self._delta_attached:
            return {}
        from deltamem.core.delta import get_delta_mem_online_state

        return get_delta_mem_online_state(self.base_model)

    def load_online_state(self, state) -> None:
        if self._delta_attached and state:
            from deltamem.core.delta import load_delta_mem_online_state

            load_delta_mem_online_state(self.base_model, state)

    def detach_state(self) -> None:
        """Detach the shared online state from the autograd graph (carry the values,
        not the history). Writes already run under no_grad, so this is a safety net."""
        if not self._delta_attached:
            return
        st = self.get_online_state()
        if st:
            self.load_online_state({k: (v.detach() if torch.is_tensor(v) else v) for k, v in st.items()})

    def state_norm(self) -> float:
        """L1 norm of the shared online state (for write-safety logging)."""
        st = self.get_online_state()
        if not st:
            return 0.0
        return float(sum(float(t.detach().abs().sum()) for t in st.values() if torch.is_tensor(t)))

    @torch.no_grad()
    def write_pass(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        """A controlled WRITE pass: read the given tokens with write_enabled=True so
        Delta-Mem updates the shared online state, then disable writes and detach.
        Never computes a selection loss; never backprops (inference-time write)."""
        if not self._delta_attached:
            return
        self.set_write_enabled(True)
        self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        self.set_write_enabled(False)
        self.detach_state()

    def write_pass_grad(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        """Differentiable WRITE pass (training only): same as write_pass but KEEPS the
        graph (no no_grad, no detach). The written state stays connected to the write
        projections (memory_k/v_proj, beta_proj) so a LATER selection loss can backprop
        into 'what to write'. Caller is responsible for bounded BPTT (detach the carried
        per-peer states at the grad-accum window boundary)."""
        if not self._delta_attached:
            return
        self.set_write_enabled(True)
        self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        self.set_write_enabled(False)
        # NOTE: no detach_state() here — that is the whole point.


    # ---- per-peer state slots (reference swap) ------------------------------------
    # Each peer gets its OWN delta state matrix set; the loop swaps the live module
    # state by REFERENCE around per-peer reads/writes, so isolation is exact.
    def _delta_module_items(self):
        from deltamem.core.delta_impl import iter_delta_mem_modules

        return list(iter_delta_mem_modules(self.base_model))

    def get_state_refs(self) -> dict:
        """Live per-module state tensors WITHOUT copy/detach (None when unset)."""
        if not self._delta_attached:
            return {}
        return {name: m.delta_state for name, m in self._delta_module_items()}

    def set_state_refs(self, state: dict | None) -> None:
        """Install per-module state tensors by reference. None entries (or state=None)
        reset modules to the uninitialized (zeros-on-first-use) condition."""
        if not self._delta_attached:
            return
        state = state or {}
        for name, m in self._delta_module_items():
            m.delta_state = state.get(name)

    def gate_lambda(self) -> torch.Tensor:
        """Current gate weight lambda = sigmoid(gate_logit) in [0,1] (M_R weight)."""
        dev = next(self.base_model.parameters()).device
        if self.gate_logit is None:
            self.gate_logit = nn.Parameter(torch.zeros(1, device=dev))  # lambda=0.5 init
        return torch.sigmoid(self.gate_logit.to(dev))

    def blend_states(self, state_r: dict | None, state_i: dict | None,
                     detach_r: bool = True, detach_i: bool = True) -> dict:
        """Blend two per-peer state dicts: M = lambda*M_R + (1-lambda)*M_I (per module).
        Missing entries on either side fall back to the other. Differentiable through
        gate_logit; detach_r/detach_i control whether each input keeps its graph (algo A
        keeps the content state R in-graph so its write projections train via the loss)."""
        lam = self.gate_lambda()
        sr, si = state_r or {}, state_i or {}
        keys = set(sr) | set(si)
        out = {}
        for k in keys:
            vr, vi = sr.get(k), si.get(k)
            if detach_r and torch.is_tensor(vr):
                vr = vr.detach()
            if detach_i and torch.is_tensor(vi):
                vi = vi.detach()
            if vr is None:
                out[k] = vi
            elif vi is None:
                out[k] = vr
            else:
                out[k] = lam.to(vr.dtype) * vr + (1.0 - lam).to(vr.dtype) * vi
        return out

    def trust_bias(self) -> torch.Tensor:
        """Scalar trust bias read from the CURRENTLY-installed per-peer state S_s.

        Flattens S_s (content-free, identity-keyed) and maps it through a small head
        to one scalar. Identity-keyed constant retrieval: depends only on the
        accumulated state, NOT on the current response hidden — so CF content swaps
        cannot scramble it. Caller installs S_s via set_state_refs BEFORE calling.
        Returns a [1] tensor (0 when no state is attached / state is empty).
        """
        if not self._delta_attached:
            return torch.zeros(1)
        items = self.get_state_refs()
        vecs = [v.reshape(-1).float() for v in items.values() if torch.is_tensor(v)]
        # device anchor from any base parameter
        dev = next(self.base_model.parameters()).device
        if not vecs:
            return torch.zeros(1, device=dev)
        flat = torch.cat(vecs).to(dev)
        if self.trust_head is None:
            self._trust_head_dim = int(flat.numel())
            self.trust_head = nn.Sequential(
                nn.Linear(self._trust_head_dim, 64), nn.Tanh(), nn.Linear(64, 1)
            ).to(dev)
        if flat.numel() != self._trust_head_dim:  # state grew/shrank; pad or trim
            if flat.numel() < self._trust_head_dim:
                flat = torch.cat([flat, flat.new_zeros(self._trust_head_dim - flat.numel())])
            else:
                flat = flat[: self._trust_head_dim]
        return self.trust_head(flat.to(self.trust_head[0].weight.dtype)).reshape(1)

    @staticmethod
    def detach_state_dict(state: dict | None) -> dict:
        """Detach a state-ref dict (truncated BPTT at the window boundary)."""
        if not state:
            return {}
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in state.items()}

    def snapshot_state_cpu(self, state: dict | None) -> dict:
        """CPU-clone a state-ref dict for checkpointing (trust_state.pt)."""
        if not state:
            return {}
        return {k: v.detach().cpu().clone() for k, v in state.items() if torch.is_tensor(v)}

    # ---- forward ----------------------------------------------------------------
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        peer_target: torch.Tensor | None = None,
        peer_spans: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if self.model_variant == VARIANT_BCE:
            return self._bce_forward(input_ids, attention_mask, peer_target, peer_spans)
        return self._ar_forward(input_ids, attention_mask, labels)

    def _ar_forward(self, input_ids, attention_mask, labels):
        out = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, use_cache=False, return_dict=True,
        )
        return JointSelectionOutput(loss=out.loss, logits=None)

    def _pool_peer_spans(self, hidden: torch.Tensor, peer_spans: torch.Tensor) -> torch.Tensor:
        """Mean-pool hidden states over each peer response token span."""
        B, P, _ = peer_spans.shape
        H = hidden.size(-1)
        pooled = hidden.new_zeros(B, P, H)
        T = hidden.size(1)
        idx = torch.arange(T, device=hidden.device)
        for p in range(P):
            start = peer_spans[:, p, 0].clamp(min=0, max=T)
            end = peer_spans[:, p, 1].clamp(min=0, max=T)
            mask = (idx.unsqueeze(0) >= start.unsqueeze(1)) & (idx.unsqueeze(0) < end.unsqueeze(1))
            mask = mask.to(hidden.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled[:, p] = (hidden * mask).sum(dim=1) / denom
        return pooled

    def _bce_forward(self, input_ids, attention_mask, peer_target, peer_spans):
        out = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False, return_dict=True,
        )
        hidden = out.hidden_states[-1]
        if peer_spans is not None:
            pooled = self._pool_peer_spans(hidden, peer_spans.to(hidden.device))
        else:
            m = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * m).sum(1) / m.sum(1).clamp_min(1.0)).unsqueeze(1).expand(-1, self.num_peers, -1)
        logits = self.peer_head(pooled.to(self.peer_head.weight.dtype)).squeeze(-1)
        loss = None
        if peer_target is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.float(), peer_target.to(logits.device, torch.float32)
            )
        return JointSelectionOutput(loss=loss, logits=logits)

    # ---- evaluation: AR candidate scoring --------------------------------------
    def score_one_candidate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate: list[int],
    ) -> torch.Tensor:
        """Summed teacher-forced log-prob of ONE candidate continuation (" Peer j").

        Differentiable (no no_grad wrapper) so the per-peer AR path can backprop
        through it. Caller is responsible for setting write_enabled / state refs
        BEFORE calling. Returns [B] log-prob.
        """
        B = input_ids.size(0)
        device = input_ids.device
        cand_t = torch.tensor(candidate, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        ext_ids = torch.cat([input_ids, cand_t], dim=1)
        ext_mask = torch.cat([attention_mask, attention_mask.new_ones(B, cand_t.size(1))], dim=1)
        out = self.base_model(input_ids=ext_ids, attention_mask=ext_mask, use_cache=False, return_dict=True)
        logits = out.logits
        plen = input_ids.size(1)
        target = ext_ids[:, plen:]
        pred = logits[:, plen - 1 : -1, :]
        logprob = torch.log_softmax(pred.float(), dim=-1)
        tok_lp = logprob.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return tok_lp.sum(dim=1)

    @torch.no_grad()
    def score_candidates(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_ids: list[list[int]],
    ) -> torch.Tensor:
        """P('Peer j' | joint prompt) for each candidate, teacher-forced. Read-only.

        Returns [B, num_candidates] summed token log-probabilities. Writes are
        disabled so candidate scoring never mutates the shared state.
        """
        self.set_write_enabled(False)
        B = input_ids.size(0)
        device = input_ids.device
        logps = []
        for cand in candidate_ids:
            cand_t = torch.tensor(cand, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            ext_ids = torch.cat([input_ids, cand_t], dim=1)
            ext_mask = torch.cat([attention_mask, attention_mask.new_ones(B, cand_t.size(1))], dim=1)
            out = self.base_model(input_ids=ext_ids, attention_mask=ext_mask, use_cache=False, return_dict=True)
            logits = out.logits  # [B, T, V]
            plen = input_ids.size(1)
            # token t is predicted by logits at position t-1
            target = ext_ids[:, plen:]                      # [B, L]
            pred = logits[:, plen - 1 : -1, :]              # [B, L, V]
            logprob = torch.log_softmax(pred.float(), dim=-1)
            tok_lp = logprob.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, L]
            logps.append(tok_lp.sum(dim=1))
        return torch.stack(logps, dim=1)  # [B, P]

    # ---- persistence ------------------------------------------------------------
    def save_feedback_adapter(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self._delta_attached:
            from deltamem.core.delta import save_delta_mem_adapter

            save_delta_mem_adapter(self.base_model, output / "delta_mem", self._delta_config)
        # No-memory control: persist the LoRA adapters so eval can reload them.
        if hasattr(self.base_model, "peft_config"):
            self.base_model.save_pretrained(output / "lora_adapter")
        torch.save(
            {"peer_head": self.peer_head.state_dict(),
             "num_peers": self.num_peers, "model_variant": self.model_variant,
             "use_shared_state": self.use_shared_state},
            output / "joint_selector_head.pt",
        )

    def load_feedback_adapter(self, checkpoint_dir: str | Path, *, map_location="cpu") -> None:
        ckpt = Path(checkpoint_dir)
        if self._delta_attached and (ckpt / "delta_mem" / "delta_mem_adapter.pt").exists():
            from deltamem.core.delta import load_delta_mem_adapter

            load_delta_mem_adapter(self.base_model, ckpt / "delta_mem")
        # No-memory control: reattach the trained LoRA adapters onto the plain base.
        lora_dir = ckpt / "lora_adapter"
        if not self.use_shared_state and lora_dir.exists() and not hasattr(self.base_model, "peft_config"):
            from peft import PeftModel

            self.base_model = PeftModel.from_pretrained(
                self.base_model, str(lora_dir), is_trainable=False
            ).to(next(self.base_model.parameters()).device)
        head_path = ckpt / "joint_selector_head.pt"
        if head_path.exists():
            payload = torch.load(head_path, map_location=map_location)
            if "peer_head" in payload:
                self.peer_head.load_state_dict(payload["peer_head"])
