"""Minimal LoRA (Hu et al., 2021) for the center model, independent of the ``peft`` package.

``y = W x + (alpha / r) * B (A x)`` with ``A in R^{r x in}`` (Kaiming-uniform init) and
``B in R^{out x r}`` (zero init), so training starts exactly at the frozen model.  Only
``A`` and ``B`` are trainable; they are kept in float32 and the update is cast to the
activation dtype.  Modules are selected by name suffix inside the decoder layers, which
covers Qwen3 (``self_attn.{q,k,v,o}_proj``) and the hybrid Qwen3.5 (``self_attn.*`` in
full-attention layers, ``linear_attn.{in_proj_qkv,in_proj_z,out_proj}`` in linear-attention
layers) with one target list.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from feedback_state.symmetric_memory import _decoder_layers

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    active: bool = True  # evidence gate: when False the layer is exactly the frozen base (see set_lora_active)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.active:
            return y
        h = self.dropout(x).to(torch.float32)
        delta = (h @ self.lora_A.t()) @ self.lora_B.t()
        return y + (delta * self.scale).to(y.dtype)


def apply_lora(model: nn.Module, *, rank: int, alpha: float, targets=DEFAULT_TARGETS, dropout: float = 0.0) -> list[str]:
    """Wrap every ``nn.Linear`` inside the decoder layers whose qualified name ends with a target."""
    wrapped: list[str] = []
    layers = _decoder_layers(model)
    for li, layer in enumerate(layers):
        for name, module in list(layer.named_modules()):
            if not isinstance(module, nn.Linear) or not any(name.endswith(t) for t in targets):
                continue
            parent_name, _, attr = name.rpartition(".")
            parent = layer.get_submodule(parent_name) if parent_name else layer
            setattr(parent, attr, LoRALinear(module, rank, alpha, dropout))
            wrapped.append(f"layers.{li}.{name}")
    if not wrapped:
        raise ValueError(f"no linear modules matched LoRA targets {tuple(targets)}")
    return wrapped


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = {n: p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n}
    missing = set(params) - set(state)
    unexpected = set(state) - set(params)
    if missing or unexpected:
        raise ValueError(f"LoRA state mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(state[n].to(device=p.device, dtype=p.dtype))


def set_lora_active(model: nn.Module, active: bool) -> None:
    """Evidence gate for the adapters: active only while peer answers / memory evidence are in the prompt.

    With the gate off every LoRALinear returns the frozen base output, so the model's behaviour
    without peers is the base model's by construction (pi_theta(.|question) == pi_base(.|question)).
    """
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.active = bool(active)
