"""Loading helpers for newer multimodal-packaged center models (Ministral-3, Qwen3.5).

These checkpoints ship as image-text-to-text models, so a plain AutoModelForCausalLM
either rejects the config (Mistral3) or needs care to get a text-only CausalLM whose
attention modules Delta-Mem can wrap. This module is import-safe on the OLD env
(transformers 4.56.2): the new model types simply aren't present and the generic path
is used.

Usage (call BEFORE importing transformers anywhere if you need the FP8 shim):
    from feedback_state.newarch_loader import apply_torch_fp8_shim
    apply_torch_fp8_shim()
"""
from __future__ import annotations

from typing import Any

import torch


def apply_torch_fp8_shim() -> None:
    """transformers>=5.x references torch.float8_e8m0fnu (added in torch 2.7) at import
    time. On torch 2.6 this raises AttributeError even though we never use FP8. Alias it
    to an existing fp8 dtype so the import succeeds; harmless because no FP8 path runs."""
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]


def _model_type(config: Any) -> str:
    return str(getattr(config, "model_type", "") or "")


def load_central_model(model_name: str, dtype, local_files_only: bool):
    """Return a text-only CausalLM for `model_name`, handling the multimodal-packaged
    Ministral-3 / Qwen3.5 checkpoints. Falls back to plain AutoModelForCausalLM for every
    ordinary model (Qwen3, SmolLM3, ...), so behaviour is unchanged on the main env."""
    apply_torch_fp8_shim()
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    mtype = _model_type(config)

    # Mistral3 / Ministral3 are multimodal: the text weights live under
    # `language_model.model.*`, so AutoModelForCausalLM can't load them directly.
    # Load the full image-text model and transplant the language model + lm_head.
    if mtype in {"mistral3", "ministral3"} and hasattr(config, "text_config"):
        from transformers import AutoModelForImageTextToText

        full = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=dtype, local_files_only=local_files_only
        )
        causal = AutoModelForCausalLM.from_config(config.text_config, dtype=dtype)
        lm = full.model.language_model if hasattr(full, "model") else full.language_model
        missing, unexpected = causal.model.load_state_dict(lm.state_dict(), strict=False)
        if missing:
            raise RuntimeError(f"Ministral text transplant missing {len(missing)} keys")
        causal.lm_head.load_state_dict(full.lm_head.state_dict())
        del full
        return causal

    # Qwen3.5 (qwen3_5) registers a *ForCausalLM and loads directly (hybrid: only the
    # full-attention layers are Delta-Mem-wrappable; the GatedDeltaNet layers are left
    # untouched). Everything else also goes through the plain loader.
    return AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, local_files_only=local_files_only
    )
