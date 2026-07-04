"""Train a candidate-wise Yes/No LoRA selector.

Each original multi-peer example is expanded into one binary SFT example per
candidate response. The prompt shows all peer responses, highlights one candidate,
and trains the LoRA adapter to answer "Yes" iff that candidate is correct.

This is peer-count invariant: the learned output space is shared Yes/No, not
fixed "Peer 0/1/2" labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from feedback_state.newarch_loader import apply_torch_fp8_shim, load_central_model

apply_torch_fp8_shim()

from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from feedback_state.data import JsonlDataset
from feedback_state.generation import dtype_from_name
from feedback_state.joint_data import yes_no_token_ids
from feedback_state.joint_prompt import build_candidate_judge_prompt
from feedback_state.permutations import apply_perm, canonical_peer_view, named_order, stable_seed
from feedback_state.utils import load_config, merge_args_with_config


def as_bool(v, default=False):
    if v is None:
        return default
    return v if isinstance(v, bool) else str(v).lower() in {"1", "true", "yes", "on"}


def parse_args():
    p = argparse.ArgumentParser(description="Train candidate-wise Yes/No LoRA selector.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--offline_data", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--central_model", default=None)
    p.add_argument("--num_peers", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--train_order", choices=["orig", "swap", "random"], default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Maximum candidate-level steps. Default: one full expanded epoch.")
    return p.parse_args()


def _example_view(record, num_peers, order):
    view = canonical_peer_view(record, num_peers, setting="A")
    keys = list(view["keys"][: view["real"]])
    names = [f"peer_{i}" for i in range(len(keys))]
    texts = list(view["texts"][: view["real"]])
    cbp = record.get("correctness_by_peer") or record.get("peer_correct") or {}
    corr = [int(round(float(cbp.get(k, 0)))) for k in keys]
    real = len(keys)
    if real <= 0:
        return keys, names, texts, corr, real
    seed = stable_seed(record.get("id") or record.get("uid") or "")
    perm = named_order(order, real, seed=seed)
    return (
        apply_perm(keys, perm),
        apply_perm(names, perm),
        apply_perm(texts, perm),
        apply_perm(corr, perm),
        real,
    )


def _target_ids(corr: int, yes_ids: list[int], no_ids: list[int]) -> list[int]:
    return yes_ids if int(corr) else no_ids


def _batch_lm_yesno_inputs(tok, prompts, targets, *, max_length: int, device):
    encoded = []
    for prompt, target in zip(prompts, targets):
        max_prompt_len = max(1, int(max_length) - len(target))
        prompt_ids = tok(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_len,
        )["input_ids"]
        encoded.append((prompt_ids, target))
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    width = max(len(prompt_ids) + len(target) for prompt_ids, target in encoded)
    input_ids = torch.full((len(encoded), width), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    labels = torch.full((len(encoded), width), -100, dtype=torch.long, device=device)
    for row, (prompt_ids, target) in enumerate(encoded):
        seq = prompt_ids + target
        n = len(seq)
        input_ids[row, :n] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[row, :n] = 1
        labels[row, len(prompt_ids):n] = torch.tensor(target, dtype=torch.long, device=device)
    return input_ids, attention_mask, labels


def main():
    args = parse_args()
    cfg = merge_args_with_config(args, load_config(args.config))
    device = torch.device(str(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    dtype = dtype_from_name(str(cfg.get("dtype", "bfloat16")))
    model_name = str(cfg.get("central_model", "/mnt/data/peilin/HF_MODEL/Qwen3-0.6B"))
    out_dir = Path(cfg.get("output_dir", "outputs/lora_yesno_q3_0.6b"))
    out_dir.mkdir(parents=True, exist_ok=True)
    num_peers = int(cfg.get("num_peers", 3))
    max_len = int(cfg.get("max_length", 8192))
    train_order = str(cfg.get("train_order", "random")).lower()

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=bool(cfg.get("local_files_only", False)))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    yes_ids, no_ids = yes_no_token_ids(tok)
    if not yes_ids or not no_ids:
        raise RuntimeError("Tokenizer produced empty Yes/No continuations")
    print(f"[lora_yesno/train] yes_ids={yes_ids} no_ids={no_ids}", flush=True)

    base = load_central_model(
        model_name,
        dtype=dtype,
        local_files_only=bool(cfg.get("local_files_only", False)),
    )
    base = get_peft_model(base, LoraConfig(
        r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
    ))
    base = base.to(device=device, dtype=dtype)
    if as_bool(cfg.get("gradient_checkpointing"), True):
        base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
    base.train()

    records = JsonlDataset(cfg["offline_data"]).records
    expanded_total = sum(max(0, min(num_peers, len(dict(r.get("peer_responses", {}))))) for r in records)
    max_steps = int(cfg.get("max_steps", -1))
    total_candidate_steps = expanded_total if max_steps <= 0 else min(max_steps, expanded_total)
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    total_optim_steps = max(1, math.ceil(total_candidate_steps / grad_accum))

    params = [p for p in base.parameters() if p.requires_grad]
    optim = AdamW(params, lr=float(cfg.get("learning_rate", 1e-4)),
                  weight_decay=float(cfg.get("weight_decay", 0.0)))
    sched = get_cosine_schedule_with_warmup(
        optim,
        int(float(cfg.get("warmup_ratio", 0.03)) * total_optim_steps),
        total_optim_steps,
    )

    print(f"[lora_yesno/train] records={len(records)} candidate_steps={total_candidate_steps} "
          f"optim_steps={total_optim_steps} train_order={train_order}", flush=True)
    print(f"[lora_yesno/train] trainable_params={sum(p.numel() for p in params)} "
          f"lr={float(cfg.get('learning_rate', 1e-4)):g}", flush=True)

    step = 0
    optim_steps = 0
    yes_count = no_count = 0
    last_loss = None
    log_every = int(cfg.get("logging_steps", 50))
    optim.zero_grad(set_to_none=True)

    while step < total_candidate_steps:
        for rec in records:
            keys, names, texts, corr, real = _example_view(rec, num_peers, train_order)
            if real <= 0:
                continue
            q = str(rec.get("problem", rec.get("question", "")))
            rag_ctx = str(rec.get("retrieved_context", rec.get("context", ""))) if as_bool(cfg.get("include_context"), True) else ""
            s = 0
            while s < real:
                if step >= total_candidate_steps:
                    break
                accum_room = grad_accum - (step % grad_accum)
                if accum_room == 0:
                    accum_room = grad_accum
                chunk = min(real - s, accum_room, total_candidate_steps - step)
                slots = range(s, s + chunk)
                prompts = [
                    build_candidate_judge_prompt(
                        q, names, texts, slot, context=rag_ctx or None,
                        include_identity=False, real=real)
                    for slot in slots
                ]
                targets = [_target_ids(corr[slot], yes_ids, no_ids) for slot in slots]
                input_ids, attention_mask, labels = _batch_lm_yesno_inputs(
                    tok, prompts, targets, max_length=max_len, device=device)
                out = base(input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels, use_cache=False, return_dict=True)
                loss = out.loss * chunk / grad_accum
                loss.backward()
                last_loss = float(out.loss.detach().float())
                yes_count += sum(int(corr[slot]) for slot in slots)
                no_count += sum(int(not corr[slot]) for slot in slots)
                step += chunk
                s += chunk
                if step % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
                    optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                    optim_steps += 1
                if step % log_every == 0:
                    print(f"[lora_yesno/train] step {step}/{total_candidate_steps} "
                          f"loss={last_loss:.4f} yes={yes_count} no={no_count}", flush=True)
            if step >= total_candidate_steps:
                break

    if step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
        optim_steps += 1

    adapter_dir = out_dir / "lora_adapter"
    base.save_pretrained(adapter_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "train_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (out_dir / "train_metrics.json").write_text(json.dumps({
        "candidate_steps": step,
        "optimizer_steps": optim_steps,
        "yes_count": yes_count,
        "no_count": no_count,
        "last_loss": last_loss,
        "yes_ids": yes_ids,
        "no_ids": no_ids,
        "train_order": train_order,
        "max_length": max_len,
    }, indent=2))
    print(f"[lora_yesno/train] saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
