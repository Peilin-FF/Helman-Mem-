"""Supervised fine-tuning of the central model to answer with peers + memory (LoRA).

Input: the prompt file written by ``scripts/build_generation_prompts.py`` (memory
state already embedded per event).  Loss: token cross-entropy on the target
(a correct peer's response, else the gold final answer), prompt tokens masked.

  PYTHONPATH=. python -m feedback_state.train_memory_generator --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B \
      --prompts outputs/gen/q3_4b/prompts_train_fixed.jsonl --mode memory --output_dir outputs/gen/q3_4b/sft_memory
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from feedback_state.newarch_loader import apply_torch_fp8_shim, dtype_from_name, load_central_model

apply_torch_fp8_shim()

from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from feedback_state.lora import DEFAULT_TARGETS, apply_lora, lora_parameters, lora_state_dict
from feedback_state.memory_generator import render_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--central_model", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--mode", choices=["memory", "peers", "solo"], default="memory")
    p.add_argument("--gated", choices=["on", "off"], default="on", help="on = adapters are gated by the presence of peer evidence at inference (solo prompts run the base model)")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_targets", default=",".join(DEFAULT_TARGETS))
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=6144)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--gradient_checkpointing", choices=["on", "off", "reentrant"], default="on")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_every", type=int, default=0, help="full-parameter runs: save an HF checkpoint every N micro-steps (learning curve over training)")
    p.add_argument("--betas", default="0.9,0.95")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.central_model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = load_central_model(args.central_model, dtype=dtype, local_files_only=True).to(device=device, dtype=dtype)
    full = args.lora_rank <= 0
    for q in base.parameters():
        q.requires_grad_(full)
    targets = tuple(t for t in args.lora_targets.split(",") if t)
    modules = [] if full else apply_lora(base, rank=args.lora_rank, alpha=args.lora_alpha, targets=targets)
    if args.gradient_checkpointing != "off" and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing == "reentrant"})
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
    base.train()
    params = [q for q in base.parameters() if q.requires_grad] if full else lora_parameters(base)

    rows = [json.loads(l) for l in args.prompts.open()]
    rows = [r for r in rows if r.get("target")]
    eos = tok.eos_token or ""
    examples = []
    skipped = 0
    for r in rows:
        prompt = render_prompt(tok, r[f"messages_{args.mode}"])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(r["target"] + eos, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(t_ids) > args.max_length:
            skipped += 1
            continue
        examples.append((p_ids, t_ids))
    print(f"[gen-sft] mode={args.mode} examples={len(examples)} skipped_long={skipped} lora_modules={len(modules)} params={sum(q.numel() for q in params)}", flush=True)
    steps_per_epoch = math.ceil(len(examples) / args.batch_size)
    total_micro = steps_per_epoch * args.epochs if args.max_steps is None else args.max_steps
    betas = tuple(float(b) for b in args.betas.split(","))
    optim = AdamW([{"params": params, "lr": args.lr, "weight_decay": 0.0}], betas=betas)
    print(f"[gen-sft] {'full-parameter' if full else 'LoRA'} training, trainable={sum(q.numel() for q in params)/1e6:.1f}M, lr={args.lr}, betas={betas}", flush=True)
    sched = get_cosine_schedule_with_warmup(optim, int(0.03 * total_micro / args.grad_accum), max(1, total_micro // args.grad_accum))
    pad = tok.pad_token_id
    micro = 0
    t0 = time.time()
    losses = []
    for epoch in range(args.epochs):
        random.shuffle(examples)
        for b in range(0, len(examples), args.batch_size):
            if micro >= total_micro:
                break
            batch = examples[b : b + args.batch_size]
            width = max(len(p) + len(t) for p, t in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long)
            lab = torch.full((len(batch), width), -100, dtype=torch.long)
            att = torch.zeros((len(batch), width), dtype=torch.long)
            for i, (p, t) in enumerate(batch):
                seq = p + t
                ids[i, : len(seq)] = torch.tensor(seq)
                att[i, : len(seq)] = 1
                lab[i, len(p) : len(seq)] = torch.tensor(t)
            out = base(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device), use_cache=False)
            (out.loss / args.grad_accum).backward()
            losses.append(float(out.loss))
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            if micro % args.logging_steps == 0:
                print(f"[gen-sft] ep{epoch} micro {micro}/{total_micro} loss={sum(losses[-args.logging_steps:]) / len(losses[-args.logging_steps:]):.4f} lr={sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)", flush=True)
            if full and args.save_every > 0 and micro % args.save_every == 0 and micro < total_micro:
                ck = args.output_dir / f"checkpoint-{micro}"
                base.save_pretrained(ck, safe_serialization=True); tok.save_pretrained(ck)
                print(f"[gen-sft] saved {ck}", flush=True)
    if micro % args.grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
    if full:
        ck = args.output_dir / "final"
        base.save_pretrained(ck, safe_serialization=True); tok.save_pretrained(ck)
        payload = {"format": "memory_generator_full_v1", "mode": args.mode, "hf_dir": str(ck), "mean_loss": sum(losses) / max(1, len(losses))}
    else:
        payload = {"format": "memory_generator_v1", "mode": args.mode, "gated": args.gated == "on", "lora": lora_state_dict(base),
                   "lora_config": {"rank": args.lora_rank, "alpha": args.lora_alpha, "targets": list(targets)},
                   "central_model": args.central_model, "prompts": str(args.prompts), "mean_loss": sum(losses) / max(1, len(losses))}
    torch.save(payload, args.output_dir / "memory_generator.pt")
    (args.output_dir / "train_summary.json").write_text(json.dumps({"examples": len(examples), "micro_steps": micro, "mean_loss": payload["mean_loss"], "args": vars(args) | {"prompts": str(args.prompts), "output_dir": str(args.output_dir)}}, indent=1))
    print(f"[gen-sft] saved {args.output_dir} mean_loss={payload['mean_loss']:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
