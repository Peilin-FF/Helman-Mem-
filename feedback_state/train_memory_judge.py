"""Train the memory-augmented judge with the competence memory in the loop.

Objective (per episode = a cold-start pass over a window of the training stream):

    L(theta) = sum_t  -log P_theta( selected candidate correct | x_t, answers_t, memory state S_{t-1} )

where S_{t-1} is the Kalman competence memory built from the feedback of the
previous events of the episode.  Minimising L is minimising the area above the
learning curve: the only way down is to read the memory better (the memory
itself has no trainable parameters).  Peer slots are randomly permuted per
event, so the judge cannot bind reliability to a prompt position; the memory is
reset at every episode, so nothing about the training peers can be stored in
the adapters except through the memory.

  PYTHONPATH=. python -m feedback_state.train_memory_judge --config configs/symmetric_memory_candidate_yesno.yaml \
      --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B --offline_data data/mixed_train_big/train.jsonl \
      --features outputs/context_features/q3_4b_big_ph/train --output_dir outputs/memjudge/q3_4b/qc
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from feedback_state.newarch_loader import apply_torch_fp8_shim, dtype_from_name, load_central_model

apply_torch_fp8_shim()

from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from feedback_state.addresses import EVIDENCE_DIM, Projection, memory_notes
from feedback_state.feature_streams import load_stream_from
from feedback_state.joint_data import batch_candidate_judge_inputs, yes_no_token_ids
from feedback_state.lora import DEFAULT_TARGETS
from feedback_state.memory_judge import MemoryJudge, selection_loss
from feedback_state.memory_runtime import MemoryRuntime
from feedback_state.permutations import random_order
from feedback_state.prompt_protocol import candidate_context_text, validate_prompt_protocol
from feedback_state.utils import load_config, merge_args_with_config


def as_bool(v, default=False):
    if v is None:
        return default
    return v if isinstance(v, bool) else str(v).lower() in {"1", "true", "yes", "on"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--central_model", default=None)
    p.add_argument("--offline_data", type=Path, default=None)
    p.add_argument("--features", type=Path, required=True, help="feature cache directory (shard*.pt) of the training stream")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--num_peers", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--legacy_prompt_protocol", choices=["on", "off"], default=None)
    p.add_argument("--design", default="qc", choices=["q", "c", "cm", "qc", "qcm"])
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--lam", type=float, default=100.0)
    p.add_argument("--memory", choices=["on", "off"], default="on", help="off = control: same judge trained without memory")
    p.add_argument("--use_steer", choices=["on", "off"], default="on")
    p.add_argument("--use_prior", choices=["on", "off"], default="on")
    p.add_argument("--memory_text", choices=["on", "off"], default="off", help="on = the memory's evidence is also written into each candidate's judge prompt as text")
    p.add_argument("--kappa_init", type=float, default=1.0)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_targets", default=",".join(DEFAULT_TARGETS))
    p.add_argument("--lora_lr", type=float, default=1e-4)
    p.add_argument("--steer_lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--episode_len", type=int, default=4096)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--slot_permute", choices=["on", "off"], default="on")
    p.add_argument("--proj_ckpt", type=Path, default=None, help="learned address (train_address.py) instead of PCA projections")
    p.add_argument("--gradient_checkpointing", choices=["on", "off", "reentrant"], default="reentrant",
                   help="reentrant = torch reentrant checkpoint (needed for hybrid Qwen3.5 layers under LoRA + hooks)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logging_steps", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = merge_args_with_config(args, load_config(args.config))
    seed = int(cfg.get("seed") or 0)
    torch.manual_seed(seed); random.seed(seed)
    device = torch.device(str(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    dtype = dtype_from_name(str(cfg.get("dtype", "bfloat16")).lower())
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(cfg.get("central_model", "models/Qwen3-4B"))
    num_peers = int(cfg.get("num_peers", 3))
    max_len = int(cfg.get("max_length", 8192))
    legacy = as_bool(cfg.get("legacy_prompt_protocol"), False)
    validate_prompt_protocol(legacy_prompt_protocol=legacy, max_length=max_len)
    include_context = as_bool(cfg.get("include_context"), False)
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    memory_on = as_bool(cfg.get("memory"), True)
    use_steer = as_bool(cfg.get("use_steer"), True) and memory_on
    use_prior = as_bool(cfg.get("use_prior"), True) and memory_on
    memory_text = as_bool(cfg.get("memory_text"), False) and memory_on
    slot_permute = as_bool(cfg.get("slot_permute"), True)
    design, dim, lam = str(cfg.get("design") or "qc"), int(cfg.get("dim") or 256), float(cfg.get("lam") or 100.0)
    lora_rank, lora_alpha = int(cfg.get("lora_rank") or 16), float(cfg.get("lora_alpha") or 32.0)
    lora_targets = tuple(t for t in str(cfg.get("lora_targets") or ",".join(DEFAULT_TARGETS)).split(",") if t)
    epochs, episode_len = int(cfg.get("epochs") or 1), int(cfg.get("episode_len") or 4096)

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=bool(cfg.get("local_files_only", False)))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = load_central_model(model_name, dtype=dtype, local_files_only=bool(cfg.get("local_files_only", False))).to(device=device, dtype=dtype)
    judge = MemoryJudge(base, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_targets=lora_targets, evidence_dim=EVIDENCE_DIM,
                        use_steer=use_steer, use_prior=use_prior, kappa_init=float(cfg.get("kappa_init") or 1.0)).to(device)
    gc_mode = str(cfg.get("gradient_checkpointing") or "reentrant").lower()
    if gc_mode in ("on", "true", "1", "yes", "reentrant") and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": gc_mode == "reentrant"})
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
    base.train()

    fs = load_stream_from(cfg["offline_data"], cfg["features"], name="train", num_peers=num_peers)
    if cfg.get("proj_ckpt"):
        ck = torch.load(Path(cfg["proj_ckpt"]), map_location="cpu", weights_only=False)
        proj_q, proj_c = Projection(state=ck["proj_q"], device=device), Projection(state=ck["proj_c"], device=device)
        print(f"[memjudge] learned address from {cfg['proj_ckpt']} (dim={proj_q.dim})", flush=True)
    else:
        proj_q = Projection(fs.sem, dim, device)
        proj_c = Projection(fs.peer_hidden.reshape(-1, fs.peer_hidden.shape[-1]), dim, device)
    runtime = MemoryRuntime(design=design, proj_q=proj_q, proj_c=proj_c, num_peers=num_peers, lam=lam, device=device)
    runtime.attach(fs)
    N = len(fs)
    labels = fs.labels.numpy()
    has_signal = np.array([0 < labels[t, : fs.real[t]].sum() < fs.real[t] for t in range(N)])
    total_events = int(cfg.get("max_steps") or N * epochs)
    planned_backward = int(has_signal.sum()) * epochs if cfg.get("max_steps") is None else int(has_signal[: total_events].sum())
    groups = judge.trainable_groups(lora_lr=float(cfg.get("lora_lr") or 1e-4), steer_lr=float(cfg.get("steer_lr") or 1e-3), weight_decay=float(cfg.get("weight_decay", 0.0)))
    params = [p for g in groups for p in g["params"]]
    optim = AdamW(groups)
    planned_optimizer_steps = max(1, math.ceil(planned_backward / grad_accum))
    warmup = int(planned_optimizer_steps * float(cfg.get("warmup_ratio", 0.03)))
    sched = get_cosine_schedule_with_warmup(optim, warmup, planned_optimizer_steps)
    print(f"[memjudge] model={model_name} design={design} dim={dim} lam={lam} memory={memory_on} steer={use_steer} prior={use_prior} "
          f"lora_rank={lora_rank} modules={len(judge.lora_modules)} trainable={sum(p.numel() for p in params)} events={total_events} "
          f"backward={planned_backward} optimizer_steps={planned_optimizer_steps} episode_len={episode_len}", flush=True)

    yes_ids, no_ids = yes_no_token_ids(tok)
    rng = np.random.default_rng(seed)
    step = backward = 0
    hist = {"hit": [], "judge_hit": [], "mem_hit": [], "loss": []}
    log_every = int(cfg.get("logging_steps", 100))
    t0 = time.time()
    done = False
    for epoch in range(epochs):
        order = rng.permutation(N)
        for ep_start in range(0, N, episode_len):
            runtime.reset()
            for t in order[ep_start : ep_start + episode_len].tolist():
                if step >= total_events:
                    done = True; break
                r = fs.real[t].item() if torch.is_tensor(fs.real[t]) else int(fs.real[t])
                if r < 1:
                    step += 1; continue
                y = labels[t, :r].copy()
                cand_texts = list(fs.texts[t][:r])
                ell, n_eff, evidence, X = runtime.read(t)
                perm = random_order(r, int(rng.integers(1 << 30))) if slot_permute else list(range(r))
                texts = [cand_texts[p] for p in perm]
                y_slot = torch.tensor([int(y[p]) for p in perm])
                ell_slot = ell[perm]; ev_slot = evidence[perm]
                rec = fs.records[t]
                q = str(rec.get("problem", rec.get("question", "")))
                ctx = candidate_context_text(rec, include_context=include_context, legacy_prompt_protocol=legacy)
                notes = [memory_notes(ell, n_eff)[p] for p in perm] if memory_text else None
                ids, mask = batch_candidate_judge_inputs(tok, q, [f"peer_{i}" for i in range(r)], texts, context=ctx or None,
                                                         include_identity=False, real=r, max_length=max_len, device=device, legacy_prompt_protocol=legacy,
                                                         slot_notes=notes)
                scores, z = judge.score(ids, mask, yes_ids, no_ids, evidence=ev_slot if use_steer else None, mem_logit=ell_slot if use_prior else None)
                loss = selection_loss(scores, y_slot)
                if loss is not None:
                    (loss / grad_accum).backward()
                    backward += 1
                    hist["loss"].append(float(loss))
                    if backward % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
                        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                hist["hit"].append(int(y_slot[int(torch.argmax(scores.detach()))]))
                hist["judge_hit"].append(int(y_slot[int(torch.argmax(z.detach()))]))
                hist["mem_hit"].append(int(y[int(torch.argmax(ell))]))
                runtime.write(t, X, y)
                step += 1
                if step % log_every == 0:
                    w = log_every
                    print(f"[memjudge] ep{epoch} step {step}/{total_events} loss={np.mean(hist['loss'][-w:]) if hist['loss'] else float('nan'):.4f} "
                          f"acc={100 * np.mean(hist['hit'][-w:]):.1f} judge={100 * np.mean(hist['judge_hit'][-w:]):.1f} mem={100 * np.mean(hist['mem_hit'][-w:]):.1f} "
                          f"kappa={float(judge.kappa):.3f} gain={float(judge.steerer.gain) if judge.steerer is not None else 0.0:.3f} "
                          f"lr={sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)", flush=True)
            if done:
                break
        if done:
            break
    if backward % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)

    payload = {
        "format": "memory_judge_v1",
        "judge": judge.state_payload(),
        "proj_q": proj_q.state(), "proj_c": proj_c.state(),
        "memory_config": runtime.config(),
        "memory_on": memory_on, "use_steer": use_steer, "use_prior": use_prior, "memory_text": memory_text,
        "lora_config": {"rank": lora_rank, "alpha": lora_alpha, "targets": list(lora_targets)},
        "train_stats": {k: (np.array(v).reshape(-1).tolist() if k != "loss" else float(np.mean(v)) if v else None) for k, v in hist.items()},
    }
    torch.save(payload, out_dir / "memory_judge.pt")
    n = len(hist["hit"]); edges = np.linspace(0, n, 11).astype(int)
    summary = {
        "events": n, "train_accuracy": float(np.mean(hist["hit"])) if n else None,
        "windows": [float(np.mean(hist["hit"][a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a],
        "judge_windows": [float(np.mean(hist["judge_hit"][a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a],
        "memory_windows": [float(np.mean(hist["mem_hit"][a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a],
        "config": {k: (None if v is None else str(v)) for k, v in cfg.items()},
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[memjudge] saved {out_dir} train_acc={100 * (summary['train_accuracy'] or 0):.2f} windows=" + " ".join(f"{100 * w:.1f}" for w in summary["windows"]), flush=True)


if __name__ == "__main__":
    main()
