"""Train the event-axis symmetric trust memory (steering coupling).

The frozen CM scores every " Peer j" with a trust-derived STEERING vector injected into
its upper-half layers (ActivationSteerer); CE to the gold slot trains the memory's
parameters (phi embedding, decay theta, write strength eta, and the steering projection +
gain). The matrices M_p / G themselves are state (buffers): they evolve by the event-axis
recurrence once per task, under no_grad.

Per example (stream order, batch=1):
  for each candidate j: steer_vec=M_pj@phi* -> logp_CM(j|steered) -> CE(gold) -> backward
  then memory.update(j, +/-1, task_type) for each peer + update_joint   (event axis)

  PYTHONPATH=. python train_symmetric_memory.py --config configs/symmetric_memory.yaml \
    --central_model Qwen/Qwen3-0.6B --output_dir outputs/sym_steer_06
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from feedback_state.newarch_loader import apply_torch_fp8_shim, load_central_model

apply_torch_fp8_shim()

from torch.optim import AdamW
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from feedback_state.data import JsonlDataset
from feedback_state.generation import dtype_from_name
from feedback_state.joint_data import VARIANT_AR, candidate_token_ids
from feedback_state.joint_models import JointDeltaMemSelector
from feedback_state.joint_prompt import PEER_SEP, build_joint_prompt
from feedback_state.permutations import canonical_peer_view
from feedback_state.symmetric_memory import ActivationSteerer, SymmetricTrustMemory, DEFAULT_TASK_TYPES, cm_context_vector
from feedback_state.tasks import task_type_of
from feedback_state.utils import load_config, merge_args_with_config


def as_bool(v, default=False):
    if v is None:
        return default
    return v if isinstance(v, bool) else str(v).lower() in {"1", "true", "yes", "on"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--offline_data", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--central_model", default=None)
    p.add_argument("--center_lora_checkpoint", type=Path, default=None,
                   help="Optional LoRA selector checkpoint to attach to the central model.")
    p.add_argument("--train_center_lora", choices=["on", "off"], default=None,
                   help="Train a LoRA adapter on the center model jointly with Sigma Mem.")
    p.add_argument("--use_joint", choices=["on", "off"], default="off")
    p.add_argument("--phi_mode", choices=["proto"], default=None)
    p.add_argument("--write_mode", choices=["addr", "carry"], default=None)
    p.add_argument("--decay_mode", choices=["scalar", "diag"], default=None)
    p.add_argument("--per_peer_decay", choices=["on", "off"], default=None)
    p.add_argument("--diff_write", choices=["on", "off"], default=None)
    p.add_argument("--peer_mode", choices=["joint", "one"], default=None,
                   help="joint: score all peers in one prompt. one: score each peer alone.")
    p.add_argument("--max_steps", type=int, default=None)
    return p.parse_args()


def _example_view(rec, num_peers):
    view = canonical_peer_view(rec, num_peers, setting="A")
    keys, names, texts, real = view["keys"], view["names"], view["texts"], view["real"]
    cbp = rec.get("correctness_by_peer") or rec.get("peer_correct") or {}
    corr = [int(round(float(cbp.get(k, 0)))) for k in keys]
    peer_ids = [int(str(k).split("_")[1]) if str(k).startswith("peer_") else i for i, k in enumerate(keys)]
    tgt = next((s for s, c in enumerate(corr) if c), None)  # gold slot = first correct peer
    return [f"peer_{i}" for i in range(len(keys))], texts, corr, real, peer_ids, tgt


def _probe_grad(params_named, label="param"):
    tot = 0.0
    for n, p in params_named:
        if p.grad is not None:
            tot += float(p.grad.detach().abs().sum())
    print(f"[grad-probe] {label} grad |.|1 sum = {tot:.4e} -> {'TRAIN' if tot > 0 else 'NO GRAD (bug)'}", flush=True)


def main():
    args = parse_args()
    cfg = merge_args_with_config(args, load_config(args.config))
    device = torch.device(str(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    dtype = dtype_from_name(str(cfg.get("dtype", "bfloat16")))
    out_dir = Path(cfg.get("output_dir", "outputs/sym")); out_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(cfg.get("central_model", "Qwen/Qwen3-0.6B"))
    lora_ckpt = Path(cfg["center_lora_checkpoint"]) if cfg.get("center_lora_checkpoint") else None
    num_peers = int(cfg.get("num_peers", 3))
    max_len = int(cfg.get("max_length", 8192))
    use_joint = str(cfg.get("use_joint", "off")) == "on"
    rank = int(cfg.get("rank", 16))
    task_types = tuple(cfg.get("task_types", DEFAULT_TASK_TYPES))
    phi_mode = str(cfg.get("phi_mode", "proto"))     # only proto (soft task-centroid address)
    phi_layer_frac = float(cfg.get("phi_layer_frac", 0.5))
    decay_mode = str(cfg.get("decay_mode", "scalar"))
    write_mode = str(cfg.get("write_mode", "addr"))  # addr (A) | carry (B: write peer-vs-gold diff direction)
    per_peer_decay = as_bool(cfg.get("per_peer_decay"), False)  # idea-3: each peer its own decay theta
    diff_write = as_bool(cfg.get("diff_write"), False)          # idea-2: read thru a differentiable write -> theta/eta learn
    peer_mode = str(cfg.get("peer_mode", "joint")).lower()
    train_center_lora = as_bool(cfg.get("train_center_lora"), False)
    whiten = as_bool(cfg.get("whiten"), write_mode == "carry")
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))

    tok_src = str(lora_ckpt) if lora_ckpt is not None and (lora_ckpt / "tokenizer_config.json").exists() else model_name
    tok = AutoTokenizer.from_pretrained(tok_src, local_files_only=bool(cfg.get("local_files_only", False)))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if PEER_SEP not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [PEER_SEP]})
    device_map = cfg.get("device_map") or None  # e.g. "auto" to shard a big model across GPUs
    # max_memory: per-device cap to force an even split (e.g. {0:"40GiB",1:"40GiB",...}).
    # Config gives it as a {gpu_index: "NNGiB"} map; keys may be str, coerce to int.
    _mm = cfg.get("max_memory") or None
    max_memory = {int(k): v for k, v in _mm.items()} if isinstance(_mm, dict) else None
    base = load_central_model(model_name, dtype=dtype, local_files_only=bool(cfg.get("local_files_only", False)),
                              device_map=device_map, max_memory=max_memory)
    base.resize_token_embeddings(len(tok))
    if lora_ckpt is not None:
        adapter_dir = lora_ckpt / "lora_adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_dir}")
        base = PeftModel.from_pretrained(
            base, str(adapter_dir), is_trainable=train_center_lora,
            local_files_only=bool(cfg.get("local_files_only", False)),
        )
        mode = "trainable" if train_center_lora else "frozen"
        print(f"[sym/steer] attached {mode} center LoRA from {adapter_dir}", flush=True)
    elif train_center_lora:
        base = get_peft_model(base, LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        ))
        print("[sym/steer] created trainable center LoRA", flush=True)
    if device_map is None:
        base = base.to(device=device, dtype=dtype)
    else:
        # device_map already placed shards across GPUs. Drive I/O from the input-embedding's
        # device so cm_context_vector / score_one_candidate put tensors where the model expects.
        device = base.get_input_embeddings().weight.device
    if train_center_lora and as_bool(cfg.get("gradient_checkpointing"), False):
        base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
    model = JointDeltaMemSelector(base, num_peers=num_peers, model_variant=VARIANT_AR,
                                  use_shared_state=False, delta_cfg=cfg, freeze_backbone=True)
    if device_map is None:
        model = model.to(device)  # device_map: base shards stay placed; selector has no own params here
    if train_center_lora:
        model.train()
    else:
        model.eval()  # backbone frozen; we never train it

    # phi source: soft task-centroid address (the only mode); centroids computed below.
    if phi_mode != "proto":
        raise ValueError(f"phi_mode must be 'proto', got {phi_mode!r}")
    proto_tau = float(cfg.get("proto_tau", 0.1))
    mem = SymmetricTrustMemory(num_peers=num_peers, rank=rank, task_types=task_types,
                               phi_mode="proto",
                               phi_in_dim=base.config.hidden_size,
                               proto_tau=proto_tau,
                               write_mode=write_mode,
                               value_in_dim=(base.config.hidden_size if write_mode == "carry" else None),
                               whiten=whiten,
                               decay_mode=decay_mode,
                               per_peer_decay=per_peer_decay,
                               use_joint=use_joint, device=device).to(device)
    mem.train()
    steerer = ActivationSteerer(base, rank=rank).to(device)

    memory_params = list(mem.parameters()) + list(steerer.parameters())
    lora_named = [(n, p) for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    lora_params = [p for _, p in lora_named]
    params = memory_params + lora_params
    named = list(mem.named_parameters()) + list(steerer.named_parameters()) + lora_named
    print(f"[sym/steer] trainable params: total={sum(p.numel() for p in params)} "
          f"(lora={sum(p.numel() for _, p in lora_named)}, peer_mode={peer_mode})", flush=True)
    memory_lr = float(cfg.get("learning_rate", 1e-3))
    lora_lr = float(cfg.get("lora_learning_rate", cfg.get("learning_rate", 1e-4)))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    optim_groups = [{"params": memory_params, "lr": memory_lr, "weight_decay": weight_decay}]
    if lora_params:
        optim_groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay})
    optim = AdamW(optim_groups)
    print(f"[sym/steer] lr: memory={memory_lr:g} lora={lora_lr:g}", flush=True)


    cand_ids = candidate_token_ids(tok, num_peers)
    records = JsonlDataset(cfg["offline_data"]).records
    total_steps = int(cfg.get("max_steps", len(records)))
    sched = get_cosine_schedule_with_warmup(optim, int(total_steps * float(cfg.get("warmup_ratio", 0.03))), total_steps)

    # phi=proto: precompute FIXED task centroids from the training data (one CM mean per task
    # present in the train stream) + global whitening stats, then install. Tasks ABSENT from
    # train get no centroid -> at eval they address as a soft mixture over the trained ones.
    by_task = {}
    cap = int(cfg.get("proto_centroid_cap", 200))
    for rec in records:
        tt = task_type_of(rec)
        if len(by_task.get(tt, [])) >= cap:
            continue
        q = str(rec.get("problem", rec.get("question", "")))
        with torch.no_grad():
            h = cm_context_vector(base, tok, q, device=device, layer_frac=phi_layer_frac)
        by_task.setdefault(tt, []).append(h.float())
    proto_tasks = sorted(by_task)
    cents = torch.stack([torch.stack(by_task[t]).mean(0) for t in proto_tasks])  # [n_proto, H]
    allh = torch.cat([torch.stack(by_task[t]) for t in proto_tasks])
    mem.set_prototypes(cents, allh.mean(0), allh.std(0))
    print(f"[sym/steer] phi=proto centroids from tasks={proto_tasks} "
          f"(n={ {t: len(by_task[t]) for t in proto_tasks} })", flush=True)

    mem.reset()
    step = 0; probed = False
    log_every = int(cfg.get("logging_steps", 50))
    while step < total_steps:
        for rec in records:
            if step >= total_steps:
                break
            slot_names, texts, corr, real, peer_ids, tgt = _example_view(rec, num_peers)
            if real < 1 or tgt is None or tgt >= real:
                step += 1
                continue
            q = str(rec.get("problem", rec.get("question", "")))
            rag_ctx = str(rec.get("retrieved_context", rec.get("context", ""))) if cfg.get("include_context") else ""
            prompt = build_joint_prompt(q, slot_names, texts, context=rag_ctx or None, include_identity=False, real=real)
            enc = tok(prompt, add_special_tokens=True, truncation=True, max_length=max_len)
            pid = torch.tensor([enc["input_ids"]], device=device)
            pmask = torch.ones_like(pid)
            one_pids = None
            if peer_mode == "one":
                one_pids = []
                for s in range(real):
                    sp = build_joint_prompt(q, [slot_names[s]], [texts[s]], context=rag_ctx or None,
                                            include_identity=False, real=1)
                    se = tok(sp, add_special_tokens=True, truncation=True, max_length=max_len)
                    one_pids.append(torch.tensor([se["input_ids"]], device=device))
            # phi context (READ direction): raw CM hidden of the problem, soft-addressed over centroids.
            phi_ctx = cm_context_vector(base, tok, q, device=device, layer_frac=phi_layer_frac)
            # carry mode (B): WRITE direction per peer = CM(peer_answer) - CM(gold). Encode gold once.
            value_vecs = None
            if write_mode == "carry":
                gold = str(rec.get("answer", ""))
                gold_h = cm_context_vector(base, tok, gold, device=device, layer_frac=phi_layer_frac)
                value_vecs = []
                for s in range(real):
                    ans_h = cm_context_vector(base, tok, str(texts[s])[:4000], device=device, layer_frac=phi_layer_frac)
                    value_vecs.append(ans_h - gold_h)
            # score each candidate WITH the steering vector in-graph; logp_CM under steering
            # carries the gradient back to the memory params (phi/phi_proj/theta/proj/gain).
            logits = []
            for s in range(real):
                pj = peer_ids[s]
                if diff_write:
                    # idea-2: read through one DIFFERENTIABLE write so loss reaches theta/eta.
                    # sign = ground-truth correctness of THIS peer (train-only signal).
                    vv = value_vecs[s] if value_vecs is not None else None
                    steerer.steer_vec = mem.steer_vector_diff(pj, phi_ctx, 1.0 if corr[s] else -1.0, value_vec=vv)
                else:
                    steerer.steer_vec = mem.steer_vector(pj, phi_ctx)
                if peer_mode == "one":
                    spid = one_pids[s]
                    lp = model.score_one_candidate(spid, torch.ones_like(spid), cand_ids[0])[0]
                else:
                    lp = model.score_one_candidate(pid, pmask, cand_ids[s])[0]
                steerer.steer_vec = None
                logits.append(lp)
            logit_vec = torch.stack(logits)
            loss = torch.nn.functional.cross_entropy(
                logit_vec.unsqueeze(0).float(), torch.tensor([tgt], device=device)) / grad_accum
            loss.backward()
            if not probed and step >= grad_accum * 3:
                # probe AFTER the matrices are warm — at step 0 M=0 so read==0 and grad is
                # legitimately zero (d/dphi of phi^T 0 phi = 0); that is not a bug.
                _probe_grad(named, "all trainable params")
                if lora_named:
                    _probe_grad(lora_named, "LoRA params")
                probed = True
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            # event-axis state update (no grad; state, not params). sign s = ground-truth
            # correctness. (A trainable judge head -> learned s was tried and dropped: from a
            # frozen-CM mid-layer feature, "answer correctness" is not linearly readable
            # (probe AUC ~0.69), so the judge could not produce a usable s; see git history.)
            signs = [1.0 if corr[s] else -1.0 for s in range(real)]
            for s in range(real):
                vv = value_vecs[s] if value_vecs is not None else None
                mem.update(peer_ids[s], signs[s], phi_ctx, value_vec=vv)
            mem.update_joint(list(signs) + [0.0] * (num_peers - real))
            step += 1
            if step % log_every == 0:
                gd = mem.gamma().detach()
                gstr = (f"{float(gd):.3f}" if gd.ndim == 0
                        else "[" + ",".join(f"{x:.3f}" for x in gd.reshape(gd.shape[0], -1).mean(-1).tolist()) + "]")
                print(f"[sym/steer] step {step}/{total_steps} loss={float(loss)*grad_accum:.4f} "
                      f"gamma={gstr} eta={float(mem.eta):.3f} gain={float(steerer.gain):.3f}", flush=True)

    # save trained params + final matrices
    payload = {"mem": mem.state_dict(), "steerer": steerer.state_dict()}
    torch.save(payload, out_dir / "sym_memory.pt")
    torch.save(mem.snapshot(), out_dir / "sym_state.pt")
    if train_center_lora:
        adapter_dir = out_dir / "lora_adapter"
        base.save_pretrained(adapter_dir)
        tok.save_pretrained(out_dir)
    (out_dir / "train_config.json").write_text(json.dumps({k: str(v) for k, v in cfg.items()}, indent=1))
    print(f"[sym/steer] saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
