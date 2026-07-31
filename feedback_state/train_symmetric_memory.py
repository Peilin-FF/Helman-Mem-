"""Train the event-axis symmetric trust memory (steering coupling).

The frozen CM scores every candidate with a trust-derived STEERING vector injected into
its upper-half layers (ActivationSteerer). Training uses cross entropy to the first
correct slot. The matrices M_p themselves are state buffers updated once per event.

Per example (stream order, batch=1):
  for each candidate j: steer_vec=M_pj@phi* -> logp_CM(j|steered) -> loss -> backward
  then memory.update(j, +/-1, task_type) for each peer   (event axis)

  python -m feedback_state.train_symmetric_memory --config configs/symmetric_memory_candidate_yesno.yaml \
    --central_model models/Qwen3-0.6B --output_dir outputs/sym_steer_06
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from feedback_state.checkpoint_manifest import write_checkpoint_manifest
from feedback_state.newarch_loader import (
    apply_torch_fp8_shim,
    dtype_from_name,
    load_central_model,
)

apply_torch_fp8_shim()

from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from feedback_state.data import JsonlDataset
from feedback_state.joint_data import batch_candidate_judge_inputs, yes_no_token_ids
from feedback_state.joint_models import CandidateUtilityScorer
from feedback_state.permutations import canonical_peer_view
from feedback_state.prompt_protocol import (
    candidate_context_text,
    candidate_tokenization_format,
    prompt_context_format,
    prompt_protocol_name,
    validate_prompt_protocol,
)
from feedback_state.symmetric_memory import (
    ActivationSteerer,
    DEFAULT_TASK_TYPES,
    SymmetricTrustMemory,
    cm_context_vector,
)
from feedback_state.tasks import task_type_of
from feedback_state.utils import load_config, merge_args_with_config


LOSS_FIRST_CORRECT_CE = "first_correct_ce"


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
    p.add_argument("--num_peers", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument(
        "--legacy_prompt_protocol",
        choices=["on", "off"],
        default=None,
        help="Reproduce the original 2,685-row prompt protocol: Python str(context) "
             "and candidate tokenizer truncation at exactly 8192 tokens.",
    )
    p.add_argument("--phi_mode", choices=["proto"], default=None)
    p.add_argument("--gamma_init", type=float, default=None,
                   help="Initial Sigma memory decay; must be in (0, 1].")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible memory/projection initialization.")
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


def _candidate_training_loss(
    logit_vec: torch.Tensor,
    correctness: list[int],
) -> torch.Tensor | None:
    """Return the peer-selection loss, or ``None`` when the objective is undefined/zero."""
    logits = logit_vec.float().reshape(-1)
    labels = torch.as_tensor(correctness, dtype=torch.float32, device=logits.device).reshape(-1)
    if logits.numel() != labels.numel():
        raise ValueError(
            f"logit/correctness size mismatch: {logits.numel()} != {labels.numel()}"
        )
    correct_slots = torch.nonzero(labels > 0.5, as_tuple=False).reshape(-1)
    if correct_slots.numel() == 0:
        return None
    target = correct_slots[0].reshape(1).to(dtype=torch.long)
    return torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)


def _skip_training_event(
    real: int,
    target: int | None,
) -> bool:
    """Return whether an event has no valid peer-selection target."""
    return real < 1 or target is None or target >= real


def _planned_backward_events(
    records: list[dict], num_peers: int, total_events: int
) -> int:
    """Count events that will contribute gradients across the planned stream."""
    if not records or total_events < 1:
        return 0
    count = 0
    for index in range(total_events):
        rec = records[index % len(records)]
        _, _, _, real, _, target = _example_view(rec, num_peers)
        count += int(not _skip_training_event(real, target))
    return count


def _probe_grad(params_named, label="param"):
    tot = 0.0
    for n, p in params_named:
        if p.grad is not None:
            tot += float(p.grad.detach().abs().sum())
    print(f"[grad-probe] {label} grad |.|1 sum = {tot:.4e} -> {'TRAIN' if tot > 0 else 'NO GRAD (bug)'}", flush=True)


def main():
    args = parse_args()
    cfg = merge_args_with_config(args, load_config(args.config))
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cfg["seed"] = seed
    device = torch.device(str(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    dtype_name = str(cfg.get("dtype", "bfloat16")).lower()
    dtype = dtype_from_name(dtype_name)
    out_dir = Path(cfg.get("output_dir", "outputs/sym")); out_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(cfg.get("central_model", "models/Qwen3-0.6B"))
    num_peers = int(cfg.get("num_peers", 3))
    max_len = int(cfg.get("max_length", 8192))
    legacy_prompt_protocol = as_bool(cfg.get("legacy_prompt_protocol"), False)
    validate_prompt_protocol(
        legacy_prompt_protocol=legacy_prompt_protocol,
        max_length=max_len,
    )
    rank = int(cfg.get("rank", 16))
    task_types = tuple(cfg.get("task_types", DEFAULT_TASK_TYPES))
    phi_mode = str(cfg.get("phi_mode", "proto"))     # only proto (soft task-centroid address)
    phi_layer_frac = float(cfg.get("phi_layer_frac", 0.5))
    gamma_init = float(cfg.get("gamma_init", 0.9))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    if not 0.0 < gamma_init <= 1.0:
        raise ValueError(f"gamma_init must be in (0, 1], got {gamma_init}")
    cfg["loss_mode"] = LOSS_FIRST_CORRECT_CE
    cfg["gamma_init"] = gamma_init
    cfg["legacy_prompt_protocol"] = "on" if legacy_prompt_protocol else "off"
    cfg["prompt_protocol"] = prompt_protocol_name(legacy_prompt_protocol)
    cfg["prompt_context_format"] = prompt_context_format(legacy_prompt_protocol)
    cfg["candidate_tokenization_format"] = candidate_tokenization_format(
        legacy_prompt_protocol
    )
    cfg["dtype"] = dtype_name
    cfg["max_length"] = max_len
    cfg["freeze_backbone"] = "true"
    cfg["peer_mode"] = "joint"
    cfg["score_mode"] = "candidate_yesno"

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=bool(cfg.get("local_files_only", False)))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device_map = cfg.get("device_map") or None  # e.g. "auto" to shard a big model across GPUs
    # max_memory: per-device cap to force an even split (e.g. {0:"40GiB",1:"40GiB",...}).
    # Config gives it as a {gpu_index: "NNGiB"} map; keys may be str, coerce to int.
    _mm = cfg.get("max_memory") or None
    max_memory = {int(k): v for k, v in _mm.items()} if isinstance(_mm, dict) else None
    base = load_central_model(model_name, dtype=dtype, local_files_only=bool(cfg.get("local_files_only", False)),
                              device_map=device_map, max_memory=max_memory)
    if device_map is None:
        base = base.to(device=device, dtype=dtype)
    else:
        # device_map already placed shards across GPUs. Drive I/O from the input-embedding's
        # device so context and candidate tensors land where the model expects.
        device = base.get_input_embeddings().weight.device
    model = CandidateUtilityScorer(base, freeze_backbone=True)
    if device_map is None:
        model = model.to(device)  # device_map: base shards stay placed; selector has no own params here
    model.eval()  # backbone frozen; we never train it

    # phi source: soft task-centroid address (the only mode); centroids computed below.
    if phi_mode != "proto":
        raise ValueError(f"phi_mode must be 'proto', got {phi_mode!r}")
    proto_tau = float(cfg.get("proto_tau", 0.1))
    mem = SymmetricTrustMemory(num_peers=num_peers, rank=rank, task_types=task_types,
                               phi_mode="proto",
                               phi_in_dim=base.config.hidden_size,
                               proto_tau=proto_tau,
                               gamma_init=gamma_init, device=device).to(device)
    mem.train()
    steerer = ActivationSteerer(base, rank=rank).to(device)

    memory_params = [
        param for param in list(mem.parameters()) + list(steerer.parameters())
        if param.requires_grad
    ]
    params = memory_params
    named = [
        (name, param)
        for name, param in list(mem.named_parameters()) + list(steerer.named_parameters())
        if param.requires_grad
    ]
    print(f"[sym/steer] trainable params: total={sum(p.numel() for p in params)} "
          f"(score_mode=candidate_yesno, loss_mode={LOSS_FIRST_CORRECT_CE})", flush=True)
    memory_lr = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    optim_groups = [{"params": memory_params, "lr": memory_lr, "weight_decay": weight_decay}]
    optim = AdamW(optim_groups)
    print(f"[sym/steer] lr: memory={memory_lr:g}", flush=True)


    yes_ids, no_ids = yes_no_token_ids(tok)
    records = JsonlDataset(cfg["offline_data"]).records
    if not records:
        raise ValueError("Training data is empty")
    total_steps = int(cfg.get("max_steps", len(records)))
    planned_backward = _planned_backward_events(records, num_peers, total_steps)
    if planned_backward < 1:
        raise ValueError("Training stream has no events with a valid objective")
    planned_optimizer_steps = math.ceil(planned_backward / grad_accum)
    warmup_steps = int(planned_optimizer_steps * float(cfg.get("warmup_ratio", 0.03)))
    sched = get_cosine_schedule_with_warmup(
        optim, warmup_steps, planned_optimizer_steps
    )
    cfg["planned_backward_events"] = planned_backward
    cfg["planned_optimizer_steps"] = planned_optimizer_steps
    cfg["warmup_optimizer_steps"] = warmup_steps
    print(
        f"[sym/steer] schedule: events={total_steps} backward={planned_backward} "
        f"optimizer_steps={planned_optimizer_steps} warmup_steps={warmup_steps}",
        flush=True,
    )

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
            h = cm_context_vector(
                base, tok, q, device=device, layer_frac=phi_layer_frac
            )
        by_task.setdefault(tt, []).append(h.float())
    proto_tasks = sorted(by_task)
    cents = torch.stack([
        torch.stack(by_task[t]).mean(0) for t in proto_tasks
    ])
    allh = torch.cat([torch.stack(by_task[t]) for t in proto_tasks])
    mem.set_prototypes(cents, allh.mean(0), allh.std(0))
    print(
        f"[sym/steer] phi=proto centroids from tasks={proto_tasks} "
        f"(n={ {t: len(by_task[t]) for t in proto_tasks} })",
        flush=True,
    )

    mem.reset()
    base_snap = mem.snapshot()
    step = 0; probed = False
    backward_steps = 0
    last_loss = None
    log_every = int(cfg.get("logging_steps", 50))
    while step < total_steps:
        for rec in records:
            if step >= total_steps:
                break
            slot_names, texts, corr, real, peer_ids, tgt = _example_view(rec, num_peers)
            # Preserve legacy CE behavior: an event with no target is skipped entirely.
            skip_objective = _skip_training_event(real, tgt)
            if skip_objective:
                step += 1
                continue
            q = str(rec.get("problem", rec.get("question", "")))
            rag_ctx = candidate_context_text(
                rec,
                include_context=as_bool(cfg.get("include_context"), False),
                legacy_prompt_protocol=legacy_prompt_protocol,
            )
            candidate_pids, candidate_mask = batch_candidate_judge_inputs(
                tok, q, slot_names, texts,
                context=rag_ctx or None,
                include_identity=False,
                real=real,
                max_length=max_len,
                device=device,
                legacy_prompt_protocol=legacy_prompt_protocol,
            )
            # phi context (READ direction): raw CM hidden of the problem, soft-addressed over centroids.
            phi_ctx = cm_context_vector(base, tok, q, device=device, layer_frac=phi_layer_frac)
            def score_candidate_batch(steer_vecs=None):
                steerer.steer_vec = steer_vecs
                try:
                    return model.score_candidate_utility(candidate_pids, candidate_mask, yes_ids, no_ids)
                finally:
                    steerer.steer_vec = None

            # Every processed event is scored with Sigma steering. There is no
            # center-only fallback; Base is an evaluation-only explicit ablation.
            steer_vecs = []
            for s in range(real):
                pj = peer_ids[s]
                # Training reads through the current correctness-driven write so
                # gradients reach the learned decay and update strengths.
                steer_vecs.append(mem.training_steer_vector(
                    pj, phi_ctx, 1.0 if corr[s] else -1.0))
            steered_logit_vec = score_candidate_batch(torch.stack(steer_vecs))
            logit_vec = steered_logit_vec
            raw_loss = _candidate_training_loss(
                logit_vec,
                corr[:real],
            )
            if raw_loss is None:
                if not skip_objective:
                    raise RuntimeError("processed event unexpectedly has no training target")
                last_loss = None
            else:
                loss = raw_loss / grad_accum
                loss.backward()
                backward_steps += 1
                last_loss = float(loss) * grad_accum
                if not probed and backward_steps >= grad_accum * 3:
                    # Probe after the matrices are warm: at step 0 M=0, so a zero read
                    # gradient is legitimate rather than evidence of a broken path.
                    _probe_grad(named, "all trainable params")
                    probed = True
                if backward_steps % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
                    optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            # Event-axis state update after selection; sign is external correctness.
            signs = [1.0 if corr[s] else -1.0 for s in range(real)]
            for s in range(real):
                mem.update(peer_ids[s], signs[s], phi_ctx)
            step += 1
            if step % log_every == 0:
                gd = mem.gamma().detach()
                gstr = (f"{float(gd):.3f}" if gd.ndim == 0
                        else "[" + ",".join(f"{x:.3f}" for x in gd.reshape(gd.shape[0], -1).mean(-1).tolist()) + "]")
                loss_msg = f"{last_loss:.4f}" if last_loss is not None else "skip"
                print(f"[sym/steer] step {step}/{total_steps} loss={loss_msg} "
                      f"gamma={gstr} eta={float(mem.eta):.3f} "
                      f"steer_gain={float(steerer.gain):.3f}", flush=True)

    # save trained params + final matrices
    if backward_steps % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
        optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
    payload = {"mem": mem.state_dict(), "steerer": steerer.state_dict()}
    torch.save(payload, out_dir / "sym_memory.pt")
    serialized_config = {
        key: (None if value is None else str(value))
        for key, value in cfg.items()
    }
    (out_dir / "train_config.json").write_text(
        json.dumps(serialized_config, indent=1) + "\n"
    )
    write_checkpoint_manifest(out_dir, Path(cfg["offline_data"]))
    print(f"[sym/steer] saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
