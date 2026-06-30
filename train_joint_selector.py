"""Train the joint-input shared-state AR selector.

Reuses the existing config system, JsonlDataset, dtype handling, and training-loop
shape. New: the JointInputCollator (one joint prompt) and JointDeltaMemSelector
(shared state S steers attention via Delta-Mem). The central agent is config-driven
(`central_model`) — pass Qwen3-0.6B or Qwen3-4B; both just work.

  PYTHONPATH=. python train_joint_selector.py --config configs/joint_ar_selector.yaml \
    --offline_data data/math_rag/train.jsonl --output_dir outputs/joint_ar
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Newer center models (Ministral-3, Qwen3.5) need transformers>=5.x, which references
# torch.float8_e8m0fnu at import time. Shim it before transformers is imported.
from feedback_state.newarch_loader import apply_torch_fp8_shim, load_central_model

apply_torch_fp8_shim()

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from feedback_state.data import JsonlDataset, counterfactual_filter_kwargs, filter_records
from feedback_state.generation import dtype_from_name
from feedback_state.joint_data import VARIANT_AR, JointInputCollator, candidate_token_ids
from feedback_state.joint_models import JointDeltaMemSelector
from feedback_state.joint_prompt import PEER_SEP, build_joint_prompt
from feedback_state.joint_write import resolve_write_policy, run_write_policy
from feedback_state.utils import load_config, merge_args_with_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train joint-input shared-state selector.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--offline_data", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--model_variant", choices=[VARIANT_AR], default=None)
    p.add_argument("--use_shared_state", type=str, default=None)
    p.add_argument("--use_lora", type=str, default=None,
                   help="Attach LoRA to the central model. Default: true iff "
                        "use_shared_state=false (no-mem control). Set true together with "
                        "use_shared_state=true for the LoRA + Delta-Mem combo.")
    p.add_argument("--central_model", default=None)
    p.add_argument("--train_order", choices=["orig", "swap", "random"], default=None)
    p.add_argument("--gradient_checkpointing", type=str, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--per_peer_state", type=str, default=None,
                   help="Give each peer identity its OWN delta state matrix set "
                        "(reads/writes swap states by reference).")
    p.add_argument("--init_lora_from", type=str, default=None,
                   help="Stage-A two-stage combo: load a PRE-TRAINED LoRA adapter from "
                        "this checkpoint dir and FREEZE it, then train ONLY delta-mem on "
                        "top (fixed strong content base + memory increment). Differs from "
                        "use_lora=true which trains LoRA + delta jointly from scratch.")
    return p.parse_args()


def as_bool(v, default=False):
    if v is None:
        return default
    return v if isinstance(v, bool) else str(v).lower() in {"1", "true", "yes", "on"}


def main() -> None:
    args = parse_args()
    cfg = merge_args_with_config(args, load_config(args.config))
    variant = str(cfg.get("model_variant", VARIANT_AR))
    use_shared = as_bool(cfg.get("use_shared_state"), True)
    output_dir = Path(cfg.get("output_dir", "outputs/joint_selector"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    dtype = dtype_from_name(str(cfg.get("dtype", "bfloat16")))
    model_name = str(cfg.get("central_model", "Qwen/Qwen3-0.6B"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=bool(cfg.get("local_files_only", False)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Special separator for multi-peer targets (only used when >1 correct peer).
    if PEER_SEP not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [PEER_SEP]})

    base = load_central_model(
        model_name, dtype=dtype, local_files_only=bool(cfg.get("local_files_only", False)),
    )
    base.resize_token_embeddings(len(tokenizer))
    # LoRA: default-on for the no-memory control; explicitly opt-in alongside the
    # shared state for the LoRA + Delta-Mem combo (use_lora=true, use_shared_state=true).
    # Stage-A two-stage combo: load a pre-trained LoRA and FREEZE it, train only delta.
    init_lora_from = cfg.get("init_lora_from")
    use_lora = as_bool(cfg.get("use_lora"), not use_shared)
    if init_lora_from:
        from peft import PeftModel

        base = PeftModel.from_pretrained(base, str(Path(init_lora_from) / "lora_adapter"),
                                         is_trainable=False)
        use_lora = True  # the tree now has LoRA; combo eval-load path expects it
        print(f"[train_joint] stage-A: loaded FROZEN LoRA from {init_lora_from}/lora_adapter")
    elif use_lora:
        from peft import LoraConfig, get_peft_model

        base = get_peft_model(base, LoraConfig(
            r=int(cfg.get("lora_r", 8)), lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)), bias="none",
            task_type="CAUSAL_LM", target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"])))
    base = base.to(device=device, dtype=dtype)

    model = JointDeltaMemSelector(
        base, num_peers=int(cfg.get("num_peers", 3)), model_variant=variant,
        use_shared_state=use_shared, delta_cfg=cfg, freeze_backbone=as_bool(cfg.get("freeze_backbone"), True),
    ).to(device)
    # Identity trust-readout head (anonymous selection + identity-only feedback channel).
    if as_bool(cfg.get("use_trust_head"), False):
        model.use_trust_head = True
        # warm up the lazy head so its params join the optimizer: install peer-0's
        # (empty) state and call once. Head is built on first trust_bias() call.
        model.set_state_refs({})
        try:
            _ = model.trust_bias()
        except Exception:
            pass
        # if state was empty (cold start), force-create the head at the flattened dim
        if model.trust_head is None:
            import torch as _t
            dim = int(cfg.get("trust_head_dim", 896))
            model._trust_head_dim = dim
            model.trust_head = _t.nn.Sequential(
                _t.nn.Linear(dim, 64), _t.nn.Tanh(), _t.nn.Linear(64, 1)
            ).to(device)
        print(f"[train_joint] trust_head enabled (dim={model._trust_head_dim})")
    # Dual-memory gate: materialize gate_logit BEFORE the optimizer so it trains.
    if as_bool(cfg.get("use_dual_memory"), False):
        model.use_dual_memory = True
        _ = model.gate_lambda()
        print(f"[train_joint] dual-memory gate enabled (gate_logit trainable)")
    if use_lora and use_shared and not init_lora_from:
        # freeze_non_delta_mem_params (called inside the selector when the shared
        # state is attached) freezes the LoRA adapters too — re-enable them so the
        # combo trains delta params + LoRA + head together. (Stage-A KEEPS LoRA frozen:
        # the loaded adapter is the fixed content base; only delta-mem trains.)
        n_lora = 0
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                n_lora += 1
        print(f"[train_joint] combo: re-enabled {n_lora} LoRA param tensors alongside delta-mem")
    if init_lora_from:
        n_lora_frozen = sum(1 for n, p in model.named_parameters() if "lora_" in n and not p.requires_grad)
        print(f"[train_joint] stage-A: {n_lora_frozen} LoRA tensors kept FROZEN; training delta-mem only")

    # Gradient checkpointing: trade compute for memory so long (8192-tok) joint
    # prompts fit. Needed for the LoRA no-mem control; the grad-accum window keeps
    # several step graphs co-resident. With a frozen backbone the input embeddings
    # must be made to require grad, else the checkpointed graph has no grad path.
    # use_cache must be off (set in forward()).
    if as_bool(cfg.get("gradient_checkpointing"), False):
        base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()

    dataset = JsonlDataset(cfg["offline_data"])
    # Counterfactual training is OPTIONAL (no rebuild). Natural diagnostics ON by default.
    n_before = len(dataset.records)
    fkw = counterfactual_filter_kwargs(cfg, "train")
    dataset.records = filter_records(dataset.records, **fkw)
    print(f"[train_joint] records: {n_before} -> {len(dataset.records)} after filter {fkw}")
    collator = JointInputCollator(
        tokenizer, num_peers=model.num_peers, variant=variant,
        peer_order=str(cfg.get("train_order", "orig")), identity_mode=str(cfg.get("identity_mode", "id")),
        max_length=int(cfg.get("max_length", 1536)), include_context=as_bool(cfg.get("include_context"), True),
    )
    loader = DataLoader(dataset, batch_size=int(cfg.get("per_device_train_batch_size", 1)),
                        shuffle=as_bool(cfg.get("shuffle"), False), collate_fn=collator,
                        num_workers=int(cfg.get("dataloader_num_workers", 0)))

    params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(params, lr=float(cfg.get("learning_rate", 1e-4)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 4))
    epochs = float(cfg.get("num_train_epochs", 1.0))
    steps_per_epoch = max(1, len(loader))
    max_steps = int(cfg.get("max_steps", -1))
    total_steps = max_steps if max_steps > 0 else int(epochs * steps_per_epoch)
    sched = get_cosine_schedule_with_warmup(optim, int(float(cfg.get("warmup_ratio", 0.03)) * total_steps), total_steps)

    # Modular Delta-Mem write protocol (see feedback_state/joint_write.py).
    train_write_policy = str(cfg.get("train_write_policy", "feedback"))
    use_feedback = as_bool(cfg.get("use_feedback_in_write"), True)
    include_selected = as_bool(cfg.get("include_selected_peer_in_feedback_prompt"), True)
    train_write_prompt_style = str(cfg.get("train_write_prompt_style", "full"))
    # Decouple identity in WRITE from identity in READ: selection can stay anonymous
    # (collator.include_identity=False) while the feedback WRITE carries the real peer
    # name, so the write matches test and gives the trust state a real identity anchor.
    write_include_identity = cfg.get("write_include_identity", None)
    debug_write = as_bool(cfg.get("debug_write"), False)
    eff_policy = resolve_write_policy(train_write_policy, use_feedback)
    print(f"[train_joint] train_write_policy={train_write_policy} (effective={eff_policy}, use_feedback_in_write={use_feedback})")

    # Per-peer state matrices.
    #  - per_peer_state: each peer identity gets its OWN delta state set; reads/writes
    #    swap the live state by reference, so peers cannot interfere in S.
    per_peer = as_bool(cfg.get("per_peer_state"), False)
    peer_models_cfg = [str(x) for x in (cfg.get("peer_models") or [])]
    if per_peer:
        assert variant == VARIANT_AR, "per_peer_state supports the AR variant"
        assert use_shared, "per_peer_state requires use_shared_state=true"
    per_peer_ar = per_peer and variant == VARIANT_AR
    cand_ids = candidate_token_ids(tokenizer, model.num_peers) if per_peer_ar else None

    def canon_peer(slot_key: str, slot: int) -> int:
        """Map a slot's canonical peer key to its state index.

        Pass slot_peer_keys (real peer_N keys), NOT the display name: under
        anon_id the display name is a randomized placeholder ("Model-A"), so
        anchoring on it would scramble per-peer state. The real key is stable.
        """
        try:
            return peer_models_cfg.index(slot_key)
        except ValueError:
            pass
        if isinstance(slot_key, str) and slot_key.startswith("peer_"):
            try:
                return int(slot_key.split("_")[1])
            except (IndexError, ValueError):
                return slot
        return slot

    print(f"[train_joint] per_peer_state={per_peer}")
    peer_states = [dict() for _ in range(model.num_peers)]
    # Dual-memory: M_R (response-only, anon write) = peer_states; M_I (response+identity,
    # named write) = peer_states_i. Scoring reads the gated blend; writes update each.
    use_dual = as_bool(cfg.get("use_dual_memory"), False)
    peer_states_i = [dict() for _ in range(model.num_peers)]

    model.train()
    model.reset_state()            # shared online S starts at zero
    model.set_write_enabled(False) # READ/selection passes never write; writes happen only in the write pass
    step = 0
    optim.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps, desc=f"train_{variant}")
    # Accumulate the loss across a grad_accum window and call backward() ONCE at the
    # window boundary. Writes run under no_grad, so the online state never carries a
    # graph; the boundary detach below only re-anchors per-peer state references.
    accum = None
    while step < total_steps:
        for batch in loader:
            write_meta = batch.get("write_meta", [])
            tb = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            # 1) SELECTION/READ phase — write disabled; AR loss only.
            if per_peer_ar:
                # Per-peer AR: score each candidate " Peer s" with peer-s's OWN state,
                # CE to the canonical correct slot. Rebuild the pure prompt (no target).
                meta0 = write_meta[0]
                names0, realn = meta0["slot_names"], int(meta0["real"])
                keys0 = meta0.get("slot_peer_keys", names0)
                tgt_slot = meta0["target_slot"]
                if tgt_slot is None or tgt_slot >= realn:
                    step += 1; progress.update(1)
                    continue  # no correct peer in scope; skip (rare)
                prompt = build_joint_prompt(
                    meta0["question"], names0, meta0["slot_texts"], context=meta0["context"],
                    include_identity=collator.include_identity, real=realn)
                enc = tokenizer(prompt, add_special_tokens=True, truncation=True,
                                max_length=int(cfg.get("max_length", 1536)))
                pid = torch.tensor([enc["input_ids"]], device=device)
                pmask = torch.ones_like(pid)
                model.set_write_enabled(False)
                logp_slots = []
                for s in range(realn):
                    j = canon_peer(keys0[s], s)
                    if use_dual:
                        model.set_state_refs(model.blend_states(peer_states[j], peer_states_i[j]))
                    else:
                        model.set_state_refs(peer_states[j])
                    lp = model.score_one_candidate(pid, pmask, cand_ids[s])[0]
                    if getattr(model, "use_trust_head", False):
                        lp = lp + model.trust_bias()[0]  # identity-keyed trust bias
                    logp_slots.append(lp)
                logits = torch.stack(logp_slots)  # [realn] log-probs as logits for CE
                full_loss = torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), torch.tensor([tgt_slot], device=device))
                loss_val = float(full_loss.detach())
                loss = full_loss / grad_accum
            else:
                out = model(**tb)
                loss_val = float(out.loss.detach().float())
                loss = out.loss / grad_accum
            accum = loss if accum is None else accum + loss
            if (step + 1) % grad_accum == 0:
                accum.backward()
                accum = None
                torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                # truncate BPTT: carry state values, drop graph history
                if per_peer:
                    peer_states = [model.detach_state_dict(st) for st in peer_states]
                    if use_dual:
                        peer_states_i = [model.detach_state_dict(st) for st in peer_states_i]
            # 2) Separate WRITE phase per example — carries state forward to the next example.
            for meta in write_meta:
                if per_peer:
                    names = meta["slot_names"]
                    keys = meta.get("slot_peer_keys", names)
                    base_inc_id = (collator.include_identity
                                   if write_include_identity is None
                                   else as_bool(write_include_identity, False))
                    for s in range(int(meta["real"])):
                        j = canon_peer(keys[s], s)
                        # M_R: response-only write (anonymous, base_inc_id).
                        model.set_state_refs(peer_states[j])
                        info = run_write_policy(
                            model, tokenizer, policy=train_write_policy, use_feedback=use_feedback,
                            question=meta["question"], context=meta["context"],
                            slot_names=[names[s]], slot_texts=[meta["slot_texts"][s]],
                            correctness_by_slot=[meta["correctness_by_slot"][s]], real=1,
                            include_identity=base_inc_id,
                            selected_slot=None,
                            selected_correct=None, include_selected_peer=False,
                            write_prompt_style=train_write_prompt_style,
                            short_answers=[meta.get("short_answers", [None] * len(names))[s]],
                            max_length=int(cfg.get("max_length", 1536)), device=device,
                            debug=debug_write,
                        )
                        peer_states[j] = model.get_state_refs()
                        # M_I: response+identity write (real name in prompt).
                        if use_dual:
                            model.set_state_refs(peer_states_i[j])
                            run_write_policy(
                                model, tokenizer, policy=train_write_policy, use_feedback=use_feedback,
                                question=meta["question"], context=meta["context"],
                                slot_names=[names[s]], slot_texts=[meta["slot_texts"][s]],
                                correctness_by_slot=[meta["correctness_by_slot"][s]], real=1,
                                include_identity=True,
                                selected_slot=None, selected_correct=None, include_selected_peer=False,
                                write_prompt_style=train_write_prompt_style,
                                short_answers=[meta.get("short_answers", [None] * len(names))[s]],
                                max_length=int(cfg.get("max_length", 1536)), device=device,
                                debug=debug_write,
                            )
                            peer_states_i[j] = model.get_state_refs()
                else:
                    info = run_write_policy(
                        model, tokenizer, policy=train_write_policy, use_feedback=use_feedback,
                        question=meta["question"], context=meta["context"],
                        slot_names=meta["slot_names"], slot_texts=meta["slot_texts"],
                        correctness_by_slot=meta["correctness_by_slot"], real=meta["real"],
                        include_identity=collator.include_identity, selected_slot=meta["target_slot"],
                        selected_correct=True, include_selected_peer=include_selected,
                        write_prompt_style=train_write_prompt_style,
                        short_answers=meta.get("short_answers"),
                        max_length=int(cfg.get("max_length", 1536)), device=device, debug=debug_write,
                    )
                if debug_write and step < 3:
                    print(f"[train_write] {info}")
            # Ablation: reset per-peer state after every example (no cross-example memory).
            if as_bool(cfg.get("reset_state_every_example"), False):
                peer_states = [dict() for _ in range(model.num_peers)]
            step += 1
            progress.update(1)
            if step % int(cfg.get("logging_steps", 10)) == 0:
                progress.set_postfix(loss=loss_val)
            if step >= total_steps:
                break
    if accum is not None:
        accum.backward()
        torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
        optim.step(); optim.zero_grad(set_to_none=True)
        if per_peer:
            peer_states = [model.detach_state_dict(st) for st in peer_states]
            if use_dual:
                peer_states_i = [model.detach_state_dict(st) for st in peer_states_i]
        else:
            model.detach_state()
    progress.close()

    model.save_feedback_adapter(output_dir)
    if getattr(model, "trust_head", None) is not None:
        torch.save({"state_dict": model.trust_head.state_dict(),
                    "dim": model._trust_head_dim}, output_dir / "trust_head.pt")
        print(f"[train_joint] saved trust_head.pt (dim={model._trust_head_dim})")
    if getattr(model, "gate_logit", None) is not None:
        torch.save({"gate_logit": model.gate_logit.detach().cpu()}, output_dir / "gate.pt")
        print(f"[train_joint] saved gate.pt (lambda={float(model.gate_lambda()):.3f})")
    # Snapshot the trained shared online state so eval can warm-start (trS).
    if as_bool(cfg.get("save_trained_state"), True):
        if per_peer:
            torch.save({"per_peer": [model.snapshot_state_cpu(st) for st in peer_states]},
                       output_dir / "trust_state.pt")
        else:
            torch.save(model.get_online_state(), output_dir / "trust_state.pt")
    tokenizer.save_pretrained(output_dir)
    (output_dir / "train_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"[train_joint] saved -> {output_dir}")


if __name__ == "__main__":
    main()
