"""Evaluate the joint-input shared-state selector + baselines (Math+RAG).

The AR selector (ar_shared_state_selector) and the no-memory control
(use_shared_state=false). State modes via --init_state {zeros,trained}
x --mode {read_only,online_feedback} = S0+RO / trS+RO / S0+ON / trS+ON.

Reuses: JsonlDataset, the joint collator's record_views, the per-peer correctness
evaluator, and the canonical slot<->peer convention. Baselines (Bayesian global/
domain/dataset, oracle) are pure functions over correctness labels.

Writes: predictions.jsonl, eval_metrics.json, eval_by_{domain,counterfactual_type,
dataset,peer}.csv, plus a 10-example joint-prompt debug preview.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

# Newer center models such as Qwen3.5 need transformers>=5.x, which references
# torch.float8_e8m0fnu at import. Shim before transformers is imported.
from feedback_state.newarch_loader import apply_torch_fp8_shim, load_central_model

apply_torch_fp8_shim()

from transformers import AutoModelForCausalLM, AutoTokenizer

from feedback_state.data import JsonlDataset, counterfactual_filter_kwargs, filter_records
from feedback_state.generation import dtype_from_name
from feedback_state.joint_data import VARIANT_AR, VARIANT_BCE, JointInputCollator, candidate_token_ids, char_to_token_spans
from feedback_state.joint_models import JointDeltaMemSelector
from feedback_state.joint_prompt import PEER_SEP, ar_target_text, peer_response_char_spans
from feedback_state.joint_write import assert_scoring_readonly, build_short_answers, resolve_write_policy, run_write_policy
from feedback_state.permutations import invert_perm, short_peer_name
from feedback_state.tasks import task_type_of
from feedback_state.utils import load_config, merge_args_with_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate joint-input shared-state selector.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--offline_data", type=Path, default=None)
    p.add_argument("--model_variant", choices=[VARIANT_AR, VARIANT_BCE], default=None)
    p.add_argument("--use_shared_state", type=str, default=None)
    p.add_argument("--per_peer_state", type=str, default=None,
                   help="Per-peer state matrices (each peer identity has its own state). "
                        "Default: auto-detect from the checkpoint's train_config.json.")
    p.add_argument("--state_bucket_by_task", type=str, default=None,
                   help="Eval-time per-(task_type, peer) state buckets: trust from one "
                        "task never pollutes another (B2-style bucketing, no retraining).")
    p.add_argument("--central_model", default=None)
    p.add_argument("--init_state", choices=["zeros", "trained"], default=None)
    p.add_argument("--mode", choices=["read_only", "online_feedback"], default=None)
    p.add_argument("--test_order", choices=["orig", "swap", "random"], default=None)
    p.add_argument("--eval_write_policy", choices=["none", "selection_tokens", "feedback"], default=None)
    # Selective-feedback gate: only WRITE feedback on examples where the gate fires.
    #   none   = write every example (current behaviour)
    #   margin = write only when CM is UNCERTAIN (top1-top2 logp margin < gate_margin_thr)
    #   trust  = write only when the selected peer's running trust is NOT already dominant
    p.add_argument("--feedback_gate", choices=["none", "margin", "trust_self", "trust_gt"], default="none")
    p.add_argument("--gate_margin_thr", type=float, default=2.0,
                   help="margin gate: write feedback only when top1-top2 logp margin < this.")
    p.add_argument("--gate_trust_thr", type=float, default=0.15,
                   help="trust gate: write only when (sel peer trust - mean others) < this "
                        "(i.e. history does NOT yet clearly favour the pick).")
    p.add_argument("--gate_trust_rho", type=float, default=0.9, help="trust EMA decay for the gate.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--stream_seed", type=int, default=None,
                   help="If set, shuffle the eval record order with this seed (for "
                        "seed experiments: online state evolution depends on stream order).")
    p.add_argument("--state_transform", choices=["none", "zeros", "randn", "shuffle", "permute_layers"],
                   default=None, help="Corrupt trained-state VALUES (keep shape) to probe "
                                      "whether the state carries trust info or is a capability switch.")
    p.add_argument("--state_seed", type=int, default=None, help="seed for --state_transform")
    return p.parse_args()


def as_bool(v, default=False):
    return v if isinstance(v, bool) else (default if v is None else str(v).lower() in {"1", "true", "yes", "on"})


def _domain(record) -> str:
    return str(record.get("domain") or task_type_of(record))


# ---- Bayesian baselines (pure) ---------------------------------------------
def peer_reliability(records, key=None):
    """Per-peer correctness rate over records (optionally within a key group)."""
    correct, total = defaultdict(float), defaultdict(float)
    for r in records:
        cbp = r.get("correctness_by_peer") or {}
        grp = key(r) if key else None
        for peer, c in cbp.items():
            total[(grp, peer)] += 1
            correct[(grp, peer)] += 1.0 if c else 0.0
    return {k: (correct[k] / total[k] if total[k] else 0.0) for k in total}


def best_peer(reliab, grp, peers):
    scored = [(reliab.get((grp, p), 0.0), p) for p in peers]
    return max(scored)[1] if scored else None


def main() -> None:
    args = parse_args()
    cfg = merge_args_with_config(args, load_config(args.config))
    variant = str(cfg.get("model_variant", VARIANT_AR))
    use_shared = as_bool(cfg.get("use_shared_state"), True)
    mode = str(cfg.get("mode", "read_only"))
    init_state = str(cfg.get("init_state", "zeros"))
    device = torch.device(str(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    dtype = dtype_from_name(str(cfg.get("dtype", "bfloat16")))
    ckpt = Path(cfg["checkpoint"]) if cfg.get("checkpoint") else None
    model_name = str(cfg.get("central_model", "Qwen/Qwen3-0.6B"))
    tok = AutoTokenizer.from_pretrained(str(ckpt) if ckpt and (ckpt / "tokenizer_config.json").exists() else model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if PEER_SEP not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [PEER_SEP]})

    base = load_central_model(model_name, dtype=dtype, local_files_only=bool(cfg.get("local_files_only", False))).to(device)
    base.resize_token_embeddings(len(tok))
    # LoRA + Delta-Mem combo: the delta adapter was saved with module names from the
    # PEFT-wrapped tree (LoRA attached BEFORE delta at train time), so mirror that
    # order — wrap with PeftModel first, then let the selector attach delta. The plain
    # no-mem LoRA path (use_shared_state=false) still loads via load_feedback_adapter.
    combo_lora = use_shared and ckpt is not None and (ckpt / "lora_adapter").exists()
    if combo_lora:
        from peft import PeftModel

        base = PeftModel.from_pretrained(base, str(ckpt / "lora_adapter"), is_trainable=False).to(device)
        print(f"[eval_joint] combo: attached LoRA adapter from {ckpt / 'lora_adapter'}")
    num_peers = int(cfg.get("num_peers", 3))
    model = JointDeltaMemSelector(base, num_peers=num_peers, model_variant=variant,
                                  use_shared_state=use_shared, delta_cfg=cfg, freeze_backbone=True).to(device)
    if combo_lora:
        # Delta attach wraps the upper-half self_attn modules into `self_attn.base.*`,
        # which is the tree shape the adapter was SAVED from — so the from_pretrained
        # load above silently dropped every upper-layer LoRA key (PEFT zero-inits
        # lora_B, i.e. those adapters became no-ops). Reload the full state dict onto
        # the post-attach tree and require that every LoRA key lands.
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file

        sd = load_file(str(ckpt / "lora_adapter" / "adapter_model.safetensors"))
        res = set_peft_model_state_dict(base, sd)
        bad = [k for k in getattr(res, "unexpected_keys", []) if "lora_" in k]
        assert not bad, f"combo LoRA reload: {len(bad)} keys failed to land, e.g. {bad[:4]}"
        print(f"[eval_joint] combo: reloaded {sum('lora_' in k for k in sd)} LoRA tensors onto delta-wrapped tree")
    if ckpt is not None:
        model.load_feedback_adapter(ckpt, map_location=device)
    # Identity trust-readout head (if the checkpoint trained one).
    if ckpt is not None and (ckpt / "trust_head.pt").exists():
        import torch.nn as _nn
        th = torch.load(ckpt / "trust_head.pt", map_location=device)
        model._trust_head_dim = int(th["dim"])
        model.trust_head = _nn.Sequential(
            _nn.Linear(model._trust_head_dim, 64), _nn.Tanh(), _nn.Linear(64, 1)
        ).to(device)
        model.trust_head.load_state_dict(th["state_dict"])
        model.trust_head.to(dtype=dtype)
        model.use_trust_head = True
        print(f"[eval_joint] loaded trust_head.pt (dim={model._trust_head_dim})")
    # Dual-memory gate (if trained): load gate_logit, enable blended scoring.
    use_dual = False
    if ckpt is not None and (ckpt / "gate.pt").exists():
        import torch.nn as _nn
        g = torch.load(ckpt / "gate.pt", map_location=device)
        model.gate_logit = _nn.Parameter(g["gate_logit"].to(device))
        model.use_dual_memory = True
        use_dual = True
        print(f"[eval_joint] loaded gate.pt (lambda={float(model.gate_lambda()):.3f})")
    model.eval()

    # Per-peer state mode: each peer identity has its OWN delta state set; reads and
    # writes swap the live state by reference. Auto-detected from the checkpoint's
    # train_config.json, overridable via --per_peer_state / config.
    per_peer = cfg.get("per_peer_state")
    if per_peer is None and ckpt is not None and (ckpt / "train_config.json").exists():
        try:
            per_peer = json.loads((ckpt / "train_config.json").read_text()).get("per_peer_state")
        except Exception:
            per_peer = None
    per_peer = as_bool(per_peer, False)
    # Task-bucketed states (eval-time only, no retraining — B2-contextual's bucketing
    # applied to the state): key the per-peer states by (task_type, peer) so trust
    # accumulated on one task never pollutes another. Fixes the cross-task transition
    # harm (math->code switches mislead a task-agnostic memory on shuffled streams).
    # Routing label: "true"/"oracle" -> ground-truth task_type (diagnostic upper bound);
    # "pred" -> the central model CLASSIFIES the task itself (honest, no oracle labels).
    _bkt = str(cfg.get("state_bucket_by_task") or "").lower()
    bucket_by_task = _bkt in {"1", "true", "yes", "on", "oracle", "pred"}
    bucket_pred = _bkt == "pred"
    peer_models_cfg = [str(x) for x in (cfg.get("peer_models") or [])]
    if per_peer:
        assert variant == VARIANT_AR, "per_peer_state eval supports only the AR variant"
        print("[eval_joint] per_peer_state mode: per-peer state matrices, per-candidate scoring passes")

    def canon_peer(slot_name: str, slot: int) -> int:
        # Prefer matching the real model name against the configured peer list.
        try:
            return peer_models_cfg.index(slot_name)
        except ValueError:
            pass
        # anon_id passes a placeholder display name; fall back to the canonical key
        # "peer_N" if given, else the slot index. This keeps per-peer state anchored
        # to the REAL peer regardless of the (possibly randomized) visible label.
        if isinstance(slot_name, str) and slot_name.startswith("peer_"):
            try:
                return int(slot_name.split("_")[1])
            except (IndexError, ValueError):
                return slot
        return slot

    # Shared state init: zeros (cold) or trained (warm-start from trust_state.pt).
    model.reset_state()
    peer_states = [dict() for _ in range(num_peers)]
    if init_state == "trained" and ckpt is not None and (ckpt / "trust_state.pt").exists():
        st = torch.load(ckpt / "trust_state.pt", map_location=device)
        if per_peer:
            assert isinstance(st, dict) and "per_peer" in st, \
                "per_peer_state eval with init_state=trained needs a per-peer trust_state.pt"
            peer_states = [
                {k: v.to(device=device, dtype=dtype) for k, v in psd.items()}
                for psd in st["per_peer"]
            ]
            st = None
        # Interpretability probe (--state_transform): corrupt the trained state's
        # VALUES while keeping its shape, to test whether the trained state carries
        # useful trust numbers or just acts as a capability switch.
        xf = str(cfg.get("state_transform", "none")).lower()
        if xf != "none" and st is not None:
            g = torch.Generator(device="cpu").manual_seed(int(cfg.get("state_seed", 0)))
            for k, v in st.items():
                vc = v.detach().cpu().float()
                if xf == "zeros":
                    vc = torch.zeros_like(vc)
                elif xf == "randn":           # random gaussian, matched per-tensor std
                    vc = torch.randn(vc.shape, generator=g) * (vc.std() + 1e-6)
                elif xf == "shuffle":         # shuffle all entries (destroys structure, keeps value set)
                    flat = vc.flatten()[torch.randperm(vc.numel(), generator=g)]
                    vc = flat.view(vc.shape)
                elif xf == "permute_layers":  # handled below (cross-layer); skip per-tensor
                    pass
                st[k] = vc.to(v.device, v.dtype)
            if xf == "permute_layers":        # reassign each layer's matrix to another layer
                keys = list(st.keys()); perm = torch.randperm(len(keys), generator=g).tolist()
                vals = [st[keys[p]] for p in perm]
                st = {keys[i]: vals[i] for i in range(len(keys))}
            print(f"[eval_joint] state_transform={xf} applied to trained state")
        if st is not None:
            model.load_online_state(st)

    # Lazily materialized state buckets: key = (task_type if bucketing else "", peer).
    # Each bucket starts from the trained per-peer init (tensors shared; writes REPLACE
    # the refs, never mutate in place, so sharing the init across buckets is safe).
    state_buckets: dict = {}
    state_buckets_i: dict = {}  # dual-memory: M_I (response+identity) buckets

    def get_peer_state(task_key: str, j: int) -> dict:
        key = (task_key if bucket_by_task else "", j)
        if key not in state_buckets:
            state_buckets[key] = dict(peer_states[j])
        return state_buckets[key]

    def get_peer_state_i(task_key: str, j: int) -> dict:
        key = (task_key if bucket_by_task else "", j)
        if key not in state_buckets_i:
            state_buckets_i[key] = dict(peer_states[j])
        return state_buckets_i[key]

    def put_peer_state(task_key: str, j: int, stref: dict) -> None:
        state_buckets[(task_key if bucket_by_task else "", j)] = stref

    def put_peer_state_i(task_key: str, j: int, stref: dict) -> None:
        state_buckets_i[(task_key if bucket_by_task else "", j)] = stref

    if per_peer and bucket_by_task:
        print(f"[eval_joint] state_bucket_by_task={'pred' if bucket_pred else 'oracle'}: "
              "per-(task, peer) state buckets")

    # Self-classified task routing ("pred"): the central model labels the task itself
    # with one short read-only forward — multiple-choice A/B/C next-token comparison
    # (robust for small models; bare label words collapse to "math" on 0.6B). The
    # retrieved-context snippet is part of the record's legitimate INPUT (the joint
    # prompt shows it too), not an oracle label.
    _TASK_LETTERS = [("math", " A"), ("code", " B"), ("rag", " C")]
    _task_letter_ids = [(tt, tok(w, add_special_tokens=False)["input_ids"][0]) for tt, w in _TASK_LETTERS]
    task_pred_total = task_pred_correct = 0
    _task_conf = defaultdict(int)  # (true, pred) -> count

    def classify_task(record) -> str:
        problem = str(record.get("problem", ""))[:1200]
        ctx = str(record.get("context") or record.get("retrieved_context") or "")[:300]
        ctx_part = f"\nRetrieved documents (snippet): {ctx}\n" if ctx.strip() else ""
        cprompt = (
            "Look at this task and classify it.\n\n"
            f"Task: {problem}\n{ctx_part}\n"
            "What kind of task is this?\n"
            "(A) a math problem to solve\n"
            "(B) a programming task: write or complete code\n"
            "(C) a knowledge question to answer (possibly using retrieved documents)\n\n"
            "Answer: ("
        )
        enc_c = tok(cprompt, return_tensors="pt", truncation=True, max_length=1024)
        enc_c = {k: t.to(device) for k, t in enc_c.items()}
        model.set_write_enabled(False)
        with torch.no_grad():
            lg = model.base_model(**enc_c, use_cache=False, return_dict=True).logits[0, -1]
        # compare both " A" and "A"-style tokens for robustness after "("
        scores = {}
        for tt, letter in [("math", "A"), ("code", "B"), ("rag", "C")]:
            ids = {tok(letter, add_special_tokens=False)["input_ids"][0],
                   tok(" " + letter, add_special_tokens=False)["input_ids"][0]}
            scores[tt] = max(float(lg[i]) for i in ids)
        return max(scores, key=scores.get)


    # Modular Delta-Mem write protocol at eval time (see feedback_state/joint_write.py).
    # eval_write_policy is authoritative; --mode online_feedback maps to "feedback" when
    # eval_write_policy is unset (back-compat).
    eval_write_policy = str(cfg.get("eval_write_policy", "feedback" if mode == "online_feedback" else "none"))
    use_feedback = as_bool(cfg.get("use_feedback_in_write"), True)
    include_selected = as_bool(cfg.get("include_selected_peer_in_feedback_prompt"), True)
    eval_write_prompt_style = str(cfg.get("eval_write_prompt_style", "full"))
    debug_write = as_bool(cfg.get("debug_write"), False)
    eff_eval_policy = resolve_write_policy(eval_write_policy, use_feedback)
    online = eff_eval_policy != "none"
    # Selective-feedback gate config + running trust (for the 'trust' gate).
    feedback_gate = str(cfg.get("feedback_gate", "none"))
    gate_margin_thr = float(cfg.get("gate_margin_thr", 2.0))
    gate_trust_thr = float(cfg.get("gate_trust_thr", 0.15))
    gate_trust_rho = float(cfg.get("gate_trust_rho", 0.9))
    gate_trust = {}            # canonical peer id -> running mean correctness (EMA)
    gate_stats = [0, 0]        # [num_written, num_total] for logging
    if online and feedback_gate != "none":
        print(f"[eval_joint] feedback_gate={feedback_gate} "
              f"(margin_thr={gate_margin_thr}, trust_thr={gate_trust_thr})")
    # Train/eval policy-mismatch diagnostic warning.
    train_policy = None
    if ckpt is not None and (ckpt / "train_config.json").exists():
        try:
            train_policy = json.loads((ckpt / "train_config.json").read_text()).get("train_write_policy")
        except Exception:
            train_policy = None
    if train_policy is not None and str(train_policy) != eval_write_policy:
        print(f"[eval_joint][WARN] train_write_policy={train_policy} != eval_write_policy={eval_write_policy} "
              f"-> train/eval write mismatch (allowed for diagnosis).")
    print(f"[eval_joint] eval_write_policy={eval_write_policy} (effective={eff_eval_policy}, online={online})")

    collator = JointInputCollator(tok, num_peers=num_peers, variant=variant,
                                  peer_order=str(cfg.get("test_order", "orig")),
                                  identity_mode=str(cfg.get("identity_mode", "id")),
                                  max_length=int(cfg.get("max_length", 1536)),
                                  include_context=as_bool(cfg.get("include_context"), True))
    cand_ids = candidate_token_ids(tok, num_peers)
    # WRITE-only identity (decoupled from anonymous READ); None -> follow read.
    _wii = cfg.get("write_include_identity", None)
    write_inc_id = (collator.include_identity if _wii is None else as_bool(_wii, False))

    records = JsonlDataset(cfg["offline_data"]).records
    # Counterfactual evaluation is OPTIONAL (independent of train). Natural ON by default.
    n_before = len(records)
    fkw = counterfactual_filter_kwargs(cfg, "eval")
    records = filter_records(records, **fkw)
    print(f"[eval_joint] records: {n_before} -> {len(records)} after filter {fkw}")
    stream_seed = cfg.get("stream_seed")
    if stream_seed is not None:
        import random as _random
        _random.Random(int(stream_seed)).shuffle(records)
        print(f"[eval_joint] stream shuffled with seed {stream_seed}")
    if cfg.get("max_samples"):
        records = records[: int(cfg["max_samples"])]
    reliab_global = peer_reliability(records)
    reliab_domain = peer_reliability(records, key=_domain)
    reliab_dataset = peer_reliability(records, key=lambda r: str(r.get("dataset", "")))

    results, debug = [], []
    correct = oracle = bayes_correct = 0
    # Running per-peer Bayesian tracker for time-varying reliability streams: Beta(1,1)
    # prior updated with EVERY peer's observed correctness after each example. Logged
    # per record (posterior at selection time, i.e. before seeing this example's labels):
    #  - bayes_posterior:       cumulative Beta posterior mean per peer id
    #  - bayes_posterior_decay: exponentially-decayed estimate (gamma=0.9), tracks drift
    bayes_a = [1.0] * num_peers; bayes_b = [1.0] * num_peers
    decay_s = [0.0] * num_peers; decay_n = [0.0] * num_peers
    BAYES_GAMMA = 0.9
    slot_picks = [0] * num_peers
    peer_picks = defaultdict(int)
    by_domain = defaultdict(lambda: [0, 0])           # [correct, total]
    by_cftype = defaultdict(lambda: [0, 0])
    by_dataset = defaultdict(lambda: [0, 0])
    by_peer = defaultdict(lambda: [0, 0])             # selected peer -> [correct, total]
    override_num = override_den = retain_num = retain_den = 0
    rag_override_num = rag_override_den = 0

    reset_state_every = as_bool(cfg.get("reset_state_every_example"), False)
    for idx, record in enumerate(tqdm(records, desc=f"eval_{variant}")):
        # Ablation: wipe per-peer state buckets before each example (no online memory).
        if reset_state_every:
            state_buckets.clear()
            state_buckets_i.clear()
        v = collator.record_views(record)
        perm, real = v["perm"], v["real"]
        # routing label for state buckets: oracle task_type, or the model's own guess
        rec_task_true = task_type_of(record)
        if bucket_by_task and bucket_pred:
            rec_task = classify_task(record)
            task_pred_total += 1
            task_pred_correct += int(rec_task == rec_task_true)
            _task_conf[(rec_task_true, rec_task)] += 1
        else:
            rec_task = rec_task_true
        prompt, char_spans = peer_response_char_spans(
            v["question"], v["slot_names"], v["slot_texts"],
            context=v["context"] or None, include_identity=collator.include_identity, real=real)
        enc = tok(prompt, add_special_tokens=True,
                  return_offsets_mapping=(variant == VARIANT_BCE),
                  truncation=True, max_length=collator.max_length)
        ids = torch.tensor([enc["input_ids"]], device=device)
        mask = torch.ones_like(ids)
        peer_spans = None
        if variant == VARIANT_BCE:
            spans = char_to_token_spans(list(enc.get("offset_mapping", [])), char_spans)
            peer_spans = torch.zeros(1, num_peers, 2, dtype=torch.long, device=device)
            for s in range(min(num_peers, len(spans))):
                peer_spans[0, s, 0], peer_spans[0, s, 1] = spans[s]
        # 1) CANDIDATE SCORING — always read-only (write_enabled=False inside
        #    score_candidates). Verify S is unchanged when debugging.
        norm_before = model.state_norm() if debug_write else 0.0
        with torch.no_grad():
            if variant == VARIANT_BCE:
                out = model(input_ids=ids, attention_mask=mask, peer_spans=peer_spans)
                logps = out.logits[0]
                if real < num_peers:
                    logps = logps.clone(); logps[real:] = float("-inf")
            elif per_peer:
                # per-peer AR: score each candidate " Peer s" with peer-s's OWN state
                rec_task_pp = task_type_of(record)
                model.set_write_enabled(False)
                lp = []
                for s in range(real):
                    j = canon_peer(v["slot_peer_keys"][s], s)
                    if use_dual:
                        model.set_state_refs(model.blend_states(
                            get_peer_state(rec_task_pp, j), get_peer_state_i(rec_task_pp, j)))
                    else:
                        model.set_state_refs(get_peer_state(rec_task_pp, j))
                    _lp = model.score_one_candidate(ids, mask, cand_ids[s])[0]
                    if getattr(model, "use_trust_head", False):
                        _lp = _lp + model.trust_bias()[0]
                    lp.append(_lp)
                logps = torch.stack(lp)
                if real < num_peers:
                    pad = torch.full((num_peers - real,), float("-inf"), device=logps.device)
                    logps = torch.cat([logps, pad])
            else:
                logps = model.score_candidates(ids, mask, cand_ids)[0]  # [P]
                if real < num_peers:
                    logps = logps.clone(); logps[real:] = float("-inf")
            selected_slot = int(torch.argmax(logps).item())
            probs = torch.softmax(logps.float(), dim=-1)
            entropy = float(-(probs * (probs.clamp_min(1e-9)).log()).sum())
            cand_scores = [float(x) for x in logps.tolist()]
        if debug_write:
            assert_scoring_readonly(model, norm_before, model.state_norm())

        selected_peer = perm[selected_slot]
        cbp_list = v["correctness_by_peer"]
        is_correct = bool(cbp_list[selected_peer]) if selected_peer < len(cbp_list) else False
        target_peers_keys = record.get("target_peers")
        # respect explicit target_peers (counterfactual diagnostics) if present
        if target_peers_keys is not None:
            keys_sorted = sorted(dict(record.get("peer_responses", {})))
            tgt_ids = [keys_sorted.index(k) for k in target_peers_keys if k in keys_sorted]
            is_correct = selected_peer in set(tgt_ids)
        else:
            tgt_ids = [i for i, c in enumerate(cbp_list) if c]

        # 2) Separate WRITE pass per eval_write_policy (AFTER correctness is known, so
        #    "feedback" can write explicit per-peer correct/incorrect labels). Carries
        #    the updated S forward to the next example.
        if eff_eval_policy != "none":
            # --- selective-feedback gate: decide whether to WRITE this example ---
            do_write = True
            if feedback_gate == "margin":
                # uncertain = small top1-top2 gap among the real candidates
                finite = sorted((s for s in cand_scores[:real]), reverse=True)
                mgn = (finite[0] - finite[1]) if len(finite) > 1 else 9.9
                do_write = mgn < gate_margin_thr
            elif feedback_gate in ("trust_self", "trust_gt"):
                # write only when history does NOT yet clearly favour the picked peer
                sel_t = gate_trust.get(selected_peer, 0.5)
                others = [gate_trust.get(perm[s], 0.5) for s in range(real) if perm[s] != selected_peer]
                mean_other = sum(others) / len(others) if others else 0.5
                do_write = (sel_t - mean_other) < gate_trust_thr
            gate_stats[1] += 1
            gate_stats[0] += int(do_write)
            # update running trust AFTER the gate decision (ATS-style EMA).
            #  trust_self: feed the CM's own per-peer score (logp -> softmax), NO ground truth.
            #  trust_gt:   feed ground-truth correctness with a miss penalty (cheating ref).
            if feedback_gate == "trust_self":
                import math as _m
                fin = [cand_scores[s] for s in range(real)]
                mx = max(fin); ex = [_m.exp(x - mx) for x in fin]; Z = sum(ex)
                for s in range(real):
                    pid = perm[s]; q = ex[s] / Z
                    gate_trust[pid] = gate_trust_rho * gate_trust.get(pid, 0.5) + (1 - gate_trust_rho) * q
            elif feedback_gate == "trust_gt":
                for s in range(real):
                    pid = perm[s]
                    y = 1.0 if cbp_list[pid] else -1.0   # miss penalty: wrong pulls trust DOWN
                    tgt_y = 1.0 if y > 0 else 0.0
                    gate_trust[pid] = gate_trust_rho * gate_trust.get(pid, 0.5) + (1 - gate_trust_rho) * tgt_y
            eval_short_answers = build_short_answers(
                record, v["slot_texts"], v.get("slot_peer_keys"), real=real) if do_write else None
            if do_write and per_peer:
                # one single-peer write per peer, into that peer's OWN state
                pass  # rec_task computed once at loop head (oracle or model-predicted)
                for s in range(real):
                    j = canon_peer(v["slot_peer_keys"][s], s)
                    # M_R: response-only write (write_inc_id, anonymous by default).
                    model.set_state_refs(get_peer_state(rec_task, j))
                    winfo = run_write_policy(
                        model, tok, policy=eval_write_policy, use_feedback=use_feedback,
                        question=v["question"], context=v["context"],
                        slot_names=[v["slot_names"][s]], slot_texts=[v["slot_texts"][s]],
                        correctness_by_slot=[v["correctness_by_slot"][s]], real=1,
                        include_identity=write_inc_id, selected_slot=None,
                        selected_correct=None, include_selected_peer=False,
                        write_prompt_style=eval_write_prompt_style,
                        short_answers=[eval_short_answers[s]],
                        max_length=collator.max_length, device=device, debug=debug_write,
                    )
                    put_peer_state(rec_task, j, model.get_state_refs())
                    # M_I: response+identity write (real name in prompt).
                    if use_dual:
                        model.set_state_refs(get_peer_state_i(rec_task, j))
                        run_write_policy(
                            model, tok, policy=eval_write_policy, use_feedback=use_feedback,
                            question=v["question"], context=v["context"],
                            slot_names=[v["slot_names"][s]], slot_texts=[v["slot_texts"][s]],
                            correctness_by_slot=[v["correctness_by_slot"][s]], real=1,
                            include_identity=True, selected_slot=None,
                            selected_correct=None, include_selected_peer=False,
                            write_prompt_style=eval_write_prompt_style,
                            short_answers=[eval_short_answers[s]],
                            max_length=collator.max_length, device=device, debug=debug_write,
                        )
                        put_peer_state_i(rec_task, j, model.get_state_refs())
            elif do_write:
                winfo = run_write_policy(
                    model, tok, policy=eval_write_policy, use_feedback=use_feedback,
                    question=v["question"], context=v["context"],
                    slot_names=v["slot_names"], slot_texts=v["slot_texts"],
                    correctness_by_slot=v["correctness_by_slot"], real=real,
                    include_identity=write_inc_id, selected_slot=selected_slot,
                    selected_correct=is_correct, include_selected_peer=include_selected,
                    write_prompt_style=eval_write_prompt_style,
                    short_answers=eval_short_answers,
                    max_length=collator.max_length, device=device, debug=debug_write,
                )
            if debug_write and idx < 3:
                print(f"[eval_write] {winfo}")

        dom = _domain(record); ds = str(record.get("dataset", "")); cft = record.get("counterfactual_type")
        correct += int(is_correct)
        oracle += int(any(cbp_list[:real]))
        # Bayesian global pick
        peers_present = list(range(real))
        bg = best_peer(reliab_global, None, [f"peer_{p}" for p in peers_present])
        bg_id = int(bg.split("_")[1]) if bg else 0
        bayes_correct += int(bg_id < len(cbp_list) and cbp_list[bg_id])
        slot_picks[selected_slot] += 1
        peer_picks[short_peer_name(v["slot_names"][selected_slot])] += 1
        by_domain[dom][0] += int(is_correct); by_domain[dom][1] += 1
        if cft:
            by_cftype[cft][0] += int(is_correct); by_cftype[cft][1] += 1
        by_dataset[ds][0] += int(is_correct); by_dataset[ds][1] += 1
        sel_short = short_peer_name(v["slot_names"][selected_slot])
        by_peer[sel_short][0] += int(is_correct); by_peer[sel_short][1] += 1
        if cft == "strong_wrong_weak_correct":
            override_den += 1; override_num += int(selected_peer in set(tgt_ids))
            if dom == "rag":
                rag_override_den += 1; rag_override_num += int(selected_peer in set(tgt_ids))
        if cft == "strong_correct_weak_wrong":
            strong = record.get("strong_peer"); keys_sorted = sorted(dict(record.get("peer_responses", {})))
            sid = keys_sorted.index(strong) if strong in keys_sorted else -1
            retain_den += 1; retain_num += int(selected_peer == sid)

        results.append({
            "example_id": record.get("id"), "domain": dom, "dataset": ds,
            "counterfactual_type": cft, "synthetic_counterfactual": bool(record.get("synthetic_counterfactual")),
            "evidence_quality": record.get("evidence_quality"),
            "slot_to_peer_id": perm, "selected_slot": selected_slot, "selected_peer_id": selected_peer,
            "selected_peer_name": v["slot_names"][selected_peer], "selected_correct": is_correct,
            "target_peer_ids": tgt_ids, "candidate_scores": cand_scores, "entropy": entropy,
            # memory-conditioned raw logits per SLOT (candidate_scores are their sigmoid/softmax)
            "candidate_logits": [float(x) for x in logps.tolist()],
            # running Bayes posterior per PEER ID at selection time (before this example's labels)
            "bayes_posterior": [round(bayes_a[j] / (bayes_a[j] + bayes_b[j]), 4) for j in range(num_peers)],
            "bayes_posterior_decay": [round(decay_s[j] / decay_n[j], 4) if decay_n[j] > 0 else 0.5
                                      for j in range(num_peers)],
            # chaotic-reliability-stream metadata (pass-through when present)
            **{k: record[k] for k in ("stream_seed", "label_pattern", "regime_id", "window_id",
                                      "phase", "window_target_marginals", "window_realized_marginals",
                                      "window_target_template") if k in record},
        })
        # Bayes tracker update AFTER logging: all peers' correctness is observable feedback
        for j in range(min(real, len(cbp_list))):
            yj = float(cbp_list[j])
            bayes_a[j] += yj; bayes_b[j] += 1.0 - yj
            decay_s[j] = BAYES_GAMMA * decay_s[j] + yj
            decay_n[j] = BAYES_GAMMA * decay_n[j] + 1.0
        if idx < 10:
            debug.append({"prompt": prompt, "ar_target": ar_target_text(
                [invert_perm(perm)[t] for t in tgt_ids][:1]) if tgt_ids else None,
                "candidates": [f"Peer {s}" for s in range(num_peers)], "candidate_scores": cand_scores,
                "state_mode": f"{'trS' if init_state=='trained' else 'S0'}+{'ON' if online else 'RO'}",
                "steering": "delta_mem_attention" if use_shared else "none"})

    n = max(1, len(results))
    metrics = {
        "variant": variant, "use_shared_state": use_shared,
        "state_mode": f"{'trS' if init_state=='trained' else 'S0'}+{'ON' if online else 'RO'}",
        "task_routing": (
            {"mode": "pred",
             "classify_accuracy": task_pred_correct / max(1, task_pred_total),
             "confusion": {f"{t}->{p}": c for (t, p), c in sorted(_task_conf.items())}}
            if (bucket_by_task and bucket_pred)
            else {"mode": "oracle"} if bucket_by_task else None
        ),
        "num_samples": len(results),
        "feedback_gate": feedback_gate,
        "gate_write_rate": (gate_stats[0] / gate_stats[1]) if gate_stats[1] else None,
        "selected_correct_rate": correct / n, "accuracy": correct / n,
        "oracle_ceiling": oracle / n, "gap_to_ceiling": (oracle - correct) / n,
        "bayesian_global_accuracy": bayes_correct / n,
        "improvement_over_bayesian": (correct - bayes_correct) / n,
        "response_override_rate": (override_num / override_den) if override_den else None,
        "trust_retention_rate": (retain_num / retain_den) if retain_den else None,
        "rag_evidence_override_rate": (rag_override_num / rag_override_den) if rag_override_den else None,
        "selected_correct_response_rate": correct / n,
        "accuracy_by_domain": {d: c / t for d, (c, t) in by_domain.items() if t},
        "accuracy_by_counterfactual_type": {k: c / t for k, (c, t) in by_cftype.items() if t},
        "slot_picks": slot_picks, "peer_picks": dict(peer_picks),
    }
    out_dir = Path(cfg.get("output", "outputs/joint_eval")).resolve()
    out_dir = out_dir if out_dir.suffix == "" else out_dir.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / "predictions.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    (out_dir / "joint_debug_preview.jsonl").write_text("\n".join(json.dumps(d) for d in debug))
    _write_csv(out_dir / "eval_by_domain.csv", by_domain)
    _write_csv(out_dir / "eval_by_counterfactual_type.csv", by_cftype)
    _write_csv(out_dir / "eval_by_dataset.csv", by_dataset)
    _write_csv(out_dir / "eval_by_peer.csv", by_peer)
    print(json.dumps(metrics, indent=2))
    print(f"[eval_joint] wrote metrics + predictions + CSVs -> {out_dir}")


def _write_csv(path: Path, table: dict) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "accuracy", "correct", "total"])
        for k, (c, t) in sorted(table.items()):
            w.writerow([k, (c / t if t else 0.0), c, t])


if __name__ == "__main__":
    main()
