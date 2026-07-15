"""Eval the event-axis symmetric trust memory on the CF unified streams.

Mirrors eval_feedback_algo.py but with no content write. The trust state is updated
once per task on the event axis, and its peer-specific readout is projected into the
frozen center model's upper decoder blocks as a residual steering vector.

Per example (stream order, batch=1):
  [reset F to baseline if reset_state=example]
  for each candidate j: steer_vec = M_pj @ phi*  -> logp_CM(j | steered)
  pick argmax -> record -> event-axis memory.update(j, +/-1, task_type) for each peer
  -> memory.update_joint(outcome_vec)

Sanity arm --ablate_memory: no steering -> must reproduce NO-WRITE.

  PYTHONPATH=. python eval_symmetric_memory.py --config configs/symmetric_memory.yaml \
    --checkpoint outputs/sym_steer_06 --central_model Qwen/Qwen3-0.6B \
    --offline_data data/v3_unified/p90.jsonl --output outputs/eval_sym/steer_p90
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from feedback_state.checkpoint_manifest import (
    MANIFEST_FILENAME,
    validate_checkpoint_manifest,
)
from feedback_state.newarch_loader import apply_torch_fp8_shim, load_central_model

apply_torch_fp8_shim()

from peft import PeftModel
from transformers import AutoTokenizer

from feedback_state.data import JsonlDataset
from feedback_state.generation import dtype_from_name
from feedback_state.joint_data import (
    VARIANT_AR,
    batch_candidate_judge_inputs,
    candidate_token_ids,
    yes_no_token_ids,
)
from feedback_state.joint_models import JointDeltaMemSelector
from feedback_state.joint_prompt import PEER_SEP, build_joint_prompt
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
    RELIABILITY_ZSCORE_EPS,
    SymmetricTrustMemory,
    cm_context_vector,
)
from feedback_state.tasks import task_type_of
from feedback_state.utils import load_config, merge_args_with_config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(v, default=False):
    if v is None:
        return default
    return v if isinstance(v, bool) else str(v).lower() in {"1", "true", "yes", "on"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
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
    p.add_argument("--center_lora_checkpoint", type=Path, default=None,
                   help="Optional LoRA selector checkpoint to attach to the central model.")
    p.add_argument("--offline_data", type=Path, default=None)
    p.add_argument("--init_state", choices=["warm", "cold"], default="cold")
    p.add_argument("--reset_state", choices=["stream", "example"], default="stream")
    p.add_argument("--task_state_mode", choices=["shared", "separate"], default=None,
                   help="shared: one online Sigma state for the whole stream. "
                        "separate: maintain one online state per broad task type.")
    p.add_argument("--ablate_memory", action="store_true")
    p.add_argument("--use_joint", choices=["on", "off"], default="off")
    p.add_argument("--graph_posterior", choices=["off", "ising"], default=None,
                   help="Optional graph posterior readout. ising: exact pairwise posterior "
                        "over peer correctness using Sigma utilities and pairwise G.")
    p.add_argument("--graph_unary_weight", type=float, default=None)
    p.add_argument("--graph_g_weight", type=float, default=None)
    p.add_argument("--graph_decay_g", type=float, default=None)
    p.add_argument("--graph_eta_g", type=float, default=None)
    p.add_argument("--graph_centered_g", choices=["on", "off"], default=None)
    p.add_argument("--phi_mode", choices=["proto"], default=None)
    p.add_argument("--write_mode", choices=["addr", "carry"], default=None)
    p.add_argument("--decay_mode", choices=["scalar", "diag"], default=None)
    p.add_argument("--per_peer_decay", choices=["on", "off"], default=None)
    p.add_argument("--peer_mode", choices=["joint", "one"], default=None,
                   help="joint: all peers in one prompt, CM compares (default). "
                        "one: each peer scored alone in its own prompt (no cross-comparison).")
    p.add_argument("--score_mode", choices=["peer_name", "candidate_yesno"], default=None,
                   help="peer_name: score 'Peer j' continuations (legacy). "
                        "candidate_yesno: score each highlighted candidate with shared Yes/No utility.")
    p.add_argument("--max_examples", type=int, default=None,
                   help="Optional cap for quick diagnostic evals; full eval by default.")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def _example_view(rec, num_peers):
    view = canonical_peer_view(rec, num_peers, setting="A")
    keys, names, texts, real = view["keys"], view["names"], view["texts"], view["real"]
    cbp = rec.get("correctness_by_peer") or rec.get("peer_correct") or {}
    corr = [int(round(float(cbp.get(k, 0)))) for k in keys]
    # peer index from canonical key "peer_N" -> identity address for the per-peer matrix
    peer_ids = [int(str(k).split("_")[1]) if str(k).startswith("peer_") else i for i, k in enumerate(keys)]
    return [f"peer_{i}" for i in range(len(keys))], texts, corr, real, peer_ids


def _ising_states(n: int, *, device, dtype=torch.float32) -> torch.Tensor:
    vals = torch.arange(2 ** int(n), device=device)
    bits = ((vals[:, None] >> torch.arange(int(n), device=device)) & 1).to(dtype)
    return bits.mul(2.0).sub(1.0)


def _zscore_1d(
    vals: torch.Tensor,
    *,
    eps: float = RELIABILITY_ZSCORE_EPS,
) -> torch.Tensor:
    """Use the shared finite near-tie peer standardization."""
    values = torch.as_tensor(vals)
    centered = values - values.mean()
    return centered / (centered.square().mean() + float(eps) ** 2).sqrt()


@torch.no_grad()
def _ising_marginal_scores(
    unary,
    G_mat,
    *,
    unary_weight: float,
    g_weight: float,
):
    """Exact posterior marginals from Sigma utilities and pairwise topology G."""
    device = G_mat.device
    u = _zscore_1d(torch.as_tensor(unary, dtype=torch.float32, device=device))
    G = torch.as_tensor(G_mat, dtype=torch.float32, device=device)
    states = _ising_states(u.numel(), device=device, dtype=torch.float32)
    logits = states @ (float(unary_weight) * u)
    if float(g_weight) != 0.0:
        pair = torch.einsum("bi,ij,bj->b", states, G, states)
        logits = logits + 0.5 * float(g_weight) * pair
    probs = torch.softmax(logits, dim=0)
    return probs @ states


def _load_shape_compatible(module, state_dict, label):
    """Load checkpoint entries whose names and shapes match the current module.

    This keeps train-3/test-5 evaluation usable: peer-shaped runtime buffers such as
    M/G are dropped separately, and any remaining peer-indexed parameter from an old
    per_peer_decay checkpoint is skipped instead of aborting the run.
    """
    current = module.state_dict()
    loadable = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in current:
            skipped.append(f"{key}: missing")
        elif tuple(current[key].shape) != tuple(value.shape):
            skipped.append(f"{key}: ckpt{tuple(value.shape)} != current{tuple(current[key].shape)}")
        else:
            loadable[key] = value
    missing, unexpected = module.load_state_dict(loadable, strict=False)
    if skipped:
        preview = "; ".join(skipped[:6])
        suffix = "" if len(skipped) <= 6 else f"; ... +{len(skipped) - 6} more"
        print(f"[eval_sym] skipped shape-incompatible {label} entries: {preview}{suffix}", flush=True)
    return missing, unexpected


def main():
    args = parse_args()
    cfg = merge_args_with_config(args, load_config(args.config))
    out = Path(cfg.get("output", "outputs/eval_sym"))
    out.mkdir(parents=True, exist_ok=True)
    # Metrics is the completion marker. Remove it before doing any expensive work so
    # a failed rerun can never look complete to the experiment runner.
    (out / "eval_metrics.json").unlink(missing_ok=True)
    device = torch.device(str(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    dtype_name = str(cfg.get("dtype", "bfloat16")).lower()
    dtype = dtype_from_name(dtype_name)
    ckpt = Path(cfg["checkpoint"]) if cfg.get("checkpoint") else None
    model_name = str(cfg.get("central_model", "Qwen/Qwen3-0.6B"))
    lora_ckpt = Path(cfg["center_lora_checkpoint"]) if cfg.get("center_lora_checkpoint") else None
    num_peers = int(cfg.get("num_peers", 3))
    max_len = int(cfg.get("max_length", 8192))
    legacy_prompt_protocol = as_bool(cfg.get("legacy_prompt_protocol"), False)
    validate_prompt_protocol(
        legacy_prompt_protocol=legacy_prompt_protocol,
        max_length=max_len,
    )
    init_state = str(cfg.get("init_state", "cold")).lower()
    reset_state = str(cfg.get("reset_state", "stream")).lower()
    task_state_mode = str(cfg.get("task_state_mode", "shared")).lower()
    ablate = bool(cfg.get("ablate_memory", False))
    offline_data_path = Path(cfg["offline_data"])
    offline_data_sha256 = _sha256_file(offline_data_path)
    checkpoint_sha256 = None
    if not ablate:
        if ckpt is None:
            raise ValueError("Sigma evaluation requires --checkpoint for provenance")
        checkpoint_file = ckpt / "sym_memory.pt"
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Sigma checkpoint not found: {checkpoint_file}")
        checkpoint_sha256 = _sha256_file(checkpoint_file)
        if (ckpt / MANIFEST_FILENAME).is_file():
            # Old steered checkpoints remain usable. When an optional manifest is
            # present, still reject a stale/tampered one.
            validate_checkpoint_manifest(ckpt)
    use_joint = str(cfg.get("use_joint", "off")) == "on"
    graph_posterior = str(cfg.get("graph_posterior", "off")).lower()
    graph_unary_weight = float(cfg.get("graph_unary_weight", 0.5))
    graph_g_weight = float(cfg.get("graph_g_weight", 1.0))
    graph_decay_g = float(cfg.get("graph_decay_g", 0.9))
    graph_eta_g = float(cfg.get("graph_eta_g", 0.1))
    graph_centered_g = as_bool(cfg.get("graph_centered_g"), True)
    rank = int(cfg.get("rank", 16))
    task_types = tuple(cfg.get("task_types", DEFAULT_TASK_TYPES))
    phi_mode = str(cfg.get("phi_mode", "proto"))     # only proto (soft task-centroid address)
    proto_tau = float(cfg.get("proto_tau", 0.1))
    phi_layer_frac = float(cfg.get("phi_layer_frac", 0.5))
    decay_mode = str(cfg.get("decay_mode", "scalar"))
    write_mode = str(cfg.get("write_mode", "addr"))  # addr (A) | carry (B)
    per_peer_decay = as_bool(cfg.get("per_peer_decay"), False)  # match train: per-peer decay theta shape
    peer_mode = str(cfg.get("peer_mode", "joint"))   # joint (compare all) | one (score each alone)
    score_mode = str(cfg.get("score_mode", "peer_name")).lower()
    whiten = as_bool(cfg.get("whiten"), write_mode == "carry")
    if graph_posterior not in {"off", "ising"}:
        raise ValueError(f"graph_posterior must be off/ising, got {graph_posterior!r}")
    if graph_posterior != "off":
        print(
            f"[eval_sym] graph_posterior={graph_posterior} base=sigma "
            f"wu={graph_unary_weight} wg={graph_g_weight} "
            f"decay_g={graph_decay_g} centered_g={graph_centered_g}",
            flush=True,
        )
    if task_state_mode not in {"shared", "separate"}:
        raise ValueError(f"task_state_mode must be shared/separate, got {task_state_mode!r}")
    if score_mode not in {"peer_name", "candidate_yesno"}:
        raise ValueError(f"score_mode must be peer_name/candidate_yesno, got {score_mode!r}")
    if score_mode == "candidate_yesno" and peer_mode != "joint":
        print("[eval_sym] score_mode=candidate_yesno uses joint candidate prompts; "
              f"peer_mode={peer_mode!r} is kept only for legacy metrics.", flush=True)

    tok_src = str(lora_ckpt) if lora_ckpt is not None and (lora_ckpt / "tokenizer_config.json").exists() else model_name
    tok = AutoTokenizer.from_pretrained(tok_src, local_files_only=bool(cfg.get("local_files_only", False)))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if PEER_SEP not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [PEER_SEP]})
    device_map = cfg.get("device_map") or None  # "auto" to shard a big model (e.g. 9B) across GPUs
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
            base, str(adapter_dir), is_trainable=False,
            local_files_only=bool(cfg.get("local_files_only", False)),
        )
        print(f"[eval_sym] attached center LoRA from {adapter_dir}", flush=True)
    if device_map is None:
        base = base.to(device=device, dtype=dtype)
    else:
        device = base.get_input_embeddings().weight.device
    # use_shared_state=False: the frozen CM does the scoring; trust enters via the memory, not delta-mem.
    model = JointDeltaMemSelector(base, num_peers=num_peers, model_variant=VARIANT_AR,
                                  use_shared_state=False, delta_cfg=cfg, freeze_backbone=True)
    if device_map is None:
        model = model.to(device)
    model.eval()

    if phi_mode != "proto":
        raise ValueError(f"phi_mode must be 'proto', got {phi_mode!r}")
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
    steerer = ActivationSteerer(base, rank=rank).to(device)
    # Load trained memory params (phi/eta/shared theta + steerer proj/gain) from checkpoint.
    # M/G are runtime state buffers and are rebuilt below for the requested num_peers.
    # Strict peer-count generalization requires peer-count-independent learned params
    # (for example per_peer_decay=off). Shape filtering below prevents old peer-indexed
    # checkpoints from crashing evaluation, but skipped params fall back to init values.
    if ckpt is not None and (ckpt / "sym_memory.pt").exists():
        payload = torch.load(ckpt / "sym_memory.pt", map_location=device)
        mem_sd = {k: v for k, v in payload["mem"].items() if k not in ("M", "G")}
        # phi=proto: install the FIXED centroids/whitening from the checkpoint FIRST, so
        # proto_proj is resized to the trained n_proto before the param load. The eval task
        # may be ABSENT from the trained centroids -> it addresses as a soft mixture (the
        # whole generalization test).
        if "proto_centroids" in mem_sd:
            cw = mem_sd["proto_centroids"]                     # already whitened at train time
            pmean, pstd = mem_sd["proto_mean"], mem_sd["proto_std"]
            # set_prototypes re-whitens, so undo: pass raw = cw*std+mean
            raw_cent = cw.to(pstd.dtype) * (pstd + 1e-6) + pmean
            mem.set_prototypes(raw_cent, pmean, pstd)
        _load_shape_compatible(mem, mem_sd, "memory")
        if steerer is not None and "steerer" in payload:
            _load_shape_compatible(steerer, payload["steerer"], "steerer")
    mem.reset()
    # warm start the matrices (rare; default cold)
    if init_state == "warm" and ckpt is not None and (ckpt / "sym_state.pt").exists():
        snap = torch.load(ckpt / "sym_state.pt", map_location=device)
        mem.restore({k: v.to(device) for k, v in snap.items()})
    base_snap = mem.snapshot()
    task_snaps = {}
    graph_G = torch.zeros(num_peers, num_peers, dtype=torch.float32, device=device)
    graph_base_snap = graph_G.detach().clone()
    graph_task_snaps = {}
    cand_ids = candidate_token_ids(tok, num_peers)
    yes_ids, no_ids = yes_no_token_ids(tok)
    records = JsonlDataset(offline_data_path).records
    if not records:
        raise ValueError("No evaluation records were loaded")
    max_examples = cfg.get("max_examples")
    if max_examples is not None:
        records = records[:int(max_examples)]
    correct = total = 0
    selections = []  # per-example: which peer was picked, was it right, per-peer scores+labels
    with torch.no_grad():
        for rec_idx, rec in enumerate(records, start=1):
            slot_names, texts, corr, real, peer_ids = _example_view(rec, num_peers)
            if real < 1:
                continue
            if reset_state == "example":
                mem.restore(base_snap)
                graph_G.copy_(graph_base_snap)
            elif task_state_mode == "separate":
                task_key = task_type_of(rec)
                mem.restore(task_snaps.get(task_key, base_snap))
                graph_G.copy_(graph_task_snaps.get(task_key, graph_base_snap))
            q = str(rec.get("problem", rec.get("question", "")))
            rag_ctx = candidate_context_text(
                rec,
                include_context=as_bool(cfg.get("include_context"), False),
                legacy_prompt_protocol=legacy_prompt_protocol,
            )
            pid = pmask = None
            # peer_mode="one": precompute a single-peer prompt per candidate (CM sees ONE answer
            # at a time, no cross-comparison). Each is scored on its lone " Peer 0" endorsement.
            one_pids = None
            candidate_pids = candidate_mask = None
            if score_mode == "candidate_yesno":
                candidate_pids, candidate_mask = batch_candidate_judge_inputs(
                    tok, q, slot_names, texts,
                    context=rag_ctx or None,
                    include_identity=False,
                    real=real,
                    max_length=max_len,
                    device=device,
                    legacy_prompt_protocol=legacy_prompt_protocol,
                )
            else:
                prompt = build_joint_prompt(q, slot_names, texts, context=rag_ctx or None,
                                            include_identity=False, real=real)
                enc = tok(prompt, add_special_tokens=True, truncation=True, max_length=max_len)
                pid = torch.tensor([enc["input_ids"]], device=device)
                pmask = torch.ones_like(pid)
            if score_mode == "peer_name" and peer_mode == "one":
                one_pids = []
                for s in range(real):
                    sp = build_joint_prompt(q, [slot_names[s]], [texts[s]], context=rag_ctx or None,
                                            include_identity=False, real=1)
                    se = tok(sp, add_special_tokens=True, truncation=True, max_length=max_len)
                    one_pids.append(torch.tensor([se["input_ids"]], device=device))
            needs_sigma_state = not ablate
            # phi context (READ): raw CM hidden vector of the problem, soft-addressed over centroids.
            phi_ctx = (cm_context_vector(base, tok, q, device=device, layer_frac=phi_layer_frac)
                       if needs_sigma_state else None)
            # carry mode (B): WRITE direction per peer = CM(peer_answer) - CM(gold). Encode gold once.
            value_vecs = None
            if write_mode == "carry" and needs_sigma_state:
                gold_h = cm_context_vector(base, tok, str(rec.get("answer", "")), device=device, layer_frac=phi_layer_frac)
                value_vecs = [cm_context_vector(base, tok, str(texts[s])[:4000], device=device, layer_frac=phi_layer_frac) - gold_h
                              for s in range(real)]

            def score_candidate_batch(steer_vecs=None):
                steerer.steer_vec = steer_vecs
                try:
                    return model.score_candidate_utility(candidate_pids, candidate_mask, yes_ids, no_ids)
                finally:
                    steerer.steer_vec = None

            def score_slot(s, steer_vec=None):
                if score_mode == "candidate_yesno":
                    if steer_vec is None:
                        lp = float(score_candidate_batch()[s])
                    else:
                        steerer.steer_vec = steer_vec
                        try:
                            lp = float(model.score_candidate_utility(
                                candidate_pids[s:s + 1], candidate_mask[s:s + 1], yes_ids, no_ids)[0])
                        finally:
                            steerer.steer_vec = None
                elif peer_mode == "one":
                    if steer_vec is not None:
                        steerer.steer_vec = steer_vec
                    # CM scores this peer ALONE: its prompt has a single answer at slot 0,
                    # so the endorsement token is always " Peer 0" (cand_ids[0]).
                    spid = one_pids[s]
                    lp = float(model.score_one_candidate(spid, torch.ones_like(spid), cand_ids[0])[0])
                else:
                    if steer_vec is not None:
                        steerer.steer_vec = steer_vec
                    lp = float(model.score_one_candidate(pid, pmask, cand_ids[s])[0])
                steerer.steer_vec = None
                return lp

            def score_center_slots():
                steerer.steer_vec = None
                if score_mode == "candidate_yesno":
                    vals = score_candidate_batch()
                    return [float(vals[s]) for s in range(real)]
                if peer_mode == "joint":
                    vals = model.score_candidates(pid, pmask, cand_ids)[0]
                    return [float(vals[s]) for s in range(real)]
                return [score_slot(s) for s in range(real)]

            sigma_scores = None
            if ablate:
                scores = score_center_slots()
            else:
                if score_mode == "candidate_yesno":
                    steer_vecs = torch.stack([
                        mem.steer_vector(peer_ids[s], phi_ctx).detach()
                        for s in range(real)
                    ])
                    vals = score_candidate_batch(steer_vecs)
                    sigma_tensor = vals[:real].float()
                    sigma_scores = [
                        float(sigma_tensor[s]) for s in range(real)
                    ]
                else:
                    sigma_scores = [
                        score_slot(s, mem.steer_vector(peer_ids[s], phi_ctx).detach())
                        for s in range(real)
                    ]
                    sigma_tensor = torch.tensor(
                        sigma_scores, dtype=torch.float32, device=device
                    )
                scores = sigma_scores
            if graph_posterior == "ising" and not ablate:
                idx = torch.tensor(peer_ids[:real], dtype=torch.long, device=device)
                graph_scores = _ising_marginal_scores(
                    sigma_tensor,
                    graph_G.index_select(0, idx).index_select(1, idx),
                    unary_weight=graph_unary_weight,
                    g_weight=graph_g_weight,
                )
                scores = [float(graph_scores[s]) for s in range(real)]
            sel = int(max(range(real), key=lambda s: scores[s]))
            total += 1
            correct += int(corr[sel] == 1)
            score_source = (
                "center"
                if ablate
                else graph_posterior
                if graph_posterior != "off"
                else "sigma"
            )
            # record which actual peer was chosen (peer_ids maps canonical slot -> peer index)
            selections.append({
                "id": str(rec.get("id") or rec.get("uid") or total),
                "task_type": str(rec.get("task_type") or rec.get("source") or ""),
                "selected_peer": int(peer_ids[sel]),          # 0/1/2 = the actual peer picked
                "selected_correct": int(corr[sel] == 1),
                # peer_scores remains the compact display copy. Audit fields below
                # preserve the exact float values used for selection/replay.
                "peer_scores": {int(peer_ids[s]): round(scores[s], 4) for s in range(real)},
                "peer_correct": {int(peer_ids[s]): int(corr[s]) for s in range(real)},
                "memory_used": not ablate,
                "score_source": score_source,
            })
            if graph_posterior != "off" and not ablate:
                selections[-1]["sigma_peer_scores"] = {
                    int(peer_ids[s]): float(sigma_scores[s]) for s in range(real)
                }
                selections[-1]["graph_scores"] = {
                    int(peer_ids[s]): float(scores[s]) for s in range(real)
                }
            # event-axis update AFTER the pick (one update per peer per task).
            # sign s: ground-truth correctness. (The trainable-judge s was dropped — correctness
            # is not linearly readable from the frozen CM feature; see git history / train script.)
            if not ablate:
                signs = [1.0 if corr[s] else -1.0 for s in range(real)]
                for s in range(real):
                    vv = value_vecs[s] if value_vecs is not None else None
                    mem.update(
                        peer_ids[s],
                        signs[s],
                        phi_ctx,
                        value_vec=vv,
                    )
                joint_outcomes = signs + [0.0] * (num_peers - real)
                mem.update_joint(joint_outcomes)
                if graph_posterior != "off":
                    o = torch.zeros(
                        num_peers, dtype=torch.float32, device=device
                    )
                    for s in range(real):
                        o[int(peer_ids[s])] = signs[s]
                    z = o.clone()
                    if graph_centered_g and real > 0:
                        idx = torch.tensor(
                            peer_ids[:real], dtype=torch.long, device=device
                        )
                        z_real = z.index_select(0, idx)
                        z.index_copy_(0, idx, z_real - z_real.mean())
                    graph_write = graph_eta_g * torch.outer(z, z)
                    graph_G.mul_(graph_decay_g).add_(graph_write)
                    graph_G.fill_diagonal_(0.0)
                if reset_state == "stream" and task_state_mode == "separate":
                    task_key = task_type_of(rec)
                    task_snaps[task_key] = mem.snapshot()
                    graph_task_snaps[task_key] = graph_G.detach().clone()
            if rec_idx == 1 or rec_idx % 100 == 0 or rec_idx == len(records):
                running_acc = correct / total if total else 0.0
                print(
                    f"[eval_sym/progress] {rec_idx}/{len(records)} "
                    f"acc={running_acc*100:.2f} ablate={ablate} "
                    f"score_mode={score_mode} peer_mode={peer_mode}",
                    flush=True,
                )

    acc = correct / total if total else 0.0
    # per-peer selection aggregate: how often each peer was chosen, and its hit-rate when chosen
    peer_chosen = {}
    for s in selections:
        p = s["selected_peer"]
        a = peer_chosen.setdefault(p, {"chosen": 0, "chosen_correct": 0})
        a["chosen"] += 1
        a["chosen_correct"] += s["selected_correct"]
    peer_selection = {str(p): {"chosen": v["chosen"],
                                "share": round(v["chosen"] / total, 4) if total else 0.0,
                                "hit_rate_when_chosen": round(v["chosen_correct"] / v["chosen"], 4) if v["chosen"] else 0.0}
                      for p, v in sorted(peer_chosen.items())}
    metrics = {
        "accuracy": acc,
        "num_samples": total,
        "coupling": "steer",
        "phi_mode": phi_mode,
        "write_mode": write_mode, "decay_mode": decay_mode,
        "num_peers": num_peers,
        "per_peer_decay": per_peer_decay,
        "peer_mode": peer_mode,
        "score_mode": score_mode,
        "max_length": max_len,
        "legacy_prompt_protocol": "on" if legacy_prompt_protocol else "off",
        "prompt_protocol": prompt_protocol_name(legacy_prompt_protocol),
        "prompt_context_format": prompt_context_format(legacy_prompt_protocol),
        "candidate_tokenization_format": candidate_tokenization_format(
            legacy_prompt_protocol
        ),
        "offline_data_sha256": offline_data_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "selection_view": "full",
        "init_state": init_state, "reset_state": reset_state, "ablate_memory": ablate, "use_joint": use_joint,
        "graph_posterior": graph_posterior,
        "task_state_mode": task_state_mode,
        "gamma": mem.gamma_scalar(),
        "eta": float(mem.eta),
        "peer_selection": peer_selection,
    }
    if graph_posterior != "off":
        metrics.update({
            "graph_base": "sigma",
            "graph_unary_weight": graph_unary_weight,
            "graph_g_weight": graph_g_weight,
            "graph_decay_g": graph_decay_g,
            "graph_eta_g": graph_eta_g,
            "graph_centered_g": graph_centered_g,
        })
    # Write detailed rows first and publish metrics last as the completion marker.
    selections_tmp = out / "selections.jsonl.tmp"
    with selections_tmp.open("w") as f:
        for s in selections:
            f.write(json.dumps(s) + "\n")
    selections_tmp.replace(out / "selections.jsonl")
    metrics_tmp = out / "eval_metrics.json.tmp"
    metrics_tmp.write_text(json.dumps(metrics, indent=1))
    metrics_tmp.replace(out / "eval_metrics.json")
    print(f"[eval_sym/steer/{phi_mode}] {cfg['offline_data']}: accuracy={acc*100:.2f} over {total} "
          f"(ablate={ablate}, reset={reset_state}, peer_mode={peer_mode}, "
          f"score_mode={score_mode}, use_joint={use_joint})")
    steerer.remove()


if __name__ == "__main__":
    main()
