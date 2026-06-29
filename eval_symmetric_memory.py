"""Eval the event-axis symmetric trust memory on the CF unified streams.

Mirrors eval_feedback_algo.py but with NO content write — the trust state is updated
once per task (event axis), and the read couples into selection as a residual STEERING
vector injected into the frozen CM's upper-half layers (ActivationSteerer). The frozen
CM scores every " Peer j"; the trust direction re-ranks.

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
import json
from pathlib import Path

import torch

from feedback_state.newarch_loader import apply_torch_fp8_shim, load_central_model

apply_torch_fp8_shim()

from transformers import AutoTokenizer

from feedback_state.data import JsonlDataset
from feedback_state.generation import dtype_from_name
from feedback_state.joint_data import VARIANT_AR, candidate_token_ids
from feedback_state.joint_models import JointDeltaMemSelector
from feedback_state.joint_prompt import PEER_SEP, build_joint_prompt
from feedback_state.permutations import canonical_peer_view
from feedback_state.symmetric_memory import ActivationSteerer, SymmetricTrustMemory, DEFAULT_TASK_TYPES, cm_context_vector
from feedback_state.utils import load_config, merge_args_with_config


def as_bool(v, default=False):
    if v is None:
        return default
    return v if isinstance(v, bool) else str(v).lower() in {"1", "true", "yes", "on"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--central_model", default=None)
    p.add_argument("--offline_data", type=Path, default=None)
    p.add_argument("--init_state", choices=["warm", "cold"], default="cold")
    p.add_argument("--reset_state", choices=["stream", "example"], default="stream")
    p.add_argument("--ablate_memory", action="store_true")
    p.add_argument("--use_joint", choices=["on", "off"], default="off")
    p.add_argument("--phi_mode", choices=["proto"], default=None)
    p.add_argument("--write_mode", choices=["addr", "carry"], default=None)
    p.add_argument("--decay_mode", choices=["scalar", "diag"], default=None)
    p.add_argument("--per_peer_decay", choices=["on", "off"], default=None)
    p.add_argument("--peer_mode", choices=["joint", "one"], default=None,
                   help="joint: all peers in one prompt, CM compares (default). "
                        "one: each peer scored alone in its own prompt (no cross-comparison).")
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


def main():
    args = parse_args()
    cfg = merge_args_with_config(args, load_config(args.config))
    device = torch.device(str(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    dtype = dtype_from_name(str(cfg.get("dtype", "bfloat16")))
    ckpt = Path(cfg["checkpoint"]) if cfg.get("checkpoint") else None
    model_name = str(cfg.get("central_model", "Qwen/Qwen3-0.6B"))
    num_peers = int(cfg.get("num_peers", 3))
    max_len = int(cfg.get("max_length", 8192))
    init_state = str(cfg.get("init_state", "cold")).lower()
    reset_state = str(cfg.get("reset_state", "stream")).lower()
    ablate = bool(cfg.get("ablate_memory", False))
    use_joint = str(cfg.get("use_joint", "off")) == "on"
    rank = int(cfg.get("rank", 16))
    task_types = tuple(cfg.get("task_types", DEFAULT_TASK_TYPES))
    phi_mode = str(cfg.get("phi_mode", "proto"))     # only proto (soft task-centroid address)
    phi_layer_frac = float(cfg.get("phi_layer_frac", 0.5))
    decay_mode = str(cfg.get("decay_mode", "scalar"))
    write_mode = str(cfg.get("write_mode", "addr"))  # addr (A) | carry (B)
    per_peer_decay = as_bool(cfg.get("per_peer_decay"), False)  # match train: per-peer decay theta shape
    peer_mode = str(cfg.get("peer_mode", "joint"))   # joint (compare all) | one (score each alone)
    whiten = as_bool(cfg.get("whiten"), write_mode == "carry")

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=bool(cfg.get("local_files_only", False)))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if PEER_SEP not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [PEER_SEP]})
    base = load_central_model(model_name, dtype=dtype, local_files_only=bool(cfg.get("local_files_only", False)))
    base.resize_token_embeddings(len(tok))
    base = base.to(device=device, dtype=dtype)
    # use_shared_state=False: the frozen CM does the scoring; trust enters via the memory, not delta-mem.
    model = JointDeltaMemSelector(base, num_peers=num_peers, model_variant=VARIANT_AR,
                                  use_shared_state=False, delta_cfg=cfg, freeze_backbone=True).to(device)
    model.eval()

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
    steerer = ActivationSteerer(base, rank=rank).to(device)
    # load trained memory params (phi/eta/theta + steerer proj/gain) from checkpoint.
    # Drop the state buffers M/G: they are the only num_peers-dependent tensors, and they
    # are cold-reset below anyway. Dropping them lets a checkpoint trained with a DIFFERENT
    # number of peers load cleanly -> train with 3 agents, evaluate with 5 (the trainable
    # params eta/gamma/phi_proj/steerer are all agent-count-independent by construction).
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
        mem.load_state_dict(mem_sd, strict=False)
        if steerer is not None and "steerer" in payload:
            steerer.load_state_dict(payload["steerer"], strict=False)
    mem.reset()
    # warm start the matrices (rare; default cold)
    if init_state == "warm" and ckpt is not None and (ckpt / "sym_state.pt").exists():
        snap = torch.load(ckpt / "sym_state.pt", map_location=device)
        mem.restore({k: v.to(device) for k, v in snap.items()})
    base_snap = mem.snapshot()

    cand_ids = candidate_token_ids(tok, num_peers)
    records = JsonlDataset(cfg["offline_data"]).records
    correct = total = 0
    selections = []  # per-example: which peer was picked, was it right, per-peer scores+labels
    with torch.no_grad():
        for rec in records:
            slot_names, texts, corr, real, peer_ids = _example_view(rec, num_peers)
            if real < 1:
                continue
            if reset_state == "example":
                mem.restore(base_snap)
            q = str(rec.get("problem", rec.get("question", "")))
            rag_ctx = str(rec.get("retrieved_context", rec.get("context", ""))) if cfg.get("include_context") else ""
            prompt = build_joint_prompt(q, slot_names, texts, context=rag_ctx or None, include_identity=False, real=real)
            enc = tok(prompt, add_special_tokens=True, truncation=True, max_length=max_len)
            pid = torch.tensor([enc["input_ids"]], device=device)
            pmask = torch.ones_like(pid)
            # peer_mode="one": precompute a single-peer prompt per candidate (CM sees ONE answer
            # at a time, no cross-comparison). Each is scored on its lone " Peer 0" endorsement.
            one_pids = None
            if peer_mode == "one":
                one_pids = []
                for s in range(real):
                    sp = build_joint_prompt(q, [slot_names[s]], [texts[s]], context=rag_ctx or None,
                                            include_identity=False, real=1)
                    se = tok(sp, add_special_tokens=True, truncation=True, max_length=max_len)
                    one_pids.append(torch.tensor([se["input_ids"]], device=device))
            # phi context (READ): raw CM hidden vector of the problem, soft-addressed over centroids.
            phi_ctx = cm_context_vector(base, tok, q, device=device, layer_frac=phi_layer_frac)
            # carry mode (B): WRITE direction per peer = CM(peer_answer) - CM(gold). Encode gold once.
            value_vecs = None
            if write_mode == "carry":
                gold_h = cm_context_vector(base, tok, str(rec.get("answer", "")), device=device, layer_frac=phi_layer_frac)
                value_vecs = [cm_context_vector(base, tok, str(texts[s])[:4000], device=device, layer_frac=phi_layer_frac) - gold_h
                              for s in range(real)]
            scores = []
            for s in range(real):
                pj = peer_ids[s]
                if not ablate:
                    steerer.steer_vec = mem.steer_vector(pj, phi_ctx).detach()
                if peer_mode == "one":
                    # CM scores this peer ALONE: its prompt has a single answer at slot 0,
                    # so the endorsement token is always " Peer 0" (cand_ids[0]).
                    spid = one_pids[s]
                    lp = float(model.score_one_candidate(spid, torch.ones_like(spid), cand_ids[0])[0])
                else:
                    lp = float(model.score_one_candidate(pid, pmask, cand_ids[s])[0])
                steerer.steer_vec = None
                scores.append(lp)
            sel = int(max(range(real), key=lambda s: scores[s]))
            total += 1
            correct += int(corr[sel] == 1)
            # record which actual peer was chosen (peer_ids maps canonical slot -> peer index)
            selections.append({
                "id": str(rec.get("id") or rec.get("uid") or total),
                "task_type": str(rec.get("task_type") or rec.get("source") or ""),
                "selected_peer": int(peer_ids[sel]),          # 0/1/2 = the actual peer picked
                "selected_correct": int(corr[sel] == 1),
                "peer_scores": {int(peer_ids[s]): round(scores[s], 4) for s in range(real)},
                "peer_correct": {int(peer_ids[s]): int(corr[s]) for s in range(real)},
            })
            # event-axis update AFTER the pick (one update per peer per task).
            # sign s: ground-truth correctness. (The trainable-judge s was dropped — correctness
            # is not linearly readable from the frozen CM feature; see git history / train script.)
            if not ablate:
                signs = [1.0 if corr[s] else -1.0 for s in range(real)]
                for s in range(real):
                    vv = value_vecs[s] if value_vecs is not None else None
                    mem.update(peer_ids[s], signs[s], phi_ctx, value_vec=vv)
                mem.update_joint(list(signs) + [0.0] * (num_peers - real))

    acc = correct / total if total else 0.0
    out = Path(cfg.get("output", "outputs/eval_sym")); out.mkdir(parents=True, exist_ok=True)
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
    (out / "eval_metrics.json").write_text(json.dumps({
        "accuracy": acc, "num_samples": total, "coupling": "steer", "phi_mode": phi_mode, "write_mode": write_mode, "decay_mode": decay_mode,
        "peer_mode": peer_mode,
        "init_state": init_state, "reset_state": reset_state, "ablate_memory": ablate, "use_joint": use_joint,
        "gamma": mem.gamma_scalar(), "eta": float(mem.eta),
        "peer_selection": peer_selection,
    }, indent=1))
    # per-example selections (id, picked peer, correctness, scores) for detailed inspection
    with (out / "selections.jsonl").open("w") as f:
        for s in selections:
            f.write(json.dumps(s) + "\n")
    print(f"[eval_sym/steer/{phi_mode}] {cfg['offline_data']}: accuracy={acc*100:.2f} over {total} "
          f"(ablate={ablate}, reset={reset_state}, joint={use_joint})")
    steerer.remove()


if __name__ == "__main__":
    main()
