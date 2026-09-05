"""Evaluate the memory-augmented judge on a stream: cold-start memory, decide-then-update.

Writes ``selections.jsonl`` (per event: selection, correctness, fused / judge / memory
scores) and ``eval_metrics.json`` (accuracy, learning-curve windows, cumulative
accuracy, judge-only and memory-only accuracies from the same pass).

  PYTHONPATH=. python -m tests.experiments.common.evaluate_memory_judge --config configs/symmetric_memory_candidate_yesno.yaml \
      --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B --checkpoint outputs/memjudge/q3_4b/qc \
      --offline_data data/ood/test.jsonl --features outputs/context_features/q3_4b_ph/ood --order shuffled0 \
      --output outputs/memjudge/q3_4b/qc/eval_ood_shuffled0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from feedback_state.newarch_loader import apply_torch_fp8_shim, dtype_from_name, load_central_model

apply_torch_fp8_shim()

from transformers import AutoTokenizer

from feedback_state.addresses import EVIDENCE_DIM, Projection, memory_notes
from feedback_state.feature_streams import load_stream_from
from feedback_state.joint_data import batch_candidate_judge_inputs, yes_no_token_ids
from feedback_state.memory_judge import MemoryJudge, selection_loss
from feedback_state.memory_runtime import MemoryRuntime
from feedback_state.prompt_protocol import candidate_context_text, validate_prompt_protocol
from feedback_state.train_memory_judge import as_bool
from feedback_state.utils import load_config, merge_args_with_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--central_model", default=None)
    p.add_argument("--checkpoint", type=Path, default=None, help="directory with memory_judge.pt; omit for the frozen judge")
    p.add_argument("--offline_data", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num_peers", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--legacy_prompt_protocol", choices=["on", "off"], default=None)
    p.add_argument("--order", default="fixed", help="fixed | shuffled<seed>")
    p.add_argument("--memory", choices=["on", "off"], default="on", help="off = judge alone (no evidence, no prior)")
    p.add_argument("--design", default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--windows", type=int, default=10)
    p.add_argument("--online_lr", type=float, default=0.0,
                   help="> 0: test-time training - after each event's feedback, one AdamW step on the judge's own "
                        "selection loss (LoRA + steerer + kappa), so the slow weights learn online alongside the memory")
    p.add_argument("--online_accum", type=int, default=1)
    p.add_argument("--memory_text", choices=["on", "off"], default="off", help="frozen judge only: write the memory's evidence into the prompt as text")
    p.add_argument("--gradient_checkpointing", choices=["on", "off", "reentrant"], default="reentrant",
                   help="reentrant = torch reentrant checkpoint (needed for hybrid Qwen3.5 layers under LoRA + hooks)")
    return p.parse_args()


def curve(hits: np.ndarray, windows: int) -> dict:
    n = len(hits); edges = np.linspace(0, n, windows + 1).astype(int)
    return {"total": float(hits.mean()), "n": int(n),
            "windows": [float(hits[a:b].mean()) for a, b in zip(edges[:-1], edges[1:]) if b > a],
            "first_half": float(hits[: n // 2].mean()), "second_half": float(hits[n // 2 :].mean()),
            "cumulative": {str(k): float(hits[:k].mean()) for k in (250, 500, 1000, 2000, 4000, 8000, 16000) if k <= n}}


def main() -> None:
    args = parse_args()
    cfg = merge_args_with_config(args, load_config(args.config))
    out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=True)
    (out / "eval_metrics.json").unlink(missing_ok=True)
    device = torch.device(str(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    dtype = dtype_from_name(str(cfg.get("dtype", "bfloat16")).lower())
    model_name = str(cfg.get("central_model", "models/Qwen3-4B"))
    num_peers = int(cfg.get("num_peers", 3))
    max_len = int(cfg.get("max_length", 8192))
    legacy = as_bool(cfg.get("legacy_prompt_protocol"), False)
    validate_prompt_protocol(legacy_prompt_protocol=legacy, max_length=max_len)
    include_context = as_bool(cfg.get("include_context"), False)
    memory_on = as_bool(cfg.get("memory"), True)
    ckpt = Path(cfg["checkpoint"]) if cfg.get("checkpoint") else None
    payload = torch.load(ckpt / "memory_judge.pt", map_location="cpu", weights_only=False) if ckpt is not None else None

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=bool(cfg.get("local_files_only", False)))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = load_central_model(model_name, dtype=dtype, local_files_only=bool(cfg.get("local_files_only", False))).to(device=device, dtype=dtype)
    if payload is not None:
        lc = payload["lora_config"]
        judge = MemoryJudge(base, lora_rank=int(lc["rank"]), lora_alpha=float(lc["alpha"]), lora_targets=tuple(lc["targets"]), evidence_dim=EVIDENCE_DIM,
                            use_steer=bool(payload["use_steer"]), use_prior=bool(payload["use_prior"]))
        judge.load_payload(payload["judge"])
        mc = payload["memory_config"]
        design, lam = str(cfg.get("design") or mc["design"]), float(cfg.get("lam") or mc["lam"])
        proj_q = Projection(state=payload["proj_q"], device=device)
        proj_c = Projection(state=payload["proj_c"], device=device)
        use_steer, use_prior = bool(payload["use_steer"]) and memory_on, bool(payload["use_prior"]) and memory_on
        memory_text = bool(payload.get("memory_text", False)) and memory_on
    else:
        judge = MemoryJudge(base, lora_rank=0, evidence_dim=EVIDENCE_DIM, use_steer=False, use_prior=memory_on)
        design, lam = str(cfg.get("design") or "qc"), float(cfg.get("lam") or 100.0)
        proj_q = proj_c = None
        use_steer, use_prior = False, memory_on
        memory_text = as_bool(cfg.get("memory_text"), False) and memory_on
    judge = judge.to(device); judge.eval(); base.eval()
    online_lr = float(cfg.get("online_lr") or 0.0)
    online_accum = int(cfg.get("online_accum") or 1)
    optim = None
    if online_lr > 0:
        if payload is None:
            raise ValueError("online adaptation needs a trained checkpoint (LoRA parameters)")
        groups = judge.trainable_groups(lora_lr=online_lr, steer_lr=online_lr * 10)
        online_params = [q for g in groups for q in g["params"]]
        optim = torch.optim.AdamW(groups)
        gc_mode = str(cfg.get("gradient_checkpointing") or "reentrant").lower()
        if gc_mode in ("on", "true", "1", "yes", "reentrant") and hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": gc_mode == "reentrant"})
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
        base.train()
        print(f"[memjudge/eval] online adaptation: lr={online_lr} accum={online_accum} params={sum(q.numel() for q in online_params)}", flush=True)

    fs = load_stream_from(cfg["offline_data"], cfg["features"], name=str(cfg["offline_data"]), num_peers=num_peers)
    if proj_q is None:  # frozen judge without a checkpoint: fit the projections on the stream itself (unsupervised)
        dim = int(cfg.get("dim") or 256)
        proj_q = Projection(fs.sem, dim, device); proj_c = Projection(fs.peer_hidden.reshape(-1, fs.peer_hidden.shape[-1]), dim, device)
    runtime = MemoryRuntime(design=design, proj_q=proj_q, proj_c=proj_c, num_peers=num_peers, lam=lam, device=device)
    runtime.attach(fs)
    runtime.reset()
    N = len(fs)
    order = np.arange(N) if str(cfg.get("order", "fixed")) == "fixed" else np.random.default_rng(int(str(cfg["order"]).replace("shuffled", ""))).permutation(N)
    if cfg.get("max_examples"):
        order = order[: int(cfg["max_examples"])]
    yes_ids, no_ids = yes_no_token_ids(tok)
    labels = fs.labels.numpy()
    hits, judge_hits, mem_hits, rows = [], [], [], []
    t0 = time.time()
    n_backward = 0
    with torch.set_grad_enabled(optim is not None):
        for i, t in enumerate(order.tolist(), start=1):
            r = int(fs.real[t])
            if r < 1:
                continue
            y = labels[t, :r].copy()
            cand_texts = list(fs.texts[t][:r])
            ell, n_eff, evidence, X = runtime.read(t)
            rec = fs.records[t]
            q = str(rec.get("problem", rec.get("question", "")))
            ctx = candidate_context_text(rec, include_context=include_context, legacy_prompt_protocol=legacy)
            ids, mask = batch_candidate_judge_inputs(tok, q, [f"peer_{k}" for k in range(r)], cand_texts, context=ctx or None,
                                                     include_identity=False, real=r, max_length=max_len, device=device, legacy_prompt_protocol=legacy,
                                                     slot_notes=(memory_notes(ell, n_eff) if memory_text else None))
            scores, z = judge.score(ids, mask, yes_ids, no_ids, evidence=evidence if use_steer else None, mem_logit=ell if use_prior else None)
            if optim is not None:
                # decide first (scores above are the decision), then learn from the revealed labels
                loss = selection_loss(scores, torch.as_tensor(y))
                if loss is not None:
                    (loss / online_accum).backward()
                    n_backward += 1
                    if n_backward % online_accum == 0:
                        torch.nn.utils.clip_grad_norm_(online_params, 1.0)
                        optim.step(); optim.zero_grad(set_to_none=True)
                scores, z = scores.detach(), z.detach()
            sel, jsel, msel = int(torch.argmax(scores)), int(torch.argmax(z)), int(torch.argmax(ell))
            hits.append(int(y[sel])); judge_hits.append(int(y[jsel])); mem_hits.append(int(y[msel]))
            rows.append({"id": fs.ids[t], "task_type": fs.task[t], "source": fs.source[t], "selected_peer": sel, "selected_correct": int(y[sel]),
                         "peer_correct": {k: int(y[k]) for k in range(r)}, "scores": [round(float(v), 4) for v in scores.tolist()],
                         "judge_scores": [round(float(v), 4) for v in z.tolist()], "memory_logits": [round(float(v), 4) for v in ell.tolist()],
                         "evidence": [round(float(v), 2) for v in n_eff.tolist()]})
            if memory_on:
                runtime.write(t, X, y)
            if i == 1 or i % 250 == 0 or i == len(order):
                print(f"[memjudge/eval] {i}/{len(order)} acc={100 * np.mean(hits):.2f} judge={100 * np.mean(judge_hits):.2f} mem={100 * np.mean(mem_hits):.2f} ({time.time() - t0:.0f}s)", flush=True)
    h = np.array(hits)
    metrics = {"accuracy": float(h.mean()), "num_samples": int(len(h)), "order": str(cfg.get("order", "fixed")), "memory": memory_on,
               "online_lr": online_lr, "online_accum": online_accum, "memory_text": memory_text,
               "checkpoint": str(ckpt) if ckpt else None, "design": design, "lam": lam, "central_model": model_name,
               "fused": curve(h, int(cfg.get("windows", 10))), "judge_only": curve(np.array(judge_hits), int(cfg.get("windows", 10))),
               "memory_only": curve(np.array(mem_hits), int(cfg.get("windows", 10))), "kappa": float(judge.kappa)}
    with (out / "selections.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (out / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[memjudge/eval] {cfg['offline_data']} order={metrics['order']} accuracy={100 * metrics['accuracy']:.2f} "
          f"judge={100 * metrics['judge_only']['total']:.2f} memory={100 * metrics['memory_only']['total']:.2f} windows=" +
          " ".join(f"{100 * w:.1f}" for w in metrics["fused"]["windows"]))
    if judge.steerer is not None:
        judge.steerer.remove()


if __name__ == "__main__":
    main()
