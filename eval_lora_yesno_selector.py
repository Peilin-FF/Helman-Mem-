"""Evaluate a candidate-wise Yes/No LoRA selector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from feedback_state.newarch_loader import apply_torch_fp8_shim, load_central_model

apply_torch_fp8_shim()

from peft import PeftModel
from transformers import AutoTokenizer

from feedback_state.data import JsonlDataset
from feedback_state.generation import dtype_from_name
from feedback_state.joint_data import batch_candidate_judge_inputs, yes_no_token_ids
from feedback_state.permutations import apply_perm, canonical_peer_view, named_order, stable_seed
from feedback_state.utils import load_config, merge_args_with_config


def as_bool(v, default=False):
    if v is None:
        return default
    return v if isinstance(v, bool) else str(v).lower() in {"1", "true", "yes", "on"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate candidate-wise Yes/No LoRA selector.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--offline_data", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--central_model", default=None)
    p.add_argument("--num_peers", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--test_order", choices=["orig", "swap", "random"], default=None)
    p.add_argument("--max_examples", type=int, default=None)
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
        return keys, names, texts, corr, real, []
    seed = stable_seed(record.get("id") or record.get("uid") or "")
    perm = named_order(order, real, seed=seed)
    return (
        apply_perm(keys, perm),
        apply_perm(names, perm),
        apply_perm(texts, perm),
        apply_perm(corr, perm),
        real,
        perm,
    )


def _peer_id(key: str, fallback: int) -> int:
    if str(key).startswith("peer_"):
        try:
            return int(str(key).split("_")[1])
        except (IndexError, ValueError):
            pass
    return int(fallback)


@torch.no_grad()
def _score_yesno_batch(model, input_ids, attention_mask, yes_id, no_id):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    last = attention_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    logits = out.logits[rows, last, :]
    logprob = torch.log_softmax(logits.float(), dim=-1)
    return logprob[:, int(yes_id)] - logprob[:, int(no_id)]


def main():
    args = parse_args()
    cfg = merge_args_with_config(args, load_config(args.config))
    ckpt = Path(cfg["checkpoint"])
    out_dir = Path(cfg.get("output", "outputs/eval_lora_yesno"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    dtype = dtype_from_name(str(cfg.get("dtype", "bfloat16")))
    model_name = str(cfg.get("central_model", "/mnt/data/peilin/HF_MODEL/Qwen3-0.6B"))
    num_peers = int(cfg.get("num_peers", 3))
    max_len = int(cfg.get("max_length", 8192))
    test_order = str(cfg.get("test_order", "orig")).lower()

    tok_src = str(ckpt) if (ckpt / "tokenizer_config.json").exists() else model_name
    tok = AutoTokenizer.from_pretrained(tok_src, local_files_only=bool(cfg.get("local_files_only", False)))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    yes_ids, no_ids = yes_no_token_ids(tok)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError(f"Expected single-token Yes/No, got {yes_ids=} {no_ids=}")

    base = load_central_model(
        model_name,
        dtype=dtype,
        local_files_only=bool(cfg.get("local_files_only", False)),
    )
    model = PeftModel.from_pretrained(
        base, str(ckpt / "lora_adapter"), is_trainable=False,
        local_files_only=bool(cfg.get("local_files_only", False)),
    ).to(device=device, dtype=dtype)
    model.eval()

    records = JsonlDataset(cfg["offline_data"]).records
    max_examples = cfg.get("max_examples")
    if max_examples is not None:
        records = records[: int(max_examples)]

    total = correct = 0
    selections = []
    for rec in records:
        keys, names, texts, corr, real, perm = _example_view(rec, num_peers, test_order)
        if real <= 0:
            continue
        q = str(rec.get("problem", rec.get("question", "")))
        rag_ctx = str(rec.get("retrieved_context", rec.get("context", ""))) if as_bool(cfg.get("include_context"), True) else ""
        input_ids, attention_mask = batch_candidate_judge_inputs(
            tok, q, names, texts,
            context=rag_ctx or None,
            include_identity=False,
            real=real,
            max_length=max_len,
            device=device,
        )
        vals = _score_yesno_batch(model, input_ids, attention_mask, yes_ids[0], no_ids[0])
        scores = [float(vals[s]) for s in range(real)]
        sel = int(max(range(real), key=lambda i: scores[i]))
        total += 1
        correct += int(corr[sel] == 1)
        peer_ids = [_peer_id(k, i) for i, k in enumerate(keys)]
        selections.append({
            "id": str(rec.get("id") or rec.get("uid") or total),
            "task_type": str(rec.get("task_type") or rec.get("source") or ""),
            "selected_peer": int(peer_ids[sel]),
            "selected_correct": int(corr[sel] == 1),
            "peer_scores": {int(peer_ids[i]): round(scores[i], 4) for i in range(real)},
            "peer_correct": {int(peer_ids[i]): int(corr[i]) for i in range(real)},
        })

    peer_chosen = {}
    for row in selections:
        p = row["selected_peer"]
        agg = peer_chosen.setdefault(p, {"chosen": 0, "chosen_correct": 0})
        agg["chosen"] += 1
        agg["chosen_correct"] += row["selected_correct"]
    peer_selection = {
        str(p): {
            "chosen": v["chosen"],
            "share": round(v["chosen"] / total, 4) if total else 0.0,
            "hit_rate_when_chosen": round(v["chosen_correct"] / v["chosen"], 4) if v["chosen"] else 0.0,
        }
        for p, v in sorted(peer_chosen.items())
    }
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "num_samples": total,
        "score_mode": "candidate_yesno",
        "arm": "lora_yesno",
        "num_peers": num_peers,
        "test_order": test_order,
        "peer_selection": peer_selection,
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / "selections.jsonl").open("w") as f:
        for row in selections:
            f.write(json.dumps(row) + "\n")
    print(f"[lora_yesno/eval] {cfg['offline_data']}: accuracy={metrics['accuracy']*100:.2f} "
          f"over {total} (test_order={test_order})", flush=True)


if __name__ == "__main__":
    main()
