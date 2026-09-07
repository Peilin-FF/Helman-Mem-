"""Batched generation + grading over a precomputed prompt stream (memory state embedded per event).

Reports answer accuracy of the central model's own generated answer along the
stream (windows / cumulative), next to the reference rates: the best selection
oracle (any peer correct) and the peers' majority.

  PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 python -m tests.experiments.common.evaluate_memory_generator \
      --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B --checkpoint outputs/gen/q3_4b/sft_memory \
      --prompts outputs/gen/q3_4b/prompts_indist_shuffled0.jsonl --mode memory --output outputs/gen/q3_4b/sft_memory/eval_indist_shuffled0
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

from feedback_state.data import JsonlDataset
from feedback_state.lora import apply_lora, load_lora_state_dict, set_lora_active
from feedback_state.memory_generator import grade, render_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--central_model", required=True)
    p.add_argument("--checkpoint", type=Path, default=None, help="directory with memory_generator.pt; omit for the frozen model")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True, help="the stream JSONL (for grading)")
    p.add_argument("--mode", choices=["memory", "peers", "solo"], default="memory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--every", type=int, default=1, help="keep every k-th event of the stream (positions and memory states unchanged; even subsample)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--windows", type=int, default=10)
    p.add_argument("--thinking", choices=["on", "off"], default="off", help="on = Qwen3 thinking mode (raise --max_new_tokens; the answer after </think> is graded)")
    p.add_argument("--engine", choices=["hf", "vllm"], default="hf", help="vllm = decode with vLLM (greedy, continuous batching; ~30x faster than HF generate); hf = transformers generate")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM engine memory share (lower it when the GPU is shared)")
    p.add_argument("--attn_gamma", type=float, default=0.0, help="the memory's attention tilt: gamma * log(p_i / max p) on peer i's block (0 = off); peers / memory modes")
    p.add_argument("--attn_bias_form", default="logratio", help="logratio | logodds (feedback_state.attn_bias)")
    p.add_argument("--attn_layers", default="all", help="HF engine only: 'all' or 'a:b'")
    p.add_argument("--swap_record", action="store_true", help="control for the tilt: the record permuted by rank (highest estimate on the least trusted peer)")
    return p.parse_args()


def curve(hits: np.ndarray, windows: int) -> dict:
    n = len(hits); edges = np.linspace(0, n, windows + 1).astype(int)
    return {"total": float(hits.mean()), "n": int(n),
            "windows": [float(hits[a:b].mean()) for a, b in zip(edges[:-1], edges[1:]) if b > a],
            "first_half": float(hits[: n // 2].mean()), "second_half": float(hits[n // 2 :].mean()),
            "cumulative": {str(k): float(hits[:k].mean()) for k in (250, 500, 1000, 2000, 4000, 8000, 16000) if k <= n}}


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "eval_metrics.json").unlink(missing_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.central_model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    hf_dir = args.checkpoint if (args.checkpoint is not None and (args.checkpoint / "config.json").exists()) else None
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    rows = [json.loads(l) for l in args.prompts.open()]
    if args.every > 1:
        rows = rows[:: args.every]
    if args.max_examples:
        rows = rows[: args.max_examples]
    prompts = [render_prompt(tok, r[f"messages_{args.mode}"], thinking=args.thinking == "on") for r in rows]
    ids_list = biases = None
    if args.attn_gamma > 0:   # the tilt: per prompt, the token positions of each peer block and its additive score
        from feedback_state.attn_bias import prompt_token_bias
        from feedback_state.memory_rl import peer_texts_in_prompt_order

        ids_list, biases = [], []
        for r, prompt in zip(rows, prompts):
            rec = records[str(r["id"])]
            probs = [float(p) for p in r["memory_prob"]]
            if args.swap_record:
                ranked = sorted(range(len(probs)), key=lambda s_: probs[s_])
                swapped = list(probs)
                for i_, s_ in enumerate(ranked):
                    swapped[s_] = probs[ranked[-1 - i_]]
                probs = swapped
            ids, b = prompt_token_bias(tok, prompt, peer_texts_in_prompt_order(rec, r["peer_order"]), probs, args.attn_gamma, args.attn_bias_form)
            ids_list.append(ids); biases.append(b)
        print(f"[gen-eval] attention tilt gamma {args.attn_gamma} ({args.attn_bias_form}{', swapped record' if args.swap_record else ''}): "
              f"{sum(bool(b.any()) for b in biases)}/{len(biases)} prompts tilted", flush=True)
    t0 = time.time()
    if args.engine == "vllm":
        if args.checkpoint is not None and hf_dir is None:
            raise SystemExit("--engine vllm needs a plain HF checkpoint directory (or no checkpoint for the frozen model); LoRA payloads need --engine hf")
        outputs = generate_vllm(str(hf_dir) if hf_dir is not None else args.central_model, prompts, args, ids_list=ids_list, biases=biases)
        print(f"[gen-eval] vllm decoded {len(prompts)} prompts ({time.time() - t0:.0f}s)", flush=True)
        write_results(args, rows, records, outputs, t0)
        return
    base = load_central_model(str(hf_dir) if hf_dir is not None else args.central_model, dtype=dtype, local_files_only=True).to(device=device, dtype=dtype)
    if hf_dir is not None:
        print(f"[gen-eval] full-parameter checkpoint {hf_dir}", flush=True)
    elif args.checkpoint is not None:
        payload = torch.load(args.checkpoint / "memory_generator.pt", map_location="cpu", weights_only=False)
        lc = payload["lora_config"]
        apply_lora(base, rank=int(lc["rank"]), alpha=float(lc["alpha"]), targets=tuple(lc["targets"]))
        load_lora_state_dict(base, payload["lora"])
        if payload.get("gated", False):
            set_lora_active(base, args.mode != "solo")   # evidence gate: no peers in the prompt -> base model exactly
            print(f"[gen-eval] gated adapters: active={args.mode != 'solo'}", flush=True)
    base.eval()
    steer = None
    if biases is not None:
        from feedback_state.attn_bias import install_hf_hooks

        steer = install_hf_hooks(base, args.attn_layers)
    outputs: list[str] = [""] * len(rows)
    # decode in length-sorted batches (order restored afterwards; the memory state is already in the prompts)
    idx = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    with torch.no_grad():
        for b in range(0, len(idx), args.batch_size):
            ids = idx[b : b + args.batch_size]
            enc = tok([prompts[i] for i in ids], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            if steer is not None:   # left padding: the prompt's tilt ends at the last prompt token
                L = enc["input_ids"].shape[1]
                bias = torch.zeros(len(ids), L + args.max_new_tokens + 1)
                for bi, i in enumerate(ids):
                    bias[bi, L - len(biases[i]): L] = torch.from_numpy(biases[i])
                steer.bias = bias.to(device)
            gen = base.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in zip(ids, texts):
                outputs[i] = t
            if (b // args.batch_size) % 25 == 0:
                print(f"[gen-eval] {min(b + args.batch_size, len(idx))}/{len(idx)} ({time.time() - t0:.0f}s)", flush=True)
    write_results(args, rows, records, outputs, t0)


def generate_vllm(model_path: str, prompts: list[str], args, ids_list=None, biases=None) -> list[str]:
    """Greedy decoding of already-rendered prompts with vLLM (same prompts, same stop condition as the HF path).
    With ``biases`` (one float32 vector per prompt over its tokens ``ids_list``) the attention tilt runs inside the engine."""
    import os

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # in-process engine: the parent already holds a CUDA context
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if biases is not None:
        from feedback_state import vllm_attn_bias

        vllm_attn_bias.install()
    from vllm import LLM, SamplingParams

    max_prompt = max(len(p) for p in prompts) // 2 + 64   # rough character->token bound only for max_model_len
    llm = LLM(model=model_path, tokenizer=args.central_model, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=min(32768, max(4096, max_prompt + args.max_new_tokens + 256)), enable_prefix_caching=True, trust_remote_code=False, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    inputs = prompts if ids_list is None else [{"prompt_token_ids": ids} for ids in ids_list]
    if biases is not None:
        for ids, b in zip(ids_list, biases):
            vllm_attn_bias.register(ids, b)
    outs = llm.generate(inputs, params, use_tqdm=True)
    return [o.outputs[0].text for o in outs]


def write_results(args, rows, records, outputs, t0) -> None:
    hits, oracle, majority, out_rows = [], [], [], []
    for r, text in zip(rows, outputs):
        rec = records[str(r["id"])]
        ok = grade(rec, text)
        hits.append(int(ok))
        pc = r["peer_correct"]
        oracle.append(int(max(pc)))
        majority.append(int(sum(pc) * 2 > len(pc)))
        out_rows.append({"pos": r["pos"], "id": r["id"], "task_type": r["task_type"], "source": r["source"], "correct": int(ok),
                         "peer_correct": pc, "memory_prob": r.get("memory_prob"), "generation": text})
    h = np.array(hits)
    metrics = {"accuracy": float(h.mean()), "num_samples": int(len(h)), "mode": args.mode, "thinking": args.thinking == "on", "max_new_tokens": args.max_new_tokens, "engine": args.engine, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
               "central_model": args.central_model, "prompts": str(args.prompts),
               "generated": curve(h, args.windows), "oracle_any_peer": curve(np.array(oracle), args.windows),
               "peer_majority_correct": curve(np.array(majority), args.windows),
               "by_task": {t: float(np.mean([hh for hh, rr in zip(hits, rows) if rr["task_type"] == t])) for t in sorted({rr["task_type"] for rr in rows})}}
    with (args.output / "generations.jsonl").open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    (args.output / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[gen-eval] mode={args.mode} accuracy={100 * metrics['accuracy']:.2f} oracle_any_peer={100 * metrics['oracle_any_peer']['total']:.2f} "
          f"by_task={ {k: round(100 * v, 1) for k, v in metrics['by_task'].items()} } windows=" + " ".join(f"{100 * w:.1f}" for w in metrics["generated"]["windows"]) + f" ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
