"""Evaluate: the central model answers every event of a stream under one condition, and the answers are graded.

A condition is a prompt mode plus the tilt settings:

    tilt   --mode peers --gamma 3          the six answers in the prompt, attention tilted by the record
    peers  --mode peers                    the same prompt, no tilt
    solo   --mode solo                     the question alone
    swap   --mode peers --gamma 3 --swap   the tilt with the record permuted by rank (control)

    PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 python -m pipeline.evaluate --model /models/Qwen3-4B \
        --record outputs/record/q3_4b/indist6/shuffled0.jsonl --stream data/indist6/test.jsonl \
        --condition tilt --mode peers --gamma 3 --output outputs/eval/q3_4b/indist6/tilt
    # a trained model: add --checkpoint outputs/train/<run>/phase2/hf/global_step_276
    # HF engine, sharded: --engine hf --shard k/N --output <out>/shard<k>, then  python -m pipeline.evaluate --merge --output <out>

Writes <output>/generations.jsonl (one graded answer per event, in stream order) and <output>/eval_metrics.json:
accuracy, per-task accuracy, the accuracy curve along the stream, the any-peer-correct and peer-majority references,
and the full condition that produced them.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from pipeline.config import shown

CONDITION_KEYS = ("condition", "mode", "gamma", "swap_record", "bias_form", "max_new_tokens", "engine",
                  "central_model", "checkpoint", "record", "stream", "every", "max_examples")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--merge", action="store_true", help="join <output>/shard*/ into <output> and stop")
    p.add_argument("--model", help="the central model's directory (its tokenizer is always used)")
    p.add_argument("--checkpoint", type=Path, default=None, help="an HF checkpoint directory of a trained central model")
    p.add_argument("--record", type=Path, help="the record file of the stream (pipeline.record output)")
    p.add_argument("--stream", type=Path, help="the stream JSONL (for grading)")
    p.add_argument("--condition", default=None, help="the condition's name, stored with the results")
    p.add_argument("--mode", choices=["peers", "solo"], default="peers")
    p.add_argument("--gamma", type=float, default=0.0, help="the tilt: gamma * log(p_i / max p) on peer i's tokens (0 = off)")
    p.add_argument("--swap", action="store_true", help="control: the record permuted by rank (highest estimate on the least trusted peer)")
    p.add_argument("--bias-form", default="logratio", help="logratio | logodds (feedback_state.attn_bias)")
    p.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--batch-size", type=int, default=8, help="HF engine only")
    p.add_argument("--attn-layers", default="all", help="HF engine only: 'all' or 'a:b'")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--every", type=int, default=1, help="keep every k-th event (positions and record states unchanged)")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--shard", default=None, help="k/N: evaluate rows k, k+N, ...")
    p.add_argument("--windows", type=int, default=10)
    return p.parse_args(argv)


def curve(hits: np.ndarray, windows: int) -> dict:
    n = len(hits); edges = np.linspace(0, n, windows + 1).astype(int)
    return {"total": float(hits.mean()), "n": int(n),
            "windows": [float(hits[a:b].mean()) for a, b in zip(edges[:-1], edges[1:]) if b > a],
            "first_half": float(hits[: n // 2].mean()), "second_half": float(hits[n // 2 :].mean()),
            "cumulative": {str(k): float(hits[:k].mean()) for k in (250, 500, 1000, 2000, 4000, 8000, 16000) if k <= n}}


def summarise(rows: list[dict], windows: int) -> dict:
    """The metrics of graded rows ({correct, peer_correct, task_type} per event, in stream order)."""
    hits = np.array([int(r["correct"]) for r in rows])
    oracle = np.array([int(max(r["peer_correct"])) for r in rows])
    majority = np.array([int(sum(r["peer_correct"]) * 2 > len(r["peer_correct"])) for r in rows])
    return {"accuracy": float(hits.mean()), "num_samples": int(len(hits)),
            "generated": curve(hits, windows), "oracle_any_peer": curve(oracle, windows),
            "peer_majority_correct": curve(majority, windows),
            "by_task": {t: float(np.mean([int(r["correct"]) for r in rows if r["task_type"] == t]))
                        for t in sorted({r["task_type"] for r in rows})}}


def merge(output: Path, windows: int) -> dict:
    shards = sorted(glob.glob(str(output / "shard*" / "generations.jsonl")))
    if not shards:
        raise SystemExit(f"no shard*/generations.jsonl under {output}")
    rows = sorted((json.loads(l) for f in shards for l in open(f)), key=lambda r: int(r["pos"]))
    meta = json.load(open(Path(shards[0]).parent / "eval_metrics.json"))
    metrics = {k: meta.get(k) for k in CONDITION_KEYS}
    metrics.update(summarise(rows, windows), shards=len(shards))
    with (output / "generations.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (output / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[evaluate] merged {len(shards)} shards, {len(rows)} events, accuracy {100 * metrics['accuracy']:.2f}", flush=True)
    return metrics


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.merge:
        merge(args.output, args.windows)
        return
    if not (args.model and args.record and args.stream):
        raise SystemExit("--model, --record and --stream are required (or --merge)")
    if args.checkpoint is not None and not (args.checkpoint / "config.json").exists():
        raise SystemExit(f"--checkpoint {args.checkpoint} is not an HF model directory (no config.json)")
    from feedback_state.newarch_loader import apply_torch_fp8_shim

    apply_torch_fp8_shim()
    from transformers import AutoTokenizer

    from feedback_state.data import JsonlDataset
    from feedback_state.memory_generator import has_chat_template, render_prompt

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "eval_metrics.json").unlink(missing_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.stream).records}
    rows = [json.loads(l) for l in args.record.open()]
    if args.every > 1:
        rows = rows[:: args.every]
    if args.max_examples:
        rows = rows[: args.max_examples]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        rows = rows[k::n]
        print(f"[evaluate] shard {k}/{n}: {len(rows)} events", flush=True)
    prompts = [render_prompt(tok, r[f"messages_{args.mode}"]) for r in rows]
    n_tok = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts[: min(len(prompts), 200)]]
    print(f"[evaluate] {args.condition or args.mode}: central model {args.model}"
          f"{f' checkpoint {args.checkpoint}' if args.checkpoint else ''}, "
          f"{'chat template' if has_chat_template(tok) else 'no chat template (plain layout)'}; prompt tokens (first {len(n_tok)}): "
          f"min {min(n_tok)} median {int(np.median(n_tok))} max {max(n_tok)}", flush=True)
    ids_list = biases = None
    if args.gamma > 0:   # the tilt: per prompt, the token positions of each peer block and its additive score
        from feedback_state.attn_bias import prompt_token_bias
        from feedback_state.memory_generator import peer_texts_in_prompt_order

        ids_list, biases = [], []
        for r, prompt in zip(rows, prompts):
            probs = [float(p) for p in r["memory_prob"]]
            if args.swap:
                ranked = sorted(range(len(probs)), key=lambda s_: probs[s_])
                probs = [probs[ranked[-1 - ranked.index(s_)]] for s_ in range(len(probs))]
            ids, b = prompt_token_bias(tok, prompt, peer_texts_in_prompt_order(records[str(r["id"])], r["peer_order"]), probs,
                                       args.gamma, args.bias_form)
            ids_list.append(ids); biases.append(b)
        print(f"[evaluate] attention tilt gamma {args.gamma} ({args.bias_form}{', swapped record' if args.swap else ''}): "
              f"{sum(bool(b.any()) for b in biases)}/{len(biases)} prompts tilted", flush=True)
    t0 = time.time()
    model_path = str(args.checkpoint) if args.checkpoint is not None else args.model
    if args.engine == "vllm":
        outputs = generate_vllm(model_path, prompts, args, tok, ids_list=ids_list, biases=biases)
    else:
        outputs = generate_hf(model_path, prompts, args, tok, biases=biases)
    print(f"[evaluate] decoded {len(prompts)} prompts ({time.time() - t0:.0f}s)", flush=True)
    write_results(args, rows, records, outputs, t0)


def generate_hf(model_path: str, prompts: list[str], args, tok, biases=None) -> list[str]:
    import torch

    from feedback_state.newarch_loader import dtype_from_name, load_central_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    base = load_central_model(model_path, dtype=dtype, local_files_only=True).to(device=device, dtype=dtype)
    base.eval()
    steer = None
    if biases is not None:
        from feedback_state.attn_bias import install_hf_hooks

        steer = install_hf_hooks(base, args.attn_layers)
    outputs: list[str] = [""] * len(prompts)
    idx = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))   # length-sorted batches, order restored below
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
            for i, t in zip(ids, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                outputs[i] = t
            if (b // args.batch_size) % 25 == 0:
                print(f"[evaluate] {min(b + args.batch_size, len(idx))}/{len(idx)}", flush=True)
    return outputs


def generate_vllm(model_path: str, prompts: list[str], args, tok, ids_list=None, biases=None) -> list[str]:
    """Greedy vLLM decoding of already-rendered prompts; with ``biases`` the tilt runs inside the patched kernels.

    Every condition hands the engine the same token ids, tokenised without added special tokens (the rendered template
    already carries BOS where the family uses one). Sliding windows are disabled: the bias kernels are plain causal, and
    every model here has a window at least as long as the contexts used.
    """
    import os

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if biases is not None:
        from feedback_state import vllm_attn_bias

        vllm_attn_bias.install()
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams

    if ids_list is None:
        ids_list = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    longest = max(len(ids) for ids in ids_list)
    model_max = int(getattr(AutoConfig.from_pretrained(model_path, local_files_only=True), "max_position_embeddings", 32768) or 32768)
    max_len = min(32768, model_max, max(4096, longest + args.max_new_tokens + 64))
    if longest + args.max_new_tokens > max_len:
        print(f"[evaluate] WARNING: longest prompt {longest} + {args.max_new_tokens} new tokens exceeds the context {max_len}", flush=True)
    llm = LLM(model=model_path, tokenizer=args.model, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=max_len, enable_prefix_caching=True, trust_remote_code=False, seed=0, disable_sliding_window=True)
    if biases is not None:
        for ids, b in zip(ids_list, biases):
            vllm_attn_bias.register(ids, b)
    outs = llm.generate([{"prompt_token_ids": ids} for ids in ids_list], SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens),
                        use_tqdm=True)
    return [o.outputs[0].text for o in outs]


def write_results(args, rows, records, outputs, t0) -> None:
    from feedback_state.memory_generator import grade

    out_rows = []
    for r, text in zip(rows, outputs):
        ok = grade(records[str(r["id"])], text)
        out_rows.append({"pos": r["pos"], "id": r["id"], "task_type": r["task_type"], "source": r["source"], "correct": int(ok),
                         "peer_correct": r["peer_correct"], "memory_prob": r.get("memory_prob"), "generation": text})
    metrics = {"condition": args.condition, "mode": args.mode, "gamma": args.gamma, "swap_record": bool(args.swap),
               "bias_form": args.bias_form if args.gamma > 0 else None,
               "max_new_tokens": args.max_new_tokens, "engine": args.engine, "central_model": args.model,
               "checkpoint": shown(args.checkpoint) if args.checkpoint else None, "record": shown(args.record),
               "stream": shown(args.stream), "every": args.every, "max_examples": args.max_examples, "shard": args.shard}
    metrics.update(summarise(out_rows, args.windows))
    with (args.output / "generations.jsonl").open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    (args.output / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"[evaluate] {args.condition or args.mode}: accuracy {100 * metrics['accuracy']:.2f}, any peer right "
          f"{100 * metrics['oracle_any_peer']['total']:.2f}, by task { {k: round(100 * v, 1) for k, v in metrics['by_task'].items()} } "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
