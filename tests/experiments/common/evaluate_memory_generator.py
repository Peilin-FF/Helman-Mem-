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
from feedback_state.evidence_seek import add_seek_instruction
from feedback_state.verdict import add_verdict_instruction, parse_verdict, strip_verdict
from feedback_state.lora import apply_lora, load_lora_state_dict, set_lora_active
from feedback_state.memory_generator import grade, has_chat_template, render_prompt


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
    p.add_argument("--shard", default=None, help="k/N: evaluate rows k, k+N, ... (one GPU per shard; scripts/merge_eval_shards.py joins the shard directories)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--windows", type=int, default=10)
    p.add_argument("--verdict", action="store_true", help="method A: ask for and score a 'Trust: ...' line before the answer (peers modes)")
    p.add_argument("--evidence_seek", action="store_true", help="method B stage 1: the reply may open with one sandbox check (two generation passes, vLLM engine); implies --verdict")
    p.add_argument("--seek_extra_tokens", type=int, default=512, help="tokens added to --max_new_tokens for the check request and the result block")
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
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        rows = rows[k::n]
        print(f"[gen-eval] shard {k}/{n}: {len(rows)} events", flush=True)
    messages = [r[f"messages_{args.mode}"] for r in rows]
    seek_infos = None
    if args.evidence_seek and args.mode != "solo":
        if args.engine != "vllm":
            raise SystemExit("--evidence_seek needs --engine vllm (two generation passes)")
        args.verdict = True   # the seek instruction asks for the Trust line too
        messages = [add_seek_instruction(m) for m in messages]
        seek_infos = [{"record": json.dumps(records[str(r["id"])]), "peer_order": list(r["peer_order"]), "prompt_source": "peers"} for r in rows]
    elif args.verdict and args.mode != "solo":
        messages = [add_verdict_instruction(m) for m in messages]
    prompts = [render_prompt(tok, m, thinking=args.thinking == "on") for m in messages]
    # the central model can be any causal LM: the record lives in the prompt files (memory_prob per peer, built from the
    # frozen Qwen3-4B judge) and the tilt only needs the peer blocks' token positions under this model's own tokenizer
    n_tok = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts[: min(len(prompts), 200)]]
    print(f"[gen-eval] central model {args.central_model}: {type(tok).__name__}, "
          f"{'chat template' if has_chat_template(tok) else 'NO chat template -> plain layout (base model)'}, "
          f"bos={tok.bos_token!r} eos={tok.eos_token!r}; prompt tokens (first {len(n_tok)}): min {min(n_tok)} median {int(np.median(n_tok))} max {max(n_tok)}", flush=True)
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
        outputs, seek_stats = generate_vllm(str(hf_dir) if hf_dir is not None else args.central_model, prompts, args, tok, ids_list=ids_list, biases=biases, seek_infos=seek_infos)
        print(f"[gen-eval] vllm decoded {len(prompts)} prompts ({time.time() - t0:.0f}s)" + (f"; checks {seek_stats}" if seek_stats else ""), flush=True)
        write_results(args, rows, records, outputs, t0, extra_metrics={"seek": seek_stats} if seek_stats else None)
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


def generate_vllm(model_path: str, prompts: list[str], args, tok, ids_list=None, biases=None, seek_infos=None) -> tuple[list[str], dict | None]:
    """Greedy decoding of already-rendered prompts with vLLM (same prompts, same stop condition as the HF path).
    With ``biases`` (one float32 vector per prompt over its tokens ``ids_list``) the attention tilt runs inside the engine.

    Every condition hands the engine the same token ids, tokenised here without added special tokens: the rendered
    template already carries BOS where the family uses one (Llama, Mistral), so the engine's own tokenisation would
    add a second BOS to the untilted conditions only.  Sliding windows are disabled (every model here has a window at
    least as long as the context we use, so this changes nothing numerically) because the bias kernels are plain causal.
    """
    import os

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # in-process engine: the parent already holds a CUDA context
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if biases is not None:
        from feedback_state import vllm_attn_bias

        vllm_attn_bias.install()
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams

    pool = None
    if seek_infos is not None:   # the checks run in spawned workers created BEFORE the engine's threads exist (a fork of the engine process can deadlock)
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context("spawn"))

    if ids_list is None:
        ids_list = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    longest = max(len(ids) for ids in ids_list)
    new_tokens = args.max_new_tokens + (args.seek_extra_tokens if seek_infos is not None else 0)
    model_max = int(getattr(AutoConfig.from_pretrained(model_path, local_files_only=True), "max_position_embeddings", 32768) or 32768)
    max_len = min(32768, model_max, max(4096, longest + new_tokens + 64))
    if longest + new_tokens > max_len:
        print(f"[gen-eval] WARNING: longest prompt {longest} + {new_tokens} new tokens exceeds the model's context {max_len}", flush=True)
    llm = LLM(model=model_path, tokenizer=args.central_model, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=max_len, enable_prefix_caching=True, trust_remote_code=False, seed=0, disable_sliding_window=True)
    params = SamplingParams(temperature=0.0, max_tokens=new_tokens)
    inputs = [{"prompt_token_ids": ids} for ids in ids_list]
    if biases is not None:
        for ids, b in zip(ids_list, biases):
            vllm_attn_bias.register(ids, b)
    if seek_infos is not None:   # method B stage 1: pass 1 to the check, the sandbox, pass 2 to the Trust line and the answer
        from feedback_state.evidence_seek import two_pass_generate
        responses, _masks, _lps, stats = two_pass_generate(llm, inputs, params, seek_infos, tok, n=1, response_length=new_tokens, pool=pool)
        pool.shutdown(wait=False, cancel_futures=True)
        return [tok.decode(ids, skip_special_tokens=True) for ids in responses], stats
    outs = llm.generate(inputs, params, use_tqdm=True)
    return [o.outputs[0].text for o in outs], None


def verdict_metrics(vrows) -> dict:
    """How good are the model's own Trust lines? agreement with the record (non-flat records), and against the
    verified labels (events where the peers disagree): AUC of the rank score and the favourite's hit rate."""
    from sklearn.metrics import roc_auc_score
    from feedback_state.verdict import rank_scores, spearman, record_is_flat
    parsed = [v for v in vrows if v[0] is not None]
    out = {"n": len(vrows), "parsed_rate": len(parsed) / max(len(vrows), 1)}
    rec = [spearman(rank_scores(rk), pr) for rk, pr, _ in parsed if pr and not record_is_flat(pr)]
    rec = [x for x in rec if x == x]
    out["spearman_with_record"] = float(np.mean(rec)) if rec else None
    dis = [(rk, lab) for rk, _, lab in parsed if 0 < sum(lab) < len(lab)]
    if dis:
        y = [l for _, lab in dis for l in lab]; sc = [x for rk, _ in dis for x in rank_scores(rk)]
        out["auc_vs_correct"] = float(roc_auc_score(y, sc))
        out["favourite_right"] = float(np.mean([lab[rk[0]] for rk, lab in dis]))
        out["spearman_with_correct"] = float(np.nanmean([spearman(rank_scores(rk), lab) for rk, lab in dis]))
        out["n_disagreeing"] = len(dis)
    return out


def write_results(args, rows, records, outputs, t0, extra_metrics: dict | None = None) -> None:
    hits, oracle, majority, out_rows = [], [], [], []
    vrows = []   # method A: (rank, memory_prob, peer_correct) per row with a parsed Trust line
    for r, text in zip(rows, outputs):
        rec = records[str(r["id"])]
        vrank = None
        if args.verdict and args.mode != "solo":
            vrank = parse_verdict(text, len(r["peer_correct"]))
            vrows.append((vrank, [float(x) for x in (r.get("memory_prob") or [])], [int(x) for x in r["peer_correct"]]))
            text_g = strip_verdict(text)
        else:
            text_g = text
        ok = grade(rec, text_g)
        hits.append(int(ok))
        pc = r["peer_correct"]
        oracle.append(int(max(pc)))
        majority.append(int(sum(pc) * 2 > len(pc)))
        out_rows.append({"pos": r["pos"], "id": r["id"], "task_type": r["task_type"], "source": r["source"], "correct": int(ok),
                         "peer_correct": pc, "memory_prob": r.get("memory_prob"), "verdict": vrank, "generation": text})
    h = np.array(hits)
    metrics = {"accuracy": float(h.mean()), "num_samples": int(len(h)), "mode": args.mode, "thinking": args.thinking == "on", "max_new_tokens": args.max_new_tokens, "engine": args.engine, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
               "central_model": args.central_model, "prompts": str(args.prompts),
               "generated": curve(h, args.windows), "oracle_any_peer": curve(np.array(oracle), args.windows),
               "peer_majority_correct": curve(np.array(majority), args.windows),
               "by_task": {t: float(np.mean([hh for hh, rr in zip(hits, rows) if rr["task_type"] == t])) for t in sorted({rr["task_type"] for rr in rows})}}
    if vrows:
        metrics["verdict"] = verdict_metrics(vrows)
    if extra_metrics:
        metrics.update(extra_metrics)
    with (args.output / "generations.jsonl").open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    (args.output / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    if "verdict" in metrics:
        print(f"[gen-eval] verdict: {metrics['verdict']}", flush=True)
    print(f"[gen-eval] mode={args.mode} accuracy={100 * metrics['accuracy']:.2f} oracle_any_peer={100 * metrics['oracle_any_peer']['total']:.2f} "
          f"by_task={ {k: round(100 * v, 1) for k, v in metrics['by_task'].items()} } windows=" + " ".join(f"{100 * w:.1f}" for w in metrics["generated"]["windows"]) + f" ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
