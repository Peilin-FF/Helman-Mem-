"""Method 2: the record steers the model through its attention (credibility-weighted attention, CrAM-style), training-free.

The prompt carries no reliability text at all (the plain peers prompt).  Instead, in every attention head of the
chosen layers the scores over the tokens of peer i's block receive an additive bias gamma * log(c_i), where
c_i = p_i / max_j p_j is the memory's estimate normalised so that the favourite is 1; softmax(scores + log c) is
exactly Norm(A ⊙ c), the attention re-weighting of CrAM (Deng et al., 2024).  Flat records (estimates within 0.1)
get no bias.  Implemented as a forward pre-hook that edits the additive attention mask; batched greedy decoding
with left padding (spans shifted by the pad length).  Output format = the standard evaluator's.

  PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 python scripts/steer_attention.py --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B \
      --prompts outputs/gen/q3_4b/prompts_ood_shuffled0.jsonl --records data/ood/test.jsonl --pos_start 4000 --pos_count 1500 \
      --gamma 3.0 --layers all --output outputs/gen/q3_4b/steer/ood/attn_g3      (add --swap_record for the control)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import _clip, render_prompt
from feedback_state.memory_rl import peer_texts_in_prompt_order
from tests.experiments.common.evaluate_memory_generator import write_results

CHAR_LIMIT = 3000
FLAT_SPREAD = 0.1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--central_model", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gamma", type=float, default=1.0, help="bias = gamma * log(p_i / max_j p_j) on peer i's tokens")
    p.add_argument("--layers", default="all", help="'all' or 'a:b' (layer indices, end exclusive)")
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--pos_start", type=int, default=None, help="slice of the stream by position")
    p.add_argument("--pos_count", type=int, default=1500)
    p.add_argument("--swap_record", action="store_true", help="control: permute the record by rank (highest credibility on the least trusted peer)")
    p.add_argument("--windows", type=int, default=10)
    return p.parse_args()


class AttentionSteer:
    """Adds a per-key-position bias to the additive attention mask of the hooked attention modules."""

    def __init__(self, layers: set[int]):
        self.layers = layers
        self.bias: torch.Tensor | None = None   # [B, max_total_len]

    def hook(self, module, args, kwargs):
        if self.bias is None or getattr(module, "_steer_idx", -1) not in self.layers:
            return None
        hs = kwargs.get("hidden_states", args[0] if args else None)
        B, Q = hs.shape[0], hs.shape[1]
        cp = kwargs.get("cache_position")
        K = int(cp[-1]) + 1 if cp is not None else Q
        add = self.bias[:B, :K][:, None, None, :].to(dtype=hs.dtype, device=hs.device)   # [B,1,1,K]
        mask = kwargs.get("attention_mask")
        neg = torch.finfo(hs.dtype).min
        if mask is None:
            mask = torch.zeros(B, 1, Q, K, dtype=hs.dtype, device=hs.device)
            if Q > 1:
                qi = torch.arange(Q, device=hs.device)[:, None] + (K - Q)
                kj = torch.arange(K, device=hs.device)[None, :]
                mask = mask.masked_fill((kj > qi)[None, None], neg)
        elif mask.dtype == torch.bool:
            mask = torch.where(mask, torch.zeros((), dtype=hs.dtype, device=hs.device), torch.full((), neg, dtype=hs.dtype, device=hs.device))
        else:
            mask = mask.to(hs.dtype)
        if mask.shape[-1] != K:
            mask = mask[..., :K] if mask.shape[-1] > K else torch.nn.functional.pad(mask, (0, K - mask.shape[-1]))
        kwargs["attention_mask"] = mask + add
        return args, kwargs


def peer_spans(rec: dict, row: dict, prompt: str) -> list[tuple[int, int]]:
    """Character spans of each peer's text inside the rendered prompt (same construction as build_messages)."""
    texts = peer_texts_in_prompt_order(rec, row["peer_order"])
    spans, cursor = [], 0
    for i, t in enumerate(texts):
        block = f"Peer {i + 1}:\n{_clip(t, CHAR_LIMIT)}"
        j = prompt.find(block, cursor)
        if j < 0:
            spans.append((0, 0))
            continue
        spans.append((j + len(f"Peer {i + 1}:\n"), j + len(block)))
        cursor = j + len(block)
    return spans


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(args.central_model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.central_model, torch_dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True).to(device).eval()
    n_layers = len(model.model.layers)
    layers = set(range(n_layers)) if args.layers == "all" else set(range(int(args.layers.split(":")[0]), int(args.layers.split(":")[1])))
    steer = AttentionSteer(layers)
    for idx, layer in enumerate(model.model.layers):
        layer.self_attn._steer_idx = idx
        layer.self_attn.register_forward_pre_hook(steer.hook, with_kwargs=True)
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    rows = sorted((json.loads(l) for l in args.prompts.open()), key=lambda r: int(r["pos"]))
    if args.pos_start is not None:
        rows = [r for r in rows if args.pos_start <= int(r["pos"]) < args.pos_start + args.pos_count]
    if args.max_examples:
        rows = rows[: args.max_examples]
    if args.swap_record:
        for r in rows:
            probs, evid = list(r["memory_prob"]), list(r.get("memory_evidence", [0.0] * len(r["memory_prob"])))
            ranked = sorted(range(len(probs)), key=lambda s_: probs[s_])
            new_p, new_e = list(probs), list(evid)
            for i, s_ in enumerate(ranked):
                new_p[s_], new_e[s_] = probs[ranked[-1 - i]], evid[ranked[-1 - i]]
            r["memory_prob"], r["memory_evidence"] = new_p, new_e
    prompts = [render_prompt(tok, r["messages_peers"], thinking=False) for r in rows]
    # per-sample bias over prompt characters -> tokens
    tok_bias: list[torch.Tensor] = []
    n_biased = 0
    for r, prompt in zip(rows, prompts):
        rec = records[str(r["id"])]
        probs = [float(p) for p in r["memory_prob"]]
        enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        b = torch.zeros(len(ids))
        if max(probs) - min(probs) > FLAT_SPREAD:
            top = max(probs)
            for (a, e), p in zip(peer_spans(rec, r, prompt), probs):
                if e <= a:
                    continue
                c = max(p, 1e-3) / max(top, 1e-3)
                val = args.gamma * math.log(c)
                for ti, (ta, tb) in enumerate(offs):
                    if tb > ta and ta >= a and tb <= e:
                        b[ti] = val
            n_biased += 1
        tok_bias.append(b)
    print(f"[attn-steer] {len(rows)} prompts, {n_biased} with a non-flat record, layers {sorted(layers)[0]}..{sorted(layers)[-1]}, gamma {args.gamma}", flush=True)
    outputs = [""] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    t0 = time.time()
    with torch.no_grad():
        for bstart in range(0, len(order), args.batch_size):
            idx = order[bstart: bstart + args.batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            L = enc["input_ids"].shape[1]
            bias = torch.zeros(len(idx), L + args.max_new_tokens + 1)
            for bi, i in enumerate(idx):
                tb = tok_bias[i]
                bias[bi, L - len(tb): L] = tb   # left padding: shift by the pad length
            steer.bias = bias.to(device)
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, L:], skip_special_tokens=True)
            for i, t in zip(idx, texts):
                outputs[i] = t
            if (bstart // args.batch_size) % 10 == 0:
                print(f"[attn-steer] {min(bstart + args.batch_size, len(order))}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
    steer.bias = None
    ns = SimpleNamespace(output=args.output, mode="peers+attention", thinking="off", max_new_tokens=args.max_new_tokens, engine="hf-attn-steer",
                         checkpoint=None, central_model=args.central_model, prompts=args.prompts, windows=args.windows)
    write_results(ns, rows, records, outputs, t0)
    (args.output / "steer_config.json").write_text(json.dumps({"gamma": args.gamma, "layers": sorted(layers), "biased_prompts": n_biased}))


if __name__ == "__main__":
    main()
