"""Where does the central model's attention go among the six peer blocks, with and without the tilt?

The question behind it: did training under the tilt teach the model to tell reliable peers from
unreliable ones on its own?  If so, at gamma = 0 (no tilt at all) the trained model's attention over
the peer blocks should line up with the record's reliability estimate, or with actual correctness,
better than the frozen model's or the no-memory control's does.

Measurement.  One HF forward pass per prompt in eager attention, a forward hook on every layer's
attention module that takes the softmaxed weights [heads, Q, K], picks the query rows of interest and
sums the columns of each peer block (token spans found exactly as the tilt finds them).  Nothing is
generated.  Two query sets: the last prompt token (the position that emits the first answer token)
and every token after the last peer block (the instruction).  Masses are averaged over heads and kept
per layer, so late-layer behaviour can be read separately.

Per prompt and model this gives m[layer, peer]; the summary reports, over prompts with a non-flat
record and disagreeing peers:
  corr(record)     mean Spearman between the peer masses and the record's p_i
  corr(correct)    mean Spearman between the peer masses and the peers' verified correctness
  favourite        mass on the record's favourite peer / mean peer mass   (1 = no preference)
  correct share    share of peer mass on correct peers / share of peers that are correct (1 = chance)
  peer share       share of all attention that lands on peer blocks at all
  entropy          normalised entropy of the mass over the six peers (1 = uniform)
With the tilt on (gamma = 3) the same numbers show what the tilt does to attention mechanically.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from feedback_state.attn_bias import bias_values, install_hf_hooks, peer_char_spans, token_bias
from feedback_state.data import JsonlDataset
from feedback_state.memory_generator import render_prompt
from feedback_state.memory_rl import peer_texts_in_prompt_order


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


class MassProbe:
    """Collects, per layer, the attention mass on each peer block for the chosen query rows."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.spans: list[tuple[int, int]] = []
        self.q_last: int = 0
        self.q_post: tuple[int, int] = (0, 0)
        self.out_last = np.zeros((n_layers, 0))
        self.out_post = np.zeros((n_layers, 0))
        self.total_last = np.zeros(n_layers)
        self.total_post = np.zeros(n_layers)
        self.keep_rows = False                      # token-resolution dump of the last query row, per layer
        self.rows: list = []

    def reset(self, spans, q_last, q_post):
        self.rows = [None] * self.n_layers
        self.spans, self.q_last, self.q_post = spans, q_last, q_post
        P = len(spans)
        self.out_last = np.zeros((self.n_layers, P)); self.out_post = np.zeros((self.n_layers, P))
        self.total_last = np.zeros(self.n_layers); self.total_post = np.zeros(self.n_layers)

    def hook_for(self, layer_idx: int):
        def hook(module, args, output):
            w = output[1]
            if w is None:
                raise RuntimeError("attention weights not returned: the model must run with attn_implementation='eager'")
            w = w[0].float()                                  # [H, Q, K]
            last = w[:, self.q_last, :].mean(0)               # [K]
            if self.keep_rows:
                self.rows[layer_idx] = last.cpu().numpy().tolist()
            a, b = self.q_post
            post = w[:, a:b, :].mean(0).mean(0) if b > a else last
            for p, (s, e) in enumerate(self.spans):
                if e > s:
                    self.out_last[layer_idx, p] = float(last[s:e].sum())
                    self.out_post[layer_idx, p] = float(post[s:e].sum())
            self.total_last[layer_idx] = float(last.sum()); self.total_post[layer_idx] = float(post.sum())
        return hook


def token_spans(offsets, char_spans) -> list[tuple[int, int]]:
    """Token index span of each peer block (first token, one past last), from the tilt's own token mapping."""
    lab = token_bias(offsets, char_spans, [float(i + 1) for i in range(len(char_spans))])
    out = []
    for i in range(len(char_spans)):
        idx = np.nonzero(lab == float(i + 1))[0]
        out.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model directory (frozen base or a trained checkpoint)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gammas", default="0,3")
    ap.add_argument("--flat", type=float, default=0.1)
    ap.add_argument("--out", default="outputs/address/attention_mass")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dump_ids", default="", help="comma-separated prompt ids to dump at token resolution (always included)")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    rows = [json.loads(l) for l in open(args.prompts)]
    # the same sample for every model: non-flat record and disagreeing peers, chosen by a fixed seed
    cand = [r for r in rows if (max(r["memory_prob"]) - min(r["memory_prob"])) > args.flat
            and 0 < sum(int(c) for c in r["peer_correct"]) < len(r["peer_correct"])]
    rng = random.Random(args.seed)
    dump_ids = {d for d in args.dump_ids.split(",") if d}
    forced = [r for r in rows if str(r["id"]) in dump_ids]
    sample = forced + [r for r in rng.sample(cand, min(args.n, len(cand))) if str(r["id"]) not in dump_ids]
    print(f"[{args.name}/{args.stream}] {len(cand)} eligible prompts, {len(sample)} sampled", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="eager",
                                                 local_files_only=True).to(args.device).eval()
    layers = model.model.layers
    probe = MassProbe(len(layers))
    for i, layer in enumerate(layers):
        layer.self_attn.register_forward_hook(probe.hook_for(i))
    steer = install_hf_hooks(model, "all")
    gammas = [float(g) for g in args.gammas.split(",")]

    results = []
    with torch.no_grad():
        for k, r in enumerate(sample):
            rec = records[str(r["id"])]
            prompt = render_prompt(tok, r["messages_peers"], thinking=False)
            enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            texts = peer_texts_in_prompt_order(rec, r["peer_order"])
            cspans = peer_char_spans(texts, prompt)
            tspans = token_spans(offs, cspans)
            if any(e <= s for s, e in tspans):
                continue
            L = len(ids)
            q_last = L - 1
            q_post = (max(e for _, e in tspans), L)
            probs = [float(p) for p in r["memory_prob"]]
            correct = [int(c) for c in r["peer_correct"]]
            x = torch.tensor([ids], device=args.device)
            per_gamma = {}
            for g in gammas:
                if g > 0:
                    b = token_bias(offs, cspans, bias_values(probs, g))
                    steer.bias = torch.tensor(b, dtype=torch.float32)[None, :]
                else:
                    steer.bias = None
                probe.keep_rows = str(r["id"]) in dump_ids
                probe.reset(tspans, q_last, q_post)
                model(input_ids=x, use_cache=False)
                if probe.keep_rows:
                    Path(args.out).mkdir(parents=True, exist_ok=True)
                    json.dump({"model": args.name, "stream": args.stream, "id": r["id"], "gamma": g, "spans": tspans,
                               "probs": probs, "correct": correct, "n_tokens": L, "row_last": probe.rows},
                              open(Path(args.out) / f"dump_{args.name}_{args.stream}_{str(r['id']).replace(':', '_')}_g{int(g)}.json", "w"))
                per_gamma[str(g)] = {"last": probe.out_last.tolist(), "post": probe.out_post.tolist(),
                                     "total_last": probe.total_last.tolist(), "total_post": probe.total_post.tolist()}
            results.append({"id": r["id"], "task": r.get("task_type"), "n_tokens": L, "probs": probs, "correct": correct,
                            "spans": tspans, "mass": per_gamma})
            if (k + 1) % 50 == 0:
                print(f"[{args.name}/{args.stream}] {k + 1}/{len(sample)} ({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.name, "path": args.model, "stream": args.stream, "n": len(results), "gammas": gammas,
               "n_layers": len(layers), "results": results}, open(out / f"{args.name}_{args.stream}.json", "w"))
    print(f"[{args.name}/{args.stream}] wrote {out / f'{args.name}_{args.stream}.json'} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
