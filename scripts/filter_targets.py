"""Likelihood filter for distillation targets: keep a target only if the current policy could plausibly have
written it under the QUESTION-ONLY prompt (mean token loss <= tau) and it does not refer to the peers.

This is the on-policy constraint made explicit: samples far from the policy (teacher solutions in a foreign
style, or answers that mention "Peer 2") carry an importance weight of ~0 and are the ones that overwrite
the model's own behaviour.  Prints per-source statistics and writes the filtered prompt file.

  PYTHONPATH=. python scripts/filter_targets.py --central_model ... --prompts in.jsonl --out out.jsonl --tau 1.0
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import torch

from feedback_state.newarch_loader import apply_torch_fp8_shim, load_central_model

apply_torch_fp8_shim()

from transformers import AutoTokenizer

from feedback_state.memory_generator import render_prompt

PEER_RE = re.compile(r"\b(peer|peers|response \d|candidate under review)\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--central_model", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=3072)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.central_model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_central_model(args.central_model, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    rows = [json.loads(l) for l in args.prompts.open()]
    items = []
    for r in rows:
        if not r.get("target"):
            continue
        p = tok(render_prompt(tok, r["messages_solo"]), add_special_tokens=False)["input_ids"]
        t = tok(r["target"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        if len(p) + len(t) <= args.max_length:
            items.append((r, p, t))
    losses = {}
    with torch.no_grad():
        for b in range(0, len(items), args.batch_size):
            chunk = items[b : b + args.batch_size]
            width = max(len(p) + len(t) for _, p, t in chunk)
            ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), width), dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.bool)
            for i, (_, p, t) in enumerate(chunk):
                seq = p + t
                ids[i, : len(seq)] = torch.tensor(seq); att[i, : len(seq)] = 1; mask[i, len(p) : len(seq)] = True
            ids, att, mask = ids.cuda(), att.cuda(), mask.cuda()
            logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1].float()
            logp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            m = mask[:, 1:].float()
            per = (-(logp * m).sum(1) / m.sum(1).clamp_min(1)).tolist()
            for (r, _, _), v in zip(chunk, per):
                losses[str(r["id"])] = v
            if (b // args.batch_size) % 100 == 0:
                print(f"[filter] {min(b + args.batch_size, len(items))}/{len(items)}", flush=True)
    kept = collections.Counter(); seen = collections.Counter(); dropped_mention = collections.Counter()
    with args.out.open("w") as f:
        for r in rows:
            src = (r.get("target_source") or "none").split("_slot")[0]
            if r.get("target") and str(r["id"]) in losses:
                seen[src] += 1
                loss = losses[str(r["id"])]
                r["target_loss"] = round(loss, 3)
                if PEER_RE.search(r["target"]):
                    dropped_mention[src] += 1; r["target"] = None
                elif loss > args.tau:
                    r["target"] = None
                else:
                    kept[src] += 1
            else:
                r["target"] = None
            f.write(json.dumps(r) + "\n")
    print(f"[filter] tau={args.tau} kept={dict(kept)} of {dict(seen)}; dropped for peer mentions {dict(dropped_mention)}", flush=True)
    print(f"[filter] wrote {args.out}")


if __name__ == "__main__":
    main()
