"""RLVR (GRPO-style) on question-only prompts, with memory-guided hint distillation.

Reward = the task verifier (math equivalence, QA normalisation, sandboxed tests for code).
Policy gradient with a group baseline (G samples per prompt, advantage = r - mean_group r),
on-policy single update per rollout batch (ratio = 1, no clipping needed), optional KL(pi || ref).

Memory-guided part (--hint_distill on): when every sample of a group fails, plain RLVR has no
signal.  We then offer the student ONE hinted rollout: the prompt additionally shows the solution
of the most reliable verified-correct peer (reliability from the memory notes in the prompt file),
asks the student to solve in its own words, and — if that hinted rollout is correct — adds a
log-likelihood term for it under the QUESTION-ONLY prompt.  The student internalises what the
reliable peer knew, in its own style, and is never trained to imitate a peer verbatim.

  PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 python -m feedback_state.train_rlvr --central_model /mnt/data/peilin/HF_MODEL/Qwen3-4B \
      --prompts outputs/gen/q3_4b/prompts_train_fixed.jsonl --records data/mixed_train_big/train.jsonl \
      --n_prompts 1500 --group 4 --hint_distill on --output_dir outputs/gen/q3_4b/rlvr_hint
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from feedback_state.newarch_loader import apply_torch_fp8_shim, dtype_from_name, load_central_model

apply_torch_fp8_shim()

from transformers import AutoTokenizer

from feedback_state.data import JsonlDataset
from feedback_state.memory_rl import hint_messages, memory_pseudo_reward as _memory_pseudo_reward, peer_texts_in_prompt_order
from feedback_state.memory_generator import grade, render_prompt


def completion_logprobs(model, tok, prompt_ids: list[list[int]], completion_ids: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced log-probs of each completion given its prompt. Returns (sum_logp [B], n_tokens [B])."""
    width = max(len(p) + len(c) for p, c in zip(prompt_ids, completion_ids))
    pad = tok.pad_token_id
    ids = torch.full((len(prompt_ids), width), pad, dtype=torch.long)
    att = torch.zeros((len(prompt_ids), width), dtype=torch.long)
    mask = torch.zeros((len(prompt_ids), width), dtype=torch.bool)
    for i, (p, c) in enumerate(zip(prompt_ids, completion_ids)):
        seq = p + c
        ids[i, : len(seq)] = torch.tensor(seq); att[i, : len(seq)] = 1
        mask[i, len(p) : len(seq)] = True
    ids, att, mask = ids.to(device), att.to(device), mask.to(device)
    out = model(input_ids=ids, attention_mask=att, use_cache=False)
    logp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    m = mask[:, 1:].float()
    return (logp * m).sum(1), m.sum(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--central_model", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--n_prompts", type=int, default=1500)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--batch_prompts", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coefficient to a frozen reference copy (0 = no KL, no reference model)")
    ap.add_argument("--hint_distill", choices=["on", "off"], default="on")
    ap.add_argument("--hint_weight", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=500, help="prompts between checkpoints")
    ap.add_argument("--verified_fraction", type=float, default=1.0, help="fraction of prompts whose verifier reward is available")
    ap.add_argument("--pseudo_reward", choices=["none", "memory"], default="none",
                    help="reward for unverified prompts: none (no signal) or memory (1 if the sample agrees with the reliability-weighted vote of the peers)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda:0")
    dtype = dtype_from_name(args.dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.central_model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    policy = load_central_model(args.central_model, dtype=dtype, local_files_only=True).to(device=device, dtype=dtype)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    policy.enable_input_require_grads()
    ref = None
    if args.beta > 0:
        ref = load_central_model(args.central_model, dtype=dtype, local_files_only=True).to(device=device, dtype=dtype).eval()
        for q in ref.parameters():
            q.requires_grad_(False)
    params = [q for q in policy.parameters() if q.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    records = {str(r.get("id") or r.get("uid")): r for r in JsonlDataset(args.records).records}
    rows = [json.loads(l) for l in args.prompts.open()]
    rng = random.Random(args.seed)
    by_task: dict[str, list] = {}
    for r in rows:
        by_task.setdefault(r["task_type"], []).append(r)
    quota = {"math": 0.4, "rag": 0.4, "code": 0.2}
    chosen = []
    for task, frac in quota.items():
        pool = by_task.get(task, []); rng.shuffle(pool)
        chosen += pool[: int(args.n_prompts * frac)]
    rng.shuffle(chosen)
    verified = {str(r["id"]): (rng.random() < args.verified_fraction) for r in chosen}
    print(f"[rlvr] prompts={len(chosen)} group={args.group} lr={args.lr} beta={args.beta} hint_distill={args.hint_distill} "
          f"verified_fraction={args.verified_fraction} pseudo_reward={args.pseudo_reward} trainable={sum(q.numel() for q in params)/1e6:.1f}M", flush=True)

    def memory_pseudo_reward(r: dict, rec: dict, text: str) -> float | None:
        """Reliability-weighted peer vote as the reward of an unverified prompt (feedback_state.memory_rl)."""
        return _memory_pseudo_reward(rec, peer_texts_in_prompt_order(rec, r["peer_order"]), r["memory_prob"], text)

    eos = tok.eos_token_id
    stats = {"reward": [], "zero_groups": 0, "hint_tried": 0, "hint_ok": 0, "loss": []}
    t0 = time.time()
    done_prompts = 0
    for b in range(0, len(chosen), args.batch_prompts):
        batch = chosen[b : b + args.batch_prompts]
        recs = [records[str(r["id"])] for r in batch]
        prompts = [render_prompt(tok, r["messages_solo"]) for r in batch]
        # ---- rollouts (no grad)
        policy.eval()
        with torch.no_grad():
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            gen = policy.generate(**enc, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0, num_return_sequences=args.group,
                                  max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
            comp = gen[:, enc["input_ids"].shape[1]:]
        prompt_ids = tok(prompts, add_special_tokens=False)["input_ids"]
        samples = []  # (prompt_ids, completion_ids, advantage, kind)
        for i, rec in enumerate(recs):
            rewards = []
            comps = []
            for g in range(args.group):
                c = comp[i * args.group + g].tolist()
                if eos in c:
                    c = c[: c.index(eos) + 1]
                c = [x for x in c if x != tok.pad_token_id] if tok.pad_token_id != eos else c
                text = tok.decode(c, skip_special_tokens=True)
                if verified[str(batch[i]["id"])]:
                    rw = float(grade(rec, text))
                elif args.pseudo_reward == "memory":
                    pr = memory_pseudo_reward(batch[i], rec, text)
                    rw = float(pr) if pr is not None else 0.0
                else:
                    rw = 0.0   # unverified and no memory: no signal (all rewards equal -> no gradient)
                rewards.append(rw); comps.append(c)
            stats["reward"] += rewards
            base = float(np.mean(rewards))
            for c, rw in zip(comps, rewards):
                if rw != base and len(c) > 0:
                    samples.append((prompt_ids[i], c, rw - base, "pg"))
            if base == 0.0 and args.hint_distill == "on" and verified[str(batch[i]["id"])]:
                stats["zero_groups"] += 1
                r = batch[i]
                correct_slots = [s for s in range(len(r["peer_correct"])) if int(r["peer_correct"][s]) == 1]
                if correct_slots:
                    s = max(correct_slots, key=lambda k: r["memory_prob"][k])
                    keys = sorted(rec.get("peer_responses", {}))
                    hint = str(rec["peer_responses"][keys[r["peer_order"][s]]])
                    hp = render_prompt(tok, hint_messages(rec, hint, float(r["memory_prob"][s]), float(r["memory_evidence"][s])))
                    stats["hint_tried"] += 1
                    with torch.no_grad():
                        henc = tok([hp], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                        hg = policy.generate(**henc, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
                        hc = hg[0, henc["input_ids"].shape[1]:].tolist()
                    if eos in hc:
                        hc = hc[: hc.index(eos) + 1]
                    htext = tok.decode(hc, skip_special_tokens=True)
                    if grade(rec, htext) and len(hc) > 0:
                        stats["hint_ok"] += 1
                        samples.append((prompt_ids[i], hc, args.hint_weight, "hint"))   # distilled under the question-only prompt
        # ---- update (single on-policy step)
        policy.train()
        if samples:
            loss_total = torch.zeros((), device=device)
            n_tok_total = sum(len(c) for _, c, _, _ in samples)
            for k in range(0, len(samples), 8):
                chunk = samples[k : k + 8]
                sum_logp, n_tok = completion_logprobs(policy, tok, [s[0] for s in chunk], [s[1] for s in chunk], device)
                adv = torch.tensor([s[2] for s in chunk], device=device, dtype=sum_logp.dtype)
                loss = -(adv * sum_logp).sum() / max(1, n_tok_total)
                if ref is not None:
                    with torch.no_grad():
                        ref_logp, _ = completion_logprobs(ref, tok, [s[0] for s in chunk], [s[1] for s in chunk], device)
                    # k3 estimator on sequence sums (approximate): exp(ref-pol) - (ref-pol) - 1
                    d = ref_logp - sum_logp
                    loss = loss + args.beta * (torch.exp(d.clamp(max=20)) - d - 1).sum() / max(1, n_tok_total)
                loss.backward()
                loss_total += loss.detach()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step(); optim.zero_grad(set_to_none=True)
            stats["loss"].append(float(loss_total))
        done_prompts += len(batch)
        if (b // args.batch_prompts) % 10 == 0:
            recent = stats["reward"][-args.batch_prompts * args.group * 10:]
            print(f"[rlvr] prompts {done_prompts}/{len(chosen)} mean_reward(recent)={np.mean(recent):.3f} zero_groups={stats['zero_groups']} hint_tried={stats['hint_tried']} hint_ok={stats['hint_ok']} loss={np.mean(stats['loss'][-10:]) if stats['loss'] else float('nan'):.4f} ({time.time() - t0:.0f}s)", flush=True)
        if args.save_every > 0 and done_prompts % args.save_every == 0 and done_prompts < len(chosen):
            ck = args.output_dir / f"checkpoint-{done_prompts}"
            policy.save_pretrained(ck, safe_serialization=True); tok.save_pretrained(ck)
            print(f"[rlvr] saved {ck}", flush=True)
    ck = args.output_dir / "final"
    policy.save_pretrained(ck, safe_serialization=True); tok.save_pretrained(ck)
    n = len(stats["reward"]); edges = np.linspace(0, n, 11).astype(int)
    summary = {"prompts": len(chosen), "group": args.group, "mean_reward": float(np.mean(stats["reward"])),
               "reward_windows": [float(np.mean(stats["reward"][a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a],
               "zero_groups": stats["zero_groups"], "hint_tried": stats["hint_tried"], "hint_ok": stats["hint_ok"], "args": {k: str(v) for k, v in vars(args).items()}}
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[rlvr] saved {ck} mean_reward={summary['mean_reward']:.3f} reward_windows=" + " ".join(f"{100*w:.1f}" for w in summary["reward_windows"]) + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
