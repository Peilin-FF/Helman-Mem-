"""Features: the frozen judge reads every event, and its hidden states become the record's addresses.

Per event, computed before any record read and without labels, dataset names or peer identities:
  q_mean / q_last   hidden states of the question alone at three layers (depth 1/3, 2/3, last), mean-pooled and last token
  peer_hidden       for each peer, the last-token hidden states of its judge prompt (feedback_state.judge_prompt), 3 layers
  sem / spread      their mean and spread over the peers
  margins           the judge's own Yes-No log-odds per peer

    PYTHONPATH=. python -m pipeline.features --stream data/indist6/test.jsonl --model /models/Qwen3-4B \
        --output outputs/features/q3_4b/indist6/shard0.pt --shards 8 --shard 0

One shard per GPU (events k, k+N, ...); pipeline.record joins the shards by their stored indices.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

FORMAT = "sigma_context_features_v1"   # the tag stored in every cache; unchanged so existing caches stay readable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_layers(count: int) -> list[int]:
    return sorted({max(1, count // 3), max(1, 2 * count // 3), count - 1})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stream", type=Path, required=True)
    ap.add_argument("--model", required=True, help="the judge: the central model's directory")
    ap.add_argument("--output", type=Path, required=True, help="<feature dir>/shard<k>.pt")
    ap.add_argument("--peers", type=int, default=6)
    ap.add_argument("--max-length", type=int, default=8192, help="tokens per judge prompt")
    ap.add_argument("--question-max-length", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=None, help="per shard")
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    if not 0 <= args.shard < args.shards:
        raise ValueError("--shard must lie in [0, --shards)")

    import torch

    from feedback_state.data import JsonlDataset
    from feedback_state.judge_prompt import context_text, judge_batch, yes_no_token_ids
    from feedback_state.newarch_loader import apply_torch_fp8_shim, dtype_from_name, load_central_model
    from feedback_state.permutations import canonical_peer_view

    apply_torch_fp8_shim()
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError("the judge needs single-token ' Yes' / ' No'")
    model = load_central_model(args.model, dtype=dtype, local_files_only=True).to(device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records = JsonlDataset(args.stream).records
    indices = list(range(args.shard, len(records), args.shards))
    if args.max_examples is not None:
        indices = indices[: args.max_examples]
    ids, original, peer_hidden_rows = [], [], []
    q_mean_rows, q_last_rows, sem_rows, spread_rows, margin_rows = [], [], [], [], []
    layers = None
    with torch.inference_mode():
        for local_index, record_index in enumerate(indices, start=1):
            record = records[record_index]
            view = canonical_peer_view(record, args.peers, setting="A")
            real = int(view["real"])
            if real < 1:
                continue
            question = str(record.get("problem", record.get("question", "")))
            enc = tokenizer(question if question.strip() else " ", return_tensors="pt", truncation=True, max_length=int(args.question_max_length))
            out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True, use_cache=False, return_dict=True)
            if layers is None:
                layers = _selected_layers(len(out.hidden_states))
            q_mean = torch.cat([out.hidden_states[l][0].float().mean(0) for l in layers])
            q_last = torch.cat([out.hidden_states[l][0, -1].float() for l in layers])

            input_ids, attention_mask = judge_batch(tokenizer, question, view["texts"][:real], context=context_text(record) or None,
                                                    max_length=args.max_length, device=device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, return_dict=True)
            last = attention_mask.long().sum(dim=1).clamp_min(1) - 1
            rows = torch.arange(real, device=device)
            logp = torch.log_softmax(out.logits[rows, last, :].float(), dim=-1)
            margins = logp[:, int(yes_ids[0])] - logp[:, int(no_ids[0])]
            hidden = torch.stack([out.hidden_states[l][rows, last, :].float() for l in layers])   # [layers, peers, H]
            padded = torch.zeros(args.peers, dtype=torch.float32, device=device)
            padded[:real] = margins
            per_peer = torch.zeros(args.peers, hidden.shape[0] * hidden.shape[2], device=device)
            per_peer[:real] = hidden.permute(1, 0, 2).reshape(real, -1)

            ids.append(str(record.get("id") or record.get("uid") or record_index + 1))
            original.append(record_index)
            q_mean_rows.append(q_mean.cpu().to(torch.float16))
            q_last_rows.append(q_last.cpu().to(torch.float16))
            sem_rows.append(hidden.mean(dim=1).reshape(-1).cpu().to(torch.float16))
            spread_rows.append(hidden.std(dim=1, unbiased=False).reshape(-1).cpu().to(torch.float16))
            margin_rows.append(padded.cpu().to(torch.float16))
            peer_hidden_rows.append(per_peer.cpu().to(torch.float16))
            if local_index % args.progress_every == 0 or local_index == len(indices):
                print(f"[features] shard {args.shard}/{args.shards}: {local_index}/{len(indices)}", flush=True)
    if not ids:
        raise ValueError("the shard produced no rows")
    payload = {
        "format": FORMAT,
        "protocol": "unsteered frozen judge; no source/domain/correctness; question-only features plus per-peer judge states",
        "input": str(args.stream.resolve()), "input_sha256": _sha256(args.stream),
        "central_model": str(Path(args.model).resolve()), "dtype": str(args.dtype), "num_peers": int(args.peers),
        "include_context": True, "num_shards": int(args.shards), "shard_index": int(args.shard), "selected_layers": layers,
        "ids": ids, "indices": torch.tensor(original, dtype=torch.int64),
        "q_mean": torch.stack(q_mean_rows), "q_last": torch.stack(q_last_rows), "sem": torch.stack(sem_rows),
        "spread": torch.stack(spread_rows), "margins": torch.stack(margin_rows), "peer_hidden": torch.stack(peer_hidden_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(f"[features] wrote {args.output}: {len(ids)} events, sem {tuple(payload['sem'].shape)}", flush=True)


if __name__ == "__main__":
    main()
