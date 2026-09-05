"""Encode frozen, unsteered center-model context features for the whitened memory.

Two feature families are written per event, both computed BEFORE any memory read
and without correctness labels, dataset names, or peer identities:

* ``q_mean`` / ``q_last``: hidden states of the question text alone (three layers,
  mean-pooled and last-token).  These support response-free M-Route.
* ``sem`` / ``spread`` / ``margins``: last-token hidden states of the Sigma-Mem
  candidate Yes/No judge prompts, aggregated permutation-invariantly over peers
  (mean and std), plus the unsteered Yes-No log-odds per peer.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from feedback_state.data import JsonlDataset
from feedback_state.joint_data import batch_candidate_judge_inputs, yes_no_token_ids
from feedback_state.newarch_loader import (
    apply_torch_fp8_shim,
    dtype_from_name,
    load_central_model,
)
from feedback_state.permutations import canonical_peer_view
from feedback_state.prompt_protocol import candidate_context_text

apply_torch_fp8_shim()

from transformers import AutoTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_id(record: dict, index: int) -> str:
    return str(record.get("id") or record.get("uid") or index + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-peers", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--question-max-length", type=int, default=2048)
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--save-peer-hidden",
        action="store_true",
        help="also store the per-candidate judge hidden states [peers, layers*H] (fp16)",
    )
    return parser.parse_args()


def _selected_layers(count: int) -> list[int]:
    return sorted({max(1, count // 3), max(1, 2 * count // 3), count - 1})


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must lie in [0, num-shards)")
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.central_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError("encoder requires single-token Yes/No")
    model = load_central_model(args.central_model, dtype=dtype, local_files_only=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records = JsonlDataset(args.input).records
    indices = list(range(args.shard_index, len(records), args.num_shards))
    if args.max_examples is not None:
        indices = indices[: args.max_examples]

    ids: list[str] = []
    original: list[int] = []
    q_mean_rows, q_last_rows, sem_rows, spread_rows, margin_rows = [], [], [], [], []
    peer_hidden_rows: list[torch.Tensor] = []
    layers: list[int] | None = None

    with torch.inference_mode():
        for local_index, record_index in enumerate(indices, start=1):
            record = records[record_index]
            view = canonical_peer_view(record, args.num_peers, setting="A")
            real = int(view["real"])
            if real < 1:
                continue
            question = str(record.get("problem", record.get("question", "")))

            # --- question-only features (response-free) ---
            enc = tokenizer(
                question if question.strip() else " ",
                return_tensors="pt",
                truncation=True,
                max_length=int(args.question_max_length),
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
            if layers is None:
                layers = _selected_layers(len(out.hidden_states))
            q_mean = torch.cat([out.hidden_states[l][0].float().mean(0) for l in layers])
            q_last = torch.cat([out.hidden_states[l][0, -1].float() for l in layers])

            # --- candidate judge features (question + peer responses, unsteered) ---
            context = candidate_context_text(
                record, include_context=args.include_context, legacy_prompt_protocol=False
            )
            input_ids, attention_mask = batch_candidate_judge_inputs(
                tokenizer,
                question,
                [f"peer_{i}" for i in range(args.num_peers)],
                view["texts"],
                context=context or None,
                include_identity=False,
                real=real,
                max_length=args.max_length,
                device=device,
                legacy_prompt_protocol=False,
            )
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last = attention_mask.long().sum(dim=1).clamp_min(1) - 1
            rows = torch.arange(real, device=device)
            logp = torch.log_softmax(out.logits[rows, last, :].float(), dim=-1)
            margins = logp[:, int(yes_ids[0])] - logp[:, int(no_ids[0])]
            hidden = torch.stack(
                [out.hidden_states[l][rows, last, :].float() for l in layers]
            )  # [layers, peers, H]
            sem = hidden.mean(dim=1).reshape(-1)
            spread = hidden.std(dim=1, unbiased=False).reshape(-1)
            padded = torch.zeros(args.num_peers, dtype=torch.float32, device=device)
            padded[:real] = margins

            ids.append(_record_id(record, record_index))
            original.append(record_index)
            q_mean_rows.append(q_mean.cpu().to(torch.float16))
            q_last_rows.append(q_last.cpu().to(torch.float16))
            sem_rows.append(sem.cpu().to(torch.float16))
            spread_rows.append(spread.cpu().to(torch.float16))
            margin_rows.append(padded.cpu().to(torch.float16))
            if args.save_peer_hidden:
                per_peer = torch.zeros(args.num_peers, hidden.shape[0] * hidden.shape[2], device=device)
                per_peer[:real] = hidden.permute(1, 0, 2).reshape(real, -1)
                peer_hidden_rows.append(per_peer.cpu().to(torch.float16))
            if local_index % args.progress_every == 0 or local_index == len(indices):
                print(
                    f"[features] shard={args.shard_index}/{args.num_shards} "
                    f"{local_index}/{len(indices)}",
                    flush=True,
                )

    if not ids:
        raise ValueError("shard produced no rows")
    payload = {
        "format": "sigma_context_features_v1",
        "protocol": (
            "unsteered frozen CM; no source/domain/correctness; question-only "
            "features plus permutation-invariant candidate-judge aggregates"
        ),
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "central_model": str(Path(args.central_model).resolve()),
        "dtype": str(args.dtype),
        "num_peers": int(args.num_peers),
        "include_context": bool(args.include_context),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "selected_layers": layers,
        "ids": ids,
        "indices": torch.tensor(original, dtype=torch.int64),
        "q_mean": torch.stack(q_mean_rows),
        "q_last": torch.stack(q_last_rows),
        "sem": torch.stack(sem_rows),
        "spread": torch.stack(spread_rows),
        "margins": torch.stack(margin_rows),
    }
    if args.save_peer_hidden:
        payload["peer_hidden"] = torch.stack(peer_hidden_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(
        f"[features] wrote {args.output} rows={len(ids)} "
        f"q_mean={tuple(payload['q_mean'].shape)} sem={tuple(payload['sem'].shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
