"""Cached frozen-center-model features + labels for one event stream.

``scripts/encode_context_features.py`` writes, per event, the unsteered center
model's representation of the question (``q_mean``/``q_last``), of the three
candidate-judge prompts (``sem`` = mean over candidates, ``peer_hidden`` = per
candidate) and the judge's own Yes/No log-odds (``margins``).  This module
aligns those tensors with the JSONL records (labels, answers, task/source) so the
memory can be simulated without a GPU forward pass.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from feedback_state.data import JsonlDataset
from feedback_state.permutations import canonical_peer_view

DATA_ROOT = Path(os.environ.get("SIGMA_DATA_ROOT", "data"))
FEATURE_ROOT = Path(os.environ.get("SIGMA_FEATURE_ROOT", "outputs/context_features"))

# stream name -> (jsonl relative to DATA_ROOT, cache directory template relative to FEATURE_ROOT)
STREAMS = {
    "train": ("mixed_train_big/train.jsonl", "{m}_big_ph/train"),
    "train_small": ("mixed_train/train.jsonl", "{m}_ph/train"),
    "indist": ("indist/test.jsonl", "{m}_indist_ph/ood"),
    "ood": ("ood/test.jsonl", "{m}_ph/ood"),
    # five-peer streams (2026-09-06): the three peers above plus Meta-Llama-3.1-8B-Instruct and DeepSeek-Coder-V2-Lite-Instruct
    "train5": ("mixed_train_big5/train.jsonl", "{m}_big5_ph/train"),
    "indist5": ("indist5/test.jsonl", "{m}_indist5_ph/ood"),
    "ood5": ("ood5/test.jsonl", "{m}_5_ph/ood"),
    # six-peer streams: the five above plus DeepSeek-R1-Distill-Qwen-7B (answer text after its think block)
    "train6": ("mixed_train_big6/train.jsonl", "{m}_big6_ph/train"),
    "indist6": ("indist6/test.jsonl", "{m}_indist6_ph/ood"),
    "ood6": ("ood6/test.jsonl", "{m}_6_ph/ood"),
}
PEERS_PER_STREAM = {"train5": 5, "indist5": 5, "ood5": 5, "train6": 6, "indist6": 6, "ood6": 6}
MODELS = {"q3_0_6b": "Qwen3-0.6B", "q3_4b": "Qwen3-4B", "q3_8b": "Qwen3-8B", "q35_4b": "Qwen3.5-4B", "q35_9b": "Qwen3.5-9B"}


@dataclass
class FeatureStream:
    name: str
    model: str
    records: list
    ids: list
    q_mean: torch.Tensor      # [N, F]
    sem: torch.Tensor         # [N, F]
    peer_hidden: torch.Tensor # [N, P, F]
    margins: torch.Tensor     # [N, P]
    labels: torch.Tensor      # [N, P] in {0, 1}
    real: torch.Tensor        # [N] number of real candidates
    task: list
    source: list
    texts: list               # [N][P] candidate texts (canonical peer order)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def num_peers(self) -> int:
        return int(self.labels.shape[1])

    def subset(self, order: np.ndarray) -> "FeatureStream":
        idx = torch.as_tensor(np.asarray(order), dtype=torch.long)
        take = lambda t: t[idx]
        return FeatureStream(
            name=self.name, model=self.model,
            records=[self.records[i] for i in order], ids=[self.ids[i] for i in order],
            q_mean=take(self.q_mean), sem=take(self.sem), peer_hidden=take(self.peer_hidden),
            margins=take(self.margins), labels=take(self.labels), real=take(self.real),
            task=[self.task[i] for i in order], source=[self.source[i] for i in order],
            texts=[self.texts[i] for i in order],
        )


def stream_paths(name: str, model: str) -> tuple[Path, Path]:
    jsonl, cache = STREAMS[name]
    return DATA_ROOT / jsonl, FEATURE_ROOT / cache.format(m=model)


def load_stream(name: str, model: str, *, num_peers: int | None = None, limit: int | None = None) -> FeatureStream:
    jsonl_path, cache_dir = stream_paths(name, model)
    num_peers = PEERS_PER_STREAM.get(name, 3) if num_peers is None else num_peers
    return load_stream_from(jsonl_path, cache_dir, name=name, model=model, num_peers=num_peers, limit=limit)


def load_stream_from(jsonl_path, cache_dir, *, name: str = "", model: str = "", num_peers: int = 3, limit: int | None = None) -> FeatureStream:
    jsonl_path, cache_dir = Path(jsonl_path), Path(cache_dir)
    records = JsonlDataset(jsonl_path).records
    shards = sorted(glob.glob(str(cache_dir / "shard*.pt")))
    if not shards:
        raise FileNotFoundError(f"no feature shards under {cache_dir}")
    payloads = [torch.load(p, map_location="cpu", weights_only=False) for p in shards]
    indices = torch.cat([p["indices"] for p in payloads])
    order = torch.argsort(indices)
    cat = lambda key: torch.cat([p[key] for p in payloads])[order]
    ids = [i for p in payloads for i in p["ids"]]
    ids = [ids[i] for i in order.tolist()]
    indices = indices[order].tolist()
    if "peer_hidden" not in payloads[0]:
        raise ValueError(f"{cache_dir} was encoded without --save-peer-hidden")
    records = [records[i] for i in indices]
    labels, real, task, source, texts = [], [], [], [], []
    for rec, rid in zip(records, ids):
        got = str(rec.get("id") or rec.get("uid") or "")
        if got != str(rid):
            raise ValueError(f"cache/record id mismatch: {rid} vs {got}")
        view = canonical_peer_view(rec, num_peers, setting="A")
        cbp = rec.get("correctness_by_peer") or rec.get("peer_correct") or {}
        labels.append([int(round(float(cbp.get(k, 0)))) for k in view["keys"]])
        real.append(int(view["real"]))
        task.append(str(rec.get("task_type") or ""))
        source.append(str(rec.get("source") or ""))
        texts.append(list(view["texts"]))
    fs = FeatureStream(
        name=name, model=model, records=records, ids=ids,
        q_mean=cat("q_mean").float(), sem=cat("sem").float(), peer_hidden=cat("peer_hidden").float(),
        margins=cat("margins").float(), labels=torch.tensor(labels, dtype=torch.long),
        real=torch.tensor(real, dtype=torch.long), task=task, source=source, texts=texts,
    )
    if limit is not None:
        fs = fs.subset(np.arange(min(int(limit), len(fs))))
    return fs
