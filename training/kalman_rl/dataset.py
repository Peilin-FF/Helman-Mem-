"""The RL dataset for the vendored verl trainer.

Prompts are rendered with ``enable_thinking=False`` so the policy sees exactly the prompt the evaluations use. With
``data.attn_gamma > 0`` each row also carries the tilt over its prompt tokens (``attn_bias``), built from the record's
estimates and the peer blocks' character spans stored by training/kalman_rl/build_rl_data.py.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

from feedback_state.attn_bias import bias_values, token_bias

def render(tokenizer, messages, thinking: bool = False) -> str:
    """Chat template with the generation prompt; Qwen3's thinking block is disabled unless ``thinking`` (same as feedback_state.memory_generator.render_prompt)."""
    messages = [dict(m) for m in messages]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(thinking))
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def kalman_collate_fn(data_list: list[dict]) -> dict:
    """Like verl's collate_fn, but non-tensor fields always become 1-D object arrays (a batch of
    equal-length token lists must not silently turn into a 2-D integer array)."""
    tensors, non_tensors = defaultdict(list), defaultdict(list)
    for d in data_list:
        for k, v in d.items():
            (tensors if isinstance(v, torch.Tensor) else non_tensors)[k].append(v)
    out = {k: torch.stack(v, dim=0) for k, v in tensors.items()}
    for k, v in non_tensors.items():
        arr = np.empty(len(v), dtype=object)
        for i, x in enumerate(v):
            arr[i] = x
        out[k] = arr
    return out


class KalmanRLDataset(RLHFDataset):
    """RLHFDataset with our prompt rendering and the tilt tensor."""

    def __init__(self, data_files, tokenizer, config, processor=None):
        self.thinking = bool(config.get("enable_thinking", False))   # Qwen3 thinking mode for every prompt of this run
        self.attn_gamma = float(config.get("attn_gamma", 0.0))         # the memory's attention tilt: gamma * log(p_i / max p) on peer i's block
        self.attn_bias_form = str(config.get("attn_bias_form", "logratio"))
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=config, processor=processor)

    def _attn_bias(self, raw: str, messages, extra: dict, att: torch.Tensor) -> torch.Tensor:
        """The tilt over the left-padded prompt tokens (row extra_info: peer_spans in the user turn, memory_prob per slot)."""
        out = torch.zeros(self.max_prompt_length, dtype=torch.float32)
        spans, probs = extra.get("peer_spans"), extra.get("memory_prob")
        if spans is None or probs is None or len(spans) == 0:
            return out
        values = bias_values([float(p) for p in probs], self.attn_gamma, self.attn_bias_form)
        if not any(values):
            return out
        content = messages[-1]["content"]
        off = raw.find(content)
        if off < 0:
            return out
        offsets = self.tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        b = token_bias(offsets, [(int(a) + off, int(e) + off) for a, e in spans], values)
        n = int(att.sum())
        b = b[-n:] if len(b) > n else b
        out[self.max_prompt_length - len(b):] = torch.from_numpy(b)
        return out

    def _read_files_and_tokenize(self):
        import datasets

        frames = [datasets.load_dataset("parquet", data_files=f)["train"] for f in self.data_files]
        self.dataframe = datasets.concatenate_datasets(frames)
        print(f"dataset len: {len(self.dataframe)}")
        if self.filter_overlong_prompts:
            tok, key, limit, think = self.tokenizer, self.prompt_key, self.max_prompt_length, self.thinking

            def fits(doc):
                return len(tok.encode(render(tok, doc[key], think), add_special_tokens=False)) <= limit

            self.dataframe = self.dataframe.filter(fits, num_proc=self.num_workers, desc=f"Filtering prompts longer than {limit} tokens")
            print(f"filter dataset len: {len(self.dataframe)}")

    def _encode(self, raw: str):
        enc = self.tokenizer(raw, return_tensors="pt", add_special_tokens=False)
        ids, att = verl_F.postprocess_data(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_length=self.max_prompt_length,
                                           pad_token_id=self.tokenizer.pad_token_id, left_pad=True, truncation=self.truncation)
        return ids[0], att[0], compute_position_id_with_mask(att)[0]

    def _raw_ids(self, raw: str) -> list[int]:
        ids = self.tokenizer.encode(raw, add_special_tokens=False)
        if len(ids) > self.max_prompt_length:
            if self.truncation == "left":
                ids = ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                ids = ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left = self.max_prompt_length // 2
                ids = ids[:left] + ids[-(self.max_prompt_length - left):]
            else:
                raise RuntimeError(f"Prompt length {len(ids)} is longer than {self.max_prompt_length}.")
        return ids

    def __getitem__(self, item):
        row: dict = dict(self.dataframe[item])
        messages = row.pop(self.prompt_key)
        raw = render(self.tokenizer, messages, self.thinking)
        ids, att, pos = self._encode(raw)
        row["input_ids"], row["attention_mask"], row["position_ids"] = ids, att, pos
        row["raw_prompt_ids"] = self._raw_ids(raw)
        if self.attn_gamma > 0:
            row["attn_bias"] = self._attn_bias(raw, messages, row.get("extra_info") or {}, att)
        row.pop("guided_prompt", None)   # a column of the parquets built before 2026-09-13 (always empty in the trained runs)
        if self.return_raw_chat:
            row["raw_prompt"] = messages
        if self.return_full_prompt:
            row["full_prompts"] = raw
        extra = row.get("extra_info") or {}
        row["index"] = extra.get("index", item)
        row["tools_kwargs"] = {}
        return row
