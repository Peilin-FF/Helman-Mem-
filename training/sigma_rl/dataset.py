"""Datasets for the vendored verl trainers.

Prompts are rendered with ``enable_thinking=False`` so the policy sees exactly the prompt our
evaluations use (Qwen3 non-thinking mode).  Each RL row may carry a second, *guided* prompt
(the question plus the peers' solutions, annotated by the memory or not, or one of them) from
which the trainer draws the central model's stream-time answer.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.dataset.sft_dataset import SFTDataset
from verl.utils.model import compute_position_id_with_mask

GUIDED_TENSOR_KEYS = ("guided_input_ids", "guided_attention_mask", "guided_position_ids", "has_guided")
GUIDED_NON_TENSOR_KEYS = ("guided_raw_prompt_ids",)


def render(tokenizer, messages) -> str:
    """Chat template with the generation prompt and the thinking block disabled (same as feedback_state.memory_generator.render_prompt)."""
    messages = [dict(m) for m in messages]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def sigma_collate_fn(data_list: list[dict]) -> dict:
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


class SigmaRLDataset(RLHFDataset):
    """RLHFDataset with (1) our prompt rendering and (2) an optional guided prompt per row.

    Row fields added on top of verl's: guided_input_ids / guided_attention_mask / guided_position_ids
    (left-padded to max_prompt_length, zeros when absent), guided_raw_prompt_ids and has_guided.
    """

    def __init__(self, data_files, tokenizer, config, processor=None):
        self.guided_key = config.get("guided_key", "guided_prompt")
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=config, processor=processor)

    def _read_files_and_tokenize(self):
        import datasets

        frames = [datasets.load_dataset("parquet", data_files=f)["train"] for f in self.data_files]
        self.dataframe = datasets.concatenate_datasets(frames)
        print(f"dataset len: {len(self.dataframe)}")
        if self.filter_overlong_prompts:
            tok, key, limit = self.tokenizer, self.prompt_key, self.max_prompt_length

            def fits(doc):
                return len(tok.encode(render(tok, doc[key]), add_special_tokens=False)) <= limit

            self.dataframe = self.dataframe.filter(fits, num_proc=self.num_workers, desc=f"Filtering prompts longer than {limit} tokens")
            print(f"filter dataset len: {len(self.dataframe)}")
        self._report_guided_budget()

    def _report_guided_budget(self, chunk: int = 512) -> None:
        """Nothing is truncated silently: say how many rows carry a guided prompt and how many lose it to the budget."""
        if self.guided_key not in self.dataframe.column_names:
            print("[sigma-data] no guided prompts in this file")
            return
        tok, limit = self.tokenizer, self.max_prompt_length
        total = with_guidance = over = 0
        longest = 0
        for start in range(0, len(self.dataframe), chunk):
            rows = self.dataframe[start : start + chunk][self.guided_key]
            texts = [render(tok, g) for g in rows if g is not None and len(g) > 0]
            total += len(rows)
            with_guidance += len(texts)
            if texts:
                lengths = [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]
                over += sum(n > limit for n in lengths)
                longest = max(longest, max(lengths))
        print(f"[sigma-data] rows={total} with_guided_prompt={with_guidance} over_budget({limit})={over} longest_guided_prompt={longest} tokens")

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
        raw = render(self.tokenizer, messages)
        ids, att, pos = self._encode(raw)
        row["input_ids"], row["attention_mask"], row["position_ids"] = ids, att, pos
        row["raw_prompt_ids"] = self._raw_ids(raw)
        guided = row.pop(self.guided_key, None)
        has_guided = 0
        guided_raw: list[int] = []
        if guided is not None and len(guided) > 0:
            graw = render(self.tokenizer, guided)
            guided_raw = self.tokenizer.encode(graw, add_special_tokens=False)
            if len(guided_raw) <= self.max_prompt_length:   # an over-long guided prompt just means "no guidance" for this row
                gids, gatt, gpos = self._encode(graw)
                has_guided = 1
        if not has_guided:
            gids, gatt, gpos = torch.full_like(ids, self.tokenizer.pad_token_id), torch.zeros_like(att), torch.zeros_like(pos)
            guided_raw = []
        row["guided_input_ids"], row["guided_attention_mask"], row["guided_position_ids"] = gids, gatt, gpos
        row["guided_raw_prompt_ids"] = guided_raw
        row["has_guided"] = torch.tensor(has_guided, dtype=torch.long)
        if self.return_raw_chat:
            row["raw_prompt"] = messages
        if self.return_full_prompt:
            row["full_prompts"] = raw
        extra = row.get("extra_info") or {}
        row["index"] = extra.get("index", item)
        row["tools_kwargs"] = {}
        return row


class SigmaSFTDataset(SFTDataset):
    """Single-turn SFT rows: ``prompt`` (chat messages, or an already rendered string) and ``response`` (text).

    Loss on the response tokens only (verl's mask convention); prompts rendered like the RL prompts.
    Used through ``data.custom_cls`` of verl.trainer.fsdp_sft_trainer.
    """

    def _read_files_and_tokenize(self):
        import pandas as pd

        frames = [pd.read_parquet(f) for f in self.parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        pk, rk = self.prompt_key[0], self.response_key[0]
        self.prompts = [self._as_prompt(p) for p in self.dataframe[pk].tolist()]
        self.responses = [str(r) for r in self.dataframe[rk].tolist()]

    @staticmethod
    def _as_prompt(p):
        if isinstance(p, str):
            return p
        return [dict(m) for m in list(p)]

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        prompt, response = self.prompts[item], self.responses[item]
        prompt_str = prompt if isinstance(prompt, str) else render(tokenizer, prompt)
        response_str = response + tokenizer.eos_token
        p = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        r = tokenizer(response_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids, prompt_att = p["input_ids"][0], p["attention_mask"][0]
        response_ids, response_att = r["input_ids"][0], r["attention_mask"][0]
        prompt_length, response_length = prompt_ids.shape[0], response_ids.shape[0]
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_att, response_att), dim=-1)
        n = input_ids.shape[0]
        if n < self.max_length:
            input_ids = torch.cat((input_ids, torch.full((self.max_length - n,), tokenizer.pad_token_id, dtype=input_ids.dtype)))
            attention_mask = torch.cat((attention_mask, torch.zeros(self.max_length - n, dtype=attention_mask.dtype)))
        elif n > self.max_length:
            if self.truncation == "right":
                input_ids, attention_mask = input_ids[: self.max_length], attention_mask[: self.max_length]
            elif self.truncation == "left":
                input_ids, attention_mask = input_ids[-self.max_length:], attention_mask[-self.max_length:]
            else:
                raise NotImplementedError(f"sequence length {n} > max_length {self.max_length} (truncation={self.truncation})")
        position_ids = compute_position_id_with_mask(attention_mask)
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0
        return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids, "loss_mask": loss_mask}
