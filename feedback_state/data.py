from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from feedback_state.permutations import (
    apply_perm,
    canonical_peer_view,
    named_order,
    peer_block_text,
    stable_seed,
)
from feedback_state.rollout import choose_committed_step
from feedback_state.tasks import peer_target_value, task_type_of
from feedback_state.utils import normalize_answer

# A record is a *counterfactual diagnostic* if it was built/labelled as one.
DIAGNOSTIC_COUNTERFACTUAL_TYPES = ("strong_wrong_weak_correct", "strong_correct_weak_wrong")


def is_counterfactual_record(record: dict[str, Any]) -> bool:
    """True for diagnostic counterfactual examples (natural or synthetic)."""
    return (
        bool(record.get("counter_trust"))
        or bool(record.get("synthetic_counterfactual"))
        or str(record.get("counterfactual_type") or "") in DIAGNOSTIC_COUNTERFACTUAL_TYPES
    )


def record_domain(record: dict[str, Any]) -> str:
    return str(record.get("domain") or ("rag" if task_type_of(record) == "rag" else task_type_of(record)))


def filter_records(
    records: list[dict[str, Any]],
    *,
    include_natural_counterfactuals: bool = True,
    include_synthetic: bool = True,
    domains: list[str] | None = None,
    counterfactual_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Optionally drop counterfactual / synthetic / off-domain / off-type records.

    Makes counterfactual training and evaluation OPTIONAL at run time (no rebuild).
    NATURAL diagnostic counterfactuals are ON by default; the SYNTHETIC ones are a
    separate, independent switch:
      * ``include_natural_counterfactuals=False`` -> drop naturally-occurring
        diagnostic counterfactuals (rare; explicit opt-out).
      * ``include_synthetic=False``               -> drop only the synthesized examples.
      * ``domains=["math"]``                       -> keep only math (or rag) examples.
      * ``counterfactual_types=[...]``             -> keep only these type labels.
    Defaults are a no-op, so existing runs (and natural diagnostics) are unaffected.
    """
    out = []
    domain_set = {str(d).lower() for d in domains} if domains else None
    type_set = {str(t) for t in counterfactual_types} if counterfactual_types else None
    for r in records:
        synth = bool(r.get("synthetic_counterfactual"))
        natural_diag = is_counterfactual_record(r) and not synth
        if not include_synthetic and synth:
            continue
        if not include_natural_counterfactuals and natural_diag:
            continue
        if domain_set is not None and record_domain(r).lower() not in domain_set:
            continue
        if type_set is not None and str(r.get("counterfactual_type") or "") not in type_set:
            continue
        out.append(r)
    return out


def counterfactual_filter_kwargs(cfg: dict[str, Any], split: str) -> dict[str, Any]:
    """Resolve filter_records kwargs for a split ('train'|'eval') from config.

    Natural diagnostic counterfactuals are ON by default. ``use_counterfactuals_<split>``
    is the SYNTHETIC master (false -> synthetic off, natural kept);
    ``use_synthetic_counterfactuals_in_<split>`` overrides it;
    ``include_natural_counterfactuals_<split>: false`` is the explicit natural opt-out.
    """
    def b(value, default):
        if value is None:
            return default
        return value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes", "on"}

    master = cfg.get(f"use_counterfactuals_{split}")
    return {
        "include_natural_counterfactuals": b(cfg.get(f"include_natural_counterfactuals_{split}"), True),
        "include_synthetic": b(cfg.get(f"use_synthetic_counterfactuals_in_{split}"), b(master, True)),
        "domains": cfg.get("domains"),
        "counterfactual_types": cfg.get("counterfactual_filter_types"),
    }


def setting_a_user_prompt(problem: str, peer_responses: dict[str, str]) -> str:
    parts = [f"Problem:\n{problem.strip()}"]
    for index, key in enumerate(sorted(peer_responses), start=1):
        parts.append(f"Peer {index}:\n{str(peer_responses[key]).strip()}")
    parts.append("Use the peer responses and provide the final answer.")
    return "\n\n".join(parts)


def setting_a_solo_prompt(problem: str) -> str:
    """Problem-only prompt (no peers) for the solo-LoRA ablation.

    Matches the direct-solve prompt style (eval_direct_math.build_prompt) so the
    solo model learns to solve from the problem alone — the control that isolates
    how much of any gain comes from peer collaboration vs. just learning to reason.
    """
    return (
        "Solve the following math problem step by step. "
        "Put your final answer within \\boxed{}.\n\n"
        f"Problem:\n{problem.strip()}"
    )


def setting_a_user_prompt_with_identity(
    problem: str, peer_responses: dict[str, str], peer_metadata: dict[str, Any]
) -> str:
    """Setting A prompt that labels each peer with its producing model identity, so
    a LoRA baseline can learn implicit per-agent trust (LoRA-identity baseline)."""
    parts = [f"Problem:\n{problem.strip()}"]
    for index, key in enumerate(sorted(peer_responses), start=1):
        identity = str(dict(peer_metadata.get(key, {})).get("model") or f"peer_{index - 1}")
        parts.append(f"Peer {index} [{identity}]:\n{str(peer_responses[key]).strip()}")
    parts.append(
        "Each peer is labeled with the model that produced it. Use the peer "
        "responses and their identities to provide the final answer."
    )
    return "\n\n".join(parts)


def peer_correctness(
    record: dict[str, Any],
    peer_keys: list[str],
    peer_responses: dict[str, str],
    *,
    setting: str = "A",
    score_map: dict[str, float] | None = None,
) -> list[float]:
    """Per-peer selection target in [0, 1], aligned to ``peer_keys`` order.

    Setting A: task-aware soft correctness via the task registry —
      * math: 1.0 iff math-equal to gold;
      * rag:  token-F1 vs gold (+aliases);
      * code: precomputed pass@1 from ``record["peer_correct"]``.
    The task is read from ``record["task_type"]`` (default "math"), so a single
    JSONL may freely mix task types.

    Setting B: the peer's forward-rollout success rate z (from ``score_map``).
    """
    score_map = score_map or {}
    if setting.upper() == "B":
        return [float(score_map.get(key, 0.0)) for key in peer_keys]
    return [peer_target_value(record, key, str(peer_responses.get(key, ""))) for key in peer_keys]


def setting_b_user_prompt(context: str, peer_responses: dict[str, str]) -> str:
    parts = [context.strip()]
    for index, key in enumerate(sorted(peer_responses), start=1):
        parts.append(f"Peer step {index}:\n{str(peer_responses[key]).strip()}")
    parts.append("Choose or write the next committed solution step.")
    return "\n\n".join(parts)


class JsonlDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: list[dict[str, Any]] = []
        with self.path.open() as handle:
            for line in handle:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class SettingBStepDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        base = JsonlDataset(path)
        self.records: list[dict[str, Any]] = []
        for problem_record in base.records:
            previous: list[str] = []
            for step in problem_record.get("steps", []):
                target = choose_committed_step(step)
                self.records.append(
                    {
                        "id": f"{problem_record.get('id')}:step{step.get('t', len(previous))}",
                        "problem": problem_record.get("problem", ""),
                        "answer": problem_record.get("answer", ""),
                        "context": step.get("context", ""),
                        "peer_step_responses": step.get("peer_step_responses", {}),
                        "rollout_scores": step.get("rollout_scores", {}),
                        "target_step": target,
                    }
                )
                if target:
                    previous.append(target)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _render_chat(tokenizer, user_prompt: str, assistant_text: str | None) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_prompt}]
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})
        kwargs = dict(
            tokenize=True,
            add_generation_prompt=assistant_text is None,
            return_tensors=None,
        )
        try:
            rendered = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except (TypeError, ValueError):
            try:
                rendered = tokenizer.apply_chat_template(messages, **kwargs)
            except ValueError:
                rendered = None
        if rendered is None:
            text = user_prompt if assistant_text is None else f"{user_prompt}\n\nAnswer:\n{assistant_text}"
            return list(tokenizer.encode(text, add_special_tokens=True))
        return _to_token_ids(rendered)
    text = user_prompt if assistant_text is None else f"{user_prompt}\n\nAnswer:\n{assistant_text}"
    return list(tokenizer.encode(text, add_special_tokens=True))


def _to_token_ids(rendered) -> list[int]:
    """Normalize apply_chat_template output to a flat list[int].

    transformers >= 5.0 returns a ``BatchEncoding`` (holding ``tokenizers.Encoding``
    objects) from ``apply_chat_template(tokenize=True, return_tensors=None)`` instead
    of a plain ``list[int]``. Handle tensors, BatchEncoding, Encoding, and nested
    single-batch lists uniformly.
    """
    if torch.is_tensor(rendered):
        return rendered.squeeze(0).tolist()
    # BatchEncoding / dict-like with input_ids
    if hasattr(rendered, "input_ids"):
        return _to_token_ids(rendered.input_ids)
    if isinstance(rendered, dict) and "input_ids" in rendered:
        return _to_token_ids(rendered["input_ids"])
    # tokenizers.Encoding object
    if hasattr(rendered, "ids"):
        return list(rendered.ids)
    seq = list(rendered)
    if not seq:
        return []
    first = seq[0]
    # Already a flat list of ints
    if isinstance(first, int):
        return seq
    # Single-batch nesting: [[...]] or [Encoding(...)]
    if len(seq) == 1:
        return _to_token_ids(first)
    if hasattr(first, "ids"):
        # list of Encoding -> take the first (single conversation)
        return list(first.ids)
    return _to_token_ids(first)


def tokenize_prompt_answer(
    tokenizer,
    user_prompt: str,
    answer: str,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_ids = _render_chat(tokenizer, user_prompt, None)
    full_ids = _render_chat(tokenizer, user_prompt, answer)
    if len(full_ids) < len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        answer_ids = tokenizer.encode(str(answer), add_special_tokens=False)
        full_ids = (prompt_ids + answer_ids)[-max_length:]
        labels = [-100] * max(0, len(full_ids) - len(answer_ids)) + answer_ids[-len(full_ids) :]
    else:
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        if len(full_ids) > max_length:
            start = len(full_ids) - max_length
            full_ids = full_ids[start:]
            labels = labels[start:]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def tokenize_text(tokenizer, text: str, *, max_length: int) -> dict[str, list[int]]:
    tokenized = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    return {
        "input_ids": list(tokenized["input_ids"]),
        "attention_mask": list(tokenized["attention_mask"]),
    }


class FeedbackDataCollator:
    def __init__(
        self,
        tokenizer,
        *,
        setting: str,
        num_peers: int,
        max_length: int = 1024,
        max_peer_length: int = 512,
        max_answer_length: int = 128,
        identity_prompt: bool = False,
        solo: bool = False,
        task: str = "selection",
        peer_order: str = "orig",
        identity_mode: str = "anon",
    ) -> None:
        self.tokenizer = tokenizer
        self.setting = setting.upper()
        self.num_peers = int(num_peers)
        self.max_length = int(max_length)
        self.identity_prompt = bool(identity_prompt)
        self.solo = bool(solo)
        # "selection" (proposed): central model picks the most trusted peer.
        # "generate" (legacy ablation): central model generates conditioned on
        # the peer-memory prefix.
        self.task = str(task).lower()
        # FIXED deterministic slot order for ALL training examples (no randomness):
        #   "orig" -> slot0=gemma, slot1=phi, slot2=ministral
        #   "swap" -> slot0=phi,   slot1=gemma, slot2=ministral
        self.peer_order = str(peer_order).lower()
        # "id" embeds each peer's model identity into its encoded block (identity
        # moves with the peer); "anon" leaks no identity (only slot is observable).
        self.identity_mode = str(identity_mode).lower()
        self.max_peer_length = int(max_peer_length)
        self.max_answer_length = int(max_answer_length)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[str]]:
        encoded_main = []
        encoded_answer = []
        encoded_peers: list[list[dict[str, list[int]]]] = []
        ids: list[str] = []
        answers: list[str] = []
        rollout_scores: list[list[float]] = []
        peer_target: list[list[float]] = []
        slot_to_peer: list[list[int]] = []
        generate = self.task == "generate"
        for feature in features:
            ids.append(str(feature.get("id", "")))
            answer = str(feature.get("answer", ""))
            answers.append(normalize_answer(answer))
            if self.setting == "B":
                peer_responses = dict(feature.get("peer_step_responses", {}))
                prompt = setting_b_user_prompt(str(feature.get("context", "")), peer_responses)
                target = str(feature.get("target_step", "") or answer)
                score_map = dict(feature.get("rollout_scores", {}))
            else:
                peer_responses = dict(feature.get("peer_responses", {}))
                if self.solo:
                    # Solo ablation: problem only, peers omitted from the prompt.
                    prompt = setting_a_solo_prompt(str(feature.get("problem", "")))
                elif self.identity_prompt:
                    prompt = setting_a_user_prompt_with_identity(
                        str(feature.get("problem", "")),
                        peer_responses,
                        dict(feature.get("peer_metadata", {})),
                    )
                else:
                    prompt = setting_a_user_prompt(str(feature.get("problem", "")), peer_responses)
                # Prefer a reasoning chain (R1 solution) as the SFT target when present;
                # fall back to the bare answer (keeps Ours / Setting B unchanged).
                target = str(feature.get("solution") or answer)
                score_map = {}
            if generate:
                encoded_main.append(
                    tokenize_prompt_answer(
                        self.tokenizer,
                        prompt,
                        target,
                        max_length=self.max_length,
                    )
                )
                encoded_answer.append(
                    tokenize_text(self.tokenizer, answer, max_length=self.max_answer_length)
                )
            # Canonical (sorted, padded) peer view — state is keyed by this peer_id.
            view = canonical_peer_view(feature, self.num_peers, setting=self.setting)
            canonical_keys = view["keys"]
            canonical_names = view["names"]
            canonical_texts = view["texts"]
            responses_by_key = {k: t for k, t in zip(canonical_keys, canonical_texts)}
            # h_j = Enc(x, r_j) for Setting A; h_{j,t} = Enc(g_t, r_{j,t}) for Setting B.
            if self.setting == "B":
                encoder_context = str(feature.get("context", "")) or str(feature.get("problem", ""))
            else:
                encoder_context = str(feature.get("problem", ""))
            # Canonical per-peer correctness target and rollout scores.
            canonical_target = peer_correctness(
                feature, canonical_keys, responses_by_key,
                setting=self.setting, score_map=score_map,
            )
            canonical_rollout = [float(score_map.get(k, 1.0)) for k in canonical_keys]
            # FIXED deterministic slot order (slot k <- canonical peer perm[k]); the
            # SAME order is applied to every training example (no per-example random).
            # FIXED deterministic slot order (slot k <- canonical peer perm[k]); the
            # SAME order is applied to every training example for orig/swap. For the
            # "random" local extension, the perm is per-example, seeded by the stable
            # record id so it is reproducible across runs.
            seed = stable_seed(feature.get("id", "")) if self.peer_order == "random" else None
            perm = named_order(self.peer_order, self.num_peers, seed=seed)
            include_identity = self.identity_mode == "id"
            slot_texts = apply_perm(canonical_texts, perm)
            slot_names = apply_perm(canonical_names, perm)
            encoded_peers.append(
                [
                    tokenize_text(
                        self.tokenizer,
                        peer_block_text(
                            encoder_context, slot_texts[k], slot_names[k],
                            include_identity=include_identity,
                        ),
                        max_length=self.max_peer_length,
                    )
                    for k in range(self.num_peers)
                ]
            )
            rollout_scores.append(apply_perm(canonical_rollout, perm))
            peer_target.append(apply_perm(canonical_target, perm))
            slot_to_peer.append(list(perm))
        batch = {
            **self._pad_peers(encoded_peers),
            "ids": ids,
            "answers": answers,
            "rollout_scores": torch.tensor(rollout_scores, dtype=torch.float32),
            "peer_target": torch.tensor(peer_target, dtype=torch.float32),
            "slot_to_peer": torch.tensor(slot_to_peer, dtype=torch.long),
        }
        if generate:
            batch.update(self._pad_main(encoded_main))
            batch.update(self._pad_answer(encoded_answer))
        return batch

    def _pad_main(self, encoded: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        pad = self.tokenizer.pad_token_id
        max_len = max(len(item["input_ids"]) for item in encoded)
        rows = []
        masks = []
        labels = []
        for item in encoded:
            pad_len = max_len - len(item["input_ids"])
            rows.append(item["input_ids"] + [pad] * pad_len)
            masks.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _pad_answer(self, encoded: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        ids, mask = self._pad_flat(encoded)
        return {"answer_input_ids": ids, "answer_attention_mask": mask}

    def _pad_peers(self, encoded: list[list[dict[str, list[int]]]]) -> dict[str, torch.Tensor]:
        flat = [peer for row in encoded for peer in row]
        ids, mask = self._pad_flat(flat)
        batch = len(encoded)
        return {
            "peer_input_ids": ids.view(batch, self.num_peers, -1),
            "peer_attention_mask": mask.view(batch, self.num_peers, -1),
        }

    def _pad_flat(self, encoded: list[dict[str, list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
        pad = self.tokenizer.pad_token_id
        max_len = max(len(item["input_ids"]) for item in encoded)
        rows = []
        masks = []
        for item in encoded:
            pad_len = max_len - len(item["input_ids"])
            rows.append(item["input_ids"] + [pad] * pad_len)
            masks.append(item["attention_mask"] + [0] * pad_len)
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.long)
