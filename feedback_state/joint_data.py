"""Dataloader for the JOINT-input AR selector.

Reuses the existing canonical peer view, fixed peer orders, and per-peer correctness
evaluator; only the *packing* is new (one joint prompt instead of K per-peer encodings).

AR: input_ids = [joint prompt + 'Peer j'], labels supervise the 'Peer j' span (prompt
masked to -100). Slot order = train/test order; the target is SLOT-indexed; eval maps
the selected slot back to the stable peer id via the same convention as the rest of the
repo.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from feedback_state.data import peer_correctness
from feedback_state.joint_write import build_short_answers
from feedback_state.joint_prompt import (
    ar_target_text,
    candidate_peer_strings,
    canonical_target_peer_id,
    peer_response_char_spans,
)
from feedback_state.permutations import apply_perm, canonical_peer_view, invert_perm, named_order

VARIANT_AR = "ar_shared_state_selector"
VARIANT_BCE = "joint_bce_shared_state_selector"


def candidate_token_ids(tokenizer, num_peers: int) -> list[list[int]]:
    """Token ids for ' Peer 0' .. ' Peer {N-1}' continuations (AR eval scoring)."""
    return [tokenizer.encode(" " + s, add_special_tokens=False) for s in candidate_peer_strings(num_peers)]


def yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Token ids for the shared Yes/No candidate-utility continuations."""
    return (
        tokenizer.encode(" Yes", add_special_tokens=False),
        tokenizer.encode(" No", add_special_tokens=False),
    )


def char_to_token_spans(offsets: list[tuple[int, int]], char_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Map [char_start, char_end) response spans to [tok_start, tok_end) spans."""
    out = []
    for cstart, cend in char_spans:
        if cstart < 0:
            out.append((0, 0))
            continue
        tok_start, tok_end = -1, -1
        for ti, (a, b) in enumerate(offsets):
            if a == b:
                continue
            if b > cstart and a < cend:
                if tok_start < 0:
                    tok_start = ti
                tok_end = ti + 1
        out.append((tok_start if tok_start >= 0 else 0, tok_end if tok_end >= 0 else 0))
    return out


class JointInputCollator:
    def __init__(
        self,
        tokenizer,
        *,
        num_peers: int,
        variant: str = VARIANT_AR,
        peer_order: str = "orig",
        identity_mode: str = "id",
        setting: str = "A",
        max_length: int = 1536,
        include_context: bool = True,
    ) -> None:
        self.tok = tokenizer
        self.num_peers = int(num_peers)
        self.variant = str(variant)
        self.peer_order = str(peer_order)
        self.identity_mode = str(identity_mode).lower()
        # id      -> real model names in the prompt (learns content prior — CF shortcut)
        # anon    -> no [..] label at all (no anchor for feedback)
        # anon_id -> a PER-EXAMPLE RANDOM placeholder label per peer: gives feedback an
        #            identity anchor to read, but the label carries no model-prior and is
        #            re-shuffled each example, so the model can't memorize "gemma is strong".
        self.include_identity = self.identity_mode in ("id", "anon_id", "id_short", "id_stable")
        self.setting = setting.upper()
        self.max_length = int(max_length)
        self.include_context = bool(include_context)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def record_views(self, feature: dict[str, Any]):
        view = canonical_peer_view(feature, self.num_peers, setting=self.setting)
        keys, names, texts, real = view["keys"], view["names"], view["texts"], view["real"]
        # train_order=random needs a PER-EXAMPLE seed, else named_order falls back to
        # seed 0 and every example gets the SAME permutation (not random at all). Use
        # a stable hash of the record id so the shuffle is per-example yet reproducible.
        seed = abs(hash(str(feature.get("id", "")))) % (2**31) if self.peer_order == "random" else None
        perm = named_order(self.peer_order, self.num_peers, seed=seed)
        score_map = dict(feature.get("rollout_scores", {})) if self.setting == "B" else {}
        responses_by_key = {keys[i]: texts[i] for i in range(self.num_peers)}
        target = peer_correctness(feature, keys, responses_by_key, setting=self.setting, score_map=score_map)
        correctness_by_peer = [t > 0.5 for t in target]
        question = str(feature.get("problem", feature.get("question", "")))
        context = str(feature.get("retrieved_context", feature.get("context", ""))) if self.include_context else ""
        slot_names = apply_perm(names, perm)
        # id_short: strip the "org/" prefix from the real model name (e.g.
        # "google/gemma-3-4b-it" -> "gemma-3-4b-it"). Tests whether the anon_id->id
        # drop is driven by the unfamiliar slash/org token shape rather than identity.
        if self.identity_mode == "id_short":
            slot_names = [n.split("/")[-1] if isinstance(n, str) else n for n in slot_names]
        # anon_id: replace real model names with per-example RANDOM placeholder labels.
        # Same example -> consistent label per slot (prompt & feedback write agree);
        # different examples -> a fresh random label assignment, so no model-prior can be
        # memorized. per-peer state is still indexed by the real key (slot_peer_keys),
        # so the feedback anchor is intact and decoupled from the visible label.
        if self.identity_mode == "anon_id":
            import random as _r
            _alpha = ["A", "B", "C", "D", "E", "F", "G", "H"]
            rseed = abs(hash(("anonid", str(feature.get("id", ""))))) % (2**31)
            labels = list(_alpha[: self.num_peers])
            _r.Random(rseed).shuffle(labels)
            slot_names = [f"Model-{labels[s]}" for s in range(len(slot_names))]
        slot_texts = apply_perm(texts, perm)
        slot_peer_keys = apply_perm(list(keys), perm)
        # id_stable: fixed placeholder label keyed to the REAL peer (peer_0 -> Model-A,
        # peer_1 -> Model-B, ...), constant across ALL examples. Gives an identity slot
        # that is as STABLE as real-name test (eliminates the anon_id train/test
        # stability mismatch that hurt 8B), while staying a semantics-free placeholder so
        # no real model-name prior is memorized. Label follows the peer key, not the slot.
        if self.identity_mode == "id_stable":
            _alpha = ["A", "B", "C", "D", "E", "F", "G", "H"]
            def _stable_label(k):
                try:
                    idx = int(str(k).split("_")[1])
                except (IndexError, ValueError):
                    idx = 0
                return f"Model-{_alpha[idx % len(_alpha)]}"
            slot_names = [_stable_label(k) for k in slot_peer_keys]
        correctness_by_slot = apply_perm(correctness_by_peer, perm)
        return dict(perm=perm, real=real, question=question, context=context,
                    slot_names=slot_names, slot_texts=slot_texts, slot_peer_keys=slot_peer_keys,
                    correctness_by_peer=correctness_by_peer, correctness_by_slot=correctness_by_slot,
                    target_floats=target)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        rows, labels_rows, span_rows, peer_targets, slot_to_peer = [], [], [], [], []
        write_meta: list[dict[str, Any]] = []  # per-example data for the separate WRITE pass
        for feature in features:
            v = self.record_views(feature)
            prompt, char_spans = peer_response_char_spans(
                v["question"], v["slot_names"], v["slot_texts"],
                context=v["context"] or None, include_identity=self.include_identity, real=v["real"],
            )
            if self.variant == VARIANT_AR:
                # AR target = the SLOT holding the canonical correct peer.
                tgt_peer = canonical_target_peer_id(v["correctness_by_peer"])
                tgt_slot = invert_perm(v["perm"])[tgt_peer] if tgt_peer is not None else 0
                tgt_ids = self.tok.encode(" " + ar_target_text([tgt_slot]), add_special_tokens=False)
                # Reserve room for the target so the joint prompt is truncated (tail
                # dropped, head kept) WITHOUT ever clipping the 'Peer j' target. The
                # prompt ends with 'Answer:', so the surviving structure is always
                # [Question ... peers ... Answer:] + [ Peer j].
                prompt_budget = max(1, self.max_length - len(tgt_ids))
                enc = self.tok(prompt, add_special_tokens=True,
                               truncation=True, max_length=prompt_budget)
                prompt_ids = list(enc["input_ids"])
                ids = prompt_ids + tgt_ids
                lbl = [-100] * len(prompt_ids) + tgt_ids
                rows.append(ids)
                labels_rows.append(lbl)
            elif self.variant == VARIANT_BCE:
                enc = self.tok(prompt, add_special_tokens=True, return_offsets_mapping=True,
                               truncation=True, max_length=self.max_length)
                rows.append(list(enc["input_ids"]))
                offsets = list(enc.get("offset_mapping", []))
                span_rows.append(char_to_token_spans(offsets, char_spans))
            else:
                raise ValueError(f"unknown joint selector variant: {self.variant}")
            peer_targets.append([1.0 if c else 0.0 for c in v["correctness_by_slot"]])
            slot_to_peer.append(list(v["perm"]))
            # Per-example data the trainer needs to build the separate write pass
            # (feedback prompt with per-peer correctness). The training "selected"
            # peer is the canonical correct target.
            tgt_peer = canonical_target_peer_id(v["correctness_by_peer"])
            tgt_slot = invert_perm(v["perm"])[tgt_peer] if tgt_peer is not None else None
            write_meta.append({
                "question": v["question"], "context": v["context"],
                "slot_names": v["slot_names"], "slot_texts": v["slot_texts"],
                "slot_peer_keys": v["slot_peer_keys"],
                "correctness_by_slot": [bool(c) for c in v["correctness_by_slot"]],
                "real": v["real"], "target_slot": tgt_slot,
                "short_answers": build_short_answers(
                    feature, v["slot_texts"], v["slot_peer_keys"], real=v["real"]),
            })

        batch: dict[str, Any] = {
            "peer_target": torch.tensor(peer_targets, dtype=torch.float32),
            "slot_to_peer": torch.tensor(slot_to_peer, dtype=torch.long),
            "write_meta": write_meta,
        }
        batch.update(self._pad(rows))
        if self.variant == VARIANT_BCE:
            P = self.num_peers
            spans = torch.zeros(len(span_rows), P, 2, dtype=torch.long)
            for b, sr in enumerate(span_rows):
                for p in range(min(P, len(sr))):
                    spans[b, p, 0], spans[b, p, 1] = sr[p][0], sr[p][1]
            batch["peer_spans"] = spans
        else:
            batch["labels"] = self._pad_labels(labels_rows, len(batch["input_ids"][0]))
        return batch

    def _pad(self, rows: list[list[int]]) -> dict[str, torch.Tensor]:
        pad = self.tok.pad_token_id
        width = max(len(r) for r in rows)
        ids = [r + [pad] * (width - len(r)) for r in rows]
        mask = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long)}

    def _pad_labels(self, rows: list[list[int]], width: int) -> torch.Tensor:
        return torch.tensor([r + [-100] * (width - len(r)) for r in rows], dtype=torch.long)
