"""Deterministic, fixed peer-position orders for the controlled swap experiment.

This is NOT random per-example shuffling. We present peers to the aggregator in a
single, fixed slot order chosen per split (``orig`` or ``swap``) so we can ask a
precise question: does the memory state ``S`` store trust by anonymous slot/source
position, or by explicit peer identity — and can identity-aware ``S`` recover when
the trusted peer appears in a different slot?

Conventions (single source of truth)
------------------------------------
* **Canonical order**: ``peer_id = index into sorted(peer_responses)`` (padded to
  ``num_peers``). The persistent state ``S[:, peer_id]`` is indexed this way when
  ``state_indexing="peer_id"`` and is NEVER permuted in storage.
* **Order** ``perm``: ``perm[slot] = peer_id`` occupying that slot (``slot_to_peer``).
  ``apply_perm(canonical_seq, perm)[slot] = canonical_seq[perm[slot]]`` maps a
  canonical-ordered sequence (responses, names, targets) into slot order.
* ``invert_perm(perm)[peer_id] = slot`` (``peer_to_slot``).

Fixed orders (peer_0=gemma, peer_1=phi, peer_2=qwen-coder):
    orig = [0, 1, 2]  -> slot0=gemma, slot1=phi,   slot2=qwen-coder
    swap = [1, 0, 2]  -> slot0=phi,   slot1=gemma, slot2=qwen-coder   (swaps the first two)
"""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

# Named fixed orders. ``swap`` exchanges the first two peers (gemma<->phi) and
# leaves the rest in place, so it generalises to >3 peers / add-remove peers.
# ``random`` is a LOCAL EXTENSION (not in the teacher's upstream): a per-example
# independent permutation, seeded by the record index for reproducibility.
ORDER_NAMES = ("orig", "swap", "random")


def identity_perm(n: int) -> list[int]:
    return list(range(n))


def random_order(num_peers: int, seed: int) -> list[int]:
    """Reproducible per-example random slot->peer_id permutation (seeded)."""
    perm = list(range(num_peers))
    random.Random(seed).shuffle(perm)
    return perm


def stable_seed(record_id: Any) -> int:
    """Deterministic, process-independent seed from a record id.

    Python's builtin hash() is randomized per process (PYTHONHASHSEED), which would
    desync train vs eval. Use a fixed hash so the same record yields the same
    per-example permutation in any run.
    """
    import hashlib

    return int(hashlib.md5(str(record_id).encode()).hexdigest()[:8], 16)


def named_order(name: str, num_peers: int, seed: int | None = None) -> list[int]:
    """Return the slot->peer_id order for ``name`` ("orig" | "swap" | "random").

    ``orig``/``swap`` are deterministic and ignore ``seed``. ``random`` requires a
    ``seed`` (per-example, e.g. the record index) and returns a reproducible
    permutation; passing no seed for ``random`` falls back to seed 0.
    """
    perm = list(range(num_peers))
    name = str(name).lower()
    if name == "orig":
        return perm
    if name == "swap":
        if num_peers >= 2:
            perm[0], perm[1] = perm[1], perm[0]  # slot0<->slot1 (phi<->gemma)
        return perm
    if name == "random":
        return random_order(num_peers, int(seed or 0))
    raise ValueError(f"Unknown peer order {name!r}; expected one of {ORDER_NAMES}")


def invert_perm(perm: Sequence[int]) -> list[int]:
    """peer_to_slot: inverse[peer_id] = slot holding that peer."""
    inverse = [0] * len(perm)
    for slot, peer_id in enumerate(perm):
        inverse[peer_id] = slot
    return inverse


def apply_perm(seq: Sequence[Any], perm: Sequence[int]) -> list[Any]:
    """Slot-order view of a canonical-ordered sequence: out[slot] = seq[perm[slot]]."""
    return [seq[p] for p in perm]


# ---------------------------------------------------------------------------
# Canonical peer view of a record + short identity names
# ---------------------------------------------------------------------------

_SHORT_NAME_TOKENS = ("gemma", "phi", "qwen", "llama", "bitcpm")


def short_peer_name(model_name: str) -> str:
    """Map a full model id to a short label (gemma / phi / qwen / ...).

    Falls back to the raw name when no known token matches, so logging is robust
    when peers are added or swapped out.
    """
    lowered = str(model_name).lower()
    for token in _SHORT_NAME_TOKENS:
        if token in lowered:
            return token
    return str(model_name)


def canonical_peer_view(record: dict[str, Any], num_peers: int, setting: str = "A") -> dict[str, Any]:
    """Canonical (sorted, padded) peer keys / names / short_names / responses + count.

    Robust to adding/removing peers: ``num_peers`` is the configured slot count;
    real peers are the sorted keys present, padded with inert ``__pad_*`` entries.
    The canonical ``peer_id`` is the index into this list and is the stable identity
    used for the per-peer state when ``state_indexing="peer_id"``.
    """
    if setting.upper() == "B":
        responses = dict(record.get("peer_step_responses", {}))
    else:
        responses = dict(record.get("peer_responses", {}))
    meta = dict(record.get("peer_metadata", {}))
    keys = sorted(responses)[:num_peers]
    real = len(keys)
    names = [str(dict(meta.get(k, {})).get("model") or k) for k in keys]
    texts = [str(responses[k]) for k in keys]
    while len(keys) < num_peers:
        pad = f"__pad_{len(keys)}"
        keys.append(pad)
        names.append(pad)
        texts.append("")
    short = [short_peer_name(n) for n in names]
    return {"keys": keys, "names": names, "short_names": short, "texts": texts, "real": real}


def peer_block_text(context: str, response: str, name: str, *, include_identity: bool) -> str:
    """Per-peer encoder text.

    ``id`` mode (include_identity=True) embeds the peer's model identity INTO its
    block, so the identity string moves with the peer (not the slot). ``anon`` mode
    leaks no identity — the only persistent source signal is the slot position.
    """
    if include_identity:
        return f"question:\n{context}\n\nPeer [{name}] response:\n{response}"
    return f"question:\n{context}\n\nPeer response:\n{response}"


# ---------------------------------------------------------------------------
# Selection diagnostics (pure; used by the evaluator)
# ---------------------------------------------------------------------------

def entropy(counts: Sequence[float]) -> float:
    """Shannon entropy (nats) of a count/probability vector; 0 for degenerate."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        p = c / total
        if p > 0:
            ent -= p * math.log(p)
    return ent


def normalized_entropy(counts: Sequence[float]) -> float:
    """Entropy / log(K) in [0,1]; 1 = uniform, 0 = always the same index."""
    k = len(counts)
    if k <= 1:
        return 0.0
    return entropy(counts) / math.log(k)


def position_bias_report(slot_counts: Sequence[float], peer_counts: Sequence[float]) -> dict[str, Any]:
    """Slot- vs peer-selection concentration.

    A slot/source-trust model (e.g. always slot0) -> max_slot_pick_rate ~ 1.0. An
    identity-trust model under a swapped test order -> slot picks track the trusted
    peer's new slot while peer picks stay concentrated on that peer.
    """
    slot_total = float(sum(slot_counts)) or 1.0
    peer_total = float(sum(peer_counts)) or 1.0
    slot_rates = [c / slot_total for c in slot_counts]
    peer_rates = [c / peer_total for c in peer_counts]
    return {
        "slot_pick_rates": slot_rates,
        "peer_pick_rates": peer_rates,
        "slot_selection_entropy": entropy(slot_counts),
        "slot_selection_entropy_normalized": normalized_entropy(slot_counts),
        "peer_selection_entropy": entropy(peer_counts),
        "peer_selection_entropy_normalized": normalized_entropy(peer_counts),
        "max_slot_pick_rate": max(slot_rates) if slot_rates else 0.0,
        "max_peer_pick_rate": max(peer_rates) if peer_rates else 0.0,
    }
