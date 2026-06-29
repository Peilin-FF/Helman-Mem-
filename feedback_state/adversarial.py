from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from feedback_state.utils import perturb_numeric_answer


def make_numeric_perturbation_response(answer: str, rng: random.Random | None = None) -> str | None:
    wrong = perturb_numeric_answer(answer, rng)
    if wrong is None:
        return None
    return f"A plausible but incorrect solution gives final answer {wrong}."


def corrupt_peer_responses(
    record: dict[str, Any],
    *,
    adversarial_peers: list[int] | tuple[int, ...],
    adversarial_rate: float,
    mode: str = "numeric_perturb",
    rng: random.Random | None = None,
) -> dict[str, Any]:
    rng = rng or random.Random()
    if adversarial_rate <= 0.0:
        return deepcopy(record)
    output = deepcopy(record)
    peers = dict(output.get("peer_responses", {}))
    metadata = dict(output.get("peer_metadata", {}))
    peer_keys = sorted(peers)
    for peer_index in adversarial_peers:
        if peer_index < 0 or peer_index >= len(peer_keys):
            continue
        if rng.random() > adversarial_rate:
            continue
        key = peer_keys[peer_index]
        replacement = None
        if mode in {"numeric_perturb", "plausible_wrong"}:
            replacement = make_numeric_perturbation_response(str(output.get("answer", "")), rng)
        if replacement is None and mode == "swap_peer" and len(peer_keys) > 1:
            choices = [other for other in peer_keys if other != key]
            replacement = peers[rng.choice(choices)]
        if replacement is None:
            replacement = "This solution is intentionally incorrect, with a final answer that differs from the reference."
        peers[key] = replacement
        peer_meta = dict(metadata.get(key, {}))
        peer_meta.update({"is_adversarial": True, "adversarial_mode": mode, "known_incorrect": True})
        metadata[key] = peer_meta
    output["peer_responses"] = peers
    output["peer_metadata"] = metadata
    return output
