"""Training-free OOD routing and reliability-weighted voting.

This module is intentionally independent from the residual-steering evaluator.  It
implements the decide-then-update algorithms in ``instruction.md`` without loading
or modifying the runtime ``M``/``G`` buffers stored in a Sigma checkpoint.  The
caller supplies the already-normalized competence direction ``phi`` produced by the
frozen encoder.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from feedback_state.tasks import (
    _mcqa_pred_label,
    _normalise_label,
    _normalize_bool_label,
    _shortqa_norm,
    boolqa_extract_answer,
    shortqa_extract_answer,
    task_type_of,
)


DEFAULT_RIDGE = 0.1
ROBUSTNESS_RIDGES = (0.05, 0.1, 0.5)
ROUTE_TIE_EPS = 1e-9
INVALID_ANSWER = "<invalid>"


@dataclass(frozen=True)
class RouteDecision:
    """A peer-routing decision made without mutating online state."""

    peer: int
    scores: np.ndarray
    tied: bool


@dataclass(frozen=True)
class FactorialVoteResult:
    """The four factorial vote arms and the exact weights they used."""

    answers: Mapping[str, str]
    weights: Mapping[str, np.ndarray]
    reliability: np.ndarray
    ridge: float
    clipped: bool


def _as_phi(phi: Any, rank: int) -> np.ndarray:
    out = np.asarray(phi, dtype=np.float64).reshape(-1)
    if out.shape != (int(rank),):
        raise ValueError(f"phi must have shape ({rank},), got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError("phi must contain only finite values")
    norm = float(np.linalg.norm(out))
    if norm == 0.0:
        raise ValueError("phi must be nonzero")
    if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"phi must be L2-normalized, got norm={norm:.8f}")
    return out


def _as_correctness(correctness: Sequence[float], num_peers: int) -> np.ndarray:
    out = np.asarray(correctness, dtype=np.float64).reshape(-1)
    if out.shape != (int(num_peers),):
        raise ValueError(
            f"correctness must have shape ({num_peers},), got {out.shape}"
        )
    if not np.isin(out, (-1.0, 1.0)).all():
        raise ValueError("correctness must contain only -1 or +1")
    return out


def canonical_answer(record: Mapping[str, Any], answer: Any) -> str:
    """Return the teacher-specified task-aware canonical answer label.

    An unparsable response is represented explicitly instead of being silently
    treated as a legitimate empty answer.  This preserves the historical OOD label
    protocol and makes parser failures auditable in the vote diagnostics.
    """

    rec = dict(record)
    task_type = task_type_of(rec)
    text = str(answer or "")
    if task_type == "mcqa":
        value = _normalise_label(_mcqa_pred_label(text, rec))
    elif task_type == "boolqa":
        value = _normalize_bool_label(boolqa_extract_answer(text))
    elif task_type == "shortqa":
        extracted = shortqa_extract_answer(text)
        value = _shortqa_option_label(rec, extracted) or _shortqa_norm(extracted)
    else:
        raise ValueError(
            "OOD answer voting supports mcqa, boolqa, and shortqa records; "
            f"got task_type={task_type!r}"
        )
    return value or INVALID_ANSWER


def canonical_gold_answer(record: Mapping[str, Any]) -> str:
    """Canonicalize the gold answer under the same frozen OOD protocol."""

    task_type = task_type_of(dict(record))
    gold = record.get("answer", "")
    if task_type == "mcqa":
        value = _normalise_label(gold)
    elif task_type == "boolqa":
        value = _normalize_bool_label(gold)
    elif task_type == "shortqa":
        value = _shortqa_option_label(record, gold) or _shortqa_norm(gold)
    else:
        raise ValueError(
            "OOD answer voting supports mcqa, boolqa, and shortqa records; "
            f"got task_type={task_type!r}"
        )
    return value or INVALID_ANSWER


def vote_is_correct(record: Mapping[str, Any], answer: str) -> bool:
    """Grade a canonical voted answer; invalid parses never count as correct."""

    gold = canonical_gold_answer(record)
    return answer != INVALID_ANSWER and gold != INVALID_ANSWER and answer == gold


def _shortqa_declared_labels(record: Mapping[str, Any]) -> list[str]:
    """Read BBH-style option labels without applying QA article removal.

    The historical short-answer normalizer maps the standalone label ``A`` to an
    empty string because it removes English articles.  Voting still needs ``(A)``
    to be the same option as ``A``.  This fixes grouping only; the evaluator keeps
    the frozen external correctness labels for scoring and memory writes.
    """

    labels = record.get("choice_labels") or []
    if labels:
        return [_normalise_label(label) for label in labels]
    problem = str(record.get("problem", ""))
    found = re.findall(r"\(([A-Z0-9])\)", problem)
    return list(dict.fromkeys(_normalise_label(label) for label in found))


def _shortqa_option_label(record: Mapping[str, Any], answer: Any) -> str:
    raw = str(answer or "").strip().strip("`").strip().rstrip(".").strip()
    match = re.fullmatch(r"\(?\s*([A-Za-z0-9]+)\s*\)?", raw)
    if not match:
        return ""
    candidate = _normalise_label(match.group(1))
    return f"opt:{candidate}" if candidate in _shortqa_declared_labels(record) else ""


def _declared_option_order(record: Mapping[str, Any]) -> dict[str, int]:
    task_type = task_type_of(dict(record))
    if task_type == "mcqa":
        labels = record.get("choice_labels") or []
        if not labels:
            labels = [
                chr(ord("A") + i) for i, _ in enumerate(record.get("choices") or [])
            ]
        return {_normalise_label(label): i for i, label in enumerate(labels)}
    if task_type == "boolqa":
        return {"yes": 0, "no": 1}

    labels = _shortqa_declared_labels(record)
    return {f"opt:{label}": i for i, label in enumerate(labels)}


def _answer_tie_key(record: Mapping[str, Any], answer: str) -> tuple:
    order = _declared_option_order(record)
    if answer in order:
        return (0, order[answer], "")
    if answer == INVALID_ANSWER:
        return (2, 0, "")
    return (1, 0, answer)


def aggregate_answer_scores(
    record: Mapping[str, Any],
    answers: Sequence[Any],
    weights: Sequence[float],
) -> dict[str, float]:
    """Sum raw peer weights for each canonical answer."""

    if len(answers) != len(weights):
        raise ValueError(
            f"answers/weights length mismatch: {len(answers)} != {len(weights)}"
        )
    totals: defaultdict[str, list[float]] = defaultdict(list)
    for answer, weight in zip(answers, weights):
        value = float(weight)
        if not np.isfinite(value):
            raise ValueError("vote weights must be finite")
        totals[canonical_answer(record, answer)].append(value)
    if not totals:
        raise ValueError("cannot vote without peer answers")
    # fsum makes aggregation invariant to peer iteration order.
    return {
        answer: float(math.fsum(values))
        for answer, values in totals.items()
    }


def vote(
    record: Mapping[str, Any],
    answers: Sequence[Any],
    weights: Sequence[float],
) -> str:
    """Select the highest-weight canonical answer.

    Raw signed weights are preserved.  On an exact aggregate-score tie, the
    lowest declared option index wins; non-option short answers use lexical order.
    """

    scores = aggregate_answer_scores(record, answers, weights)
    best_score = max(scores.values())
    tied = [answer for answer, score in scores.items() if score == best_score]
    return min(tied, key=lambda answer: _answer_tie_key(record, answer))


def gls_weights(matrix: Any, rhs: Sequence[float], ridge: float) -> np.ndarray:
    """Solve ``(G + ridge I) w = rhs`` exactly; no pseudo-inverse fallback."""

    G = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if G.shape != (b.size, b.size):
        raise ValueError(f"G shape {G.shape} is incompatible with rhs {b.shape}")
    if float(ridge) <= 0.0:
        raise ValueError("ridge must be positive")
    return np.linalg.solve(G + float(ridge) * np.eye(b.size, dtype=np.float64), b)


class OODRoutingState:
    """Float64 M/G state for pre-hoc routing and factorial voting."""

    def __init__(
        self,
        *,
        num_peers: int,
        rank: int,
        gamma: float,
        eta: float,
        gamma_g: float,
        eta_g: float,
    ) -> None:
        self.num_peers = int(num_peers)
        self.rank = int(rank)
        if self.num_peers < 1 or self.rank < 1:
            raise ValueError("num_peers and rank must be positive")
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.gamma_g = float(gamma_g)
        self.eta_g = float(eta_g)
        if not (0.0 <= self.gamma <= 1.0 and 0.0 <= self.gamma_g <= 1.0):
            raise ValueError("gamma and gamma_g must lie in [0, 1]")
        if self.eta < 0.0 or self.eta_g < 0.0:
            raise ValueError("eta and eta_g must be nonnegative")
        self.M = np.zeros(
            (self.num_peers, self.rank, self.rank), dtype=np.float64
        )
        self.G = np.eye(self.num_peers, dtype=np.float64)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self.M.copy(), self.G.copy()

    def reliability(self, phi: Any) -> np.ndarray:
        direction = _as_phi(phi, self.rank)
        return np.einsum("i,pij,j->p", direction, self.M, direction)

    def route(self, phi: Any, event_index: int) -> RouteDecision:
        scores = self.reliability(phi)
        tied = float(scores.max() - scores.min()) < ROUTE_TIE_EPS
        peer = int(event_index) % self.num_peers if tied else int(np.argmax(scores))
        return RouteDecision(peer=peer, scores=scores.copy(), tied=tied)

    def route_mg(
        self,
        phi: Any,
        event_index: int,
        *,
        ridge: float = DEFAULT_RIDGE,
    ) -> RouteDecision:
        reliability = self.reliability(phi)
        scores = gls_weights(self.G, reliability, ridge)
        tied = float(scores.max() - scores.min()) < ROUTE_TIE_EPS
        peer = int(event_index) % self.num_peers if tied else int(np.argmax(scores))
        return RouteDecision(peer=peer, scores=scores.copy(), tied=tied)

    def all_votes(
        self,
        phi: Any,
        record: Mapping[str, Any],
        answers: Sequence[Any],
        *,
        ridge: float = DEFAULT_RIDGE,
        clipped: bool = False,
    ) -> FactorialVoteResult:
        if len(answers) != self.num_peers:
            raise ValueError(
                f"expected {self.num_peers} peer answers, got {len(answers)}"
            )
        reliability = self.reliability(phi)
        ones = np.ones(self.num_peers, dtype=np.float64)
        weights = {
            "maj": ones / self.num_peers,
            "G": gls_weights(self.G, ones, ridge),
            "M": reliability.copy(),
            "MG": gls_weights(self.G, reliability, ridge),
        }
        if clipped:
            weights = {name: np.maximum(values, 0.0) for name, values in weights.items()}
        selections = {
            name: vote(record, answers, arm_weights)
            for name, arm_weights in weights.items()
        }
        return FactorialVoteResult(
            answers=selections,
            weights={name: values.copy() for name, values in weights.items()},
            reliability=reliability.copy(),
            ridge=float(ridge),
            clipped=bool(clipped),
        )

    def robustness_votes(
        self,
        phi: Any,
        record: Mapping[str, Any],
        answers: Sequence[Any],
        *,
        ridges: Sequence[float] = ROBUSTNESS_RIDGES,
    ) -> dict[tuple[float, bool], FactorialVoteResult]:
        """Return the declared lambda sweep for both raw and clipped weights."""

        return {
            (float(ridge), clipped): self.all_votes(
                phi,
                record,
                answers,
                ridge=float(ridge),
                clipped=clipped,
            )
            for ridge in ridges
            for clipped in (False, True)
        }

    def update(self, phi: Any, correctness: Sequence[float]) -> None:
        """Apply Eqs. 1--3 after all decisions for the event are logged."""

        direction = _as_phi(phi, self.rank)
        c = _as_correctness(correctness, self.num_peers)
        outer = np.outer(direction, direction)
        self.M = (
            self.gamma * self.M
            + self.eta * c[:, None, None] * outer[None, :, :]
        )
        q = c - c.mean()
        self.G = self.gamma_g * self.G + self.eta_g * np.outer(q, q)
        np.fill_diagonal(self.G, 1.0)

    def decay_without_feedback(self) -> None:
        """Advance one event while masking the unavailable label innovation."""

        self.M = self.gamma * self.M
        self.G = self.gamma_g * self.G
        np.fill_diagonal(self.G, 1.0)


class DecayedLabelDictionary:
    """True-dataset-name oracle baseline with post-decision updates."""

    def __init__(self, *, num_peers: int, gamma: float) -> None:
        self.num_peers = int(num_peers)
        self.gamma = float(gamma)
        self.numerator: defaultdict[tuple[str, int], float] = defaultdict(float)
        self.denominator: defaultdict[tuple[str, int], float] = defaultdict(float)

    def rates(self, dataset_name: str) -> np.ndarray:
        name = str(dataset_name)
        result = np.zeros(self.num_peers, dtype=np.float64)
        for peer in range(self.num_peers):
            key = (name, peer)
            den = self.denominator[key]
            result[peer] = self.numerator[key] / den if den > 0.0 else 0.0
        return result

    def route(self, dataset_name: str, event_index: int) -> RouteDecision:
        scores = self.rates(dataset_name)
        tied = float(scores.max() - scores.min()) < ROUTE_TIE_EPS
        peer = int(event_index) % self.num_peers if tied else int(np.argmax(scores))
        return RouteDecision(peer=peer, scores=scores.copy(), tied=tied)

    def update(self, dataset_name: str, correctness: Sequence[float]) -> None:
        c = _as_correctness(correctness, self.num_peers)
        name = str(dataset_name)
        for peer in range(self.num_peers):
            key = (name, peer)
            self.numerator[key] = (
                self.gamma * self.numerator[key] + float(c[peer] == 1.0)
            )
            self.denominator[key] = self.gamma * self.denominator[key] + 1.0
