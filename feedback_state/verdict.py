"""The verdict line: the central model states its own judgement of the peers before it answers.

Method A of the judgement work (docs section 22): the model's reply must begin with one line

    Trust: 4 > 1 > 6 > 2 > 3 > 5

ranking the peers from most to least trustworthy for this question, by their numbers in prompt
order.  During training the line is rewarded for agreeing with the record's per-peer estimate
(distillation: the record teaches, the model graduates); at test time there is no record and the
ranking must come from the content.  The line is stripped before the answer is graded.
"""
from __future__ import annotations

import math
import re
from typing import Sequence

VERDICT_INSTRUCTION = (
    "Begin your reply with exactly one line of the form 'Trust: a > b > c > ...' that ranks all the peers from most to "
    "least trustworthy for this question, using their numbers. Then answer."
)
_LINE = re.compile(r"^\s*Trust\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUM = re.compile(r"\d+")


def add_verdict_instruction(messages: list[dict]) -> list[dict]:
    """Append the verdict instruction to the user turn's 'Instruction:' line (returns a new list)."""
    out = [dict(m) for m in messages]
    user = out[-1]
    if user.get("role") != "user" or VERDICT_INSTRUCTION in user["content"]:
        return out
    user["content"] = user["content"].rstrip() + " " + VERDICT_INSTRUCTION
    return out


def parse_verdict(text: str, n_peers: int) -> list[int] | None:
    """The ranking as 0-based peer indices, most trusted first; None if absent or malformed.

    Tolerates a partial ranking (missing peers are appended in prompt order) but rejects duplicates,
    out-of-range numbers, and lines with fewer than two peers.
    """
    m = _LINE.search(text or "")
    if not m:
        return None
    nums = [int(x) for x in _NUM.findall(m.group(1))]
    rank = []
    for k in nums:
        i = k - 1
        if i < 0 or i >= n_peers or i in rank:
            return None
        rank.append(i)
    if len(rank) < 2:
        return None
    rank += [i for i in range(n_peers) if i not in rank]
    return rank


def strip_verdict(text: str) -> str:
    """The reply without its Trust line, for grading."""
    return _LINE.sub("", text or "", count=1).lstrip("\n")


def rank_scores(rank: Sequence[int]) -> list[float]:
    """Per-peer score implied by a ranking: n-1 for the most trusted down to 0."""
    n = len(rank)
    s = [0.0] * n
    for pos, i in enumerate(rank):
        s[i] = float(n - 1 - pos)
    return s


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties; nan if either side is constant."""
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = ranks(list(a)), ranks(list(b))
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    va = sum((x - ma) ** 2 for x in ra); vb = sum((x - mb) ** 2 for x in rb)
    if va == 0 or vb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(va * vb)


def verdict_agreement(rank: Sequence[int] | None, target: Sequence[float]) -> float:
    """0.5 (1 + Spearman) between the ranking and a per-peer target (the record's p, or the labels); 0 if no ranking."""
    if rank is None or len(target) != len(rank):
        return 0.0
    rho = spearman(rank_scores(rank), target)
    return 0.0 if rho != rho else 0.5 * (1.0 + rho)


def record_is_flat(probs: Sequence[float], flat: float = 0.1) -> bool:
    return (not probs) or (max(probs) - min(probs) <= flat)
