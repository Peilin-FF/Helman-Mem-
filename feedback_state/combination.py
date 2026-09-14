"""Combination during training: per question, before its rollouts, the prompt is peers + memory or question alone.

The state is the one combination uses at test time (pipeline.combination), kept online along the training order:
  the record over the seven answers (the six peers and the central model's own answer in the last slot), started cold on
  the addresses pipeline.record --save-addresses wrote (the judge's features stay frozen), and
  the reading line (rho, delta) per task type (feedback_state.reading_line).
A step reads the state for all its questions, chooses, and writes once the rewards are in:
  peers + memory chosen   the samples are readings: the reading line takes their accuracy y at (T, kappa)
  question alone chosen   the samples are the model's own answers: the record takes their accuracy at the own slot
  either way              the six peers' verified labels go into the record
Questions of one step do not see each other's outcomes: the whole step is read before any of it is written.
"""
from __future__ import annotations

import collections

import numpy as np
import torch

from feedback_state.attn_bias import bias_values
from feedback_state.memory_runtime import MemoryRuntime
from feedback_state.reading_line import ReadingLine

PEERS_MEMORY, QUESTION_ALONE = "peers+memory", "question alone"


class TrainingCombination:
    def __init__(self, saved: dict, *, prior: tuple[float, float] = (0.5, 0.0), lam: float = 1.0, gamma: float = 3.0,
                 bias_form: str = "logratio"):
        self.runtime = MemoryRuntime.from_addresses(saved)
        self.index = {str(i): t for t, i in enumerate(saved["ids"])}
        self.labels = saved["labels"]
        self.own = self.runtime.P - 1
        self.prior, self.lam, self.gamma, self.bias_form = tuple(float(x) for x in prior), float(lam), float(gamma), str(bias_form)
        self.lines: dict[str, ReadingLine] = {}
        self.pending: list[dict] = []

    def line(self, task: str) -> ReadingLine:
        return self.lines.setdefault(task, ReadingLine(self.prior, self.lam))

    @torch.no_grad()
    def choose(self, ids, tasks, peer_orders) -> list[dict]:
        """Read the state for one step's questions and choose each one's prompt; the tilt values come with peers + memory."""
        self.pending = []
        for qid, task, order in zip(ids, tasks, peer_orders):
            t = self.index[str(qid)]
            ell, _, _, X = self.runtime.read(t)
            p = torch.sigmoid(ell).tolist()
            trust, kappa = max(p[c] for c in range(len(p)) if c != self.own), p[self.own]
            line = self.line(str(task))
            rho, delta = line.estimate()
            value = line.value(trust, kappa)
            take = value >= kappa
            shown = [p[int(c)] for c in order]              # the peers in prompt-slot order
            self.pending.append({"t": t, "X": X, "id": str(qid), "task": str(task), "trust": trust, "own_prob": kappa, "rho": rho,
                                 "delta": delta, "peers_memory_value": value, "choice": PEERS_MEMORY if take else QUESTION_ALONE,
                                 "memory_prob": shown, "bias": bias_values(shown, self.gamma, self.bias_form) if take else [0.0] * len(shown)})
        return self.pending

    @torch.no_grad()
    def update(self, accuracy) -> tuple[list[dict], dict]:
        """Write one step: accuracy[i] is the mean reward of question i's samples, in the order choose() saw them."""
        if len(accuracy) != len(self.pending):
            raise ValueError(f"{len(accuracy)} accuracies for {len(self.pending)} chosen questions")
        rows = []
        for d, y in zip(self.pending, accuracy):
            y = float(y)
            targets = {c: 1.0 if int(self.labels[d["t"], c]) else -1.0 for c in range(self.runtime.real[d["t"]]) if c != self.own}
            if d["choice"] == QUESTION_ALONE:
                targets[self.own] = 2.0 * y - 1.0
            self.runtime.write_targets(d["X"], targets)
            if d["choice"] == PEERS_MEMORY:
                self.line(d["task"]).write(d["trust"], d["own_prob"], y)
            rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k not in ("t", "X", "bias", "memory_prob")} | {"accuracy": round(y, 4)})
        chosen = collections.defaultdict(list)
        for r in rows:
            chosen[r["choice"]].append(r["accuracy"])
        metrics = {"combination/share_peers_memory": len(chosen[PEERS_MEMORY]) / max(1, len(rows)),
                   "combination/trust": float(np.mean([r["trust"] for r in rows])), "combination/own_prob": float(np.mean([r["own_prob"] for r in rows]))}
        for name, key in ((PEERS_MEMORY, "peers_memory"), (QUESTION_ALONE, "question_alone")):
            if chosen[name]:
                metrics[f"combination/accuracy_{key}"] = float(np.mean(chosen[name]))
        for task, line in self.lines.items():
            metrics[f"combination/rho_{task}"], metrics[f"combination/delta_{task}"] = line.estimate()
        self.pending = []
        return rows, metrics

    def state_dict(self) -> dict:
        return {"record": self.runtime.mem.snapshot(), "reading_line": {t: {"P": l.P.copy(), "q": l.q.copy(), "events": l.events} for t, l in self.lines.items()}}
