"""The swarm online: every condition a track of one run, the record read before and written after every tick.

A tick is one step of the swarm on every track: the sub-tasks (events) the benchmark hands out, the peers' answers to each, the
central model's own answer, the record's estimates (read before any label of the tick is written), the central model's
reading of the peers (tilted on record tracks), the answer it commits, every answer's verified label, the writes, the
commits. Tracks:
  solo          the central model's own answer is committed (question alone)
  peers         the central model reads the peers' answers and commits its answer (question + peers)
  tilt          the same with the record's tilt on its attention (peers + memory)
  combination   per event peers + memory or question alone, chosen by the reading line (feedback_state.reading_line)
A record track keeps its own record over the peers' answers and the central model's own answer (recorded, never shown),
cold at the start, addressed by the frozen judge's features of the event, computed live.

A benchmark plugs in as an adapter (feedback_state.classeval_online.ClassEvalAdapter, feedback_state.marble_online.MarbleDBAdapter):
  num_peers                          the pool's size
  events(tracks) -> [(track, lane, event)]   this tick's sub-tasks, every track's (none: the run is over)
  peers(items) -> per item the peers' answers, each {response, ...}, in peer order (items: (track, lane, event))
  own(items) -> per item the central model's own answer (its question-alone answer)
  display(event, text) -> an answer as the judge and the central model read it
  grade(event, text) -> the answer's verified label, 0/1, never dependent on what any track committed
  commit(track, lane, event, text, row)      the committed answer enters the track's state; extra fields go into row
  end_tick(tracks)                           after every commit of the tick (a planner reading the verdicts, ...)
  unit_metrics(track) -> dict, progress(track) -> str      the task-level results (classes built, diagnoses)
The model calls are injected (central_fn, features_fn; the adapter's own), so the loop runs on CPU in the tests:
  central_fn(requests) -> texts; a request is {messages, texts (slot order) | None, probs (slot order) | None, gamma}
  features_fn(event, texts) -> {sem, peer_hidden, margins} (feedback_state.judge_features.event_features)
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

KINDS = ("solo", "peers", "tilt", "combination")
READS_PEERS = ("peers", "tilt", "combination")
RECORDS = ("tilt", "combination")
OWN = ("solo",) + RECORDS
PEERS_MEMORY, QUESTION_ALONE = "peers+memory", "question alone"


class OnlineRecord:
    """The record over one event's answers, from features computed live: the same rows, read and write as pipeline.record
    (feedback_state.memory_runtime).

    The address projections (standardise + PCA, label-free) are either given (fit beforehand on another stream's features) or
    fit on the swarm's own first `warmup` events: until then every read is the cold record (0.5 everywhere, no tilt) and the
    events' features and labels are kept; at the warmup-th write the projections are fit on those features and their labels
    are written in order, so from then on the record holds every past event, as if it had been fit from the start. Nothing
    of an event is used before it happened."""

    def __init__(self, proj_q, proj_c, num_answers: int, *, design: str = "qc", lam: float = 100.0, dim: int = 64, warmup: int = 0):
        if design == "q":
            raise ValueError("the online record needs a design with one row per answer (qc, c, cm, qcm)")
        if (proj_q is None) != (proj_c is None) or (proj_q is None and warmup < 1):
            raise ValueError("give both projections, or none and a warmup of at least one event to fit them on")
        self.P, self.design, self.lam, self.dim, self.warmup = int(num_answers), str(design), float(lam), int(dim), int(warmup)
        self.buffer: list[tuple[dict, list[int]]] = []
        self.fitted_at = 0 if proj_q is not None else None
        self.mem = None
        if proj_q is not None:
            self._start(proj_q, proj_c)

    @property
    def ready(self) -> bool:
        return self.mem is not None

    def _start(self, proj_q, proj_c) -> None:
        from feedback_state.addresses import design_dim
        from feedback_state.kalman_memory import KalmanMemory

        self.proj_q, self.proj_c = proj_q, proj_c
        self.mem = KalmanMemory(design_dim(self.design, self.P, proj_q.dim, proj_c.dim), 1, lam=self.lam)

    def rows(self, f: dict) -> torch.Tensor:
        from feedback_state.addresses import design_rows

        psi_q = self.proj_q(f["sem"].float().cpu().reshape(1, -1))[0].to(torch.float64)
        psi_c = self.proj_c(f["peer_hidden"].float().cpu()).to(torch.float64)
        z = f["margins"].float().cpu().to(torch.float64)
        return design_rows(self.design, psi_q, psi_c[: self.P], z[: self.P], list(range(self.P)), self.P)

    @torch.no_grad()
    def read(self, f: dict) -> tuple[list[float], list[float]]:
        """(P(correct), evidence) per answer, from the events written so far."""
        if not self.ready:
            return [0.5] * self.P, [0.0] * self.P
        X = self.rows(f)
        mu, var = self.mem.read(X)
        p = self.mem.prob(mu[:, 0], var)
        n = self.mem.evidence(X, var)
        return [float(x) for x in p], [float(x) for x in n]

    @torch.no_grad()
    def write(self, f: dict, labels: list[int]) -> None:
        """One event's verified labels (one per answer), after its read."""
        if not self.ready:
            self.buffer.append((f, list(labels)))
            if len(self.buffer) < self.warmup:
                return
            from feedback_state.addresses import Projection

            sem = torch.stack([b[0]["sem"].float().cpu() for b in self.buffer])
            hidden = torch.cat([b[0]["peer_hidden"].float().cpu()[: self.P] for b in self.buffer])
            self._start(Projection(sem, self.dim, torch.device("cpu")), Projection(hidden, self.dim, torch.device("cpu")))
            self.fitted_at = len(self.buffer)
            pending, self.buffer = self.buffer, []
            for g, y in pending:
                self._write(self.rows(g), y)
            return
        self._write(self.rows(f), labels)

    def _write(self, X: torch.Tensor, labels: list[int]) -> None:
        for c, y in enumerate(labels):
            self.mem.write(X[c], torch.tensor([1.0 if int(y) else -1.0], dtype=torch.float64))


@dataclass
class Track:
    name: str
    kind: str
    gamma: float = 3.0
    record: OnlineRecord | None = None
    line: object | None = None                          # ReadingLine (combination)
    seed: int = 0
    rows: list = field(default_factory=list)            # one per event, in the order the track met them
    units: list = field(default_factory=list)           # one per finished task (a class built, a database diagnosed)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"track {self.name}: kind {self.kind!r} is not one of {KINDS}")
        if self.kind in RECORDS and self.record is None:
            raise ValueError(f"track {self.name}: a {self.kind} track needs a record")
        if self.kind == "combination" and self.line is None:
            raise ValueError(f"track {self.name}: a combination track needs a reading line")
        self.rng = random.Random(self.seed)


def run_online(adapter, tracks: list[Track], *, central_fn, features_fn, workers: int = 16, log=print) -> None:
    """Run every track to the end of the adapter's stream (see the module doc). Rows and task results accumulate on the tracks."""
    from feedback_state.memory_generator import build_messages

    P = int(adapter.num_peers)
    pos = {t.name: 0 for t in tracks}
    tick = 0
    while True:
        batch = adapter.events(tracks)
        if not batch:
            break
        evs = [b[2] for b in batch]

        # 1. the peers' answers, 2. the central model's own answer
        need = [i for i, b in enumerate(batch) if b[0].kind in READS_PEERS]
        answers = dict(zip(need, adapter.peers([batch[i] for i in need]))) if need else {}
        shown = {i: [adapter.display(evs[i], a["response"]) for a in answers[i]] for i in need}
        own_i = [i for i, b in enumerate(batch) if b[0].kind in OWN]
        own_text = dict(zip(own_i, adapter.own([batch[i] for i in own_i]))) if own_i else {}

        # 3. the record: features of the answers and the own answer, read before any of this tick's labels is written
        F, probs, evidence = {}, {}, {}
        for i, b in enumerate(batch):
            if b[0].kind in RECORDS:
                texts = shown[i] + [adapter.display(evs[i], own_text[i])]
                try:
                    F[i] = features_fn(evs[i], texts)
                except Exception as e:   # one event the judge cannot read must not end the run: it gets the cold record, no write
                    log(f"[online] {evs[i]['id']}: the judge failed ({type(e).__name__}: {str(e)[:160]}); no record read or write")
                    F[i] = None
                probs[i], evidence[i] = b[0].record.read(F[i]) if F[i] is not None else ([0.5] * (P + 1), [0.0] * (P + 1))

        # 4. the central model reading the peers (tilted on record tracks)
        slots = {i: batch[i][0].rng.sample(range(P), P) for i in need}
        reqs = []
        for i in need:
            t = batch[i][0]
            texts = [shown[i][p] for p in slots[i]]
            tilt = t.kind in RECORDS and t.gamma > 0
            reqs.append({"messages": build_messages(evs[i], texts, mode="peers"), "texts": texts,
                         "probs": [probs[i][p] for p in slots[i]] if tilt else None, "gamma": t.gamma if tilt else 0.0})
        read_text = dict(zip(need, central_fn(reqs))) if reqs else {}

        # 5. what each track commits
        choice, final = {}, {}
        for i, b in enumerate(batch):
            t = b[0]
            if t.kind == "solo":
                final[i] = own_text[i]
            elif t.kind in ("peers", "tilt"):
                final[i] = read_text[i]
            else:
                trust, kappa = max(probs[i][:P]), probs[i][P]
                value = t.line.value(trust, kappa)
                choice[i] = {"take": value >= kappa, "trust": trust, "own_prob": kappa, "value": value, "estimate": t.line.estimate()}
                final[i] = read_text[i] if choice[i]["take"] else own_text[i]

        # 6. every answer's verified label
        jobs = [(i, f"peer{p}", a["response"]) for i in need for p, a in enumerate(answers[i])]
        jobs += [(i, "own", own_text[i]) for i in own_i] + [(i, "read", read_text[i]) for i in need]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            graded = list(ex.map(lambda j: int(bool(adapter.grade(evs[j[0]], j[2]))), jobs))
        lab: dict[int, dict] = {i: {} for i in range(len(batch))}
        for (i, what, _), ok in zip(jobs, graded):
            lab[i][what] = ok

        # 7. write (in batch order), commit, record the rows
        for i, (t, l, ev) in enumerate(batch):
            got = lab[i]
            peer_labels = [got[f"peer{p}"] for p in range(len(answers.get(i, [])))]
            if t.kind in RECORDS and F[i] is not None:
                t.record.write(F[i], peer_labels + [got["own"]])
            if t.kind == "combination":
                t.line.write(choice[i]["trust"], choice[i]["own_prob"], got["read"])
            correct = got["own"] if t.kind == "solo" else got["read"] if t.kind in ("peers", "tilt") else (got["read"] if choice[i]["take"] else got["own"])
            row = {"pos": pos[t.name], "tick": tick, "lane": l, "id": ev["id"], "task_type": ev["task_type"], "source": ev.get("source"),
                   "correct": int(correct), "generation": final[i]}
            if i in own_text:
                row.update(own_correct=got["own"], own_answer=adapter.display(ev, own_text[i]))
            if i in answers:
                row.update(peer_order=list(slots[i]), peer_correct=[peer_labels[p] for p in slots[i]], peer_correct_by_peer=peer_labels,
                           read_correct=got["read"], peer_answers=shown[i],
                           **{k: [a.get(k) for a in answers[i]] for k in ("turns",) if all(k in a for a in answers[i])})
                if all("visible" in a for a in answers[i]):
                    row["peer_visible_passed"] = [bool(a["visible"] and a["visible"][-1]["passed"]) for a in answers[i]]
            if t.kind in RECORDS:
                row.update(memory_prob=[round(probs[i][p], 4) for p in slots[i]], memory_evidence=[round(evidence[i][p], 1) for p in slots[i]],
                           own_prob=round(probs[i][P], 4))
            if t.kind == "combination":
                c = choice[i]
                row.update(choice=PEERS_MEMORY if c["take"] else QUESTION_ALONE, trust=round(c["trust"], 4), peers_memory_value=round(c["value"], 4),
                           rho=round(c["estimate"][0], 4), delta=round(c["estimate"][1], 4))
            adapter.commit(t, l, ev, final[i], row)
            t.rows.append(row)
            pos[t.name] += 1
        adapter.end_tick(tracks)
        tick += 1
        log(f"[online] tick {tick}: {len(batch)} events; " + "; ".join(
            f"{t.name} {sum(r['correct'] for r in t.rows)}/{len(t.rows)} right, {adapter.progress(t)}" for t in tracks))


def metrics(track: Track, adapter=None, windows: int = 8) -> dict:
    """A track's results: the events' accuracy (each committed answer on its own), the peers', the record's, the reading
    line's, and the adapter's task-level results."""
    from pipeline.evaluate import curve

    rows = sorted(track.rows, key=lambda r: r["pos"])
    hits = np.array([r["correct"] for r in rows]) if rows else np.zeros(0)
    out = {"condition": track.name, "mode": "online", "kind": track.kind, "gamma": track.gamma if track.kind in RECORDS else None,
           "accuracy": float(hits.mean()) if len(hits) else 0.0, "num_samples": len(rows), "generated": curve(hits, windows) if len(hits) else {}}
    peered = [r for r in rows if "peer_correct_by_peer" in r]
    if peered:
        n_peers = len(peered[0]["peer_correct_by_peer"])
        out["peers"] = {"accuracy": [float(np.mean([r["peer_correct_by_peer"][p] for r in peered])) for p in range(n_peers)],
                        "oracle_any_peer": float(np.mean([max(r["peer_correct_by_peer"]) for r in peered]))}
        if all("turns" in r for r in peered):
            out["peers"]["mean_turns"] = float(np.mean([np.mean(r["turns"]) for r in peered]))
    if any("own_correct" in r for r in rows):
        out["own_accuracy"] = float(np.mean([r["own_correct"] for r in rows if "own_correct" in r]))
    if track.kind in RECORDS and rows:
        from pipeline.record import quality

        q = quality(rows, windows, name=track.name)
        out["record"] = {k: q[k] for k in ("memory", "running_peer", "n_mixed", "chance_on_mixed", "windows") if k in q}
        out["record"]["addresses_fit_after_events"] = track.record.fitted_at
    if track.kind == "combination" and rows:
        rho, delta = track.line.estimate()
        out["combination"] = {"share_peers_memory": float(np.mean([r["choice"] == PEERS_MEMORY for r in rows])), "rho": rho, "delta": delta,
                              "peers_memory_accuracy": float(np.mean([r["read_correct"] for r in rows])),
                              "question_alone_accuracy": float(np.mean([r["own_correct"] for r in rows]))}
    if adapter is not None:
        out.update(adapter.unit_metrics(track))
    return out
