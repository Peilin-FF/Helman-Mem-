"""The record next to reliability tables: the obvious alternative to a record addressed by content.

A table predicts a peer's correctness on an event from its running accuracy (Laplace: (right + 1) / (seen + 2)) in a cell:
the peer alone, or the peer and a key of the event (its task, its sub-task type, the peer's own verdict, ...). Every table is
read before and written after each event, along the same order as the record, so they see exactly the same history; an
empty cell falls back to the peer's own history.

    compare(rows, cells)   rows: {pos, peer_order, peer_correct, memory_prob, ...} per event (a record file, or an online
                           track's generations); cells: {name: cell(row, slot) -> key}; returns per estimate (the record and
                           every table) the AUC, how often its top peer is right, and the same where the peers disagree.
"""
from __future__ import annotations

import collections

import numpy as np


def compare(rows: list[dict], cells: dict) -> dict:
    from pipeline.record import auc

    cells = {"per peer": (lambda r, s: ""), **cells}
    rows = sorted(rows, key=lambda r: int(r["pos"]))
    counts = {k: collections.defaultdict(lambda: [0, 0]) for k in cells}
    preds = {k: [] for k in ["record"] + list(cells)}
    picks = {k: [0, 0, 0, 0] for k in preds}      # [right on all, all, right on mixed, mixed]
    for r in rows:
        ys = r["peer_correct"]
        est = {"record": [float(p) for p in r["memory_prob"]]}
        keys = {name: [cell(r, s) for s in range(len(r["peer_order"]))] for name, cell in cells.items()}
        for name in cells:
            vals = []
            for s, pid in enumerate(r["peer_order"]):
                hit, n = counts[name][(pid, keys[name][s])]
                if n == 0:                               # an empty cell falls back to the peer's own history
                    hit, n = counts["per peer"][(pid, "")]
                vals.append((hit + 1) / (n + 2))
            est[name] = vals
        mixed = 0 < sum(ys) < len(ys)
        for name, ps in est.items():
            preds[name] += list(zip(ps, ys))
            top = max(range(len(ps)), key=lambda s: ps[s])
            picks[name][0] += ys[top]; picks[name][1] += 1
            if mixed:
                picks[name][2] += ys[top]; picks[name][3] += 1
        for name in cells:                               # write: the event's labels, after every estimate was read
            for s, pid in enumerate(r["peer_order"]):
                counts[name][(pid, keys[name][s])][0] += ys[s]
                counts[name][(pid, keys[name][s])][1] += 1
    n_peers = len(rows[0]["peer_order"]) if rows else 0
    best = max(range(n_peers), key=lambda p: sum(r["peer_correct"][r["peer_order"].index(p)] for r in rows)) if rows else None
    out = {name: {"auc": auc(preds[name]), "pick_right": picks[name][0] / max(1, picks[name][1]),
                  "pick_right_on_mixed": picks[name][2] / max(1, picks[name][3])} for name in preds}
    out["_reference"] = {"events": len(rows), "mixed": picks["record"][3], "best_peer_in_hindsight": best,
                         "best_peer_accuracy": float(np.mean([r["peer_correct"][r["peer_order"].index(best)] for r in rows])) if rows else None,
                         "any_peer_right": float(np.mean([max(r["peer_correct"]) for r in rows])) if rows else None}
    return out


def markdown(title: str, rel: dict) -> list[str]:
    ref = rel["_reference"]
    lines = ["", f"Reliability along the stream, {title} ({ref['events']} events, {ref['mixed']} where the peers disagree; best peer in "
             f"hindsight {100 * (ref['best_peer_accuracy'] or 0):.1f}, any peer right {100 * (ref['any_peer_right'] or 0):.1f}):", "",
             "| estimate | AUC | top peer right | top peer right where the peers disagree |", "|---|---:|---:|---:|"]
    for name, v in rel.items():
        if not name.startswith("_"):
            lines.append(f"| {name} | {v['auc']:.3f} | {100 * v['pick_right']:.1f} | {100 * v['pick_right_on_mixed']:.1f} |")
    return lines
