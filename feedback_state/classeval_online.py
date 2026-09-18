"""ClassEval in the online swarm (feedback_state.online_swarm): every class built for real, one method at a time, and what the
central model commits is what the class's next methods are written against.

An event is a method of a class the track is building (`lanes` classes at once per track, each class's methods in order): the
class so far is the track's committed methods before it and the later methods' signatures. The peers work on it as agents
against the track's sandbox (the class with the committed methods in place: feedback_state.classeval.visible_check); the own
answer is the central model's method from the question alone.

Credit. Every answer (each peer's, the own answer, the committed one) is labelled by its method's hidden tests with the answer
put into the GOLD class, so nobody is charged for a method committed before theirs. The committed method is also tested
inside the committed class (in_context_correct), and a finished class runs all its test classes on the committed class
(class pass): that is where committed errors propagate.

Injected, besides the engine's: peers_fn(events) -> per event, per peer {response, turns, visible}; central_fn for the own
answers; grade_fn(event, text, overrides) -> bool; class_test_fn(event, committed) -> bool.
"""
from __future__ import annotations

import numpy as np

from feedback_state.online_swarm import Track


def failing_stub(cls: dict, name: str) -> str:
    """What is committed when an answer holds no usable method: its signature and docstring, and a body that raises."""
    from feedback_state.classeval import _description

    m = next(x for x in cls["methods_info"] if x["method_name"] == name)
    desc = _description(m)
    indent = next((len(l) - len(l.lstrip()) for l in desc.splitlines()[1:] if l.strip()), 4) or 4
    return desc.rstrip() + "\n" + " " * indent + "raise NotImplementedError"


def commit_source(cls: dict, ev: dict, text: str) -> str:
    from feedback_state.classeval import extract_method

    found = extract_method(text, ev["classeval"]["method_name"])
    return found["method"] if found else failing_stub(cls, ev["classeval"]["method_name"])


class ClassEvalAdapter:
    def __init__(self, classes: list[dict], *, lanes: int, peers_fn, central_fn, grade_fn, class_test_fn, visible: dict | None = None,
                 num_peers: int = 6):
        self.classes, self.num_peers = list(classes), int(num_peers)
        self.peers_fn, self.central_fn, self.grade_fn, self.class_test_fn = peers_fn, central_fn, grade_fn, class_test_fn
        self.visible = visible or {}
        self.lanes = max(1, int(lanes))
        self.state: dict[str, dict] = {}
        self.committed: dict[str, dict[str, dict[str, str]]] = {}    # track -> class id -> {method: committed source}

    def _state(self, t: Track) -> dict:
        if t.name not in self.state:
            self.state[t.name] = {"queue": list(range(len(self.classes))), "lanes": [None] * self.lanes}
            self.committed[t.name] = {}
        return self.state[t.name]

    def events(self, tracks: list[Track]) -> list[tuple]:
        from feedback_state.classeval import event

        batch = []
        for t in tracks:
            st = self._state(t)
            for l in range(self.lanes):
                if st["lanes"][l] is None and st["queue"]:
                    st["lanes"][l] = (st["queue"].pop(0), 0)
                if st["lanes"][l] is None:
                    continue
                ci, k = st["lanes"][l]
                cls = self.classes[ci]
                ev = event(cls, k, committed=self.committed[t.name].get(cls["task_id"]) or None)
                if ev["id"] in self.visible:
                    ev["classeval"]["visible"] = self.visible[ev["id"]]
                batch.append((t, l, ev))
        return batch

    def peers(self, items):
        return self.peers_fn([ev for _, _, ev in items])

    def own(self, items):
        from feedback_state.memory_generator import build_messages

        return self.central_fn([{"messages": build_messages(ev, [], mode="solo"), "texts": None, "probs": None, "gamma": 0.0} for _, _, ev in items])

    def display(self, ev: dict, text: str) -> str:
        from feedback_state.classeval import display

        return display(ev, text)

    def grade(self, ev: dict, text: str) -> int:
        return int(bool(self.grade_fn(ev, text, None)))

    def commit(self, t: Track, l: int, ev: dict, text: str, row: dict) -> None:
        st = self._state(t)
        ci, k = st["lanes"][l]
        cls = self.classes[ci]
        row.update(class_id=cls["task_id"], step=k,
                   in_context_correct=int(bool(self.grade_fn(ev, text, ev["classeval"].get("committed") or {}))))
        committed = self.committed[t.name].setdefault(cls["task_id"], {})
        committed[ev["classeval"]["method_name"]] = commit_source(cls, ev, text)
        if k + 1 < len(cls["methods_info"]):
            st["lanes"][l] = (ci, k + 1)
            return
        st["lanes"][l] = None
        mine = [r for r in t.rows if r.get("class_id") == cls["task_id"]] + [row]
        t.units.append({"class_id": cls["task_id"], "methods": len(cls["methods_info"]), "events": len(mine),
                        "passed": int(bool(self.class_test_fn(ev, committed))), "methods_correct": sum(r["correct"] for r in mine),
                        "in_context_correct": sum(r["in_context_correct"] for r in mine)})

    def end_tick(self, tracks: list[Track]) -> None:
        return None

    def progress(self, t: Track) -> str:
        return f"{sum(c['passed'] for c in t.units)}/{len(t.units)} classes"

    def unit_metrics(self, t: Track) -> dict:
        rows = t.rows
        return {"in_context_accuracy": float(np.mean([r["in_context_correct"] for r in rows])) if rows else 0.0,
                "classes": len(t.units), "class_pass": float(np.mean([c["passed"] for c in t.units])) if t.units else 0.0}
