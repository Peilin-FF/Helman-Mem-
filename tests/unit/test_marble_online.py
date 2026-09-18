"""The database swarm online (feedback_state.marble_online on feedback_state.online_swarm, pipeline.marble_online): the planner's
order and the forced coverage of the five causes, the verdicts the planner reads, the labels against the injected anomaly,
the record, the task-level diagnoses, the failures that must not stop a run, the report's tables and the job expansion.

CPU only: the benchmark's roles (the planner, the investigators, the injection) and the models are stand-ins.
"""
import collections
import re
import threading
from pathlib import Path

import pytest
import torch

from feedback_state.marble_online import MarbleDBAdapter, diagnosis, grade, verdict
from feedback_state.online_swarm import OnlineRecord, Track, metrics, run_online
from feedback_state.reading_line import ReadingLine

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"
LABELS = ["INSERT_LARGE_DATA", "LOCK_CONTENTION", "VACUUM", "REDUNDANT_INDEX", "FETCH_LARGE_DATA"]
PEERS = ["p0", "p1", "p2", "p3", "p4", "p5"]


def task(i: int, causes: list[str]) -> dict:
    return {"id": f"T{i}", "scenario": "EDUCATION" if i % 2 else "FINANCE", "task": f"Database {i} is slow. Find out why.",
            "labels": list(LABELS), "root_causes": causes, "number_of_labels_pred": 2, "max_iterations": 5,
            "output_format": "Choose the most likely causes.",
            "agents": [{"agent_id": f"agent{k + 1}", "profile": f"agent{k + 1} will explore the possibility of {c} as a root cause."}
                       for k, c in enumerate(LABELS)]}


TASKS = [task(0, ["LOCK_CONTENTION"]), task(1, ["VACUUM", "FETCH_LARGE_DATA"]), task(2, ["INSERT_LARGE_DATA"])]


class Planner:
    """Names LOCK and INSERT (as 'Agent 2', 'agent_1'), then LOCK again (nothing new), then stops."""
    made = []

    def __init__(self, track, lane, t):
        self.track, self.task, self.updates, self.calls = track.name, t, [], 0
        Planner.made.append(self)

    def assign(self):
        self.calls += 1
        return [{"Agent 2": "check pg_locks for waits", "agent_1": "look for large INSERTs"}, {"agent2": "again"}][min(self.calls, 2) - 1]

    def update(self, summary):
        self.updates.append(summary)

    def decide(self, results):
        return self.calls < 2

    def final(self, summary):   # it decides from the verdicts it was given: the causes whose verdict says yes
        causes = [LABELS[int(k) - 1] for k, v in re.findall(r"agent(\d)': '([^']*)'", summary) if verdict(v) == "YES"]
        return "Final answer: " + (", ".join(causes) or "NONE")


def finding(t: dict, cause: str, say_yes: bool) -> str:
    return f"I queried the database about {cause} in {t['id']}.\nFinal answer: {'yes' if say_yes else 'no'}"


def investigate(lane, model, t, agent, assignment):
    cause = re.search(r"possibility of ([A-Z_]+)", agent["profile"]).group(1)
    truth = cause in t["root_causes"]
    say = {"p0": truth, "p1": False, "central": True}.get(model, (hash((model, t["id"], cause)) % 2 == 0))
    return {"text": finding(t, cause, say), "evidence": f"{model} queried"}


def central_fn(requests):   # reading: the finding the record trusts most (the first slot without a record)
    out = []
    for q in requests:
        probs = q["probs"] or [1.0] + [0.0] * (len(q["texts"]) - 1)
        out.append(q["texts"][max(range(len(probs)), key=lambda s: probs[s])])
    return out


def features_fn(ev, texts):   # the question part shared, the answer part the peer's identity
    return {"sem": torch.ones(4), "peer_hidden": torch.eye(len(PEERS) + 1, 11) * 2.0, "margins": torch.zeros(len(PEERS) + 1)}


class _Projection:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, X):
        return X[:, : self.dim].float()


def adapter(lanes=1, prepared=None, **kw):
    prepared = prepared if prepared is not None else []
    lock = threading.Lock()

    def prepare(lane, t):
        with lock:
            prepared.append((lane, t["id"]))
        return {"prepared": t["id"]}

    args = dict(peers=PEERS, own_model="central", prepare=prepare, planner_for=lambda tr, l, t: Planner(tr, l, t),
                investigate=investigate, lanes=lanes, workers=8, log=lambda *_: None)
    args.update(kw)
    return MarbleDBAdapter(TASKS, **args)


def tracks():
    q, c = _Projection(4), _Projection(11)
    return [Track("solo", "solo"), Track("peers", "peers", seed=1),
            Track("tilt", "tilt", record=OnlineRecord(q, c, len(PEERS) + 1, lam=1.0), seed=2),
            Track("combination", "combination", record=OnlineRecord(q, c, len(PEERS) + 1, lam=1.0), line=ReadingLine(), seed=3)]


def test_every_task_investigates_all_five_causes_in_the_planners_order_then_the_rest():
    Planner.made = []
    prepared = []
    a, ts = adapter(prepared=prepared), tracks()
    run_online(a, ts, central_fn=central_fn, features_fn=features_fn, workers=8, log=lambda *_: None)

    assert prepared == [(0, "T0"), (0, "T1"), (0, "T2")]                   # one injection per task, for every track
    for t in ts:
        assert len(t.rows) == 15 and len(t.units) == 3
        for tid in ("T0", "T1", "T2"):
            rows = [r for r in t.rows if r["task_id"] == tid]
            assert sorted(r["cause"] for r in rows) == sorted(LABELS)       # all five, once each
            by_iteration = collections.defaultdict(list)
            for r in rows:
                by_iteration[r["iteration"]].append((r["cause"], r["forced"]))
            assert by_iteration[0] == [("LOCK_CONTENTION", False), ("INSERT_LARGE_DATA", False)]   # the planner's names, its order
            assert by_iteration[1] == [("VACUUM", True)]                    # it named only a done cause: the next in the benchmark's order
            assert by_iteration[2] == [("REDUNDANT_INDEX", True), ("FETCH_LARGE_DATA", True)]   # it stopped: the rest together
            assert [r["assignment"] for r in rows][:2] == ["check pg_locks for waits", "look for large INSERTs"]
    solo_planners = [p for p in Planner.made if p.track == "solo"]
    assert len(solo_planners) == 3 and all(len(p.updates) == 3 for p in solo_planners)
    assert "Final answer: yes" in solo_planners[0].updates[0] and "agent2" in solo_planners[0].updates[0]   # it read the verdicts
    # the solo track commits the central model's own findings (always yes): every cause yes, so the planner names them all
    assert all(set(u["verdicts"].values()) == {"YES"} for u in ts[0].units)
    # it lists them in the order it received them; the benchmark's rule keeps the first two allowed guesses
    assert all(u["planner"]["predicted"] == ["LOCK_CONTENTION", "INSERT_LARGE_DATA"] for u in ts[0].units)
    assert [u["planner"]["correct"] for u in ts[0].units] == [1, 0, 1]
    # the verdicts' own diagnosis keeps the first allowed guesses in the benchmark's order: T1's causes are not among them
    assert ts[0].units[1]["verdict_diagnosis"]["predicted"] == ["INSERT_LARGE_DATA", "LOCK_CONTENTION"] and ts[0].units[1]["verdict_diagnosis"]["f1"] == 0


def test_labels_follow_the_injected_anomaly_and_the_record_learns_the_reliable_peer():
    a, ts = adapter(), tracks()
    run_online(a, ts, central_fn=central_fn, features_fn=features_fn, workers=8, log=lambda *_: None)
    solo, peers, tilt, comb = ts

    for t in (peers, tilt, comb):
        for r in t.rows:
            truth = r["truth"] == "yes"
            assert r["peer_correct_by_peer"][0] == 1 and r["peer_correct_by_peer"][1] == int(not truth)   # p0 right, p1 always no
            assert r["peer_verdicts"][0] == ("YES" if truth else "NO")
    assert all(r["own_verdict"] == "YES" for r in solo.rows) and [r["correct"] for r in solo.rows] == [int(r["truth"] == "yes") for r in solo.rows]
    first, last = min(tilt.rows, key=lambda r: r["pos"]), max(tilt.rows, key=lambda r: r["pos"])
    assert set(first["memory_prob"]) == {0.5}
    assert last["memory_prob"][last["peer_order"].index(0)] == max(last["memory_prob"])
    assert sum(r["correct"] for r in tilt.rows) > sum(r["correct"] for r in peers.rows)
    m = metrics(tilt, a)
    assert m["tasks"] == 3 and m["balanced_accuracy"] is not None and 0 <= m["planner"]["accuracy"] <= 1 and "verdict_diagnosis" in m
    assert metrics(solo, a)["no_recall"] == 0.0 and metrics(solo, a)["yes_recall"] == 1.0
    assert "combination" in metrics(comb, a) and all(r["choice"] in ("peers+memory", "question alone") for r in comb.rows)


def test_tasks_run_on_several_lanes_with_one_injection_each():
    prepared = []
    a, ts = adapter(lanes=2, prepared=prepared), tracks()[:2]
    run_online(a, ts, central_fn=central_fn, features_fn=features_fn, workers=8, log=lambda *_: None)
    assert sorted(t for _, t in prepared) == ["T0", "T1", "T2"] and {l for l, _ in prepared} == {0, 1}
    assert all(len(t.units) == 3 and len(t.rows) == 15 for t in ts)


def test_a_failing_planner_or_investigator_costs_its_part_not_the_run():
    class Broken(Planner):
        def assign(self):
            raise ValueError("JSON parsing failed")

        def decide(self, results):
            raise ValueError("no JSON")

    def flaky(lane, model, t, agent, assignment):
        if model == "p3":
            raise ConnectionError("server gone")
        return investigate(lane, model, t, agent, assignment)

    a, ts = adapter(planner_for=lambda tr, l, t: Broken(tr, l, t), investigate=flaky), tracks()[1:2]
    run_online(a, ts, central_fn=central_fn, features_fn=features_fn, workers=8, log=lambda *_: None)
    rows = ts[0].rows
    assert len(rows) == 15 and all(r["forced"] for r in rows)                # no planner: the benchmark's order, one per iteration
    assert [r["cause"] for r in rows if r["task_id"] == "T0"] == LABELS
    assert all(r["peer_correct_by_peer"][3] == 0 and r["peer_verdicts"][3] is None for r in rows)   # a failed investigation is wrong


def test_the_verdicts_are_parsed_and_graded_like_the_benchmarks_findings():
    ev = {"answer": "yes"}
    assert verdict("evidence...\nFinal answer: yes") == "YES" and verdict("Root cause investigated: VACUUM. Verdict: NO") == "NO"
    assert verdict("<think>maybe yes</think>The locks pile up, so no.") == "NO" and verdict("I could not decide.") is None
    assert grade(ev, "Final answer: yes") == 1 and grade(ev, "Final answer: no") == 0 and grade(ev, "nothing") == 0
    assert diagnosis({"VACUUM": "Final answer: yes", "LOCK_CONTENTION": "Final answer: yes", "FETCH_LARGE_DATA": "no"}, LABELS) \
        == "Final answer: LOCK_CONTENTION, VACUUM"


def test_the_report_sets_the_record_against_verdict_aware_tables():
    from feedback_state.reliability_tables import compare
    from pipeline.marble_online import CELLS

    rows = []
    for pos in range(40):   # peer 0's YES is always right, its NO never; peer 1 is right half the time whatever it says
        truth = pos % 2 == 0
        v0 = "YES" if pos % 4 == 0 else "NO"
        rows.append({"pos": pos, "id": f"T{pos}::VACUUM", "cause": "VACUUM", "source": "S", "peer_order": [0, 1], "memory_prob": [0.5, 0.5],
                     "peer_verdicts": [v0, "NO"], "peer_correct": [int((v0 == "YES") == truth and v0 == "YES"), int(not truth)]})
    rel = compare(rows, CELLS)
    assert rel["per peer and its verdict"]["auc"] > rel["per peer"]["auc"]
    assert set(rel) >= {"record", "per peer", "per peer and cause", "per peer, cause and its verdict", "per peer and scenario", "_reference"}


def test_the_database_swarm_expands_to_its_jobs(tmp_path):
    from pipeline.config import load
    from pipeline.run import Plan

    over = [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", f"paths.logs={tmp_path}/logs"]
    plan = Plan(load(EXPERIMENTS / "marble_db_online.yaml", over), "marble_db_online.yaml", smoke=False, gpus=list(range(8)))
    *features, job = plan.online()
    assert all(f.wave == 0 and "mixed_train_big6" in f.cmd for f in features) and job.wave == 1
    assert job.gpus == 8 and "pipeline.marble_online online" in job.cmd and "datasets/marble_db/tasks.jsonl" in job.cmd
    assert "--queries 3" in job.cmd and "--lanes 1" in job.cmd and "--pg-data /mnt/data/peilin/pg" in job.cmd and "--served-name Qwen3-4B" in job.cmd
    assert "--fit-peers 6" in job.cmd and '"reasoning": true' in job.cmd and "--limit" not in job.cmd
    [report] = plan.diagnose()
    assert "pipeline.marble_online report" in report.cmd and "--eval online_tilt=" in report.cmd
    smoke = Plan(load(EXPERIMENTS / "marble_db_online.yaml", over), "marble_db_online.yaml", smoke=True, gpus=list(range(8)))
    *_, job = smoke.online()
    assert "--limit 2" in job.cmd and "/smoke/" in job.cmd


@pytest.mark.parametrize("lanes", [1, 3])
def test_the_run_ends_when_every_task_is_done(lanes):
    a, ts = adapter(lanes=lanes), tracks()[:1]
    run_online(a, ts, central_fn=central_fn, features_fn=features_fn, workers=4, log=lambda *_: None)
    assert len(ts[0].units) == 3 and a.events(ts) == []
