"""The agent swarm: grading of diagnoses and findings, the stream built from the events, and the swarm step's jobs."""
import json
from pathlib import Path

import pytest

from feedback_state.swarm import (cause_of, exact, finding_correct, graded, hit, predicted_causes, stream_record, summarize_results,
                                  verdict_answer, verdict_of)
from feedback_state.tasks import peer_is_correct
from pipeline.config import load
from pipeline.layout import Layout
from pipeline.run import Plan

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"
TASKS = Path(__file__).resolve().parents[2] / "datasets" / "marble_db" / "tasks.jsonl"


def test_the_benchmark_tasks_are_released_with_the_repository():
    tasks = [json.loads(l) for l in TASKS.open()]
    assert len(tasks) == 100 and len({t["id"] for t in tasks}) == 100
    assert {len(t["root_causes"]) for t in tasks} == {1, 2} and {t["max_iterations"] for t in tasks} == {5}
    assert [cause_of(a["profile"]) for a in tasks[0]["agents"]] == ["INSERT_LARGE_DATA", "LOCK_CONTENTION", "VACUUM", "REDUNDANT_INDEX", "FETCH_LARGE_DATA"]
    assert all(set(t["root_causes"]) <= set(t["labels"]) for t in tasks)


def test_a_diagnosis_is_graded_by_the_benchmarks_hit_rule_and_the_exact_set():
    text = '{"most likely causes": ["LOCK_CONTENTION", "Redundant index", "VACUUM"]}\nFinal answer: LOCK_CONTENTION, REDUNDANT_INDEX'
    assert predicted_causes(text) == ["LOCK_CONTENTION", "REDUNDANT_INDEX"]                   # the Final answer line wins
    assert predicted_causes('{"causes": ["fetch large data", "INSERT_LARGE_DATA", "INSERT_LARGE_DATA"]}') == ["FETCH_LARGE_DATA", "INSERT_LARGE_DATA"]
    assert hit(text, ["REDUNDANT_INDEX"], 2) and not hit(text, ["VACUUM"], 2)
    assert exact(text, ["LOCK_CONTENTION", "REDUNDANT_INDEX"], 3) and not exact(text, ["LOCK_CONTENTION"], 3)
    assert graded("nothing found", {"root_causes": ["VACUUM"], "number_of_labels_pred": 2}) == {"correct": 0, "exact": 0, "predicted": []}
    record = {"task_type": "dbdiag", "answer": ["VACUUM"], "number_of_labels_pred": 2, "peer_correct": {"peer_0": 1.0}}
    assert peer_is_correct(record, None, "Final answer: INSERT_LARGE_DATA, VACUUM") and not peer_is_correct(record, None, "Final answer: INSERT_LARGE_DATA, LOCK_CONTENTION")
    assert peer_is_correct(record, "peer_0", "anything")                                        # a finding keeps its own label


def test_a_finding_is_right_when_its_verdict_matches_the_ground_truth():
    yes = "Root cause investigated: LOCK_CONTENTION. Verdict: YES\npg_locks shows many waits."
    assert verdict_of(yes) == "YES" and verdict_of("Root cause investigated: VACUUM. **Verdict: no**") == "NO" and verdict_of("unsure") is None
    assert finding_correct(yes, "LOCK_CONTENTION", ["LOCK_CONTENTION", "VACUUM"]) == 1
    assert finding_correct(yes, "LOCK_CONTENTION", ["VACUUM"]) == 0
    assert finding_correct("no verdict given", "VACUUM", ["INSERT_LARGE_DATA"]) == 0            # an unreadable finding counts as wrong
    findings = [{"cause": c, "text": f"Root cause investigated: {c}. Verdict: {v}"} for c, v in
                (("INSERT_LARGE_DATA", "NO"), ("LOCK_CONTENTION", "YES"), ("VACUUM", "NO"), ("REDUNDANT_INDEX", "YES"), ("FETCH_LARGE_DATA", "NO"))]
    assert verdict_answer(findings) == "Final answer: LOCK_CONTENTION, REDUNDANT_INDEX"
    assert verdict_answer(findings[:1]) == "Final answer: NONE"
    assert summarize_results([{"agent1": "x" * 2000}]).count("x") == 1000 - len("- {'agent1': '")


def test_the_stream_holds_the_findings_as_the_answers_with_their_labels():
    task = json.loads(TASKS.open().readline())
    findings = [{"agent_id": f"agent{i + 1}", "cause": c, "text": f"Root cause investigated: {c}. Verdict: {'YES' if c in task['root_causes'] else 'NO'}",
                 "verdict": "YES" if c in task["root_causes"] else "NO", "correct": 1}
                for i, c in enumerate(["INSERT_LARGE_DATA", "LOCK_CONTENTION", "VACUUM", "REDUNDANT_INDEX", "FETCH_LARGE_DATA"])]
    rec = stream_record({"swarm": {"findings": findings}}, task, "Qwen3-4B")
    assert rec["id"] == task["id"] and rec["task_type"] == "dbdiag" and rec["answer"] == task["root_causes"]
    assert sorted(rec["peer_responses"]) == [f"peer_{k}" for k in range(5)] and rec["peer_correct"]["peer_1"] == 1.0
    assert rec["peer_metadata"]["peer_4"]["cause"] == "FETCH_LARGE_DATA" and "Possible root causes:" in rec["problem"]


def test_the_swarm_experiment_runs_the_benchmark_then_the_usual_steps(tmp_path):
    cfg = load(EXPERIMENTS / "swarm.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models"])
    plan = Plan(cfg, "swarm.yaml", smoke=False, gpus=[0, 1, 2])

    assert Layout(cfg).registry.problems() == []
    sw = {j.name: j for j in plan.swarm()}
    assert sorted(sw) == ["swarm_merge_q3_4b_marble_db", "swarm_q3_4b_marble_db_0", "swarm_q3_4b_marble_db_1"]     # shards: 2
    first = sw["swarm_q3_4b_marble_db_0"]
    assert f"--tasks {TASKS} --model /models/Qwen3-4B --served-name Qwen3-4B --iterations 5 --anomaly-duration 60" in first.cmd
    assert "--shard 0/2 --port 8123 --pg-port 5432 --pg-data /mnt/data/peilin/pg/5432" in first.cmd and first.gpus == 1
    assert "--shard 1/2 --port 8124 --pg-port 5433 --pg-data /mnt/data/peilin/pg/5433" in sw["swarm_q3_4b_marble_db_1"].cmd
    merge = sw["swarm_merge_q3_4b_marble_db"]
    assert merge.gpus == 0 and merge.wave == 1 and merge.done == tmp_path / "data/marble_db/test.jsonl" and "--partial" not in merge.cmd
    assert (f"--stream {tmp_path}/data/marble_db/test.jsonl --solo {tmp_path}/out/eval/q3_4b/marble_db/solo "
            f"--swarm {tmp_path}/out/eval/q3_4b/marble_db+own/swarm --verdicts {tmp_path}/out/eval/q3_4b/marble_db+own/verdicts") in merge.cmd
    own = {j.name: j for j in plan.own()}
    assert own["own_q3_4b_marble_db"].cmd.endswith(f"--eval {tmp_path}/out/eval/q3_4b/marble_db/solo --out {tmp_path}/data/marble_db+q3_4b/test.jsonl")
    assert all("--peers 6 " in j.cmd for j in plan.features()) and plan.record()[0].cmd.endswith("--own-slot 5")
    assert sorted(j.name for j in plan.evaluate()) == ["eval_q3_4b_marble_db_peers", "eval_q3_4b_marble_db_tilt"]
    with pytest.raises(SystemExit, match="released model"):
        cfg2 = load(EXPERIMENTS / "swarm.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "central=[comb3b]",
                                                 "trained={comb3b: {init: q3_4b, judge: q3_4b}}"])
        p2 = Plan(cfg2, "swarm.yaml", smoke=False, gpus=[0]); p2.dry = True
        (tmp_path / "out/train/comb3b/hf/global_step_1").mkdir(parents=True)
        p2.swarm()


def test_another_central_model_runs_the_swarm_on_its_own_stream_ports_and_clusters(tmp_path):
    cfg = load(EXPERIMENTS / "swarm_qwen3_8b.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models"])
    plan = Plan(cfg, "swarm_qwen3_8b.yaml", smoke=False, gpus=[0, 5, 6, 7])
    sw = {j.name: j for j in plan.swarm()}
    assert sorted(sw) == [f"swarm_merge_qwen3_8b_marble_db_qwen3_8b"] + [f"swarm_qwen3_8b_marble_db_qwen3_8b_{k}" for k in range(4)]
    first = sw["swarm_qwen3_8b_marble_db_qwen3_8b_0"]
    assert "--model /models/Qwen3-8B --served-name Qwen3-8B" in first.cmd and "--shard 0/4 --port 8150 --pg-port 5450 --pg-data /mnt/data/peilin/pg/5450" in first.cmd
    assert "--port 8153 --pg-port 5453" in sw["swarm_qwen3_8b_marble_db_qwen3_8b_3"].cmd
    assert sw["swarm_merge_qwen3_8b_marble_db_qwen3_8b"].done == tmp_path / "data/marble_db_qwen3_8b/test.jsonl"
    assert cfg["swarm"]["iterations"] == 5 and cfg["swarm"]["tasks"] == "datasets/marble_db/tasks.jsonl"       # inherited from swarm.yaml
    import yaml

    tags = yaml.safe_load((EXPERIMENTS.parent / "datasets" / "marble_db_qwen3_8b.yaml").read_text())["peers"]
    peers = Layout(cfg).peer_models(tags)
    assert [p["role"] for p in peers] == ["INSERT_LARGE_DATA", "LOCK_CONTENTION", "VACUUM", "REDUNDANT_INDEX", "FETCH_LARGE_DATA"]
    assert {p["model"] for p in peers} == {"qwen3_8b"} and Layout(cfg).stream("marble_db_qwen3_8b")["peers"] == 5


def test_a_swarm_smoke_run_builds_its_stream_under_smoke(tmp_path):
    cfg = load(EXPERIMENTS / "swarm.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data"])
    plan = Plan(cfg, "swarm.yaml", smoke=True, gpus=[0])
    jobs = plan.swarm()
    assert len(jobs) == 1 and "--limit 4" in jobs[0].cmd and jobs[0].done == tmp_path / "out/smoke/data/marble_db/test.jsonl"
    assert jobs[0].cmd.endswith("--max-new-tokens 768 --partial")                     # the merge takes the four tasks that exist
    assert all(str(j.done).startswith(str(tmp_path / "out/smoke/")) for j in plan.own() + plan.features() + plan.record())


def test_the_swarm_table_lists_the_benchmark_and_ours_side_by_side(tmp_path):
    from pipeline.table import build

    cfg = load(EXPERIMENTS / "swarm.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data"])
    L = Layout(cfg)
    for c, acc in (("swarm", 0.5), ("verdicts", 0.4), ("tilt", 0.7), ("combination", 0.72)):
        d = L.eval_dir("q3_4b", "marble_db", c)
        d.mkdir(parents=True)
        (d / "eval_metrics.json").write_text(json.dumps({"accuracy": acc, "share_peers_memory": 0.8, "reading_line": {}}))
    text = build(cfg, smoke=False)
    assert "| row | marble_db: question alone | marble_db: MARBLE swarm | marble_db: agents' verdicts | marble_db: question + peers | marble_db: peers + memory | marble_db: combination |" in text
    assert "| q3_4b |   -   |  50.0 |  40.0 |   -   |  70.0 |  72.0 |" in text
    assert "MARBLE swarm (`swarm`): the benchmark as-is" in text
