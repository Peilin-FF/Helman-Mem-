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


def test_a_team_of_different_models_serves_each_once_and_gives_every_agent_its_own(tmp_path):
    from feedback_state.swarm import route_for

    cfg = load(EXPERIMENTS / "swarm_team.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models"])
    assert Layout(cfg).registry.problems() == []
    jobs = Plan(cfg, "swarm_team.yaml", smoke=False, gpus=[0, 1, 2, 3, 4, 5, 6, 7]).swarm()
    assert [j.name for j in jobs] == ["swarm_q3_4b_marble_db_team"] and jobs[0].gpus == 6                # the planner's model and five agents'
    cmd = jobs[0].cmd
    assert "python -m pipeline.swarm team --tasks" in cmd and "--model /models/Qwen3-4B --served-name Qwen3-4B" in cmd and "--tool-parser hermes" in cmd
    assert "--workers 5 --port 8170 --pg-port 5460 --pg-data /mnt/data/peilin/pg && python -m pipeline.swarm merge" in cmd
    agents = json.loads(cmd.split("--agents '")[1].split("' --workers")[0])
    assert agents["agent1"] == {"name": "Meta-Llama-3.1-8B-Instruct", "path": "/models/Meta-Llama-3.1-8B-Instruct", "parser": "llama3_json"}
    assert [agents[f"agent{i}"]["parser"] for i in range(1, 6)] == ["llama3_json", "mistral", "hermes", "mistral", "hermes"]
    assert jobs[0].done == tmp_path / "data/marble_db_team/test.jsonl"
    smoke = Plan(cfg, "swarm_team.yaml", smoke=True, gpus=[0, 1, 2]).swarm()[0]
    assert "--workers 1 " in smoke.cmd and smoke.gpus == 3 and smoke.cmd.endswith("--partial")            # fewer GPUs than models: they share
    routes = {"Qwen3-4B": "http://localhost:8170/v1", "Qwen2.5-7B-Instruct": "http://localhost:8173/v1"}
    assert route_for("openai/Qwen2.5-7B-Instruct", routes, "x") == "http://localhost:8173/v1" and route_for("openai/unknown", routes, "x") == "x"
    assert route_for("openai/Qwen3-4B", None, "http://one") == "http://one"


def test_a_finished_run_becomes_sub_step_questions_for_the_pool_of_peers(tmp_path):
    from pipeline.swarm import cmd_steps, evidence_of

    tasks = [json.loads(l) for l in TASKS.open()][:2]
    shard = tmp_path / "run" / "shard0"; shard.mkdir(parents=True)
    with (shard / "events.jsonl").open("w") as f:
        for t in tasks:
            its = [{"iteration": 1, "results": [{"agent2": "Result from the model: checked pg_locks\nResult from the function: []"}]},
                   {"iteration": 2, "results": [{"agent2": "again"}, {"agent5": "Result from the function: big SELECTs"}]}]
            f.write(json.dumps({"id": t["id"], "model": "Qwen3-4B", "swarm": {"iterations": its}}) + "\n")
    ev = json.loads((shard / "events.jsonl").open().readline())
    assert evidence_of(ev, "agent2").startswith("[iteration 1] Result from the model: checked pg_locks") and "[iteration 2] again" in evidence_of(ev, "agent2")
    assert evidence_of(ev, "agent1") == "(the agent ran no query on this task)"
    out = tmp_path / "q" / "test.jsonl"
    cmd_steps(type("A", (), {"events": tmp_path / "run", "tasks": TASKS, "limit": 2, "out": out})())
    rows = [json.loads(l) for l in out.open()]
    assert len(rows) == 10 and rows[0]["id"] == f"{tasks[0]['id']}::INSERT_LARGE_DATA" and rows[0]["task_type"] == "boolqa"
    assert [r["answer"] for r in rows[:5]] == ["yes" if c in tasks[0]["root_causes"] else "no" for c in
                                               ["INSERT_LARGE_DATA", "LOCK_CONTENTION", "VACUUM", "REDUNDANT_INDEX", "FETCH_LARGE_DATA"]]
    assert rows[1]["problem"] == "Is LOCK_CONTENTION a root cause of this database's performance issue?" and "checked pg_locks" in rows[1]["context"]
    assert rows[1]["peer_responses"] == {} and rows[1]["task_id"] == tasks[0]["id"] and rows[1]["evidence_model"] == "Qwen3-4B"


def test_the_steps_experiment_has_the_pool_answer_then_the_usual_steps_then_the_diagnosis(tmp_path):
    cfg = load(EXPERIMENTS / "swarm_steps.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models"])
    assert Layout(cfg).registry.problems() == []
    plan = Plan(cfg, "swarm_steps.yaml", smoke=False, gpus=[0, 1, 2, 3])
    build = {j.name: j for j in plan.swarm()}
    assert sorted(build) == ["steps_marble_db_steps_q", "steps_marble_db_steps_qwen3_8b_q"] and all(j.gpus == 0 for j in build.values())
    assert f"--events {tmp_path}/out/swarm/q3_4b/marble_db --tasks {TASKS} --out {tmp_path}/data/marble_db_steps_q/test.jsonl" in build["steps_marble_db_steps_q"].cmd
    assert f"--events {tmp_path}/out/swarm/qwen3_8b/marble_db_qwen3_8b " in build["steps_marble_db_steps_qwen3_8b_q"].cmd
    peers = plan.peers()
    assert len(peers) == 12 and all("--mode honest" in j.cmd and "--shards 1 --shard 0" in j.cmd for j in peers)                # six peers, two streams, one shard each
    assert any(f"--stream {tmp_path}/data/marble_db_steps_q/test.jsonl --output {tmp_path}/data/marble_db_steps_answers/gemma-3-4b-it" in j.cmd for j in peers)
    assert any("DeepSeek-R1-Distill-Qwen-7B" in j.cmd and j.cmd.rstrip().endswith("--reasoning") for j in peers)
    add = {j.name: j for j in plan.streams()}["stream_marble_db_steps"]
    assert add.cmd.startswith(f"python -m pipeline.streams add --base {tmp_path}/data/marble_db_steps_q/test.jsonl --answers {tmp_path}/data/marble_db_steps_answers --peer gemma-3-4b-it")
    assert add.cmd.endswith(f"--peer DeepSeek-R1-Distill-Qwen-7B --out {tmp_path}/data/marble_db_steps/test.jsonl") and add.gpus == 0
    dg = {j.name: j for j in plan.diagnose()}["diagnose_q3_4b_marble_db_steps"]
    assert f"--eval solo={tmp_path}/out/eval/q3_4b/marble_db_steps/solo --eval tilt={tmp_path}/out/eval/q3_4b/marble_db_steps+own/tilt" in dg.cmd
    assert f"--eval combination={tmp_path}/out/eval/q3_4b/marble_db_steps+own/combination" in dg.cmd and f"--events {tmp_path}/out/swarm/q3_4b/marble_db " in dg.cmd
    assert all("--peers 7 " in j.cmd for j in plan.features()) and plan.record()[0].cmd.endswith("--own-slot 6")                # the pool of six and the model's own answer
    smoke = Plan(cfg, "swarm_steps.yaml", smoke=True, gpus=[0])
    sj = smoke.swarm()[0]
    assert sj.cmd.endswith("--limit 4") and f"--events {tmp_path}/out/swarm/q3_4b/marble_db " in sj.cmd and str(sj.done).startswith(f"{tmp_path}/out/smoke/data/")


def test_a_pool_of_peers_investigates_every_sub_step_and_becomes_a_stream(tmp_path):
    from feedback_state.swarm import verdict_of
    from pipeline.swarm import cmd_merge_pool

    assert verdict_of("pg_locks shows no waits.\nFinal answer: no") == "NO" and verdict_of("<think>maybe no</think>Waits found. Final answer: yes") == "YES"
    assert verdict_of("Root cause investigated: VACUUM. Verdict: YES\nFinal answer: no") == "YES" and verdict_of("unsure") is None
    cfg = load(EXPERIMENTS / "swarm_pool.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models"])
    assert Layout(cfg).registry.problems() == []
    plan = Plan(cfg, "swarm_pool.yaml", smoke=False, gpus=list(range(8)))
    job = plan.swarm()[0]
    assert job.name == "swarm_q3_4b_marble_db_pool" and job.gpus == 7 and "--force-tool query_db --workers 5 --port 8180 --pg-port 5470" in job.cmd
    teams = json.loads(job.cmd.split("--teams '")[1].split("' --force-tool")[0])
    assert list(teams) == ["Qwen3-4B", "gemma-3-4b-it", "Phi-4-mini-instruct", "Qwen2.5-Coder-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
                           "DeepSeek-Coder-V2-Lite-Instruct", "DeepSeek-R1-Distill-Qwen-7B"]
    assert teams["DeepSeek-R1-Distill-Qwen-7B"]["reasoning"] and teams["DeepSeek-Coder-V2-Lite-Instruct"]["prefix_caching"] is False
    assert teams["DeepSeek-Coder-V2-Lite-Instruct"]["env_vars"] == {"VLLM_USE_V1": 0}
    assert "merge-pool" in job.cmd and "--peers gemma-3-4b-it,Phi-4-mini-instruct," in job.cmd and f"--solo {tmp_path}/out/eval/q3_4b/marble_db_pool/solo" in job.cmd
    assert f"--events {tmp_path}/out/swarm/q3_4b/marble_db_pool " in plan.diagnose()[0].cmd and all("--peers 7 " in j.cmd for j in plan.features())

    tasks = [json.loads(l) for l in TASKS.open()][:2]
    pool = ["A", "B"]
    shard = tmp_path / "run" / "shard0"; shard.mkdir(parents=True)
    with (shard / "events.jsonl").open("w") as f:
        for t in tasks:
            def team(always):
                fs = [{"agent_id": a["agent_id"], "cause": c, "text": f"evidence\nFinal answer: {always}", "verdict": always.upper(),
                       "correct": int((always == "yes") == (c in t["root_causes"]))} for a, c in zip(t["agents"], ["INSERT_LARGE_DATA", "LOCK_CONTENTION", "VACUUM", "REDUNDANT_INDEX", "FETCH_LARGE_DATA"])]
                return {"final": "", "findings": fs, "errors": [], "correct": 0, "exact": 0}
            f.write(json.dumps({"id": t["id"], "teams": {"Qwen3-4B": team("no"), "A": team("yes"), "B": team("no")}}) + "\n")
    ns = type("A", (), {"out": tmp_path / "run", "tasks": TASKS, "stream": tmp_path / "s" / "test.jsonl", "solo": tmp_path / "solo", "peers": ",".join(pool),
                        "model": "/models/Qwen3-4B", "served_name": "Qwen3-4B", "max_new_tokens": 768, "windows": 10, "partial": True})()
    cmd_merge_pool(ns)
    rows = [json.loads(l) for l in ns.stream.open()]
    assert len(rows) == 10 and rows[0]["task_type"] == "boolqa" and sorted(rows[0]["peer_responses"]) == ["peer_0", "peer_1"]
    first = rows[[r["cause"] for r in rows[:5]].index(tasks[0]["root_causes"][0])]
    assert first["answer"] == "yes" and first["peer_correct"] == {"peer_0": 1.0, "peer_1": 0.0} and first["peer_metadata"]["peer_0"]["model"] == "A"
    own = [json.loads(l) for l in (ns.solo / "generations.jsonl").open()]
    assert len(own) == 10 and sum(r["correct"] for r in own) == 10 - sum(len(t["root_causes"]) for t in tasks)      # the own team says no everywhere
    assert json.load((ns.solo / "eval_metrics.json").open())["mode"] == "solo" and json.load((ns.stream.parent / "teams.json").open())["A"]["yes_rate"] == 1.0


def test_shards_claim_tasks_and_a_stopped_shard_gives_its_unfinished_ones_back(tmp_path):
    from pipeline.swarm import all_done_ids, claim, release_stale_claims

    claims = tmp_path / "claims"; claims.mkdir()
    assert claim(claims, "T1", "shard0") and not claim(claims, "T1", "shard1") and claim(claims, "T2", "shard1")
    (tmp_path / "shard0").mkdir(); (tmp_path / "shard1").mkdir()
    (tmp_path / "shard0" / "events.jsonl").write_text(json.dumps({"id": "T1"}) + "\n")
    assert all_done_ids(tmp_path) == {"T1"}
    assert release_stale_claims(claims, "shard1", done_ids := all_done_ids(tmp_path)) == ["T2"]      # shard1 stopped before T2's event
    assert release_stale_claims(claims, "shard0", done_ids) == [] and claim(claims, "T2", "shard0")   # T1 stays claimed, T2 is free again


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
