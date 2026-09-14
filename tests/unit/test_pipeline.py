"""The pipeline's config, layout and job expansion: what `bash run.sh <experiment> --dry-run` would run.

CPU only, no models and no data: the expansion must be right before any GPU time is spent.
"""
import json
from pathlib import Path

import pytest
import yaml

from pipeline.config import deep_merge, load
from pipeline.layout import Layout
from pipeline.registry import load_registry
from pipeline.run import Plan, stale, streams_command

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"


def test_a_config_inherits_its_base_and_overrides_win():
    cfg = load(EXPERIMENTS / "main.yaml", ["evaluation.max_new_tokens=1024", "eval_conditions=[tilt]"])

    assert cfg["name"] == "main"
    assert cfg["datasets"] == ["indist6", "ood6"]                         # the experiment's own value
    assert cfg["record"]["dim"] == 256                                   # inherited from base.yaml
    assert cfg["evaluation"]["max_new_tokens"] == 1024                   # the override
    assert cfg["evaluation"]["engine"] == "vllm"                         # merged, not replaced
    assert cfg["eval_conditions"] == ["tilt"]


def test_deep_merge_merges_mappings_and_replaces_lists():
    assert deep_merge({"a": {"x": 1, "y": 2}, "l": [1, 2]}, {"a": {"y": 3}, "l": [9]}) == {"a": {"x": 1, "y": 3}, "l": [9]}


def test_every_experiment_config_loads_and_expands(tmp_path):
    for f in sorted(EXPERIMENTS.glob("*.yaml")):
        cfg = load(f, [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", f"paths.logs={tmp_path}/logs"])
        plan = Plan(cfg, str(f), smoke=False, gpus=[0, 1])
        for step in cfg["steps"]:
            if step == "evaluate" and cfg.get("arms"):
                continue   # needs a trained checkpoint on disk: covered by the training test below
            jobs = getattr(plan, step)()
            assert jobs, f"{f.name}: step {step} expands to no jobs"


def test_the_registry_is_consistent_and_groups_expand():
    reg = load_registry()

    assert reg.problems() == []
    assert "_misleading" not in reg.datasets                              # a template is not a dataset
    rates = reg.expand(["misleading_rates"])
    assert len(rates) == 10 and rates[0] == "indist6_misleading_p000" and rates[-1] == "ood6_misleading_p100"
    assert reg.expand(["indist6", "misleading_rates", "indist6"]) == ["indist6"] + rates   # in order, each once
    assert reg.dataset("ood6_misleading_p025")["answers"] == "ood6_misleading"            # {base} filled from the template
    assert reg.dataset("indist6")["peers"] == ["gemma3_4b", "phi4_mini", "qwen25_coder_7b", "llama31", "deepseek_coder_v2_lite", "r1_distill_qwen_7b"]
    assert reg.peer("r1_distill_qwen_7b")["reasoning"] and "reasoning" not in reg.model("r1_distill_qwen_7b")      # a peer-only setting
    with pytest.raises(KeyError, match="did you mean"):
        reg.dataset("indist6_misleading_p05")


def test_the_registry_reports_broken_references(tmp_path):
    for sub in ("datasets", "models", "peers"):
        (tmp_path / sub).mkdir()
    (tmp_path / "models/m.yaml").write_text("path: M\n")
    (tmp_path / "peers/p.yaml").write_text("model: m\n")
    (tmp_path / "peers/q.yaml").write_text("model: ghost\n")
    (tmp_path / "datasets/s.yaml").write_text("path: s/test.jsonl\npeers: [p, q, nobody]\n")
    (tmp_path / "datasets/s_bad.yaml").write_text("kind: misleading\nbase: s\nanswers: nowhere\nregime: p150\n")
    problems = "\n".join(load_registry(tmp_path).problems())

    assert "ghost" in problems and "nobody" in problems and "nowhere" in problems and "regime must be pNNN" in problems


def test_registered_datasets_resolve_to_data_and_smoke_builds_are_isolated(tmp_path):
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.data={tmp_path}/data", f"paths.outputs={tmp_path}/out"])
    real, smoke = Layout(cfg), Layout(cfg, smoke=True)

    assert real.stream("indist6")["path"] == tmp_path / "data/indist6/test.jsonl"
    assert real.stream("train6")["path"] == tmp_path / "data/mixed_train_big6/train.jsonl"
    p050 = real.stream("indist6_misleading_p050")
    assert p050["path"] == tmp_path / "data/indist6_misleading_p050/test.jsonl"
    assert (p050["kind"], p050["base"], p050["answers"], p050["regime"], p050["peers"]) == ("misleading", "indist6", "indist6_misleading", "p050", 6)
    assert real.stream("indist6_misleading")["path"] == tmp_path / "data/indist6_misleading"
    assert smoke.stream("indist6_misleading_p050")["path"] == tmp_path / "out/smoke/data/indist6_misleading_p050/test.jsonl"
    assert smoke.stream("indist6")["path"] == real.stream("indist6")["path"]          # the released stream is read, never written
    assert str(smoke.eval_dir("q3_4b", "indist6", "tilt")).startswith(str(tmp_path / "out/smoke"))
    with pytest.raises(KeyError):
        real.stream("nope")
    with pytest.raises(KeyError, match="group"):
        real.stream("misleading_rates")


def test_the_record_file_names_its_fit_and_any_non_default_setting(tmp_path):
    base = load(EXPERIMENTS / "main.yaml", [f"paths.outputs={tmp_path}"])
    assert Layout(base).record_file("q3_4b", "indist6").name == "shuffled0.fit-train6.jsonl"
    other = load(EXPERIMENTS / "main.yaml", [f"paths.outputs={tmp_path}", "record.dim=64"])
    assert Layout(other).record_file("q3_4b", "indist6").name == "shuffled0.fit-train6.qc-d64-lam100.jsonl"


def test_main_expands_to_the_reference_pipeline(tmp_path):
    cfg = load(EXPERIMENTS / "main.yaml", [f"paths.outputs={tmp_path}/out", "paths.models_root=/models"])
    plan = Plan(cfg, "main.yaml", smoke=False, gpus=[0, 1, 2, 3])

    feats = plan.features()
    assert len(feats) == 3 * 4                                            # train6 (fit) + indist6 + ood6, four shards each
    assert "--model /models/Qwen3-4B" in feats[0].cmd and "--shards 4 --shard 0" in feats[0].cmd
    rec = [j for j in plan.record() if j.name == "record_q3_4b_indist6"][0]
    assert "--fit-features" in rec.cmd and "train6" in rec.cmd and "--dim 256 --lam 100.0" in rec.cmd
    ev = {j.name: j for j in plan.evaluate()}
    assert len(ev) == 2 * 4
    assert "--mode peers --gamma 3.0 --bias-form logratio" in ev["eval_q3_4b_indist6_tilt"].cmd
    assert "--swap" in ev["eval_q3_4b_ood6_swap"].cmd and "--swap" not in ev["eval_q3_4b_ood6_tilt"].cmd
    assert "--mode solo --gamma 0.0" in ev["eval_q3_4b_indist6_solo"].cmd


def test_misleading_expands_datasets_into_answers_streams_and_evaluations(tmp_path):
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models",
                                                 "datasets=[indist6_misleading_p050, ood6_misleading_p100]"])
    plan = Plan(cfg, "misleading.yaml", smoke=False, gpus=[0])

    assert plan.eval_datasets() == ["indist6_misleading_p050", "ood6_misleading_p100"]
    assert plan.answer_datasets() == ["indist6_misleading", "ood6_misleading"]
    streams = {j.name: j for j in plan.streams()}
    cmd = streams["stream_indist6_misleading_p050"].cmd
    assert '"kind": "fraction"' in cmd and "--regime p050 " in cmd and f"--answers {tmp_path}/data/indist6_misleading " in cmd
    assert ("--peer-dirs gemma-3-4b-it,Phi-4-mini-instruct,Qwen2.5-Coder-7B-Instruct,Meta-Llama-3.1-8B-Instruct,"
            "DeepSeek-Coder-V2-Lite-Instruct,DeepSeek-R1-Distill-Qwen-7B ") in cmd
    assert all(j.gpus == 0 for j in streams.values())
    peers = plan.peers()
    assert len(peers) == 2 * 6                                            # two answers datasets, six peers, one shard each
    r1 = [j for j in peers if "DeepSeek-R1" in j.name][0]
    assert "--reasoning" in r1.cmd and "--model /models/DeepSeek-R1-Distill-Qwen-7B " in r1.cmd
    assert r1.done == tmp_path / "data/indist6_misleading/DeepSeek-R1-Distill-Qwen-7B/shard0of1.jsonl"
    coder = [j for j in peers if "DeepSeek-Coder" in j.name][0]
    assert coder.env == {"VLLM_USE_V1": "0"} and "--no-prefix-caching" in coder.cmd
    assert all("--no-prefix-caching" not in j.cmd for j in peers if "DeepSeek-Coder" not in j.name)
    assert all("--fit-features" not in j.cmd for j in plan.record())      # fit: self


def test_a_new_regime_is_a_new_dataset_file(tmp_path):
    root = Path(__file__).resolve().parents[2] / "configs"
    for sub in ("models", "peers"):
        (tmp_path / sub).symlink_to(root / sub)
    (tmp_path / "datasets").mkdir()
    for f in ("_misleading.yaml", "indist6.yaml", "indist6_misleading.yaml"):
        (tmp_path / "datasets" / f).write_text((root / "datasets" / f).read_text())
    (tmp_path / "datasets/indist6_saboteurs.yaml").write_text(
        "include: _misleading.yaml\nbase: indist6\nregime: {kind: fraction, rate: 1.0, peers: [1, 4]}\ndrop_forced: true\n")
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.registry={tmp_path}", f"paths.data={tmp_path}/data", "datasets=[indist6_saboteurs]"])
    L = Layout(cfg)

    assert L.registry.problems() == []
    cmd = streams_command(L, "indist6_saboteurs")
    assert '"peers": [1, 4]' in cmd and "--regime indist6_saboteurs " in cmd and "--drop-forced" in cmd
    assert Plan(cfg, "misleading.yaml", smoke=False, gpus=[0]).answer_datasets() == ["indist6_misleading"]


def test_smoke_uses_one_dataset_one_shard_and_a_small_record(tmp_path):
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.outputs={tmp_path}/out"])
    plan = Plan(cfg, "misleading.yaml", smoke=True, gpus=[0, 1, 2, 3])

    assert plan.eval_datasets() == ["indist6_misleading_p050"]
    assert all(str(j.done).startswith(str(tmp_path / "out/smoke/data/indist6_misleading/")) for j in plan.peers())
    assert all("--shards 1" in j.cmd and "--max-examples 48" in j.cmd for j in plan.features())
    assert all("--dim 32" in j.cmd for j in plan.record())
    assert all("--limit 48" in j.cmd for j in plan.streams())


def test_a_stored_evaluation_with_other_settings_is_reported_not_reused(tmp_path):
    m = tmp_path / "eval_metrics.json"
    m.write_text(json.dumps({"mode": "peers", "gamma": 3.0, "swap_record": False, "max_new_tokens": 768,
                             "record": "r.jsonl", "checkpoint": None, "accuracy": 0.5}))
    same = {"mode": "peers", "gamma": 3.0, "swap_record": False, "max_new_tokens": 768, "record": "r.jsonl", "checkpoint": None}

    assert stale(m, same) is None
    assert "gamma" in stale(m, dict(same, gamma=5.0))
    assert stale(tmp_path / "missing.json", same) is None
    moved = dict(same, record="/mnt/data/peilin/.rproj-jobs/sigma-mem/20260101-000000/r.jsonl")
    m.write_text(json.dumps(dict(moved, accuracy=0.5)))
    assert stale(m, same) is None   # written from a job snapshot that is gone by now: still the same record


def test_training_expands_records_data_and_one_run_per_arm_in_waves(tmp_path):
    cfg = load(EXPERIMENTS / "train_tilt.yaml", [f"paths.outputs={tmp_path}/out", "paths.models_root=/models"])
    plan = Plan(cfg, "train_tilt.yaml", smoke=False, gpus=list(range(8)))
    jobs = {j.name: j for j in plan.train()}

    assert jobs["record_q3_4b_train6_fixed"].wave == 0 and "fixed.fit-train6.jsonl" in jobs["record_q3_4b_train6_fixed"].cmd
    assert "--prompt peers --solo-fraction 0.25" in jobs["data_q3_4b_train6_peers_solo25"].cmd
    assert "--prompt solo" in jobs["data_q3_4b_train6_solo"].cmd
    tilt = jobs["train_run3b_tilt"]
    assert (tilt.wave, tilt.gpus) == (2, 8)
    assert "MODEL=/models/Qwen3-4B" in tilt.cmd and f"OUT={tmp_path}/out/train/run3b_tilt " in tilt.cmd
    assert "data.attn_gamma=3.0" in tilt.cmd and "attn_implementation=sdpa" in tilt.cmd
    assert "data.attn_gamma" not in jobs["train_ctrl3b_peers"].cmd
    assert len([j for j in jobs if j.startswith("train_")]) == 3


def test_trained_runs_are_evaluated_from_their_last_checkpoint(tmp_path):
    cfg = load(EXPERIMENTS / "train_tilt.yaml", [f"paths.outputs={tmp_path}/out", "paths.models_root=/models"])
    plan = Plan(cfg, "train_tilt.yaml", smoke=False, gpus=[0])
    for arm in cfg["arms"]:
        for step in (40, 276):
            (tmp_path / "out/train" / arm / "hf" / f"global_step_{step}").mkdir(parents=True)
    ev = {j.name: j for j in plan.evaluate()}

    assert len(ev) == 3 * 2 * 4
    assert "--checkpoint " + str(tmp_path / "out/train/run3b_tilt/hf/global_step_276") in ev["eval_run3b_tilt_indist6_tilt"].cmd
    assert "--record " + str(tmp_path / "out/record/q3_4b/indist6/shuffled0.fit-train6.jsonl") in ev["eval_run3b_tilt_indist6_tilt"].cmd


def test_a_smoke_training_run_overrides_the_config(tmp_path):
    cfg = load(EXPERIMENTS / "train_tilt.yaml", [f"paths.outputs={tmp_path}/out", "overrides=[trainer.total_training_steps=100]"])
    plan = Plan(cfg, "train_tilt.yaml", smoke=True, gpus=[0, 1])
    runs = [j for j in plan.train() if j.name.startswith("train_")]
    args = runs[0].cmd.split("train_grpo.sh", 1)[1]

    assert args.rindex("trainer.total_training_steps=2") > args.rindex("trainer.total_training_steps=100")   # hydra: the last one wins
    assert runs[0].gpus == 2 and len(runs) == 1                                                            # one arm in a smoke run


def test_the_central_prompts_are_the_ones_every_stored_result_used():
    from feedback_state.memory_generator import build_messages

    rec = {"task_type": "math", "problem": "2 + 2?", "answer": "4"}
    peers = build_messages(rec, ["4", "5"], mode="peers")
    solo = build_messages(rec, ["4", "5"], mode="solo")

    assert peers[0]["content"] == ("You are the central model of a multi-agent system. Several peer models answered the same question. "
                                   "Treat their answers as evidence, verify them yourself, and produce your own final answer.")
    assert solo[0]["content"] == "Answer the question."
    assert peers[1]["content"].startswith("Question:\n2 + 2?\n\nPeer answers:\n\nPeer 1:\n4\n\nPeer 2:\n5\n\nInstruction: ")


def test_paths_written_into_outputs_are_relative_to_the_repository():
    from pipeline.config import REPO, shown

    assert shown(REPO / "data" / "ood6_misleading_p050" / "test.jsonl") == "data/ood6_misleading_p050/test.jsonl"
    assert shown("/models/Qwen3-4B") == "/models/Qwen3-4B"
    assert shown(None) is None
    assert shown("/mnt/data/peilin/.rproj-jobs/sigma-mem/20260913-162846/data/ood6/test.jsonl") == "data/ood6/test.jsonl"
    assert shown("outputs/record/q3_4b/ood6/shuffled0.jsonl") == "outputs/record/q3_4b/ood6/shuffled0.jsonl"


def test_a_failed_job_is_retried_once_before_it_counts_as_failed(tmp_path):
    from pipeline.run import Job, Scheduler

    cfg = load(EXPERIMENTS / "main.yaml", [f"paths.outputs={tmp_path}/out", f"paths.logs={tmp_path}/logs"])
    L = Layout(cfg)
    L.run_dir("t").mkdir(parents=True)
    flag, out = tmp_path / "tried", tmp_path / "result"
    flaky = Job("x", "flaky", f"if [ -e {flag} ]; then touch {out}; else touch {flag}; kill -9 $$; fi", gpus=0, done=out)
    sched = Scheduler([0], "t", L, dry=False)
    sched.run(flaky)
    assert out.exists() and sched.failed == []

    broken = Job("x", "broken", "exit 3", gpus=0, done=tmp_path / "never")
    sched.run(broken)
    assert sched.failed == ["broken"] and "attempt 2" in L.log("t", "broken").read_text()


def test_question_only_is_shared_by_a_stream_and_its_misleading_variants(tmp_path):
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data",
                                                 "datasets=[indist6_misleading_p025, indist6_misleading_p050]"])
    plan = Plan(cfg, "misleading.yaml", smoke=False, gpus=[0])
    ev = {j.name: j for j in plan.evaluate()}

    assert sorted(n for n in ev if n.endswith("_solo")) == ["eval_q3_4b_indist6_solo"]          # one job, on the base stream
    assert f"--stream {tmp_path}/data/indist6/test.jsonl " in ev["eval_q3_4b_indist6_solo"].cmd
    assert "eval_q3_4b_indist6_misleading_p050_tilt" in ev and len(ev) == 2 * 2 + 1
    stored = tmp_path / "out/eval/q3_4b/indist6/solo"
    stored.mkdir(parents=True)
    (stored / "eval_metrics.json").write_text(json.dumps({"mode": "solo", "gamma": 0.0, "swap_record": False, "max_new_tokens": 768,
                                                          "checkpoint": None, "record": "outputs/record/q3_4b/indist6/shuffled0.fit-train6.jsonl"}))
    assert ev["eval_q3_4b_indist6_solo"].check() is None                                     # the main experiment's result is reused


def test_the_families_experiment_evaluates_every_model_with_its_own_record(tmp_path):
    cfg = load(EXPERIMENTS / "misleading_families.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models"])
    plan = Plan(cfg, "misleading_families.yaml", smoke=False, gpus=[0, 1])

    assert cfg["central"] == ["llama31", "ministral", "qwen25", "phi4", "qwen3_14b"]
    assert "peers" not in cfg["steps"]                                                     # nothing is generated
    ev = {j.name: j for j in plan.evaluate()}
    assert len(ev) == 5 * (10 * 2 + 2)                                                     # tilt + peers per dataset, solo per base stream
    job = ev["eval_qwen3_14b_ood6_misleading_p050_tilt"]
    assert "--model /models/Qwen3-14B " in job.cmd and "/record/qwen3_14b/ood6_misleading_p050/shuffled0.fit-self.jsonl" in job.cmd
    assert all("--fit-features" not in j.cmd for j in plan.record())


def test_a_regime_table_has_one_block_per_central_model(tmp_path):
    from pipeline.table import build

    cfg = load(EXPERIMENTS / "misleading_families.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data",
                                                          "central=[llama31, qwen25]", "datasets=[indist6_misleading_p050]"])
    L = Layout(cfg)
    for m, acc in (("llama31", 0.61), ("qwen25", 0.58)):
        for c in ("tilt", "peers"):
            d = L.eval_dir(m, "indist6_misleading_p050", c)
            d.mkdir(parents=True)
            (d / "eval_metrics.json").write_text(json.dumps({"accuracy": acc}))
    text = build(cfg, smoke=False)

    assert "| llama31 · p050 |  61.0 |  61.0 |" in text and "| qwen25 · p050 |  58.0 |  58.0 |" in text


def test_a_families_smoke_run_reads_the_released_datasets_and_writes_to_smoke(tmp_path):
    cfg = load(EXPERIMENTS / "misleading_families.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "central=[qwen25]"])
    plan = Plan(cfg, "misleading_families.yaml", smoke=True, gpus=[0, 1])

    assert plan.eval_datasets() == ["indist6_misleading_p050"]
    feats = plan.features()
    assert all(f"--stream {tmp_path}/data/indist6_misleading_p050/test.jsonl " in j.cmd and "--max-examples 48" in j.cmd for j in feats)
    assert all(str(j.done).startswith(str(tmp_path / "out/smoke/")) for j in feats + plan.record() + plan.evaluate())


def test_a_new_peer_is_a_new_file_and_a_stream_lists_its_peers(tmp_path):
    root = Path(__file__).resolve().parents[2] / "configs"
    (tmp_path / "models").symlink_to(root / "models")
    (tmp_path / "peers").mkdir()
    for f in (root / "peers").glob("*.yaml"):
        (tmp_path / "peers" / f.name).write_text(f.read_text())
    (tmp_path / "peers/ministral.yaml").write_text("model: ministral\n")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets/indist7.yaml").write_text(
        "path: indist7/test.jsonl\npeers: [gemma3_4b, phi4_mini, qwen25_coder_7b, llama31, deepseek_coder_v2_lite, r1_distill_qwen_7b, ministral]\n")
    (tmp_path / "datasets/indist7_misleading.yaml").write_text("kind: answers\nbase: indist7\nmode: misleading\n")
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.registry={tmp_path}", f"paths.data={tmp_path}/data", "paths.models_root=/models",
                                                 "datasets=[indist7_misleading]"])
    plan = Plan(cfg, "misleading.yaml", smoke=False, gpus=[0])

    assert Layout(cfg).registry.problems() == []
    jobs = plan.peers()
    assert len(jobs) == 7 and "--model /models/Ministral-8B-Instruct-2410 " in jobs[-1].cmd
    assert [j for j in jobs if "DeepSeek-R1" in j.name][0].cmd.count("--reasoning") == 1


def test_the_reading_line_adds_the_own_answer_before_the_record_and_decides_after_the_reading(tmp_path):
    cfg = load(EXPERIMENTS / "reading_line.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models",
                                                   "datasets=[indist6_misleading_p025, indist6_misleading_p050]"])
    plan = Plan(cfg, "reading_line.yaml", smoke=False, gpus=[0, 1])

    own = plan.own()
    solo = [j for j in own if j.wave == 0]
    assert [j.name for j in solo] == ["eval_q3_4b_indist6_solo"]                   # once per base stream, needing no record
    assert "--record" not in solo[0].cmd and "--order shuffled0 " in solo[0].cmd
    assert solo[0].done == tmp_path / "out/eval/q3_4b/indist6/solo/eval_metrics.json"   # shared with the main experiment
    add = {j.name: j for j in own if j.wave == 1}
    assert add["own_q3_4b_indist6_misleading_p050"].cmd == (
        f"python -m pipeline.streams add --base {tmp_path}/data/indist6_misleading_p050/test.jsonl "
        f"--eval {tmp_path}/out/eval/q3_4b/indist6/solo --out {tmp_path}/data/indist6_misleading_p050+q3_4b/test.jsonl")
    feats = plan.features()
    assert len(feats) == 2 * 2 and all("+q3_4b/test.jsonl --model /models/Qwen3-4B " in j.cmd and "--peers 7 " in j.cmd for j in feats)
    rec = {j.name: j for j in plan.record()}["record_q3_4b_indist6_misleading_p050"]
    assert "--peers 7 " in rec.cmd and rec.cmd.endswith("--own-slot 6")
    assert rec.done == tmp_path / "out/record/q3_4b/indist6_misleading_p050+own/shuffled0.fit-self.jsonl"
    ev = {j.name: j for j in plan.evaluate()}
    assert sorted(ev) == ["eval_q3_4b_indist6_misleading_p025_tilt", "eval_q3_4b_indist6_misleading_p050_tilt"]   # question only ran in own
    tilt = ev["eval_q3_4b_indist6_misleading_p050_tilt"]
    assert "/record/q3_4b/indist6_misleading_p050+own/" in tilt.cmd
    assert tilt.done == tmp_path / "out/eval/q3_4b/indist6_misleading_p050+own/tilt/eval_metrics.json"
    dec = {j.name: j for j in plan.decide()}["decide_q3_4b_indist6_misleading_p050"]
    assert f"--reading {tmp_path}/out/eval/q3_4b/indist6_misleading_p050+own/tilt --own {tmp_path}/out/eval/q3_4b/indist6/solo " in dec.cmd
    assert "--prior 0.5,0.0 --lam 1.0" in dec.cmd and dec.gpus == 0
    with pytest.raises(SystemExit, match="fit: self"):
        Plan(load(EXPERIMENTS / "reading_line.yaml", ["record.fit=train6"]), "reading_line.yaml", smoke=False, gpus=[0])


def test_a_reading_line_smoke_run_reads_the_released_stream_and_writes_to_smoke(tmp_path):
    cfg = load(EXPERIMENTS / "reading_line.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data"])
    plan = Plan(cfg, "reading_line.yaml", smoke=True, gpus=[0])
    own = plan.own()

    assert "--limit 48" in own[0].cmd and f"--base {tmp_path}/data/indist6_misleading_p050/test.jsonl " in own[-1].cmd
    jobs = own + plan.features() + plan.record() + plan.evaluate() + plan.decide()
    assert all(str(j.done).startswith(str(tmp_path / "out/smoke/")) for j in jobs)


def test_the_reading_line_table_shows_the_decision_next_to_reading_and_question_only(tmp_path):
    from pipeline.table import build

    cfg = load(EXPERIMENTS / "reading_line.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "datasets=[indist6_misleading_p050]"])
    L = Layout(cfg)
    for c, acc in (("tilt", 0.70), ("decide", 0.74)):
        d = L.eval_dir("q3_4b", "indist6_misleading_p050", c)
        d.mkdir(parents=True)
        (d / "eval_metrics.json").write_text(json.dumps({"accuracy": acc, "read_share": 0.8,
                                                         "reading_line": {"math": {"rho": 0.9, "delta": 0.3, "reads_at_mean_own_prob": "T >= 0.50"}}}))
    d = L.eval_dir("q3_4b", "indist6", "solo")
    d.mkdir(parents=True)
    (d / "eval_metrics.json").write_text(json.dumps({"accuracy": 0.69}))
    text = build(cfg, smoke=False)

    assert "| p050 |  70.0 |  69.0 |  74.0 |" in text and "| p050 · indist6 | 80% | math 0.90 / 0.30 / T >= 0.50 |" in text
