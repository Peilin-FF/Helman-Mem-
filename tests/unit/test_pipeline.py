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
    assert reg.peer_set("six") == ["gemma3_4b", "phi4_mini", "qwen25_coder_7b", "llama31", "deepseek_coder_v2_lite", "r1_distill_qwen_7b"]
    with pytest.raises(KeyError, match="did you mean"):
        reg.dataset("indist6_misleading_p05")


def test_the_registry_reports_broken_references(tmp_path):
    for sub in ("datasets", "models", "peers"):
        (tmp_path / sub).mkdir()
    (tmp_path / "models/m.yaml").write_text("path: M\n")
    (tmp_path / "peers/two.yaml").write_text("models: [m, ghost]\n")
    (tmp_path / "datasets/s.yaml").write_text("path: s/test.jsonl\npeers: two\n")
    (tmp_path / "datasets/s_bad.yaml").write_text("kind: misleading\nbase: s\nanswers: nowhere\nregime: p150\n")
    problems = "\n".join(load_registry(tmp_path).problems())

    assert "ghost" in problems and "nowhere" in problems and "regime must be pNNN" in problems


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
    assert coder.env == {"VLLM_USE_V1": "0"}
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
