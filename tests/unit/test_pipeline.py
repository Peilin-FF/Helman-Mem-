"""The pipeline's config, layout and job expansion: what `bash run.sh <experiment> --dry-run` would run.

CPU only, no models and no data: the expansion must be right before any GPU time is spent.
"""
import json
from pathlib import Path

import pytest
import yaml

from pipeline.config import deep_merge, load
from pipeline.layout import Layout
from pipeline.run import Plan, stale

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"


def test_a_config_inherits_its_base_and_overrides_win():
    cfg = load(EXPERIMENTS / "main.yaml", ["evaluation.max_new_tokens=1024", "eval_conditions=[tilt]"])

    assert cfg["name"] == "main"
    assert cfg["eval_streams"] == ["indist6", "ood6"]
    assert "evaluate" not in cfg["streams"]                             # experiment keys never leak into a registry          # the experiment's own value
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


def test_derived_streams_live_next_to_their_base_and_smoke_is_isolated(tmp_path):
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.data={tmp_path}/data", f"paths.outputs={tmp_path}/out"])
    real, smoke = Layout(cfg), Layout(cfg, smoke=True)

    assert real.stream("indist6")["path"] == tmp_path / "data/indist6/test.jsonl"
    assert real.stream("indist6_adv_p050")["path"] == tmp_path / "data/indist6_adv_p050/test.jsonl"
    assert real.stream("train6_adv_all100")["path"] == tmp_path / "data/mixed_train_big6_adv_all100/train.jsonl"
    assert smoke.stream("indist6_adv_p050")["path"] == tmp_path / "out/smoke/data/indist6_adv_p050/test.jsonl"
    assert smoke.stream("indist6")["path"] == real.stream("indist6")["path"]          # the released stream is read, never written
    assert str(smoke.eval_dir("q3_4b", "indist6", "tilt")).startswith(str(tmp_path / "out/smoke"))
    with pytest.raises(KeyError):
        real.stream("nope")


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


def test_misleading_expands_regimes_into_streams_and_evaluations(tmp_path):
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.outputs={tmp_path}/out", "regimes_run=[all100, k2]"])
    plan = Plan(cfg, "misleading.yaml", smoke=False, gpus=[0])

    assert plan.eval_streams() == ["ood6_adv_all100", "indist6_adv_all100", "ood6_adv_k2", "indist6_adv_k2"]
    streams = {j.name: j for j in plan.streams()}
    assert '"kind": "count"' in streams["stream_indist6_adv_k2"].cmd
    assert all(j.gpus == 0 for j in streams.values())
    peers = plan.peers()
    assert len(peers) == 2 * 6                                            # two base streams, six peers, one shard each
    r1 = [j for j in peers if "DeepSeek-R1" in j.name][0]
    assert "--reasoning" in r1.cmd
    coder = [j for j in peers if "DeepSeek-Coder" in j.name][0]
    assert coder.env == {"VLLM_USE_V1": "0"}
    assert all("--fit-features" not in j.cmd for j in plan.record())      # fit: self


def test_smoke_uses_one_stream_one_shard_and_a_small_record(tmp_path):
    cfg = load(EXPERIMENTS / "misleading.yaml", [f"paths.outputs={tmp_path}/out"])
    plan = Plan(cfg, "misleading.yaml", smoke=True, gpus=[0, 1, 2, 3])

    assert plan.eval_streams() == ["indist6_adv_p050"]
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

    assert shown(REPO / "data" / "ood6_adv_p050" / "test.jsonl") == "data/ood6_adv_p050/test.jsonl"
    assert shown("/models/Qwen3-4B") == "/models/Qwen3-4B"
    assert shown(None) is None
    assert shown("/mnt/data/peilin/.rproj-jobs/sigma-mem/20260913-162846/data/ood6/test.jsonl") == "data/ood6/test.jsonl"
    assert shown("outputs/record/q3_4b/ood6/shuffled0.jsonl") == "outputs/record/q3_4b/ood6/shuffled0.jsonl"
