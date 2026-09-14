"""Combination during training: the online state, the prompt data, the tilt's token map and the jobs (CPU only)."""
import json
from pathlib import Path

import pandas as pd
import torch

from feedback_state.attn_bias import token_slots
from feedback_state.combination import PEERS_MEMORY, QUESTION_ALONE, TrainingCombination
from pipeline.config import load
from pipeline.run import Plan

EXPERIMENTS = Path(__file__).resolve().parents[2] / "configs" / "experiments"


def _saved(n=4, answers=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"design": "qc", "num_peers": answers, "lam": 1.0, "psi_q": torch.randn(n, 2, generator=g), "psi_c": torch.randn(n, answers, 2, generator=g),
            "z": torch.zeros(n, answers), "real": [answers] * n, "ids": [f"e{i}" for i in range(n)],
            "labels": torch.tensor([[1, 0, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]])}


def test_a_step_is_read_before_it_is_written():
    comb = TrainingCombination(_saved(), prior=(0.5, 0.0), gamma=3.0)
    first = comb.choose(["e0", "e1"], ["math", "math"], [[0, 1], [1, 0]])
    assert [d["choice"] for d in first] == [PEERS_MEMORY, PEERS_MEMORY]              # cold record: flat 0.5, reading ties the own answer
    assert first[0]["memory_prob"] == first[1]["memory_prob"] and not any(first[0]["bias"])   # a flat record does not tilt
    rows, metrics = comb.update([0.75, 0.25])
    assert [r["accuracy"] for r in rows] == [0.75, 0.25] and metrics["combination/share_peers_memory"] == 1.0
    assert comb.lines["math"].events == 2 and comb.runtime.mem.writes == 2 * 2        # two peers' labels per question, no own label
    again = comb.choose(["e0"], ["math"], [[0, 1]])
    assert again[0]["memory_prob"] != first[0]["memory_prob"]                         # the written labels move the next read


def test_question_alone_writes_the_own_answer_and_not_the_reading_line():
    right, wrong = TrainingCombination(_saved(), prior=(0.2, 0.5)), TrainingCombination(_saved(), prior=(0.2, 0.5))   # reading looks bad
    for comb, accuracy in ((right, 1.0), (wrong, 0.0)):
        chosen = comb.choose(["e2"], ["rag"], [[1, 0]])[0]
        assert chosen["choice"] == QUESTION_ALONE and not any(chosen["bias"])
        comb.update([accuracy])
        assert comb.runtime.mem.writes == 3 and comb.lines["rag"].events == 0          # two peers and the own answer; no reading outcome
    own = lambda comb: comb.choose(["e2"], ["rag"], [[1, 0]])[0]["own_prob"]
    assert own(right) > own(wrong)                                                    # the own answers' accuracy moves the own estimate


def test_the_tilt_follows_the_slot_of_every_token():
    offsets = [(0, 3), (3, 8), (8, 12), (12, 20), (20, 25)]
    assert token_slots(offsets, [(3, 12), (12, 25)]).tolist() == [-1, 0, 0, 1, 1]


def test_combination_rows_carry_both_prompts_and_no_stored_estimates(tmp_path):
    from training.kalman_rl import build_rl_data

    rec = {"id": "q1", "task_type": "math", "problem": "1+1?", "answer": "2", "peer_responses": {"peer_0": "It is 2.", "peer_1": "It is 3.", "peer_2": "2"}}
    stream = tmp_path / "stream.jsonl"
    stream.write_text(json.dumps(rec) + "\n")
    user = "Question:\n1+1?\n\nPeer answers:\n\n[Peer 1]\nIt is 3.\n\n[Peer 2]\nIt is 2.\n\nInstruction: answer."
    row = {"pos": 0, "id": "q1", "task_type": "math", "source": "s", "peer_order": [1, 0], "peer_correct": [0, 1], "memory_prob": [0.4, 0.6],
           "memory_evidence": [1.0, 1.0], "own_prob": 0.5, "own_correct": 1,
           "messages_peers": [{"role": "system", "content": "s"}, {"role": "user", "content": user}],
           "messages_solo": [{"role": "system", "content": "s"}, {"role": "user", "content": "Question:\n1+1?\n\nInstruction: answer."}]}
    record = tmp_path / "record.jsonl"
    record.write_text(json.dumps(row) + "\n")
    out = tmp_path / "data.parquet"
    build_rl_data.main(["--record", str(record), "--stream", str(stream), "--prompt", "combination", "--out", str(out)])

    df = pd.read_parquet(out)
    extra = df.iloc[0]["extra_info"]
    assert list(df.iloc[0]["prompt_solo"])[-1]["content"].startswith("Question:") and extra["prompt_source"] == "combination"
    assert "memory_prob" not in extra and len(extra["peer_spans"]) == 2 and list(extra["peer_order"]) == [1, 0]


def test_train_combination_expands_to_own_answers_addresses_and_one_online_run(tmp_path):
    cfg = load(EXPERIMENTS / "train_combination.yaml", [f"paths.outputs={tmp_path}/out", f"paths.data={tmp_path}/data", "paths.models_root=/models"])
    plan = Plan(cfg, "train_combination.yaml", smoke=False, gpus=list(range(8)))

    solo = [j for j in plan.own() if j.wave == 0]
    assert len(solo) == 1 and "--record" not in solo[0].cmd and f"--stream {tmp_path}/data/mixed_train_big6/train.jsonl " in solo[0].cmd
    rec = plan.record()[0]
    assert "--own-slot 6" in rec.cmd and f"--save-addresses {tmp_path}/out/record/q3_4b/train6+own/shuffled0.fit-self.addresses.pt" in rec.cmd
    jobs = {j.name: j for j in plan.train()}
    assert not any(n.startswith("record_") for n in jobs)                              # the record step made the record
    data = jobs["data_q3_4b_train6_combination"]
    assert "--prompt combination" in data.cmd and f"--stream {tmp_path}/data/train6+q3_4b/test.jsonl " in data.cmd
    train = jobs["train_comb3b"]
    assert f"data.combination.addresses={tmp_path}/out/record/q3_4b/train6+own/shuffled0.fit-self.addresses.pt" in train.cmd
    assert "data.combination.prior=[0.5,0.0]" in train.cmd and "data.shuffle=False" in train.cmd and "data.attn_gamma=3.0" in train.cmd
    assert train.gpus == 8 and train.wave == 2
