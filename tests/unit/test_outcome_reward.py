import json

import numpy as np
import pytest
import torch

from training.sigma_rl.outcome_protocol import PROTOCOL
from training.sigma_rl.outcome_reward import OutcomeRewardManager, center_outcome, compute_center_score, verifier_record


@pytest.mark.parametrize("peer_correct", [[0, 0, 0], [1, 1, 1], [0, 1, 0]])
def test_center_reward_is_independent_of_peer_correctness(peer_correct):
    record = {"task_type": "math", "problem": "What is 2 + 2?", "answer": "4",
              "peer_correct": peer_correct, "peer_responses": {"one": "Final answer: 5"},
              "memory_prob": [1.0, 1.0, 1.0]}
    info = {"protocol": PROTOCOL, "record": json.dumps(record)}
    good = compute_center_score("math", "Final answer: 4", "4", info)
    bad = compute_center_score("math", "Final answer: 5", "4", info, acc=1.0)
    assert good == {"score": 1.0, "acc": 1.0, "verified": 1.0, "pseudo": 0.0}
    assert bad == {"score": 0.0, "acc": 0.0, "verified": 1.0, "pseudo": 0.0}
    assert "peer_correct" not in verifier_record(record) and "peer_responses" not in verifier_record(record)


def test_old_data_and_unverified_pseudo_reward_are_rejected():
    with pytest.raises(ValueError, match="legacy"):
        compute_center_score("math", "Final answer: 4", "4", {"guided_mode": "memory"})
    with pytest.raises(ValueError, match="peer-vote fallback"):
        compute_center_score("math", "Final answer: 4", "4", {"protocol": PROTOCOL, "verified": False})


def test_verifier_task_mismatch_and_unknown_task_are_rejected():
    with pytest.raises(ValueError, match="disagree"):
        compute_center_score("math", "answer", "4", {"protocol": PROTOCOL, "record": {"task_type": "code"}})
    with pytest.raises(ValueError, match="no verifier registered"):
        verifier_record({"task_type": "unknown", "answer": "4"})


@pytest.mark.parametrize("record", [
    {"task_type": "math"}, {"task_type": "math", "answer": ""},
    {"task_type": "code", "code_format": "io", "test_cases": []},
    {"task_type": "code", "code_format": "io", "test_cases": [{"input": "1"}]},
    {"task_type": "code", "code_format": "asserts", "test": ""},
    {"task_type": "code", "code_format": "functional", "test_cases": [{"input": "1", "output": "1"}]},
    {"task_type": "code", "code_format": "unknown"},
])
def test_missing_verification_data_does_not_produce_fabricated_scores(record):
    with pytest.raises(ValueError):
        center_outcome(record, "some response")


def test_verifier_exception_is_not_swallowed(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("unavailable verifier")

    monkeypatch.setattr("training.sigma_rl.outcome_reward.grade", fail)
    with pytest.raises(OSError):
        center_outcome({"task_type": "math", "answer": "4"}, "Final answer: 4")


def make_batch():
    from verl import DataProto

    extra = np.empty(2, dtype=object)
    extra[:] = [{"protocol": PROTOCOL}, {"protocol": PROTOCOL}]
    references = np.empty(2, dtype=object)
    references[:] = [{"ground_truth": "4"}, {"ground_truth": "4"}]
    return DataProto.from_dict(
        tensors={"prompts": torch.tensor([[10, 20], [10, 20]]), "responses": torch.tensor([[1, 3, 0], [2, 3, 0]]),
                 "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])},
        non_tensors={"extra_info": extra, "reward_model": references, "data_source": np.array(["math", "math"], dtype=object)},
    )


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "Final answer: 4" if int(ids[0]) == 1 else "Final answer: 5"


def test_parallel_reward_manager_rewards_only_last_real_response_token():
    result = OutcomeRewardManager(Tokenizer(), max_workers=2)(make_batch(), return_dict=True)
    assert torch.equal(result["reward_tensor"], torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    assert result["reward_extra_info"]["acc"] == [1.0, 0.0]
    assert result["reward_extra_info"]["pseudo"] == [0.0, 0.0]


def test_reward_model_shortcut_is_rejected():
    batch = make_batch()
    batch.batch["rm_scores"] = torch.ones_like(batch.batch["responses"])
    with pytest.raises(ValueError, match="precomputed"):
        OutcomeRewardManager(Tokenizer())(batch)


def test_no_response_and_bad_masks_are_not_assigned_a_reward():
    batch = make_batch()
    batch.batch["attention_mask"][:, 2:] = 0
    with pytest.raises(ValueError, match="missing generated"):
        OutcomeRewardManager(Tokenizer())(batch)
    batch.batch["attention_mask"][:, 2:] = torch.tensor([1, 0, 1])
    with pytest.raises(ValueError, match="right-padded"):
        OutcomeRewardManager(Tokenizer())(batch)
