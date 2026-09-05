import numpy as np
import pytest
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

from training.sigma_rl.outcome_batch import POLICY_TENSORS, actor_inputs, add_outcome_advantages, checked_generation, policy_batch
from training.sigma_rl.outcome_protocol import PolicyObservation


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "public prompt"

    def encode(self, text, **kwargs):
        return [1, 2, 3]


def observation(peers=3, position=0):
    return PolicyObservation(position, (("user", "question and all peer responses"),), tuple(range(peers)), torch.ones(peers, 4))


def generated_fixture():
    prompt = policy_batch([observation()], Tokenizer(), max_length=16)
    repeated = prompt.repeat(repeat_times=2, interleave=True)
    responses = torch.tensor([[7, 9], [8, 9]])
    ids = torch.cat([repeated.batch["input_ids"], responses], dim=-1)
    mask = torch.cat([repeated.batch["attention_mask"], torch.ones_like(responses)], dim=-1)
    result = DataProto.from_dict(tensors={
        "prompts": repeated.batch["input_ids"], "responses": responses, "input_ids": ids,
        "attention_mask": mask, "position_ids": compute_position_id_with_mask(mask),
        "peer_evidence": repeated.batch["peer_evidence"], "peer_mask": repeated.batch["peer_mask"],
    })
    return prompt, result


def test_worker_receives_no_grader_records_current_labels_or_gold():
    batch = policy_batch([observation()], Tokenizer())
    assert batch.batch["input_ids"].shape == (1, 8192)
    assert batch.batch["attention_mask"].sum() == 3
    batch.non_tensor_batch["extra_info"] = np.array([{"answer": "SECRET", "peer_correct": [1, 0, 1]}], dtype=object)
    batch.meta_info["answer"] = "SECRET"
    worker = actor_inputs(batch)
    assert set(worker.batch.keys()) == set(POLICY_TENSORS)
    assert set(worker.non_tensor_batch) == {"raw_prompt_ids"}
    assert not worker.meta_info


def test_variable_peer_count_preserves_all_original_evidence_rows():
    batch = policy_batch([observation(3), observation(5), observation(10)], Tokenizer(), max_length=16)
    assert batch.batch["peer_evidence"].shape == (3, 10, 4)
    assert batch.batch["peer_mask"].sum(-1).tolist() == [3, 5, 10]
    assert torch.equal(batch.batch["peer_evidence"][0, 3:], torch.zeros(7, 4, dtype=torch.float64))


def test_exact_on_policy_generation_is_accepted():
    prompt, generated = generated_fixture()
    assert checked_generation(prompt, generated, samples_per_question=2) is generated


@pytest.mark.parametrize("key", ["peer_evidence", "peer_mask"])
def test_legacy_backends_cannot_silently_drop_the_memory(key):
    prompt, generated = generated_fixture()
    generated.batch.pop(key)
    with pytest.raises(ValueError, match="integration is not implemented"):
        checked_generation(prompt, generated, samples_per_question=2)


def test_relabeling_guided_answer_under_solo_prompt_is_rejected():
    prompt, generated = generated_fixture()
    generated.batch["prompts"][0, -1] = 99
    with pytest.raises(ValueError, match="different prompt"):
        checked_generation(prompt, generated, samples_per_question=2)


def test_using_post_feedback_snapshot_for_the_policy_loss_is_rejected():
    prompt, generated = generated_fixture()
    generated.batch["peer_evidence"][0, 0, 0] = 10
    with pytest.raises(ValueError, match="policy condition"):
        checked_generation(prompt, generated, samples_per_question=2)


def test_parallel_padding_cannot_change_the_grpo_group_size():
    prompt, generated = generated_fixture()
    padded = generated.select_idxs([0, 1, 0])
    with pytest.raises(ValueError, match="unpad scheduling"):
        checked_generation(prompt, padded, samples_per_question=2)


def test_only_center_outcomes_enter_repository_grpo_advantages():
    _, generated = generated_fixture()
    result = add_outcome_advantages(generated, torch.tensor([[0.0, 0.0], [0.0, 1.0]]), ["q0", "q0"])
    assert (result.batch["advantages"][0] < 0).all()
    assert (result.batch["advantages"][1] > 0).all()
    assert len(result) == 2


def test_all_wrong_group_has_zero_rl_signal_without_fallback_or_dropping():
    _, generated = generated_fixture()
    result = add_outcome_advantages(generated, torch.zeros(2, 2), ["q0", "q0"])
    assert torch.equal(result.batch["advantages"], torch.zeros(2, 2)) and len(result) == 2


def test_step_labels_are_rejected():
    _, generated = generated_fixture()
    with pytest.raises(ValueError, match="not per-step supervision"):
        add_outcome_advantages(generated, torch.tensor([[1.0, 0.0], [0.0, 1.0]]), ["q0", "q0"])


@pytest.mark.parametrize("key", ["peer_evidence", "prompts"])
def test_a_grpo_group_cannot_mix_different_prompts_or_memory(key):
    _, generated = generated_fixture()
    generated.batch[key][1].add_(1)
    with pytest.raises(ValueError, match="mixes different policy conditions"):
        add_outcome_advantages(generated, torch.tensor([[0.0, 0.0], [0.0, 1.0]]), ["q0", "q0"])
