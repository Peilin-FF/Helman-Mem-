"""Protocol regression tests; these do not claim a generation worker is integrated."""
import numpy as np
import pytest
import torch

from feedback_state.addresses import Projection
from feedback_state.feature_streams import FeatureStream
from feedback_state.memory_runtime import MemoryRuntime
from training.sigma_rl.outcome_protocol import CausalPeerEpisode, checked_prompt_ids, public_messages


def memory_fixture(peers=3, design="qc", device="cpu"):
    dim, count = 2, 3
    state = {"mean": torch.zeros(dim), "std": torch.ones(dim), "center": torch.zeros(dim),
             "basis": torch.eye(dim), "scale": 1.0}
    projection = Projection(state=state, device=device)
    stream = FeatureStream(
        name="fixture", model="fixture", records=[{"problem": "What is 2 + 2?", "task_type": "math"}] * count,
        ids=[str(i) for i in range(count)], q_mean=torch.ones(count, dim), sem=torch.ones(count, dim),
        peer_hidden=torch.ones(count, peers, dim), margins=torch.zeros(count, peers),
        labels=torch.zeros(count, peers, dtype=torch.long), real=torch.full((count,), peers),
        task=["math"] * count, source=["fixture"] * count,
        texts=[[f"peer {i} reasoning" for i in range(peers)] for _ in range(count)],
    )
    runtime = MemoryRuntime(design=design, proj_q=projection, proj_c=projection, num_peers=peers, lam=3.0, device=device)
    runtime.attach(stream)
    return runtime, stream


class PublicOnly(dict):
    private = {"answer", "solution", "target", "peer_correct", "correctness_by_peer", "scores", "memory_prob"}

    def get(self, key, default=None):
        assert key not in self.private, f"private field accessed: {key}"
        return super().get(key, default)

    def __getitem__(self, key):
        assert key not in self.private, f"private field accessed: {key}"
        return super().__getitem__(key)


def test_prompt_never_accesses_labels_gold_targets_or_reliability():
    rec = PublicOnly(problem="What is 2 + 2?", task_type="math", answer="PRIVATE_GOLD",
                     peer_correct=[0, 1], scores=[0.3, 0.9], memory_prob=[0.1, 0.99], solution="PRIVATE_SOLUTION")
    a = public_messages(rec, ["reasoning one", "reasoning two"])
    rec.update(answer="CHANGED_GOLD", peer_correct=[1, 0], scores=[1, 0])
    assert a == public_messages(rec, ["reasoning one", "reasoning two"])
    assert "PRIVATE_" not in str(a) and "estimated probability" not in str(a)


@pytest.mark.parametrize("peers", [1, 3, 5, 10])
def test_all_peer_text_survives_without_character_clipping(peers):
    texts = [f"START_{i}" + "x" * 4500 + f"END_{i}" for i in range(peers)]
    prompt = public_messages({"problem": "q"}, texts)[1]["content"]
    assert all(text in prompt for text in texts)
    assert all(prompt.count(f"Peer {i + 1}:\n") == 1 for i in range(peers))


def test_rag_context_is_not_silently_cut_at_8000_characters():
    ctx = "x" * 9000 + "CONTEXT_END"
    result = public_messages({"problem": "q", "task_type": "rag", "context": ctx}, ["p"])
    assert ctx in result[1]["content"]


def test_mcqa_options_and_labels_stay_public():
    result = public_messages({"problem": "q", "task_type": "mcqa", "choices": ["red", "blue"],
                              "choice_labels": ["X", "Y"], "answer": "PRIVATE_GOLD"}, ["p"])
    assert "(X) red" in result[1]["content"] and "(Y) blue" in result[1]["content"]
    assert "PRIVATE_GOLD" not in str(result)


def test_exact_8192_token_boundary_is_checked_on_the_whole_rendered_prompt():
    class Tokenizer:
        length = 8192

        def apply_chat_template(self, messages, **kwargs):
            assert len(messages) == 2
            return "rendered"

        def encode(self, text, **kwargs):
            assert text == "rendered" and kwargs == {"add_special_tokens": False}
            return [1] * self.length

    tokenizer = Tokenizer()
    messages = public_messages({"problem": "q"}, ["p"])
    assert len(checked_prompt_ids(tokenizer, messages)) == 8192
    tokenizer.length = 8193
    with pytest.raises(ValueError, match="no truncation"):
        checked_prompt_ids(tokenizer, messages)


def test_current_labels_cannot_change_current_observation_but_change_next_question():
    observations, next_observations = [], []
    for labels in ([0, 0, 0], [1, 0, 1]):
        runtime, fs = memory_fixture()
        episode = CausalPeerEpisode(runtime, feedback=lambda t: labels, rollouts_per_question=2)
        observations.append(episode.begin(0, fs.records[0], fs.texts[0]))
        episode.complete(["incorrect", "correct"], score=lambda text: float(text == "correct"))
        next_observations.append(episode.begin(1, fs.records[1], fs.texts[1]))
    assert observations[0].messages == observations[1].messages
    assert torch.equal(observations[0].evidence, observations[1].evidence)
    assert next_observations[0].messages == next_observations[1].messages
    assert not torch.equal(next_observations[0].evidence, next_observations[1].evidence)


def test_future_feedback_is_not_requested():
    runtime, fs = memory_fixture()
    requested = []

    def feedback(position):
        requested.append(position)
        return [0, 1, 0]

    episode = CausalPeerEpisode(runtime, feedback=feedback, rollouts_per_question=2)
    episode.begin(0, fs.records[0], fs.texts[0])
    assert requested == [] and runtime.mem.writes == 0
    episode.complete(["a", "b"], score=lambda text: 0)
    episode.begin(1, fs.records[1], fs.texts[1])
    assert requested == [0]


def test_all_center_samples_are_scored_before_peer_labels_are_accessed():
    runtime, fs = memory_fixture()
    events = []

    def feedback(position):
        events.append("feedback")
        assert runtime.mem.writes == 0
        return [0, 1, 0]

    def score(text):
        events.append(text)
        assert runtime.mem.writes == 0
        return 0

    episode = CausalPeerEpisode(runtime, feedback=feedback, rollouts_per_question=2)
    episode.begin(0, fs.records[0], fs.texts[0])
    with pytest.raises(RuntimeError, match="previous question"):
        episode.begin(1, fs.records[1], fs.texts[1])
    episode.complete(["center_1", "center_2"], score=score)
    assert events == ["center_1", "center_2", "feedback"]
    assert runtime.mem.writes == 3 and episode.next_position == 1
    with pytest.raises(RuntimeError, match="begin a question"):
        episode.complete(["center_1", "center_2"], score=score)


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1], [0, 1, 0]])
@pytest.mark.parametrize("design", ["q", "qc"])
def test_no_peer_label_pattern_filters_own_responses_or_changes_the_reward(labels, design):
    runtime, fs = memory_fixture(design=design)
    expected, _ = memory_fixture(design=design)
    episode = CausalPeerEpisode(runtime, feedback=lambda t: labels, rollouts_per_question=2)
    episode.begin(0, fs.records[0], fs.texts[0])
    event = episode.complete(["wrong", "right"], score=lambda s: int(s == "right"))
    expected.write(0, expected.rows(0), labels)
    assert event.responses == ("wrong", "right") and event.rewards == (0.0, 1.0)
    assert not hasattr(event, "peer_correct")
    assert torch.equal(runtime.mem.b, expected.mem.b)
    assert torch.equal(runtime.mem.P, expected.mem.P)


def test_center_reward_never_becomes_a_peer_memory_write():
    states = []
    for center_correct in (0, 1):
        runtime, fs = memory_fixture()
        episode = CausalPeerEpisode(runtime, feedback=lambda t: [1, 0, 1], rollouts_per_question=2)
        episode.begin(0, fs.records[0], fs.texts[0])
        episode.complete(["a", "b"], score=lambda text: center_correct)
        states.append(runtime.mem.snapshot())
    assert torch.equal(states[0]["b"], states[1]["b"])
    assert torch.equal(states[0]["P"], states[1]["P"])


@pytest.mark.parametrize("peers", [3, 5, 10])
def test_permutation_keeps_peer_tensor_alignment_and_canonical_memory_writes(peers):
    runtime, fs = memory_fixture(peers=peers)
    labels = [i % 2 for i in range(peers)]
    runtime.write(0, runtime.rows(0), labels)
    reference = runtime.mem.snapshot()
    _, _, evidence, rows = runtime.read(1)
    order = tuple(reversed(range(peers)))
    episode = CausalPeerEpisode(runtime, feedback=lambda t: labels, rollouts_per_question=2, start_position=1)
    observed = episode.begin(1, fs.records[1], fs.texts[1], peer_order=order)
    assert torch.equal(observed.evidence, evidence[list(order)])
    event = episode.complete(["a", "b"], score=lambda s: 0)
    actual = runtime.mem.snapshot()
    runtime.mem.restore(reference)
    runtime.write(1, rows, labels)
    assert torch.equal(actual["b"], runtime.mem.b)
    assert torch.equal(actual["P"], runtime.mem.P)
    assert torch.equal(event.observation.evidence, evidence[list(order)])


def test_worker_cannot_mutate_the_saved_policy_context():
    runtime, fs = memory_fixture()
    episode = CausalPeerEpisode(runtime, feedback=lambda t: [1, 0, 1], rollouts_per_question=2)
    obs = episode.begin(0, fs.records[0], fs.texts[0])
    original = obs.evidence.clone()
    obs.evidence.add_(999)
    event = episode.complete(["a", "b"], score=lambda text: 0)
    assert torch.equal(event.observation.evidence, original)
    assert not event.observation.evidence.requires_grad


def test_failed_verifier_does_not_consume_feedback_or_advance_memory():
    runtime, fs = memory_fixture()
    episode = CausalPeerEpisode(runtime, feedback=lambda t: pytest.fail("too early"), rollouts_per_question=2)
    episode.begin(0, fs.records[0], fs.texts[0])

    def fail(text):
        raise OSError("verifier infrastructure unavailable")

    with pytest.raises(OSError):
        episode.complete(["a", "b"], score=fail)
    assert episode.awaiting_response and episode.next_position == 0 and runtime.mem.writes == 0


@pytest.mark.parametrize("labels", [[0, 1], [0, None, 1], [0, 0.3, 1], [0, float("nan"), 1]])
def test_invalid_or_missing_peer_feedback_is_not_assumed_incorrect(labels):
    runtime, fs = memory_fixture()
    episode = CausalPeerEpisode(runtime, feedback=lambda t: labels, rollouts_per_question=2)
    episode.begin(0, fs.records[0], fs.texts[0])
    with pytest.raises(ValueError):
        episode.complete(["a", "b"], score=lambda text: 0)
    assert runtime.mem.writes == 0 and episode.awaiting_response


def test_partial_original_memory_write_is_rolled_back_on_failure(monkeypatch):
    runtime, fs = memory_fixture()
    snapshot = runtime.mem.snapshot()
    episode = CausalPeerEpisode(runtime, feedback=lambda t: [1, 0, 1], rollouts_per_question=2)
    episode.begin(0, fs.records[0], fs.texts[0])

    def partial_write(t, rows, labels):
        runtime.mem.write(rows[0], torch.ones(1))
        raise OSError("injected write failure")

    monkeypatch.setattr(runtime, "write", partial_write)
    with pytest.raises(OSError):
        episode.complete(["a", "b"], score=lambda text: 0)
    assert runtime.mem.writes == 0 and episode.awaiting_response
    assert torch.equal(runtime.mem.P, snapshot["P"]) and torch.equal(runtime.mem.b, snapshot["b"])


def test_missing_or_filtered_rollouts_are_rejected():
    runtime, fs = memory_fixture()
    episode = CausalPeerEpisode(runtime, feedback=lambda t: pytest.fail("too early"), rollouts_per_question=2)
    episode.begin(0, fs.records[0], fs.texts[0])
    with pytest.raises(ValueError, match="not a selected best answer"):
        episode.complete(["the only correct response"], score=lambda text: 1)


def test_grpo_reuses_repository_advantage_and_has_no_all_wrong_fallback():
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    rewards = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    advantage, _ = compute_grpo_outcome_advantage(rewards, torch.ones_like(rewards), np.array(["q0", "q0", "q1", "q1"]))
    assert torch.equal(advantage[:2], torch.zeros_like(advantage[:2]))
    assert (advantage[2] < 0).all() and (advantage[3] > 0).all()
