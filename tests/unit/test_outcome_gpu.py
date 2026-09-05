"""Opt-in CUDA COMPONENT tests, not an end-to-end training benchmark.

SIGMA_OUTCOME_GPU_TEST=1 CUDA_VISIBLE_DEVICES=<free GPU> PYTHONPATH=.:training/verl \
    python -u -m pytest -s -q tests/unit/test_outcome_gpu.py

No trained model, existing experiment output or original memory code is modified.
"""
import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("SIGMA_OUTCOME_GPU_TEST") != "1", reason="explicit opt-in CUDA component test")


def test_original_kalman_closed_loop_on_cuda():
    from tests.unit.test_outcome_protocol import memory_fixture
    from training.sigma_rl.outcome_protocol import CausalPeerEpisode

    runtime, stream = memory_fixture(peers=10, device="cuda")
    episode = CausalPeerEpisode(runtime, feedback=lambda t: [i % 2 for i in range(10)], rollouts_per_question=2)
    first = episode.begin(0, stream.records[0], stream.texts[0])
    assert first.evidence.is_cuda
    torch.cuda.synchronize()
    start = time.perf_counter()
    completed = episode.complete(["own a", "own b"], score=lambda text: int(text == "own b"))
    second = episode.begin(1, stream.records[1], stream.texts[1])
    torch.cuda.synchronize()
    print(f"[component] ten-peer feedback write + next read: {(time.perf_counter() - start) * 1000:.3f} ms")
    assert completed.rewards == (0.0, 1.0)
    assert runtime.mem.writes == 10
    assert not torch.equal(first.evidence, second.evidence)


def test_original_steerer_8192_tokens_backward_and_checkpoint_recomputation():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from feedback_state.symmetric_memory import ActivationSteerer

    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                        max_position_embeddings=9000, attention_dropout=0.0, use_cache=False)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).cuda().float()
    steerer = ActivationSteerer(model, rank=4, gain_init=1.0, proj_std=1e-2).cuda()
    inputs = torch.randint(2, 128, (2, 8192), device="cuda")
    evidence = torch.full((2, 4), 0.3, device="cuda")
    # Original single-vector-per-sequence interface ONLY. This is not the
    # pending multi-peer joint-generation extension.
    gradients, durations = [], []
    for checkpointing in (False, True):
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        model.zero_grad(set_to_none=True)
        steerer.zero_grad(set_to_none=True)
        steerer.steer_vec = evidence
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            logits = model(input_ids=inputs, logits_to_keep=1).logits.float()
            loss = torch.nn.functional.cross_entropy(logits[:, -1], torch.tensor([3, 7], device="cuda"))
            loss.backward()  # KEEP the original context until recomputation ends
        finally:
            steerer.steer_vec = None
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        gradients.append(steerer.proj.weight.grad.detach().clone())
        assert torch.isfinite(gradients[-1]).all() and gradients[-1].norm() > 0
        assert model.model.embed_tokens.weight.grad.norm() > 0
    assert torch.allclose(gradients[0], gradients[1], atol=1e-5, rtol=1e-4)
    print(f"[component] random 2-layer / hidden=64 Qwen3, B=2, input=8192, forward+backward seconds: {durations}")
    print("[scope] NOT Qwen3-4B throughput; NOT joint peer generation; NOT distributed training")
    # Demonstrate the interface mismatch instead of silently aggregating peers.
    steerer.steer_vec = torch.zeros(2, 3, 4, device="cuda")
    try:
        with pytest.raises(ValueError, match="steer_vec must be"):
            model(input_ids=inputs[:, :4])
    finally:
        steerer.steer_vec = None
        steerer.remove()


def test_repository_grpo_advantages_and_trajectory_guards_on_cuda():
    from tests.unit.test_outcome_batch import generated_fixture
    from training.sigma_rl.outcome_batch import add_outcome_advantages, checked_generation

    prompt, generated = generated_fixture()
    prompt.to("cuda")
    generated.to("cuda")
    checked_generation(prompt, generated, samples_per_question=2)
    result = add_outcome_advantages(generated, torch.tensor([[0.0, 0.0], [0.0, 1.0]], device="cuda"), ["q0", "q0"])
    assert result.batch["advantages"].is_cuda and torch.isfinite(result.batch["advantages"]).all()
    assert result.batch["advantages"][0, 0] < 0 < result.batch["advantages"][1, 0]
