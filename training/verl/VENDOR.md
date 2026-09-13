# Vendored verl (0.3.1.dev, ARPO fork)

Source: github.com/RUC-NLPIR/ARPO (`verl_arpo_entropy`, a fork of volcengine/verl 0.3.1). Only the `verl/` package,
`scripts/`, `LICENSE`, `Notice.txt` and `requirements.txt` are kept. It is used through `PYTHONPATH=training/verl` (no
install), so every job snapshot carries its own copy.

Used: `verl.trainer.ppo.ray_trainer.RayPPOTrainer` (subclassed in `training/kalman_rl/trainer.py`), the FSDP actor and
rollout workers, the vLLM spmd rollout (`rollout.mode=sync`). Not used: the ARPO / AEPO agentic and entropy machinery
(`sync_with_tool` rollouts, `workers/agent/`, `tools/`, branching knobs); it stays untouched and `configs/train/grpo.yaml`
never selects it.

## Local patches (grep `kalman:`)

1. `verl/__init__.py`: `pkg_resources` replaced by `importlib.metadata` for the version check.
2. `verl/workers/fsdp_workers.py`: `attn_implementation` of the actor and critic comes from the config instead of being
   hard-coded; with `model.attn_bias` the HF attention-bias hooks (`feedback_state.attn_bias.install_hf_hooks`) are
   installed on the actor and reference models (requires `use_remove_padding=False`).
3. `verl/workers/actor/dp_actor.py`: the per-micro-batch tilt (`attn_bias`) is set on the hooks before each forward, and
   padding shared by a whole micro-batch is trimmed (with the tilt trimmed alike).
4. `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`: with `rollout.attn_bias` the patched Triton kernels are
   installed (`feedback_state.vllm_attn_bias.install`) and every prompt's tilt is registered with its token ids.
5. `verl/trainer/ppo/ray_trainer.py`: the tilt travels with the validation prompts into the rollout.
6. `verl/trainer/fsdp_sft_trainer.py`: `attn_implementation` from the config.
7. `verl/utils/checkpoint/fsdp_checkpoint_manager.py`: fp32 model / optimizer / extra shards are written only when listed
   in `checkpoint.contents`; `['hf_model']` keeps the HF copy only (a full checkpoint of a 4B model is 63 GB).

Everything else is byte-identical to the fork.
