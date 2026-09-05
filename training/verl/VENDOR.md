# Vendored verl (0.3.1.dev, ARPO fork)

Source: `/mnt/peilin/ARPO/ARPO/verl_arpo_entropy` (github.com/RUC-NLPIR/ARPO, a fork of
volcengine/verl 0.3.1).  Only the `verl/` python package, `scripts/` (checkpoint merger, diagnostics),
`LICENSE`, `Notice.txt` and `requirements.txt` are kept; `__pycache__`, docs, tests, docker and
examples are not.  The package is used through `PYTHONPATH=training/verl` (no install), so every
job snapshot carries its own copy.

What we use: `verl.trainer.ppo.ray_trainer.RayPPOTrainer` (subclassed in
`training/sigma_rl/trainer.py`), the FSDP actor/rollout workers, the vLLM spmd rollout
(`rollout.mode=sync`), `verl.trainer.fsdp_sft_trainer`, `scripts/model_merger.py`.

What we do not use: the ARPO / AEPO agentic machinery — `rollout.mode=sync_with_tool`
(`workers/rollout/vllm_rollout/vllm_rollout_with_tools.py`), `workers/agent/`, `tools/`, the
entropy-based branching knobs (`initial_rollouts`, `beam_size`, `branch_probability`,
`entropy_weight`), the deep-research reward.  They stay in the tree untouched; our config
(`training/configs/grpo_sigma.yaml`) simply never selects them.

## Local patches (grep `sigma:`)

1. `verl/workers/fsdp_workers.py`: the actor's and critic's `attn_implementation` come from
   `model.attn_implementation` (default `flash_attention_2`) instead of being hard-coded, so the
   stack also runs with `sdpa` when flash-attn is unavailable.
2. `verl/trainer/fsdp_sft_trainer.py`: same for SFT.

Everything else is byte-identical to the fork.
3. `verl/__init__.py`: `pkg_resources` (absent from the `sigma` env) replaced by
   `importlib.metadata` for the version check.
