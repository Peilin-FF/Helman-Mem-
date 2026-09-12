"""Entry point: multi-GPU GRPO on the central model with the memory hooks.

  PYTHONPATH=.:training/verl python -m training.sigma_rl.main_grpo \
      data.train_files=... data.val_files=... actor_rollout_ref.model.path=... trainer.n_gpus_per_node=4 [any hydra override]

Use training/scripts/train_grpo.sh, which sets the environment (CUDA_VISIBLE_DEVICES, PYTHONPATH,
Ray temp dir, code-execution flag) and tees the log.  Config: training/configs/grpo_sigma.yaml.
"""
from __future__ import annotations

import os

import hydra
import ray
from omegaconf import OmegaConf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(ROOT, "training", "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="grpo_sigma", version_base=None)
def main(config):
    run(config)


def run(config) -> None:
    if not ray.is_initialized():
        env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            "FEEDBACK_CODE_EXEC_ALLOW": os.environ.get("FEEDBACK_CODE_EXEC_ALLOW", "1"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        }
        ray_cfg = config.get("ray_init", {}) or {}
        temp_dir = ray_cfg.get("temp_dir", None) or os.environ.get("RAY_TMPDIR", None)
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
        ray.init(runtime_env={"env_vars": env_vars}, num_cpus=ray_cfg.get("num_cpus", None), include_dashboard=False, _temp_dir=temp_dir)
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.main_ppo import create_rl_sampler
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        from training.sigma_rl.dataset import SigmaRLDataset, sigma_collate_fn
        from training.sigma_rl.reward import SigmaRewardManager, compute_score, stdin_to_devnull
        from training.sigma_rl.trainer import SigmaRayPPOTrainer

        stdin_to_devnull()   # sandboxed graders must never wait on the actor's stdin
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        assert config.actor_rollout_ref.actor.strategy in ("fsdp", "fsdp2"), "this entry point uses the FSDP workers"
        assert config.actor_rollout_ref.rollout.mode == "sync", "the memory hint pass uses the synchronous rollout"
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker), Role.Critic: ray.remote(CriticWorker)}
        pool_id = "global_pool"
        resource_pool_spec = {pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRollout: pool_id, Role.Critic: pool_id}
        if (config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss) and not bool(config.memory.get("self_distill", False)):
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = pool_id
        workers = int(config.reward_model.get("max_workers", 32))
        manager = str(config.reward_model.get("reward_manager", "sigma"))
        if manager == "outcome":   # strict post-answer verification only (no pseudo-reward, labels-after rows only)
            from training.sigma_rl.outcome_reward import OutcomeRewardManager

            reward_fn = OutcomeRewardManager(tokenizer, max_workers=workers)
            val_reward_fn = OutcomeRewardManager(tokenizer, max_workers=workers)
        else:
            reward_fn = SigmaRewardManager(tokenizer, num_examine=0, compute_score=compute_score, reward_fn_key=config.data.reward_fn_key, max_workers=workers)
            val_reward_fn = SigmaRewardManager(tokenizer, num_examine=1, compute_score=compute_score, reward_fn_key=config.data.reward_fn_key, max_workers=workers)
        train_dataset = SigmaRLDataset(config.data.train_files, tokenizer, config.data)
        val_dataset = SigmaRLDataset(config.data.val_files, tokenizer, config.data)
        trainer = SigmaRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=None,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping),
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=sigma_collate_fn,
            train_sampler=create_rl_sampler(config.data, train_dataset),
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
