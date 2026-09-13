"""GRPO of the central model on verl's Ray PPO trainer, with the record's attention tilt in every forward.

Each prompt of the stream is answered by rollout.n samples (question + peers' answers, or the question alone), graded by
the task verifier, and trained with verl's GRPO (group-mean baseline, PPO clip, KL loss to the reference). With
``data.attn_gamma > 0`` the tilt rides with the prompt into the rollout (patched vLLM kernels) and into the actor and
reference forwards (additive attention mask). On top of verl this trainer only writes metrics.jsonl and keeps a bf16 HF
copy of every checkpoint under hf/global_step_N (the fp32 copy verl writes is removed).
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics


def hf_copy(src: str, dst: str, dtype: str = "bfloat16") -> None:
    """Copy an HF checkpoint directory, casting the weights (verl writes the fp32 master copy: 16 GB for a
    4B model) to ``dtype`` (8 GB in bf16).  ``dtype=none`` copies the files as they are."""
    if dtype in ("none", "", None):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(dst, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=getattr(_torch, dtype), low_cpu_mem_usage=True, local_files_only=True)
    model.save_pretrained(dst, safe_serialization=True)
    del model
    try:
        AutoTokenizer.from_pretrained(src, local_files_only=True).save_pretrained(dst)
    except Exception:
        for name in os.listdir(src):
            if not name.endswith((".safetensors", ".bin", ".json")) or name in ("config.json", "generation_config.json"):
                continue
            shutil.copy2(os.path.join(src, name), dst)


class KalmanRayPPOTrainer(RayPPOTrainer):
    # ------------------------------------------------------------------ logging helpers
    @staticmethod
    def _acc_metrics(batch: DataProto, metrics: dict) -> None:
        nt = batch.non_tensor_batch
        if "acc" not in nt:
            return
        acc = np.asarray(nt["acc"], dtype=float)
        metrics["reward/acc"] = float(acc.mean())
        for task in sorted(set(nt["data_source"].tolist())):
            metrics[f"reward/acc/{task}"] = float(acc[nt["data_source"] == task].mean())

    @staticmethod
    def _finish_tracking(logger) -> None:
        """Close the wandb run before the actor exits (otherwise wandb's teardown at interpreter exit
        raises a BrokenPipeError and the job ends with exit code 1 although training completed)."""
        try:
            wb = getattr(logger, "logger", {}).get("wandb")
            if wb is not None:
                wb.finish()
        except Exception as exc:
            print(f"[kalman] wandb finish failed: {exc}")

    def _append_metrics_file(self, metrics: dict) -> None:
        name = self.config.trainer.get("metrics_file", "metrics.jsonl")
        if not name:
            return
        path = os.path.join(self.config.trainer.default_local_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in metrics.items()}) + "\n")

    def _save_checkpoint(self):
        super()._save_checkpoint()
        if not self.config.trainer.get("keep_hf_checkpoints", True):
            return
        src = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}", "actor", "huggingface")
        dst = os.path.join(self.config.trainer.default_local_dir, "hf", f"global_step_{self.global_steps}")
        if os.path.isdir(src) and any(f.endswith(".safetensors") for f in os.listdir(src)):
            hf_copy(src, dst, dtype=str(self.config.trainer.get("hf_checkpoint_dtype", "bfloat16")))
            print(f"[kalman] HF checkpoint kept at {dst}")
            if self.config.trainer.get("remove_fp32_hf_copy", True):   # the 16 GB fp32 copy is redundant once the bf16 one exists
                for name in os.listdir(src):
                    if name.endswith(".safetensors") or name == "model.safetensors.index.json":
                        os.remove(os.path.join(src, name))
                print(f"[kalman] fp32 weights removed from {src}")

    @staticmethod
    def _shutdown_dataloader(iterator) -> None:
        """Stop the dataloader worker processes before the actor returns (otherwise their teardown
        raises a harmless but alarming 'DataLoader worker killed' in Ray's main loop)."""
        try:
            iterator._shutdown_workers()
        except Exception:
            pass

    # ------------------------------------------------------------------ training loop
    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(project_name=self.config.trainer.project_name, experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger, config=OmegaConf.to_container(self.config, resolve=True))
        self.global_steps = 0
        self._load_checkpoint()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            self._append_metrics_file({"training/global_step": self.global_steps, **val_metrics})
            if self.config.trainer.get("val_only", False):
                return
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        for epoch in range(self.config.trainer.total_epochs):
            dl_iter = iter(self.train_dataloader)
            for batch_dict in dl_iter:
                metrics, timing_raw = {}, {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"],
                                      non_tensor_batch_keys=[k for k in ("raw_prompt_ids", "raw_prompt", "tools_kwargs", "multi_modal_data") if k in batch.non_tensor_batch])
                if "attn_bias" in batch.batch.keys():   # the memory's attention tilt: rides with the prompts into the rollout and stays for the actor / ref forwards
                    gen_batch.batch["attn_bias"] = batch.batch["attn_bias"]
                is_last_step = self.global_steps >= self.total_training_steps
                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    batch = batch.repeat(repeat_times=rollout_n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    with _timer("reward", timing_raw):
                        reward_tensor, reward_extra = compute_reward(batch, self.reward_fn)
                    batch.batch["token_level_scores"] = reward_tensor
                    if reward_extra:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra.items()})
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=batch.batch["response_mask"], loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode)
                        metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch) if self.ref_in_actor else self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            batch = batch.union(self.critic_wg.compute_values(batch))
                    with _timer("adv", timing_raw):
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=rollout_n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            extras = {k: batch.non_tensor_batch[k].tolist() for k in ("acc", "data_source") if k in batch.non_tensor_batch}
                            self._dump_generations(inputs=inputs, outputs=outputs, scores=scores, reward_extra_infos_dict=extras, dump_path=rollout_data_dir)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=self.resource_pool_manager.get_n_gpus()))
                self._acc_metrics(batch, metrics)
                logger.log(data=metrics, step=self.global_steps)
                self._append_metrics_file(metrics)
                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    self._shutdown_dataloader(dl_iter)
                    self._finish_tracking(logger)
                    return
