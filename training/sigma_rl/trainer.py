"""GRPO with the central model's stream-time answer, on verl's Ray PPO trainer.

Protocol (one event of the stream = one prompt).  The central model receives the question and the
peers' solutions; it does not know which solution is right.  It decides what to trust from the
memory (each peer's reliability on this question address, learned from the feedback of *earlier*
events) and its own judgment, and answers.  Only then is the answer graded, the memory updated.
This is the "labels after the answer" regime.  The classical regime ("labels before": rejection
sampling / distillation on solutions that are known to be correct) is the baseline, obtained from the
same code by building the guided prompt from a *verified-correct* peer (build_rl_data --guided
hint_label / hint_random) and keeping only verified guided answers (memory.guided_filter=verified).

Per step, on top of verl's plain GRPO (group-mean baseline, PPO clip, no critic, KL off):

  solo samples     n rollouts from the question-only prompt (exploration, and what is evaluated later)
  guided samples   rollouts from the guided prompt: the stream-time answer.  Graded post hoc like every
                   other sample.  Trained under the guided prompt (learning to use peers + memory) and/or
                   re-labelled under the question-only prompt (internalisation), same GRPO group as the
                   solo samples so the group baseline is shared.

The memory enters only through the guided prompt (which solutions, with which reliability notes) and,
when no verifier is available after the answer, through the pseudo-reward (training/sigma_rl/reward.py).
Everything else (advantages, actor update, checkpoints, validation) is untouched verl.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask

from training.sigma_rl.dataset import GUIDED_NON_TENSOR_KEYS, GUIDED_TENSOR_KEYS


def _take(extra: dict, idx) -> dict:
    return {k: [v[i] for i in idx] for k, v in extra.items()}


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


class SigmaRayPPOTrainer(RayPPOTrainer):
    # ------------------------------------------------------------------ guided pass (the stream-time answer)
    def _memory_cfg(self) -> dict:
        m = self.config.get("memory", None)
        return dict(m) if m is not None else {}

    @staticmethod
    def _clone(dp: DataProto) -> DataProto:
        return dp.select_idxs(list(range(len(dp))))

    def _guided_pass(self, batch: DataProto, guided_batch: DataProto | None, reward_tensor: torch.Tensor, reward_extra: dict, metrics: dict):
        mcfg = self._memory_cfg()
        rollouts = int(mcfg.get("guided_rollouts", 0))
        batch.non_tensor_batch["sample_kind"] = np.array(["solo"] * len(batch), dtype=object)
        scores = reward_tensor.sum(-1)
        uids = batch.non_tensor_batch["uid"]
        group_scores: dict[str, list[float]] = defaultdict(list)
        first_row: dict[str, int] = {}
        for i, u in enumerate(uids):
            group_scores[u].append(float(scores[i]))
            first_row.setdefault(u, i)
        zero = {u for u, s in group_scores.items() if max(s) <= 0.0}
        metrics["memory/groups"] = len(group_scores)
        metrics["memory/zero_groups"] = len(zero)
        for k in ("guided_groups", "guided_samples", "guided_success", "guided_samples_added"):
            metrics[f"memory/{k}"] = 0
        if rollouts <= 0 or guided_batch is None:
            return batch, reward_tensor, reward_extra
        only_zero = str(mcfg.get("guided_groups", "all")) == "zero"
        verified_only = bool(mcfg.get("guided_verified_only", True))
        loss_prompt = str(mcfg.get("guided_loss_prompt", "both"))
        assert loss_prompt in ("solo", "guided", "both"), loss_prompt
        guided_row: dict[str, int] = {}
        has = guided_batch.batch["has_guided"]
        infos = guided_batch.non_tensor_batch.get("extra_info", None)
        for j, u in enumerate(guided_batch.non_tensor_batch["uid"]):
            if int(has[j]) != 1:
                continue
            if verified_only and infos is not None and not bool((infos[j] or {}).get("verified", True)):
                continue
            guided_row[u] = j
        sel = [u for u in first_row if u in guided_row and (not only_zero or u in zero)][: int(mcfg.get("max_guided_groups", 1_000_000))]
        metrics["memory/guided_groups"] = len(sel)
        if not sel:
            return batch, reward_tensor, reward_extra
        # --- rollout from the guided prompts
        src = guided_batch.select_idxs([guided_row[u] for u in sel])
        raw = np.empty(len(sel), dtype=object)
        for i, x in enumerate(src.non_tensor_batch["guided_raw_prompt_ids"]):
            raw[i] = [int(t) for t in x]
        hg = DataProto.from_dict(
            tensors={"input_ids": src.batch["guided_input_ids"], "attention_mask": src.batch["guided_attention_mask"], "position_ids": src.batch["guided_position_ids"]},
            non_tensors={"raw_prompt_ids": raw},
            meta_info={"do_sample": str(mcfg.get("guided_sampling", "greedy")) != "greedy"},
        )
        world_size = self.actor_rollout_wg.world_size
        hg_padded, pad = pad_dataproto_to_divisor(hg, world_size)
        out_full = self.actor_rollout_wg.generate_sequences(hg_padded)
        reps = len(out_full) // len(hg_padded)
        out = out_full[: len(sel) * reps]
        base_rows = [first_row[u] for u in sel for _ in range(reps)]
        responses = out.batch["responses"]
        response_att = out.batch["attention_mask"][:, -responses.shape[1]:]
        # --- rows under the guided prompt, exactly as generated
        g = batch.select_idxs(base_rows)
        for key in ("prompts", "responses", "input_ids", "attention_mask", "position_ids"):
            g.batch[key] = out.batch[key]
        g.batch["response_mask"] = response_att
        if "rollout_log_probs" in g.batch.keys() and "rollout_log_probs" in out.batch.keys():
            g.batch["rollout_log_probs"] = out.batch["rollout_log_probs"]
        g.non_tensor_batch["sample_kind"] = np.array(["guided"] * len(g), dtype=object)
        # --- the same answers re-labelled under the question-only prompt (internalisation)
        sb = batch.select_idxs(base_rows)
        prompt_solo = sb.batch["prompts"]
        attention_mask = torch.cat([sb.batch["attention_mask"][:, : prompt_solo.shape[1]], response_att], dim=1)
        sb.batch["responses"] = responses
        sb.batch["input_ids"] = torch.cat([prompt_solo, responses], dim=1)
        sb.batch["attention_mask"] = attention_mask
        sb.batch["position_ids"] = compute_position_id_with_mask(attention_mask)
        sb.batch["response_mask"] = response_att
        if "rollout_log_probs" in sb.batch.keys() and "rollout_log_probs" in out.batch.keys():
            sb.batch["rollout_log_probs"] = out.batch["rollout_log_probs"]
        sb.non_tensor_batch["sample_kind"] = np.array(["guided_solo"] * len(sb), dtype=object)
        # --- grade after the answer (once; both views share the answer)
        h_reward, h_extra = compute_reward(g, self.reward_fn)
        h_scores = h_reward.sum(-1)
        metrics["memory/guided_samples"] = int(len(g))
        metrics["memory/guided_success"] = int((h_scores > 0).sum())
        # the central model's stream-time answers, graded after the fact (before any filtering)
        g_acc = np.asarray(h_extra["acc"], dtype=float) if "acc" in h_extra else h_scores.gt(0).float().numpy()
        metrics["reward/acc_guided"] = float(g_acc.mean())
        metrics["reward/score_guided"] = float(h_scores.mean())
        g_tasks = g.non_tensor_batch["data_source"]
        for task in sorted(set(g_tasks.tolist())):
            metrics[f"reward/acc_guided/{task}"] = float(g_acc[g_tasks == task].mean())
        keep = torch.nonzero(h_scores > 0).flatten().tolist() if str(mcfg.get("guided_filter", "none")) == "verified" else list(range(len(g)))
        if not keep:
            return batch, reward_tensor, reward_extra
        parts = []
        if loss_prompt in ("guided", "both"):
            parts.append(g)
        if loss_prompt in ("solo", "both"):
            parts.append(sb)
        need = (-(len(keep) * len(parts))) % world_size   # every dp rank must receive the same number of rows
        added = DataProto.concat([part.select_idxs(keep) for part in parts])
        added = added.select_idxs(list(range(len(added))) + [i % len(added) for i in range(need)])
        add_reward = torch.cat([h_reward[keep] for _ in parts], dim=0)
        add_reward = torch.cat([add_reward, add_reward[[i % len(add_reward) for i in range(need)]]], dim=0) if need else add_reward
        add_extra = {k: [v[i] for i in keep] * len(parts) for k, v in h_extra.items()}
        add_extra = {k: v + [v[i % len(v)] for i in range(need)] for k, v in add_extra.items()}
        metrics["memory/guided_samples_added"] = int(len(added))
        merged = DataProto.concat([batch, added])
        reward_tensor = torch.cat([reward_tensor, add_reward], dim=0)
        extra = {k: list(v) + list(add_extra.get(k, [0.0] * len(added))) for k, v in reward_extra.items()}
        for k, v in add_extra.items():
            if k not in extra:
                extra[k] = [0.0] * len(batch) + list(v)
        return merged, reward_tensor, extra

    # ------------------------------------------------------------------ logging helpers
    def _memory_metrics(self, batch: DataProto, metrics: dict) -> None:
        nt = batch.non_tensor_batch
        scores = batch.batch["token_level_scores"].sum(-1).numpy()
        kind = nt.get("sample_kind", np.array(["solo"] * len(batch), dtype=object))
        acc = np.asarray(nt["acc"], dtype=float) if "acc" in nt else scores
        for name, label in (("solo", "solo"), ("guided", "guided_kept")):   # guided_kept = the guided rows that entered the update
            m = kind == name
            if not m.any():
                continue
            metrics[f"reward/score_{label}"] = float(scores[m].mean())
            metrics[f"reward/acc_{label}"] = float(acc[m].mean())
            if name == "solo":
                for task in sorted(set(nt["data_source"].tolist())):
                    mt = m & (nt["data_source"] == task)
                    if mt.any():
                        metrics[f"reward/acc_solo/{task}"] = float(acc[mt].mean())
        solo = kind == "solo"
        if "verified" in nt and solo.any():
            ver = np.asarray(nt["verified"], dtype=float) > 0.5
            metrics["reward/verified_frac"] = float(ver[solo].mean())
            unv = solo & ~ver
            if unv.any():
                metrics["reward/pseudo_score_unverified"] = float(scores[unv].mean())
                metrics["reward/true_acc_unverified"] = float(acc[unv].mean())
                if "pseudo" in nt:
                    metrics["reward/pseudo_available_frac"] = float(np.asarray(nt["pseudo"], dtype=float)[unv].mean())

    @staticmethod
    def _finish_tracking(logger) -> None:
        """Close the wandb run before the actor exits (otherwise wandb's teardown at interpreter exit
        raises a BrokenPipeError and the job ends with exit code 1 although training completed)."""
        try:
            wb = getattr(logger, "logger", {}).get("wandb")
            if wb is not None:
                wb.finish()
        except Exception as exc:
            print(f"[sigma] wandb finish failed: {exc}")

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
            print(f"[sigma] HF checkpoint kept at {dst}")

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
                guided_batch = None
                if "has_guided" in batch.batch.keys():
                    guided_batch = batch.pop(batch_keys=[k for k in GUIDED_TENSOR_KEYS if k in batch.batch.keys()],
                                             non_tensor_batch_keys=[k for k in GUIDED_NON_TENSOR_KEYS if k in batch.non_tensor_batch])
                    guided_batch.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"].copy()
                    if "extra_info" in batch.non_tensor_batch:
                        guided_batch.non_tensor_batch["extra_info"] = batch.non_tensor_batch["extra_info"].copy()
                gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"],
                                      non_tensor_batch_keys=[k for k in ("raw_prompt_ids", "raw_prompt", "tools_kwargs", "multi_modal_data") if k in batch.non_tensor_batch])
                is_last_step = self.global_steps >= self.total_training_steps
                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    batch = batch.repeat(repeat_times=rollout_n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    with _timer("reward", timing_raw):
                        reward_tensor, reward_extra = compute_reward(batch, self.reward_fn)
                    with _timer("guided", timing_raw):
                        batch, reward_tensor, reward_extra = self._guided_pass(batch, guided_batch, reward_tensor, reward_extra, metrics)
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
                            extras = {k: batch.non_tensor_batch[k].tolist() for k in ("acc", "verified", "pseudo", "sample_kind", "data_source") if k in batch.non_tensor_batch}
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
                self._memory_metrics(batch, metrics)
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
