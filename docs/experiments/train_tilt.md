# train_tilt: GRPO of the central model with and without the record's tilt

```bash
bash run.sh configs/experiments/train_tilt.yaml --smoke     # one arm, 2 GPUs, 2 steps, 48-event evaluation
bash run.sh configs/experiments/train_tilt.yaml             # the three arms, 8 GPUs each, then evaluation
```

Three arms, identical data and schedule, differing only in the prompt and the tilt (design log section 21):

| arm | prompt | tilt in rollouts and updates |
|---|---|---|
| `run3b_tilt` | question + six peers' answers (25% of events question-only) | on, γ = 3 |
| `ctrl3b_peers` | the same | off |
| `solo3b_q` | the question alone | off |

The steps of the experiment:

| wave | what runs | where it lands |
|---|---|---|
| records | the judge's record on train6 in file order (`fixed`) and on indist6 (`shuffled0`), if missing | `outputs/record/q3_4b/...` |
| data | `training/kalman_rl/build_rl_data.py`: one parquet per prompt kind; validation = every 8th indist6 event, 512 events | `outputs/train/data/` |
| train | GRPO from Qwen3-4B, one epoch of train6 (276 steps) | `outputs/train/<arm>/` |
| evaluate | the last checkpoint in tilt / peers / solo / swap on indist6 and ood6 | `outputs/eval/<arm>/<stream>/<condition>/` |

The GRPO settings are the defaults of `configs/train/grpo.yaml`: 64 prompts × 8 samples per step, mini-batch 16, KL loss
0.01 to the initial model, prompt 4,608 + response 768 tokens, thinking off, validation every 20 steps, a bf16 HF checkpoint
every 40 steps under `hf/global_step_N`. The tilt arm adds `data.attn_gamma=3.0`, the attention bias in the actor and the
vLLM rollout, and `sdpa` attention without padding removal (the bias needs the padded mask). A run counts as done when its
`complete.json` exists.

Results of the stored arms (last checkpoints, whole streams):

| model | indist6 tilt / peers / solo | ood6 tilt / peers / solo |
|---|---|---|
| frozen Qwen3-4B | 67.2 / 64.8 / 60.5 | 74.0 / 69.2 / 67.8 |
| run3b_tilt | 75.9 / 75.1 / 73.4 | 75.2 / 72.5 / 70.7 |
| ctrl3b_peers | 74.9 / 74.4 / 71.4 | 75.5 / 73.3 / 72.7 |
| solo3b_q | 73.0 / 72.5 / 71.5 | 75.1 / 71.2 / 72.9 |

Caveats worth keeping in view:
- The stored arms were not trained from Qwen3-4B directly: each started from a 40-step checkpoint of an earlier run of
  the same arm (removed on 2026-09-13), with data seed 2. A run of this config starts from Qwen3-4B with data seed 1, so it
  is the recipe going forward and will not reproduce the stored numbers exactly.
- The validation set is drawn from `indist6`, the stream the checkpoints are later evaluated on. It selects nothing
  (the last step is evaluated), but it is not a clean held-out split; `val_data` in the config is where to change it.

Sizing and lessons (Qwen3-4B, A100-80GB): about 55 s per step on 8 GPUs, so one epoch in about 4-5 hours per arm. vLLM
0.8.5 counts memory held by other processes against `rollout.gpu_memory_utilization` (0.6); on a crowded GPU it must
exceed (others + ~12 GB) / 80 GB. Sandboxed code grading runs in Ray task workers with stdin on /dev/null
(`training/kalman_rl/reward.py`): forking from the many-threaded trainer actor can hang the reward step.

Analyses of these runs live in `analysis/`: `run_attention_mass.sh` (where attention goes among the peer blocks),
`tilt_training_effect.py` (the tilt's lift by record spread), `compare_training_dynamics.py` (training curves).
