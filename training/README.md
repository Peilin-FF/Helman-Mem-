# training/ — multi-GPU RL and SFT for the central model

Systematic, multi-GPU training of the central model on the vendored [verl](verl/VENDOR.md)
stack (Ray + FSDP actor + vLLM rollouts), taken from the ARPO repository.  The algorithm is
deliberately basic — GRPO (group-mean baseline, PPO clip, no critic, no KL) — so that the
only non-standard ingredient is the memory's function.  The ARPO / AEPO agentic and
entropy machinery of the fork is not used.

```
training/
  sigma_rl/            our code on top of verl
    build_rl_data.py     prompt stream (+ memory state)  ->  train / val parquet
    build_sft_data.py    verified generations            ->  SFT parquet
    dataset.py           SigmaRLDataset (question-only + hinted prompt per row), SigmaSFTDataset
    reward.py            verifier reward | memory pseudo-reward; parallel reward manager
    trainer.py           SigmaRayPPOTrainer = verl GRPO + the memory hint pass
    main_grpo.py         entry point (hydra)
  configs/grpo_sigma.yaml, sft_sigma.yaml
  scripts/build_data.sh, train_grpo.sh, train_sft.sh, eval_hf.sh
  setup_env.sh         installs the missing packages into the `sigma` env (done once already)
  verl/                vendored verl 0.3.1 (see VENDOR.md; two one-line patches)
```

## The protocol: labels only after the answer

At event t of the stream the central model receives the question and the peers' solutions,
unlabeled.  It decides what to trust from the **memory** (each peer's reliability on this question
address, the Kalman posterior built from the feedback of events < t — precomputed per event by
`scripts/build_generation_prompts.py`) and from its own judgment, and answers.  Only then are its
answer and the peers' solutions graded and the memory updated.  Its own answer is what we train
and evaluate: along the stream (`reward/acc_guided`) and afterwards alone, question only, on the
held-out streams (`eval_hf.sh`).  Previous methods learn from solutions annotated as correct
*before* learning; that regime is the baseline here.

Per step (`training/sigma_rl/trainer.py`, plain GRPO underneath):

1. n **solo** samples from the question-only prompt (exploration, and what is evaluated later);
2. one **guided** sample from the guided prompt — the stream-time answer;
3. every sample is graded after the fact; the guided answer is trained under the guided prompt
   (learning to use peers + memory) and re-labelled under the question-only prompt
   (internalisation), in the same GRPO group as the solo samples.

The regime is chosen when the data is built (`build_rl_data.py --guided`) plus one filter flag:

| regime | `--guided` | what the model sees before answering | labels before the answer | `memory.guided_filter` |
|---|---|---|---|---|
| **ours** | `memory` | all peer solutions + the memory's reliability notes | no | `none` |
| ours, single hint | `hint_memory` | the solution of the peer the memory trusts most | no | `none` |
| labels-after, no memory | `peers` | all peer solutions, no notes | no | `none` |
| classical baseline | `hint_label` | the most reliable *verified-correct* solution | yes | `verified` |
| classical, no memory | `hint_random` | a random verified-correct solution | yes | `verified` |
| plain RLVR | `none` | the question only | no | – |
| imitation baseline | `train_sft.sh` on `build_sft_data.py --use_targets on` | the annotated solution as target | yes | – |

Where no verifier is available after the answer (`--verified_fraction p`), the reward of a solo
sample is the memory's reliability-weighted vote over the peers' answers (`reward.py`).  The stream
is visited in order (`data.shuffle: False`), so the memory of event t never knows later feedback.

## Strict outcome layer (review of the separately contributed modules, 2026-09-05)

`outcome_protocol.py`, `outcome_batch.py`, `outcome_reward.py` and `audit_outcome_data.py` were added
on the server by another model (their unit tests were removed as redundant).  They are kept, with
these findings:

- **What they are.** A label-private formulation of the same protocol: `public_messages` (question +
  all peer responses, no reliability notes, no clipping), `CausalPeerEpisode` (a live wrapper around
  `MemoryRuntime` enforcing read -> answer -> grade -> write, with rollback), `OutcomeRewardManager`
  (post-answer verification only; refuses pseudo-rewards and untagged rows), DataProto guards, and a
  token-budget audit.
- **Not runnable as a trainer.** `outcome_batch.checked_generation` requires the rollout to return
  `peer_evidence`/`peer_mask` tensors (the memory injected as a tensor, ActivationSteerer-style).  No
  rollout backend does that (vLLM cannot), and the modules are not wired into `main_grpo.py`.  They
  are a contract for a future tensor-memory path, not a replacement for `train_grpo.sh`.
- **Conflicts with the running design, resolved as follows.** Their guards reject (a) memory notes as
  text, (b) re-labelling the guided answer under the question-only prompt, and (c) mixed GRPO
  groups.  None of these is required by the protocol: the label still arrives only after the answer;
  (b) is the STaR-style rationalisation update and (c) is what gives an all-wrong solo group its
  signal.  The running pipeline keeps all three; the strict manager is available as
  `reward_model.reward_manager=outcome` for fully verified labels-after data (rows now carry a
  `protocol` tag: `peer_outcome_v1`, or `labels_before_v1` for the classical baseline).
- **Adopted.** The 8192-token input budget (their `audit_outcome_data.py` on the training stream:
  question + all three unclipped peer responses is at most 3963 tokens, p99 2125, so nothing is ever
  dropped; the old 3000-character clipping touched 0.1% of responses), and a load-time report of how
  many guided prompts would exceed the budget (`[sigma-data] ... over_budget=...`).
- **Equivalence note.** The live episode wrapper and the precomputed memory trajectory in the prompt
  files give identical observations: the memory's inputs are the peers' labels and frozen features,
  never the central model's outputs, and the prompt builder reads the memory before writing each
  event.  Visiting the stream in order (`data.shuffle: False`) is what keeps that causal.

## Workflow (all on the server, through rproj)

```bash
# 0. once: dependencies (already installed in env `sigma`)
rproj run 'bash training/setup_env.sh'

# 1. data: one parquet per regime (stream order kept), a 30%-verified variant, and the validation set
rproj run 'MODEL_TAG=q3_4b bash training/scripts/build_data.sh'

# 2. GRPO (check `rproj gpu` first and pick idle devices); VAL=outputs/rl/data/q3_4b/val_indist.parquet everywhere
rproj submit 'GPUS=0,1,2,3 EXP=q3_4b_grpo_memory TRAIN=outputs/rl/data/q3_4b/train_memory.parquet     VAL=... bash training/scripts/train_grpo.sh'                                     # ours
rproj submit 'GPUS=0,1,2,3 EXP=q3_4b_grpo_peers  TRAIN=outputs/rl/data/q3_4b/train_peers.parquet      VAL=... bash training/scripts/train_grpo.sh'                                     # labels after, no memory
rproj submit 'GPUS=0,1,2,3 EXP=q3_4b_grpo_label  TRAIN=outputs/rl/data/q3_4b/train_hint_label.parquet VAL=... bash training/scripts/train_grpo.sh memory.guided_filter=verified'       # classical baseline
rproj submit 'GPUS=0,1,2,3 EXP=q3_4b_grpo_plain  TRAIN=outputs/rl/data/q3_4b/train_none.parquet       VAL=... bash training/scripts/train_grpo.sh memory.guided_rollouts=0'            # plain RLVR
rproj submit 'GPUS=0,1,2,3 EXP=q3_4b_grpo_v30    TRAIN=outputs/rl/data/q3_4b/train_v30_memory.parquet VAL=... bash training/scripts/train_grpo.sh'                                     # verifier on 30% only

# 3. full evaluation of any HF checkpoint (question-only, indist + OOD)
rproj submit 'GPU=1 CKPT=outputs/rl/q3_4b_grpo_hint/hf/global_step_40 bash training/scripts/eval_hf.sh'

# 4. optional SFT stage on memory-chosen hinted solutions (multi-GPU replacement of scripts/generate_hinted.py + train_memory_generator.py)
rproj run 'PYTHONPATH=. python -m training.sigma_rl.build_sft_data --prompts outputs/gen/q3_4b/prompts_train_fixed.jsonl --records data/mixed_train_big/train.jsonl --generations outputs/gen/q3_4b/<hinted>/generations.jsonl --out outputs/rl/data/q3_4b/sft_hinted_memory.parquet'
rproj submit 'GPUS=0,1,2,3 EXP=q3_4b_sft_hinted_memory TRAIN=outputs/rl/data/q3_4b/sft_hinted_memory.parquet bash training/scripts/train_sft.sh'
```

Every hydra key of `configs/grpo_sigma.yaml` can be appended to the launch command
(`actor_rollout_ref.rollout.n=4`, `data.train_batch_size=32`, `trainer.total_training_steps=100`,
`memory.hint_sampling=sample`, ...).

## Outputs (`outputs/rl/<EXP>/`)

- `train.log` — the full log; `metrics.jsonl` — one JSON line per step with
  `reward/acc_guided` (the stream-time answer, and per task), `reward/acc_solo` (and per task),
  `memory/guided_groups`, `memory/guided_success`, `memory/guided_samples_added`,
  `val-core/<task>/acc/mean@1`, timings.
- `global_step_N/actor/` — sharded FSDP checkpoint (resumable; the last 2 are kept);
  `hf/global_step_N/` — a plain HF directory (kept for every save), loadable by
  `tests/experiments/common/evaluate_memory_generator.py --checkpoint`.
- `hydra/` — the resolved config.

## Sizing (Qwen3-4B, 4 x A100-80GB)

Default: 64 prompts x 8 samples per step, mini-batch 16 prompts (4 optimizer steps per
rollout batch), prompt 3072 + response 768 tokens, vLLM at 60% of each GPU, flash-attn with
padding removed, dynamic batching at 16k tokens per GPU.  vLLM 0.8.5 counts the memory other
processes already hold on the device against `gpu_memory_utilization`, so on a GPU where others
use U GB the value must exceed (U + ~12 GB) / 80 GB, or the engine fails with "No available
memory for the cache blocks"; `actor_rollout_ref.actor.fsdp_config.param_offload=True
optimizer_offload=True` frees another ~12 GB per GPU when the box is crowded.  Validation on 512 in-distribution
prompts every 20 steps, checkpoints every 20 steps.  One epoch of the training stream
(~14.6k prompts) is ~230 steps.

Shared box rules apply: set `GPUS` to idle devices only (`rproj gpu`), never all 8; Ray uses
its own temp dir (`/mnt/data/peilin/.ray`) and at most 32 CPU cores; a killed job stops its
own Ray workers (`trap` in `train_grpo.sh`).

## Measured (smoke runs, 2026-09-05)

| run | GPUs | model | per step (8 prompts x 4 samples) | notes |
|---|---|---|---|---|
| smoke2_memory (ours) | 2 (shared) | Qwen3-0.6B | 13-48 s | every event gets its guided answer (8/8, 7/8, 7/8), both views added, no label filtering; stream accuracy of the guided answer 0.38 -> 0.57 -> 0.86 over 3 steps (tiny sample) |
| smoke2_label (classical baseline) | 1 | Qwen3-0.6B | 10-16 s | `--guided hint_label` + `guided_filter=verified`: only verified guided answers enter (5/5, 3/8, 5/7) |
| smoke_q3_4b | 2 (shared), param + optimizer offload | Qwen3-4B | 48 s (gen 14 s, guided 12 s, update 18 s) | peak torch memory 59 GB allocated / 68 GB reserved; sharded checkpoint 123 s |
| smoke8_q3_4b | 8 (all idle), no offload, default config (8192-token prompts) | Qwen3-4B | 40-58 s for 32 prompts x 4 samples + 32 guided answers (gen 18 s, guided 12-18 s, update 4-6 s) | peak 52.6 GB allocated / 61 GB reserved per GPU; validation before and after; all GPUs released at exit |

Fixed costs dominate at this size; at the default 64 x 8 batch on 4 GPUs expect roughly 3-5 min
per step, i.e. one epoch of the training stream (~230 steps) in about half a day.  A sharded
checkpoint of the 4B model is ~70 GB (fp32 shards + optimizer), so only the last one is kept;
HF copies are cast to bf16 (8 GB) and kept for every save.

Two lessons baked into the code: (1) vLLM's memory budget counts other users' processes, see
above; (2) sandboxed code grading must not fork from the many-threaded trainer actor and must
not inherit its stdin — it runs in Ray task workers with stdin on /dev/null
(`training/sigma_rl/reward.py`), otherwise the reward step can hang forever.

## Models

- **Qwen3-4B**: fully supported (vLLM 0.8.5, transformers 4.56 in env `sigma`).
- **Qwen3.5-4B**: needs transformers 5.x (env `sigma3_5`), which vLLM 0.8.5 / verl 0.3.1 do
  not support; a newer vLLM (with torch 2.8+) in a separate env is required before the
  multi-GPU stack can train it.  Until then Qwen3.5 stays on the single-GPU trainers
  (`feedback_state/train_rlvr.py`, `feedback_state/train_memory_generator.py`), which share
  the memory hooks with this stack.
