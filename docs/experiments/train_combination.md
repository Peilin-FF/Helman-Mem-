# Train combination: GRPO of Qwen3-4B with combination choosing every rollout's prompt

Question: when Qwen3-4B is trained under combination, does it gain reading ability (peers + memory) or its own ability
(question alone), and how does that move combination's choice between them?

## Setup

- **Training stream.** `train6` (17.7k questions, six honest peers) in the record's order (`shuffled0`), one epoch,
  64 questions and 8 samples per question per step, from the frozen Qwen3-4B. GRPO settings as `configs/train/grpo.yaml`.
- **Before training.** The frozen Qwen3-4B answers every training question alone; one record over the seven answers
  (six peers, its own answer last) gets its judge features and projected addresses (`record --save-addresses`). The judge
  stays frozen: during training only the labels change, never the addresses.
- **Per step, before the rollouts.** For each question, combination reads its state: T (top trust among the peers) and
  κ (the own answer's estimate) from the record, ρ and δ from the reading line of its task type. It picks peers + memory
  when T·ρ + (1 − T)(κ − δ) ≥ κ, otherwise question alone. All 8 samples use that prompt; peers + memory gets the tilt
  (γ = 3) from the record's current estimates.
- **Per step, after the rewards.** The record writes the six peers' verified labels. With question alone chosen it also
  writes the own answer's label, 2y − 1 with y the samples' accuracy; with peers + memory chosen the reading line takes
  (T, κ, y). Questions of one step are all read before any is written.
- **Logs.** `outputs/train/comb3b/combination.jsonl` (every question: T, κ, ρ, δ, the choice, the samples' accuracy),
  `metrics.jsonl` (per step: share of peers + memory, accuracy of each kind, ρ and δ per task), `combination_state/`.

## Run

```bash
bash run.sh configs/experiments/train_combination.yaml --smoke     # 48 questions, 2 GPUs, 2 steps
bash run.sh configs/experiments/train_combination.yaml             # own answers, features, record, then the epoch
```

The trained checkpoint is then evaluated like the frozen model (question alone, question + peers, peers + memory,
combination on the 0–100% misleading datasets), next to the frozen Qwen3-4B, with the frozen Qwen3-4B's features addressing
both records:

```bash
bash run.sh configs/experiments/combination_trained.yaml
```
