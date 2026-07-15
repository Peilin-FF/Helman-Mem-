<h1 align="center">
    Feedback-Supervised Persistent State (on δ-mem)
</h1>

A multi-peer math-solving extension built on top of the δ-mem repository. A central Qwen3-4B model is a pure **aggregator**: given K candidate answers/steps from smaller peer models, it does not generate an answer from scratch — it **selects** the most trustworthy peer's response (a K-way multiple-choice over peers). A **persistent per-peer matrix state** is the trust model: it regresses, online, how reliable each peer has been, and its read scores drive the selection. The state updates online from feedback: ground-truth answers (Setting A) or forward-rollout outcomes (Setting B).

Because the central model only *selects*, its accuracy is bounded above by **any-peer-correct coverage** (no selector can pick a right answer when none exists) and should beat the **random-peer** and **best-fixed-peer** floors — those three are reported alongside accuracy (see `eval_baselines.py`). The frozen backbone is used only as an encoder `Enc(x, r_j)`, so the trust state is the only learnable driver of accuracy. A legacy generative path (`task: generate`) is retained for ablations.

The original δ-mem implementation it builds on still lives under [`deltamem/`](deltamem/) and is unmodified — the released δ-mem adapter is **not** required to use this code.

## What is in this repo

```text
feedback_state/                 # core module
├── state.py                    # FeedbackPersistentState — per-peer r×r matrix state (+ save/load trained S)
├── model.py                    # FeedbackStateQwenForCausalLM — frozen encoder + per-peer trust scorer (selection); legacy prefix-memory generation
├── data.py                     # JSONL datasets + FeedbackDataCollator
├── tasks.py                    # task registry: math/rag/code correctness, extraction, prompts (add a TaskSpec to extend)
├── datasets.py                 # 7-benchmark math registry + load_math_dataset()
├── eval_datasets.py            # RAG + code loaders (hotpotqa, triviaqa, humaneval, mbpp, bigcodebench, livecodebench)
├── code_exec.py                # sandboxed code execution (asserts/unittest/io/functional)
├── scenarios.py                # controlled offline data scenarios (counter-trust); DatasetScenario interface
├── permutations.py             # fixed peer orders (orig/swap), slot↔peer_id mapping, identity encoding, bias metrics
├── joint_prompt.py             # joint-input prompt / AR target / candidate id helpers (pure)
├── joint_data.py               # JointInputCollator (one joint prompt; AR labels / BCE spans)
├── joint_models.py             # JointDeltaMemSelector (Variants A & B; shared-state attention steering via Delta-Mem)
├── math_synthesis.py           # optional synthetic math counterfactual generation utilities
├── plans.py                    # Setting B step schemas (llm | semantic_4 | semantic_5)
├── rollout.py                  # Setting B helpers (build_step_context, score_rollout_texts)
├── baselines.py                # majority vote, aggregator, self-judge, oracle, text-memory, + selection floors/ceiling
├── cooling.py                  # apply_cooling_off (batched encode + sequential write)
├── evaluation.py               # FeedbackSchedule + FeedbackEvalRunner (learning-curve protocol)
├── metrics.py                  # bootstrap CI, paired sign test, slope
├── adversarial.py              # peer corruption (numeric perturb / plausible-wrong / swap_peer)
├── generation.py               # TextGenerator wrapping Transformers or vLLM
└── utils.py                    # config loader, answer extractor/normalizer, math_equal

configs/                        # YAML configs
├── benchmarks.yaml             # per-benchmark dataset settings
├── generate_setting_{a,b}.yaml # offline data generation
├── feedback_state_{a,b}.yaml   # training (incl. cooling_off_fraction)
├── eval_feedback_state.yaml    # clean proposed-method eval (+ cooling-off impact)
├── eval_adversarial.yaml       # opt-in controlled-corruption robustness eval
├── eval_baselines.yaml         # eval of any of the 6 baselines
└── eval_schedule.yaml          # learning-curve protocol (multi-seed, multi-checkpoint)

scripts/
├── generate_setting_a_peers.py     # task-aware peer responses (math/rag/code; RAG context deprivation) → JSONL
├── generate_setting_b_rollouts.py  # plans + per-step peer proposals + forward rollouts
├── score_code_peers.py             # offline per-peer code pass@1 → record["peer_correct"]
├── build_counter_trust.py          # construct N%-counter-trust train/test data (offline)
├── mix_task_datasets.py            # interleave/concat/shuffle per-task JSONLs → one mixed eval stream
└── plot_trust_shift.py             # per-task peer selection rate vs. feedback count (the trust-shift curve)

train_feedback_state.py
eval_feedback_state.py
eval_baselines.py
deltamem/tests/test_feedback_state.py   # unit tests
```

## How it works (one paragraph each)

**Selection over peers (proposed).** The state stores *trust*, not answer content. The write target is a **signed correctness** value `v_j = (2c_j − 1)·τ` along a fixed trust direction `τ` (`trust_value`), where `c_j ∈ [0,1]` is peer j's correctness and confidence `= |2c_j − 1|`. At test time the central model reads `m_j = S_j · q_j` for every peer, maps it through a scalar head `score_j = w·m_j`, and **selects `argmax_j score_j`**, emitting that peer's extracted answer. Training is **per-peer BCE**: `Σ_j BCE(σ(score_j), c_j)`, backpropagated through the online reads (BPTT across the feedback stream). The projections `W_q, W_k` and the score head are learned by this loss; `W_β` (the trust adaptation rate) stays learned; the backbone is frozen and only encodes `h_j = Enc(x, r_j)`. This unifies the two settings — they differ only in how `c_j` is obtained.

**Setting A — outcome-supervised trust.** For each problem `x` with ground-truth answer `y` and K peer responses `r_j`, `c_j = 1` iff `r_j`'s answer is math-equal to `y`, else `0`. The central model selects with the *current* state, then (no label leak) the state is updated with a delta-rule write: `S_j ← Diag(λ_j)·S_j + Diag(β_j)·(v_j − S_j·k_j)·k_jᵀ`, with `β_j = |2c_j−1|·σ(W_β·h_j + b)`, `λ_j = 1 − β_j`, `h_j = Enc(x, r_j)`.

**Setting B — rollout-supervised trust.** Each problem is decomposed into a fixed T-step plan. At step t, peers propose `r_{j,t}`; each proposal is forced as the next step and the remaining steps are rolled out M times. The rollout success rate `z_{j,t}` *is* the soft correctness `c_{j,t}` — so the signed target `v_{j,t} = (2z − 1)·τ` and confidence `|2z − 1|` are exactly the Setting A write with a continuous label. Successful rollouts pull the peer's trust up, failed rollouts push it down, uncertain rollouts barely move it.

## Engineer runbook

Run everything from the repo root. **All data — train and test — is built
offline and statically**, then the trust state is trained and evaluated on those
fixed JSONLs. The pipeline has four stages; each stage is one command and writes a
JSONL you can inspect:

```text
1. GENERATE   peer responses for problems            scripts/generate_setting_a_peers.py   -> pool JSONL
   (task-aware: math / rag / code; RAG context withheld from chosen peers;
    samples_per_peer>1 builds a sample pool for counterfactual pairing)
2. CONSTRUCT  optional offline data shaping
     - code pass@1 labels                            scripts/score_code_peers.py           -> + peer_correct
     - counter-trust fraction N%                     scripts/build_counter_trust.py        -> diagnostic JSONL
     - mix task types into one stream                scripts/mix_task_datasets.py          -> mixed JSONL
3. TRAIN      the trust state on the train JSONL      train_feedback_state.py               -> checkpoint (+ trust_state.pt)
4. EVALUATE   on the test JSONL                       eval_feedback_state.py                -> metrics JSON
     baselines / floors / ceiling                     eval_baselines.py
     trust-shift plot                                 scripts/plot_trust_shift.py
```

Minimum path: **GENERATE → TRAIN → EVALUATE** (stage 2 is only for the multi-task,
code, and counter-trust experiments). The model is a frozen-backbone *selector*:
training learns the per-peer trust state only — no LoRA, no central-model
fine-tuning, by default.

### 0. Prepare the environment

Use Python 3.10+ with a CUDA-enabled PyTorch build on the training machine:

```bash
cd /path/to/delta-Mem-main
pip install -r requirements.txt
# Recommended on CUDA generation hosts:
pip install vllm
huggingface-cli login
```

Accept the Hugging Face license for `google/gemma-3-4b-it` before generation.
The default model set is:

| Role | Model |
| --- | --- |
| Central model | `Qwen/Qwen3-4B-Instruct-2507` |
| Peer 0 | `google/gemma-3-4b-it` |
| Peer 1 | `microsoft/Phi-4-mini-instruct` |
| Peer 2 | `Qwen/Qwen2.5-Coder-7B-Instruct` |

These are current instruct checkpoints in the roughly 4B class, from different
model families. Gemma 3 is multimodal-capable, but this project sends text-only
prompts. Keep `use_vllm: true` for production generation.

Verify the local code before launching GPU jobs:

```bash
PYTHONPATH=. python -m compileall -q feedback_state scripts \
  train_feedback_state.py eval_feedback_state.py eval_baselines.py deltamem/tests
PYTHONPATH=. python -m pytest -q deltamem/tests
```

Tests that execute generated code need `FEEDBACK_CODE_EXEC_ALLOW=1` set (others skip).

### 1. Run the lightweight smoke pipeline

The checked-in smoke configs use `sshleifer/tiny-gpt2`, CPU execution, and a
three-row local math file. They verify execution only, not math quality.

```bash
# Setting A: generate -> train -> evaluate
PYTHONPATH=. python scripts/generate_setting_a_peers.py \
  --config configs/smoke_generate_setting_a.yaml \
  --output data/smoke_generated_setting_a.jsonl
PYTHONPATH=. python train_feedback_state.py \
  --config configs/smoke_feedback_state_a.yaml \
  --offline_data data/smoke_generated_setting_a.jsonl \
  --output_dir outputs/smoke_feedback_state_a
PYTHONPATH=. python eval_feedback_state.py \
  --config configs/smoke_eval_feedback_state.yaml \
  --checkpoint outputs/smoke_feedback_state_a \
  --offline_data data/smoke_generated_setting_a.jsonl \
  --output outputs/smoke_eval_a.json \
  --mode online_feedback

# Setting B: fixed plan -> rollouts -> train -> evaluate
PYTHONPATH=. python scripts/generate_setting_b_rollouts.py \
  --config configs/smoke_generate_setting_b.yaml \
  --output data/smoke_generated_setting_b.jsonl
PYTHONPATH=. python train_feedback_state.py \
  --config configs/smoke_feedback_state_b.yaml \
  --offline_data data/smoke_generated_setting_b.jsonl \
  --output_dir outputs/smoke_feedback_state_b
PYTHONPATH=. python eval_feedback_state.py \
  --config configs/smoke_eval_feedback_state.yaml \
  --checkpoint outputs/smoke_feedback_state_b \
  --offline_data data/smoke_generated_setting_b.jsonl \
  --output outputs/smoke_eval_b.json \
  --setting B --mode setting_b_rollout
```

### 2. Generate Setting A offline artifacts

DeepMath-103K exposes a single Hugging Face split named `train`. Use
`--start_index` and `--max_samples` to create disjoint train, validation, and
test artifacts from that source split:

```bash
# Train: source rows [0, 5000)
PYTHONPATH=. python scripts/generate_setting_a_peers.py \
  --config configs/generate_setting_a.yaml \
  --split train --start_index 0 --max_samples 5000 \
  --output data/deepmath_setting_a_train.jsonl

# Validation: source rows [5000, 5500)
PYTHONPATH=. python scripts/generate_setting_a_peers.py \
  --config configs/generate_setting_a.yaml \
  --split train --start_index 5000 --max_samples 500 \
  --output data/deepmath_setting_a_val.jsonl

# Test: source rows [5500, 6000)
PYTHONPATH=. python scripts/generate_setting_a_peers.py \
  --config configs/generate_setting_a.yaml \
  --split train --start_index 5500 --max_samples 500 \
  --output data/deepmath_setting_a_test.jsonl
```

Generation is offline, cached, resumable, and shardable. Re-running the same
command skips completed IDs. Use `--shard_index N --num_shards M` to divide a
slice across workers.

Each Setting A row contains the problem, normalized answer, full peer
responses, model identities, adversarial metadata, and generation parameters.
The default generation config uses ordinary real-model peer responses
(`adversarial_rate: 0.0`). Controlled corruption is an optional robustness
experiment, not the default pipeline.

### 3. Train Setting A

```bash
PYTHONPATH=. python train_feedback_state.py \
  --config configs/feedback_state_a.yaml \
  --offline_data data/deepmath_setting_a_train.jsonl \
  --output_dir outputs/feedback_state_a
```

The backbone is **frozen** (encoder only); only the trust state + scoring head are
trained (`use_lora: false`, `state_dim: 8`). `gradient_accumulation_steps: 4` is
intentional: the projections `W_q`/`W_k`/`W_β` receive gradients through the state
carried into later reads (BPTT across the feedback stream); with accumulation `1`
the state is detached before those reads, so keep it ≥ 2.

Training also saves the **trained belief state** `S` to
`outputs/feedback_state_a/trust_state.pt` (`save_trained_state: true`). `S` is
runtime memory, not a weight — eval can warm-start from it (see *Warm-starting eval*).

### 4. Evaluate Setting A

Run clean online-feedback evaluation first:

```bash
PYTHONPATH=. python eval_feedback_state.py \
  --config configs/eval_feedback_state.yaml \
  --checkpoint outputs/feedback_state_a \
  --offline_data data/deepmath_setting_a_test.jsonl \
  --output outputs/feedback_state_a_clean.json
```

Then measure robustness under adversarial peers:

```bash
for r in 0.25 0.5 0.75 1.0; do
  PYTHONPATH=. python eval_feedback_state.py \
    --config configs/eval_adversarial.yaml \
    --checkpoint outputs/feedback_state_a \
    --offline_data data/deepmath_setting_a_test.jsonl \
    --output outputs/feedback_state_a_adversarial_r${r}.json \
    --mode adversarial_peers --adversarial_rate $r \
    --adversarial_peers 0 1
done
```

Run directly comparable Setting A baselines on the same offline test file. All
are **answer-selection** tasks over the same peer pool: `majority_vote`,
`self_judge`, and the LLM `aggregator`s pick among peer answers; `random_peer`
and `best_fixed_peer` are the selection floors; `any_peer_correct` is the
selection ceiling (coverage) that bounds every method including ours.

```bash
for baseline in any_peer_correct random_peer best_fixed_peer majority_vote \
                aggregator aggregator_with_ids self_judge text_memory; do
  PYTHONPATH=. python eval_baselines.py \
    --config configs/eval_baselines.yaml \
    --offline_data data/deepmath_setting_a_test.jsonl \
    --output outputs/baseline_${baseline}.json \
    --baseline $baseline --setting A
done
```

### 5. Generate Setting B offline artifacts

For the project objective, use `semantic_4`. It defines the step roles before
solving and enforces the same four-step semantics for every problem:

1. `Understand`
2. `Plan`
3. `Compute`
4. `Verify`

The planner fills in problem-specific content for those roles before any peer
proposal or rollout is generated. The full plan is stored in each JSONL row.

```bash
# Train: source rows [0, 2000)
PYTHONPATH=. python scripts/generate_setting_b_rollouts.py \
  --config configs/generate_setting_b.yaml \
  --split train --start_index 0 --max_samples 2000 \
  --step_schema semantic_4 --fixed_steps 4 --rollout_count 1 \
  --output data/deepmath_setting_b_train.jsonl

# Validation: source rows [2000, 2200)
PYTHONPATH=. python scripts/generate_setting_b_rollouts.py \
  --config configs/generate_setting_b.yaml \
  --split train --start_index 2000 --max_samples 200 \
  --step_schema semantic_4 --fixed_steps 4 --rollout_count 1 \
  --output data/deepmath_setting_b_val.jsonl

# Test: source rows [2200, 2400)
PYTHONPATH=. python scripts/generate_setting_b_rollouts.py \
  --config configs/generate_setting_b.yaml \
  --split train --start_index 2200 --max_samples 200 \
  --step_schema semantic_4 --fixed_steps 4 --rollout_count 1 \
  --output data/deepmath_setting_b_test.jsonl
```

Use `--rollout_count 3` for a higher-quality `z_{j,t}` estimate at about three
times the rollout cost. Setting B stores the plan, step contexts, peer step
proposals, rollout completions, extracted rollout answers, and scores.

### 6. Train and evaluate Setting B

```bash
PYTHONPATH=. python train_feedback_state.py \
  --config configs/feedback_state_b.yaml \
  --offline_data data/deepmath_setting_b_train.jsonl \
  --output_dir outputs/feedback_state_b

PYTHONPATH=. python eval_feedback_state.py \
  --config configs/eval_feedback_state.yaml \
  --checkpoint outputs/feedback_state_b \
  --offline_data data/deepmath_setting_b_test.jsonl \
  --output outputs/feedback_state_b_rollout.json \
  --setting B --mode setting_b_rollout

PYTHONPATH=. python eval_baselines.py \
  --config configs/eval_baselines.yaml \
  --offline_data data/deepmath_setting_b_test.jsonl \
  --output outputs/baseline_oracle_rollout.json \
  --baseline oracle_rollout --setting B
```

### 7. Optional cooling-off experiment

To reserve delayed-feedback examples, set `cooling_off_fraction: 0.2` in
`configs/feedback_state_a.yaml` before training. Training writes the holdout to
`outputs/feedback_state_a/cooling_holdout.jsonl`.

Measure the effect of state-only cooling writes:

```bash
PYTHONPATH=. python eval_feedback_state.py \
  --config configs/eval_feedback_state.yaml \
  --checkpoint outputs/feedback_state_a \
  --offline_data data/deepmath_setting_a_test.jsonl \
  --output outputs/feedback_state_a_cooling_impact.json \
  --cooling_off_rate 1.0 --measure_cooling_impact
```

### Artifact checklist

After a full run, the important artifacts are:

```text
data/deepmath_setting_a_{train,val,test}.jsonl
data/deepmath_setting_b_{train,val,test}.jsonl
outputs/feedback_state_a/feedback_state_adapter.pt
outputs/feedback_state_b/feedback_state_adapter.pt
outputs/feedback_state_a_clean.json
outputs/feedback_state_a_adversarial_r*.json
outputs/feedback_state_b_rollout.json
outputs/baseline_*.json
```

## CLI reference

Useful training overrides:

| Flag | Effect |
| --- | --- |
| `--max_steps 200` | Stop early for a GPU smoke run. |
| `--resume_from_checkpoint outputs/feedback_state_a` | Load the trust adapter before training. |
| `--train_memory_only true` | Train only the trust params (`feedback_state.*`, `peer_score_head`, `trust_value`). |
| `--use_lora true` | (Ablation) LoRA-tune the encoder — reintroduces a backbone confound; off by default. |

Config keys worth knowing: `task: selection` (default; `generate` is the legacy
ablation), `num_peers`, `save_trained_state: true`, `samples_per_peer` (generation).

Evaluation: `--init_state {zeros,trained}` chooses the starting belief, `--mode`
chooses what happens during the test pass:

| `--mode` | Behaviour (selection) |
| --- | --- |
| `online_feedback` | Select with the current state, then write the feedback signal after each example (state keeps adapting). |
| `read_only` | Select with the current state; **no** writes during the test pass. |
| `no_memory` | Ignore the state; pick a peer at random (the no-trust floor). |
| `adversarial_peers` | Corrupt selected peers, then continue online writes. |
| `setting_b_rollout` | Setting B: write the rollout-score signal step-wise. |

| `--init_state` | Starting belief |
| --- | --- |
| `zeros` (default) | Cold start. |
| `trained` | Warm-start from `<checkpoint>/trust_state.pt` (or `--trained_state PATH`). |

The two warm-start settings are `--init_state trained` with `--mode online_feedback`
(keep updating) or `--mode read_only` (freeze). Peer corruption is gated by
`--mode adversarial_peers`; a nonzero `--adversarial_rate` in a clean mode is
ignored and reported as `ignored_adversarial_rate`.

## Benchmarks

`feedback_state/datasets.py` ships a registry that normalises every supported math benchmark to `{"id", "problem", "answer", "source"}`. The generation scripts and the proposed-method eval both go through it, so apples-to-apples comparison across benchmarks is straightforward.

| `dataset:` key | HF path | Default split | Notes |
| --- | --- | --- | --- |
| `deepmath` | `zwhe99/DeepMath-103K` | `train` | Used for training **and** the cooling-off holdout. |
| `math500` | `HuggingFaceH4/MATH-500` | `test` | |
| `aime2024` | `Maxwell-Jia/AIME_2024` | `train` | 30 problems, single split. |
| `amc` | `AI-MO/aimo-validation-amc` | `train` | |
| `olympiadbench` | `Hothan/OlympiadBench` | `train` | Pass `dataset_config: OE_TO_maths_en_COMP` (English math, open-ended, competition). Answer field is a list — first non-empty value is used. |
| `minerva` | `math-ai/minervamath` | `test` | |
| `college_math` | `math-ai/college-math` | `test` | |

If a mirror's column names drift, override at the call site:

```yaml
# configs/generate_setting_a.yaml
dataset: math500
field_overrides:
  problem: question         # if a fork renamed it
  answer: target
```

Or programmatically:

```python
from feedback_state.datasets import load_math_dataset
records = load_math_dataset("aime2024", split="train", max_samples=10)
```

### Per-benchmark generation

Generate Setting A peer responses for any benchmark by overriding `dataset:` (and `--split`):

```bash
PYTHONPATH=. python scripts/generate_setting_a_peers.py \
  --config configs/generate_setting_a.yaml \
  --dataset math500 --split test --max_samples 500 \
  --output data/math500_setting_a_test.jsonl

PYTHONPATH=. python scripts/generate_setting_a_peers.py \
  --config configs/generate_setting_a.yaml \
  --dataset olympiadbench --split train --max_samples 200 \
  --output data/olympiadbench_setting_a.jsonl
```

Setting B works the same way; add `--step_schema` to pick the decomposition (see next section).

## Setting B step decomposition

The "what is one step" question matters: within a problem, all peers see the same step plan so peer-vs-peer comparison is well-defined, but across problems the meaning of "step 0" depends on the planner's choice. Pick the schema that matches your analysis:

| `step_schema` | What "step t" means | Use when |
| --- | --- | --- |
| `llm` (code fallback) | Whatever the central planner produced for that specific problem | You want max per-problem quality, comparisons stay within a problem |
| `semantic_4` (shipped config) | Step 0 = Understand, 1 = Plan, 2 = Compute, 3 = Verify (across every problem) | Project default; predefined steps and cross-problem step-level metrics |
| `semantic_5` | Adds a Decompose role between Understand and Plan | Same as above but with an explicit sub-problem decomposition step |

Set it in the generator config or via CLI:

```bash
PYTHONPATH=. python scripts/generate_setting_b_rollouts.py \
  --config configs/generate_setting_b.yaml \
  --dataset math500 --step_schema semantic_4 \
  --output data/math500_setting_b_test.jsonl
```

With a fixed schema, every step in the saved JSONL is prefixed with its role (e.g. `"[Understand] restate ..."`), so the same `t` index across problems shares semantics. Each record's `step_schema` field records which choice was used — eval/baselines can branch on it.

## Baselines

Six baselines live in [`feedback_state/baselines.py`](feedback_state/baselines.py) and run through [`eval_baselines.py`](eval_baselines.py). All operate on the same offline JSONL files the proposed method uses, so numbers are directly comparable.

| `--baseline` | Setting | What it does |
| --- | --- | --- |
| `majority_vote` | A | Extract every peer's final answer, return the most common. No central model call. |
| `aggregator` | A | Central model picks the best from {peer responses, its own attempt}. |
| `aggregator_with_ids` | A | Same as `aggregator` but candidate labels include the producer's model name — tests whether the central model implicitly learns to trust certain identities. |
| `self_judge` | A | Central scores each candidate 0–10 and picks the highest. |
| `text_memory` | A | Prepends a running text log of per-peer correctness to the prompt. Sequential, log updates **after** each prediction (no leak). Tests whether prompt-level memory matches the latent persistent state. |
| `oracle_rollout` | B | At each step, pick the peer with the highest rollout score. Upper bound on rollout-label informativeness; no model involved. |

### Run a baseline

```bash
# 1. Pure post-processing (no model load) — fastest sanity check.
PYTHONPATH=. python eval_baselines.py \
  --config configs/eval_baselines.yaml \
  --baseline majority_vote \
  --offline_data data/deepmath_setting_a_test.jsonl

# 2. Aggregator with peer identities.
PYTHONPATH=. python eval_baselines.py \
  --config configs/eval_baselines.yaml \
  --baseline aggregator_with_ids \
  --offline_data data/deepmath_setting_a_test.jsonl

# 3. Setting B oracle upper bound.
PYTHONPATH=. python eval_baselines.py \
  --config configs/eval_baselines.yaml \
  --baseline oracle_rollout --setting B \
  --offline_data data/deepmath_setting_b_test.jsonl
```

Output schema mirrors `eval_feedback_state.py` (`summary.accuracy`, `records[]`), so all four numbers — proposed method, majority vote, aggregator(+ids), self-judge, oracle, text-memory — drop into one comparison table.

### How baselines map to claims

- **Beats `majority_vote`** → peers carry signal that simple voting misses.
- **Beats `aggregator` and `aggregator_with_ids`** → latent persistent state outperforms one-shot identity-aware aggregation.
- **Beats `self_judge`** → the model isn't just doing implicit self-rating; the state matters.
- **Beats `text_memory`** → latent state beats prompt-level memory at the same information content.
- **Approaches `oracle_rollout` (Setting B)** → rollouts are informative *and* the model successfully exploits them.

## Cooling-off period (delayed feedback)

A cooling-off period is a window between training and the proper test where **belief states are updated from delayed feedback, but model parameters are frozen**. The use case: in production, ground-truth (or rollout) signals often arrive after the model has already shipped. You want to know how much the persistent state updates from that delayed feedback help the next batch of test predictions.

### How the split works

In the training config, set `cooling_off_fraction`. Train uses the first `(1 − fraction)` of `--offline_data`; the remaining `fraction` is written to `<output_dir>/cooling_holdout.jsonl` and **never gets a backward pass**. The split is at the JSONL line (problem) level, so for Setting B no problem's steps straddle the train/cooling boundary.

```yaml
# configs/feedback_state_a.yaml
cooling_off_fraction: 0.2     # 80% train, 20% cooling holdout
```

After training:

```text
outputs/feedback_state_a/
├── feedback_state_adapter.pt
├── cooling_holdout.jsonl        # (100 - M)% of train, held out for cooling
├── _train_split.jsonl
└── train_config.json
```

### Cooling-off semantics

`feedback_state/cooling.py` exports `apply_cooling_off(model, records, tokenizer, ...)`. For each holdout record it runs **`torch.no_grad()` encode → `write_all(...)`** — exactly the same write equation as training (`S ← Diag(λ)·S + Diag(β)·(v − S·k)·kᵀ`), but with no autograd graph and no optimizer step. Setting B uses the per-step rollout scores so writes are sign-weighted by `u = 2z − 1` and confidence `c = |u|`.

**Batching**: sequential, `batch_size = 1`. State is shared across the batch and order-dependent, so parallel batching would lose the online semantics.

**Delayed feedback rate**: `cooling_off_rate ∈ [0, 1]` (default 1.0). At each holdout record, with probability `1 − rate` the record is *seen but skipped* (no write) — this simulates real-world feedback that arrives sparsely. The number of writes actually applied is reported in `cooling_off.applied / skipped / total`.

### Run cooling + test in one shot

If `--cooling_off_data` is not passed, eval auto-detects `<checkpoint>/cooling_holdout.jsonl`. State is reset to zeros, the cooling pass runs, then the regular test loop runs:

```bash
PYTHONPATH=. python eval_feedback_state.py \
  --config configs/eval_feedback_state.yaml \
  --checkpoint outputs/feedback_state_a \
  --offline_data data/deepmath_setting_a_test.jsonl \
  --mode read_only \
  --cooling_off_rate 0.5            # only 50% of holdout carries feedback
```

### Measure the impact of the cooling-off period

`--measure_cooling_impact` runs the test **twice** in `read_only` mode (no test-time writes — so the comparison isolates the cooling effect):

1. **Baseline**: reset state, no cooling, run test → per-record correctness `A`.
2. **Post-cooling**: reset state, apply cooling-off, run test → per-record correctness `B`.

```bash
PYTHONPATH=. python eval_feedback_state.py \
  --config configs/eval_feedback_state.yaml \
  --checkpoint outputs/feedback_state_a \
  --offline_data data/deepmath_setting_a_test.jsonl \
  --cooling_off_rate 1.0 \
  --measure_cooling_impact
```

The resulting `outputs/feedback_state_eval.json` `summary.cooling_impact` contains:

| Field | Meaning |
| --- | --- |
| `baseline_accuracy` | accuracy with no cooling |
| `post_cooling_accuracy` | accuracy after cooling writes |
| `delta_accuracy` | post − baseline |
| `helped` / `helped_pct` | test records that went **wrong → right** |
| `hurt` / `hurt_pct` | test records that went **right → wrong** |
| `unchanged` | the rest |
| `net_helped_pct` | `(helped − hurt) / num_test` |
| `cooling_writes_applied` | how many holdout records carried feedback |
| `cooling_writes_skipped` | skipped because `random() > cooling_off_rate` |

Plus a `cooling_impact_per_record` array with per-test-item tags (`helped`, `hurt`, `unchanged`) so you can correlate impact with problem difficulty, peer makeup, etc.

> **Note on attribution.** This measures the *aggregate* effect of the whole cooling window on test accuracy, not "did record `i` help?" individually. Per-cooling-record attribution would need leave-one-out re-runs at O(N_cooling × N_test) cost.

### ICLR-grade protocol (`--schedule`)

For paper-quality results you want a **learning curve** across feedback checkpoints, with multiple seeds, confidence intervals, and a significance test — not a single before/after number. The runner in [`feedback_state/evaluation.py`](feedback_state/evaluation.py) does this:

```bash
PYTHONPATH=. python eval_feedback_state.py \
  --config configs/eval_schedule.yaml \
  --checkpoint outputs/feedback_state_a \
  --offline_data data/deepmath_setting_a_test.jsonl \
  --schedule "0,50,100,250,500,1000" \
  --num_seeds 3 \
  --shuffle_feedback_per_seed \
  --cooling_off_rate 0.75 \
  --encode_batch_size 16
```

What happens:

1. For each seed in `[0, 1, 2]`:
   - Reset state to zeros.
   - Shuffle the feedback pool with that seed.
   - Evaluate the test set at `t=0` (pre-cooling baseline, `read_only`).
   - Apply 50 delayed-feedback writes (bernoulli-filtered by `cooling_off_rate`) → eval at `t=50`.
   - Apply 50 more (cumulative 100) → eval at `t=100`. ... continue through the schedule.
2. Aggregate across seeds → mean, std, **percentile-bootstrap 95% CI** per checkpoint.
3. Compare each checkpoint to `t=0` per record: **paired sign test** p-value, helped/hurt counts.
4. Fit the learning curve: least-squares **slope of accuracy vs feedback_count** as a single headline number.

Output schema (`outputs/feedback_state_eval.json`):

```json
{
  "summary": {
    "schedule": [0, 50, 100, 250, 500, 1000],
    "num_seeds": 3,
    "aggregate": {
      "feedback_counts": [0, 50, 100, 250, 500, 1000],
      "accuracy_mean":    [0.42, 0.45, 0.48, 0.51, 0.54, 0.56],
      "accuracy_std":     [0.01, 0.02, 0.02, 0.02, 0.03, 0.03],
      "accuracy_ci95_low":  [...],
      "accuracy_ci95_high": [...],
      "delta_vs_t0_mean":   [0.0, 0.03, 0.06, 0.09, 0.12, 0.14],
      "helped_mean":        [0,   18,   32,   54,   71,   83],
      "hurt_mean":          [0,    4,    7,   10,   13,   16],
      "paired_sign_p_vs_t0_mean": [1.0, 0.04, 0.002, 1e-5, 1e-8, 1e-12],
      "learning_curve_slope": 0.000128
    }
  },
  "per_seed":         [{"seed": 0, "checkpoints": [...]}, ...],
  "per_record_impact": [...]
}
```

Plotting `accuracy_mean ± ci95` against `feedback_counts` gives you a Figure 3-style learning curve out of the box.

**Performance:** `apply_cooling_off` now batches the encoder forward (`encode_batch_size`, default 16) — mathematically identical to the sequential implementation (verified by the `test_batched_cooling_matches_sequential` unit test) and ~10–50× faster in wall-clock for large holdouts because the expensive op is the base-model forward, not the state write.

**What this protocol enables a paper to claim:**

- "Accuracy improves monotonically with the number of delayed feedback signals" — read the slope.
- "Improvement is significant at every checkpoint past *X*" — read the p-values.
- "Variance across seeds shrinks as cooling progresses" — read the std.
- "On *Y*% of test problems, cooling helps; on *Z*%, it hurts" — read helped/hurt.

### Suggested sweep

To answer "what cooling rate is worth waiting for," sweep `--cooling_off_rate`:

```bash
for r in 0.1 0.25 0.5 0.75 1.0; do
  PYTHONPATH=. python eval_feedback_state.py \
    --config configs/eval_feedback_state.yaml \
    --checkpoint outputs/feedback_state_a \
    --offline_data data/deepmath_setting_a_test.jsonl \
    --cooling_off_rate $r --measure_cooling_impact \
    --output outputs/cooling_sweep_r${r}.json
done
```

Plot `net_helped_pct` vs `cooling_off_rate` to see the curve.

## Warm-starting eval from the trained state S

`S` (the per-peer belief) is *runtime memory*, not a learned weight, so training now
snapshots it to `<output_dir>/trust_state.pt` (`save_trained_state: true`). Eval can
warm-start from it instead of zeros via `--init_state trained`. The two warm-start
settings:

```bash
# 1) S = trained, KEEP updating online during eval
PYTHONPATH=. python eval_feedback_state.py --config configs/eval_feedback_state.yaml \
  --checkpoint outputs/feedback_state_a --offline_data data/deepmath_setting_a_test.jsonl \
  --init_state trained --mode online_feedback --output outputs/eval_trained_online.json

# 2) S = trained, FROZEN during eval (no online update)
PYTHONPATH=. python eval_feedback_state.py --config configs/eval_feedback_state.yaml \
  --checkpoint outputs/feedback_state_a --offline_data data/deepmath_setting_a_test.jsonl \
  --init_state trained --mode read_only --output outputs/eval_trained_frozen.json
```

`--init_state zeros` (default) keeps the old cold-start behaviour.

## Multi-task mixtures (Math + RAG, Math + Code, …)

The trust state is meant to be **context-sensitive**: Gemma's success on math should
not raise its trust on RAG-QA, and a coder peer's strength on code should not depend
on its (mediocre) math. The pipeline is now task-generic — every task reduces to a
per-peer correctness label `c_j ∈ [0,1]`, dispatched by `record["task_type"]` through
`feedback_state/tasks.py`:

| task_type | correctness | datasets |
| --- | --- | --- |
| `math` | symbolic `math_equal` | deepmath, math500, … |
| `rag`  | QA token-F1 (soft) / EM (binary), alias-aware | hotpotqa, triviaqa |
| `code` | precomputed pass@1 (offline exec) | humaneval, mbpp, bigcodebench, livecodebench |

A single JSONL may freely mix task types; nothing in the encoder/selection model
changes. Eval reports a per-task breakdown in `summary.selection.by_task` —
`accuracy`, `peer_selection_rate` (the trust-shift signal), and `peer_correctness`.

**Always evaluate on a MIXED stream.** The context-sensitivity claim — distinct,
*coexisting* beliefs per task — is only observable when task types interleave in one
feedback stream. Build one with `scripts/mix_task_datasets.py` and watch the curves
with `scripts/plot_trust_shift.py` (in `--mode online_feedback`, stream position =
cumulative feedback count):

```bash
# Interleave per-task JSONLs into one eval stream
PYTHONPATH=. python scripts/mix_task_datasets.py \
  --inputs data/v2_math_test.jsonl data/v2_code_scored.jsonl \
  --output data/v2_mixed.jsonl --mode interleave
# Eval on the MIXED stream (warm-started + online)
PYTHONPATH=. python eval_feedback_state.py --config configs/eval_v2_code.yaml \
  --checkpoint outputs/fs_mathcode --offline_data data/v2_mixed.jsonl \
  --init_state trained --mode online_feedback --output outputs/eval_mixed.json
# Plot per-task peer selection rate vs. feedback count
PYTHONPATH=. python scripts/plot_trust_shift.py --eval outputs/eval_mixed.json \
  --window 50 --peer-names gemma phi qwen-coder \
  --output outputs/trust_shift.png
```

`summary.selection.by_task` then shows, in one run, the coder peer dominating
`peer_selection_rate` on `code` while the math peers keep theirs on `math`; the plot
shows *when* the shift happens as feedback accrues.

**Math + RAG (context-sensitivity).** Gemma is deprived of retrieved documents for
RAG (kept good at math), so the state must hold *different* trust in Gemma per task.

```bash
# Train on math (3 peers)
PYTHONPATH=. python train_feedback_state.py --config configs/feedback_state_a.yaml \
  --offline_data data/deepmath_setting_a_train.jsonl --output_dir outputs/fs_mathrag
# Generate RAG peers (Gemma gets NO context)
PYTHONPATH=. python scripts/generate_setting_a_peers.py --config configs/generate_v2_rag.yaml \
  --output data/v2_rag.jsonl
# MIX math + rag into one stream, then eval on it (warm-started + online).
PYTHONPATH=. python scripts/mix_task_datasets.py \
  --inputs data/v2_math_test.jsonl data/v2_rag.jsonl \
  --output data/v2_mixed_mathrag.jsonl --mode interleave
PYTHONPATH=. python eval_feedback_state.py --config configs/eval_v2_rag.yaml \
  --checkpoint outputs/fs_mathrag --offline_data data/v2_mixed_mathrag.jsonl --output outputs/eval_mathrag.json
# Gemma (peer_0) should be high in by_task["math"].peer_selection_rate but LOW in
# by_task["rag"].peer_selection_rate — coexisting per-task beliefs.
PYTHONPATH=. python scripts/plot_trust_shift.py --eval outputs/eval_mathrag.json \
  --peer-names gemma phi qwen-coder
```

**Math + Code (trust transfer to a specialist).** Add Qwen2.5-Coder as a 4th peer,
train on **math only**, eval on code — trust should shift to the coder on code.

```bash
# Train on math with 4 peers (coder slot present but unremarkable on math)
PYTHONPATH=. python scripts/generate_setting_a_peers.py --config configs/generate_v2_mathcode_train.yaml \
  --output data/v2_math_train.jsonl
PYTHONPATH=. python train_feedback_state.py --config configs/feedback_state_a_mathcode.yaml \
  --offline_data data/v2_math_train.jsonl --output_dir outputs/fs_mathcode
# Generate code peers, then score pass@1 OFFLINE (writes peer_correct into the JSONL)
PYTHONPATH=. python scripts/generate_setting_a_peers.py --config configs/generate_v2_code.yaml \
  --output data/v2_code.jsonl
FEEDBACK_CODE_EXEC_ALLOW=1 PYTHONPATH=. python scripts/score_code_peers.py \
  --input data/v2_code.jsonl --output data/v2_code_scored.jsonl
# MIX math + code into one stream, eval on it, and plot the trust shift.
PYTHONPATH=. python scripts/mix_task_datasets.py \
  --inputs data/v2_math_test.jsonl data/v2_code_scored.jsonl \
  --output data/v2_mixed_mathcode.jsonl --mode interleave
PYTHONPATH=. python eval_feedback_state.py --config configs/eval_v2_code.yaml \
  --checkpoint outputs/fs_mathcode --offline_data data/v2_mixed_mathcode.jsonl --output outputs/eval_mathcode.json
PYTHONPATH=. python scripts/plot_trust_shift.py --eval outputs/eval_mathcode.json \
  --window 50 --peer-names gemma phi qwen-coder
# by_task["code"].peer_selection_rate for the coder peer should climb over the stream.
```

**Code execution safety.** Scoring runs model-generated code in a `python -I`
subprocess with timeout + RLIMIT caps. It is gated behind `FEEDBACK_CODE_EXEC_ALLOW=1`
— run data prep on a disposable/containerised host. BigCodeBench uses the `unittest`
harness; LiveCodeBench uses the `io`/`functional` harness (public test cases).

**Add a new task or mixture.** Register one `TaskSpec` in `feedback_state/tasks.py`
(give it `target_fn`/`correct_fn`/`extract_fn`/`prompt_fn`) and, for new datasets, one
loader in `feedback_state/eval_datasets.py`. No model, training, or eval-loop changes.

## Counter-trust: response-dependent trust (controlled fraction)

Trust must depend on the **actual response**, not just peer identity: a
historically strong peer can be wrong on a given problem, and a historically weak
peer can be right — the state must reject the strong-wrong response and recover the
weak-right one. The **counter-trust** scenario builds exactly these cases, offline,
at a fraction you control. Same tool for train and test data; just change
`--fraction`.

```bash
# Build N%-counter-trust TRAIN data, and SAVE the peer prior (strong->weak ranking)
# computed on the train pool so the test build reuses the SAME identities.
PYTHONPATH=. python scripts/build_counter_trust.py \
  --input data/pool_train.jsonl --output data/ct_train.jsonl \
  --fraction 0.5 --seed 0 --save-prior outputs/peer_prior.json \
  --metrics outputs/ct_train_metrics.json
# Build a STRICT diagnostic TEST set, REUSING the train prior (no strong/weak drift)
PYTHONPATH=. python scripts/build_counter_trust.py \
  --input data/pool_test.jsonl --output data/ct_test.jsonl \
  --fraction 0.3 --strict --seed 1 --prior outputs/peer_prior.json \
  --metrics outputs/ct_test_metrics.json
# Train + eval as usual
PYTHONPATH=. python train_feedback_state.py --config configs/feedback_state_a.yaml \
  --offline_data data/ct_train.jsonl --output_dir outputs/fs_ct
PYTHONPATH=. python eval_feedback_state.py --config configs/eval_feedback_state.yaml \
  --checkpoint outputs/fs_ct --offline_data data/ct_test.jsonl \
  --mode online_feedback --output outputs/eval_ct.json
```

`--save-prior` writes the ranking used; `--prior` loads it. Without this, each
build ranks peers by *its own* pool, so the "strong" peer could differ between
train and test. Inline `--peer-prior peer_0 peer_1 peer_2` overrides both.

**How examples are constructed** (`feedback_state/scenarios.py`):
- Historical reliability is computed per peer across the pool → strong peer = top,
  weak peers = the rest. Use `--save-prior` on the train build and `--prior` on the
  test build to fix this ranking across splits (or pass an explicit `--peer-prior`).
- **Natural** example: keep a problem where the strong peer's *actual* response is
  wrong and ≥1 weak peer's *actual* response is right (`--strict`: exactly one).
- **Counterfactual** (fallback when natural examples are scarce; needs a sample
  pool from `samples_per_peer>1`): under the **same problem**, pair a wrong strong
  sample with a correct weak sample. Never pairs across different problems.
- `target_peers` = the correct peers; a prediction counts correct iff the selected
  peer ∈ `target_peers`.

**Logged metrics.** Construction (`build_counter_trust.py` → metrics JSON):
`num_natural_counter_trust_examples`, `num_counterfactual_same_question_pairings`,
`counter_trust_fraction`. Model behaviour (`eval_feedback_state.py` →
`summary.selection.counter_trust`): `weak_correct_recovery_rate` (selected a
weak-correct peer) and `strong_wrong_rejection_rate` (avoided the strong-wrong
peer). High values on both = the state has learned *response-dependent* trust.

**Add another scenario.** Implement the `DatasetScenario` protocol in
`feedback_state/scenarios.py` and register it; `compose_dataset` /
`compose_fraction` and the `build_counter_trust.py` CLI (`--scenario`) reuse it
unchanged.

## Controlled peer-position swap experiment

Does the memory state `S` store trust by anonymous **slot/source position** or by
explicit **peer identity** — and can identity-aware `S` recover when the trusted
peer moves to a different slot? This uses **deterministic fixed orders** (no random
shuffling): one fixed slot order for the whole training run, one for the whole eval
run, independently configurable.

Stable peer ids (always, regardless of displayed slot): `peer_0`=gemma
(`google/gemma-3-4b-it`), `peer_1`=phi (`microsoft/Phi-4-mini-instruct`),
`peer_2`=qwen-coder (`Qwen2.5-Coder-7B-Instruct`).

| knob | values | meaning |
| --- | --- | --- |
| `train_order` / `test_order` | `orig` \| `swap` | `orig`: slot0=gemma,slot1=phi,slot2=qwen-coder. `swap`: slot0=phi,slot1=gemma,slot2=qwen-coder (swaps the first two). **Independently set.** |
| `identity_mode` | `anon` \| `id` | `anon`: encoded block shows only `question` + `Peer response` (no name). `id`: `Peer [google/gemma-3-4b-it] response: …` — the identity string moves with the peer, not the slot. |
| `state_indexing` | `slot` \| `peer_id` | `slot`: `S` follows the slot position. `peer_id`: `S` follows the stable peer (read/written via `slot_to_peer`). |
| state mode (eval) | `init_state` × `mode` | `S0+RO`/`trS+RO`/`S0+ON`/`trS+ON` = (zeros\|trained S) × (read_only\|online_feedback). |

Run the four train×test combinations (do **not** fix test_order to orig):

```bash
for TR in orig swap; do
  PYTHONPATH=. python train_feedback_state.py --config configs/feedback_state_a.yaml \
    --offline_data data/deepmath_setting_a_train.jsonl \
    --output_dir outputs/fs_${TR}_id_peerid \
    --train_order $TR --identity_mode id --state_indexing peer_id   # via config keys
done

for TR in orig swap; do for TE in orig swap; do for ST in "zeros read_only" "trained read_only" "zeros online_feedback" "trained online_feedback"; do
  set -- $ST
  PYTHONPATH=. python eval_feedback_state.py --config configs/eval_feedback_state.yaml \
    --checkpoint outputs/fs_${TR}_id_peerid --offline_data data/deepmath_setting_a_test.jsonl \
    --train_order $TR --test_order $TE --identity_mode id --state_indexing peer_id \
    --init_state $1 --mode $2 \
    --output outputs/eval_${TR}_${TE}_id_peerid_${1}_${2}.json
done; done; done
```

(`identity_mode`/`state_indexing` at eval must match the checkpoint. Pass values via
the config or as CLI flags.)

**What to expect** (the diagnostic is `test_order=swap`):
- `S0+RO` (S=0): all memory reads are zero → logits tie → argmax defaults to **slot0**. With `test_order=swap`, slot0=phi → picks phi → accuracy drops. (A one-time `[sanity]` line confirms `‖S‖≈0` and the tie.)
- `id` + `peer_id` + `trS+RO`: trust is stored against the peer identity, so the model follows **gemma into slot1** and recovers best-fixed/gemma-level accuracy — independent of train/test order.
- `anon` + `slot` + `trS+RO`: trust is stored against the slot, so behavior depends on the train/test order mapping (e.g. train_order=swap put gemma in slot1; test_order=orig puts phi in slot1 → may select slot1 and drop).

**Logging.** Every prediction logs `example_id, dataset, identity_mode, train_order,
test_order, state_mode, state_indexing, slot_to_peer_id, slot_to_peer_name,
selected_slot, selected_peer_id, selected_peer_name, selected_correct,
correctness_by_slot, correctness_by_peer_id, logits_by_slot, logits_by_peer_id`.

**Metrics** (`summary.selection`): `accuracy`; `slot_picks` p0/p1/p2 (by SLOT) and
`peer_picks_by_name` (by identity: gemma/phi/qwen-coder); `accuracy_by_selected_slot`
/ `accuracy_by_selected_peer`; `position_bias` (slot vs peer entropy). The printed
one-row table is `Dataset | identity_mode | train_order | test_order | state_mode |
state_indexing | accuracy | slot p0/p1/p2 | peer picks`.

**Note (selection architecture).** Because each peer is scored independently by a
shared head, with `state_indexing=peer_id` the model is permutation-equivariant:
`test_order=swap` selects the *same peer* (its slot just changes), so accuracy is
unchanged and `slot_picks` track the peer's new slot. The slot-vs-identity contrast
shows up by comparing `state_indexing=peer_id` against `state_indexing=slot` (and
`id` vs `anon`). This is intentionally a controlled fixed-swap test, not general
permutation invariance — random/all-permutation sweeps can be added later.

## Math counterfactual trust (response-driven trust)

Tests whether the aggregator overrides *historical* trust using the *current
response*: even though Gemma is the historically strongest math peer, the model
should reject Gemma when its current answer is wrong and pick a weaker peer that is
right. **Math only.** This reuses the existing math evaluator, dataloader, trainer,
evaluator, and `head_only` baseline — the only new code is a taxonomy classifier
(`feedback_state/scenarios.classify_math_counterfactual`) and an offline builder.

**1. Build per-type splits + stats** from an existing local peer-response dataset
(no new responses are generated; field names are auto-normalised — `question/problem`,
`answer/gold/reference_answer`, `peer_responses/model_outputs`, `gemma/google/gemma-3-4b-it`):

```bash
python scripts/build_math_counterfactuals.py --config configs/math_counterfactual.yaml \
  --source-dir data/math_peer_responses --output-dir data/math_cf
```

Each example is classified (priority): `all_wrong` > `all_correct` >
`strong_wrong_weak_correct` (the key diagnostic) > `strong_correct_weak_wrong` >
`multi_correct` > `other`, with strict subsets (`strong_wrong_weak_correct_strict`
= exactly one weak peer correct; `strong_correct_weak_wrong_strict` = Gemma the only
correct). Writes `normal.jsonl`, the typed/strict files, `rejected_or_unparsed.jsonl`,
and `counterfactual_dataset_stats.json`. If natural `strong_wrong_weak_correct` is
scarce it **warns** (no fabrication); `--enable-synthetic` (off by default) reuses
`adversarial.make_numeric_perturbation_response` to make easy ones.

**2. Mix train/eval splits** per `counterfactual_type_distribution` (default normal
0.5 / strong_correct_weak_wrong 0.25 / strong_wrong_weak_correct 0.25 — keeping both
diagnostics so the model can't learn "always weak" or "always strong"):

```bash
python scripts/build_math_counterfactuals.py --config configs/math_counterfactual.yaml \
  --output-dir data/math_cf --make-splits        # -> data/math_cf/{train,eval}.jsonl
```

**3. Train + 4. Eval** use the unchanged trainer/evaluator (the emitted records carry
`counterfactual_type`/`strict_counterfactual`/`target_peers`/`strong_peer`, consumed
as-is). The four state modes are `--init_state {zeros,trained}` × `--mode {read_only,online_feedback}`
= `S0+RO / trS+RO / S0+ON / trS+ON`. Peer order (`orig`/`swap`), `identity_mode`
(`anon`/`id`), and `state_indexing` (`slot`/`peer_id`) all apply.

```bash
python train_feedback_state.py --config configs/feedback_state_a.yaml \
  --offline_data data/math_cf/train.jsonl --output_dir outputs/fs_math_cf \
  --train_order orig --identity_mode id --state_indexing peer_id
python eval_feedback_state.py --config configs/eval_feedback_state.yaml \
  --checkpoint outputs/fs_math_cf --offline_data data/math_cf/eval.jsonl \
  --init_state trained --mode read_only --identity_mode id --state_indexing peer_id \
  --output outputs/cf_eval.json
```

**Baselines (already implemented).** The static no-memory aggregators are
`--task head_only` (stateless `direct_head` on the encoded response): `identity_mode=anon`
= **NoMem-Resp**, `identity_mode=id` = **NoMem-ID**. Compare δ-Mem (`task=selection`,
`trS+RO`) against these and against `S0+RO`.

**Optional synthetic augmentation** (off by default; natural examples are never
replaced). When natural `strong_wrong_weak_correct` is too scarce, synthesize
controlled examples into **separate** files. Set `enable_synthetic_math_counterfactuals: true`
(or pass `--enable-synthetic`); generation only fires if `synthetic_only_if_insufficient`
and the natural count is below `min_strong_wrong_weak_correct_{train,eval}`.

```bash
python scripts/build_math_counterfactuals.py --config configs/math_counterfactual.yaml \
  --source-dir data/math_peer_responses --output-dir data/math_cf --enable-synthetic
python scripts/build_math_counterfactuals.py --config configs/math_counterfactual.yaml \
  --output-dir data/math_cf --make-splits     # mixes synthetic at 30% (train) / 50% (eval)
```

- The strong (Gemma) response is corrupted by a configurable mode
  (`wrong_final_answer`, `contradiction_final_answer`, `irrelevant_answer`,
  `malformed_answer`, `wrong_reasoning_plausible`) and **verified WRONG** by the math
  evaluator; the weak-correct response is sourced by priority (natural correct →
  same-question reassign → `"The correct answer is {gold}."` → optional stronger-model
  hook) and **verified CORRECT**. A `strong_correct_weak_wrong` **reverse control**
  (70/30 split) keeps the model from learning "always pick weak". Difficulty:
  easy (irrelevant/malformed) / medium (wrong/contradiction) / hard (wrong-reasoning,
  not default). All same-question only; every synthetic record carries full provenance
  (`synthetic_type`/`synthetic_difficulty`/`synthesis_mode`/`weak_correct_response_source`/…).
- Files: `synthetic_strong_wrong_weak_correct.jsonl`, `synthetic_strong_correct_weak_wrong.jsonl`,
  `combined_counterfactual_{train,eval}.jsonl`, `synthetic_sample_preview.jsonl`.
- Eval reports **natural / synthetic / combined separately**:
  `weak_correct_recovery_rate_{natural,synthetic,combined}`,
  `strong_correct_retention_rate_{…}`, plus `accuracy_by_synthetic_difficulty` and
  `accuracy_by_synthesis_mode`. Synthetic examples are controlled diagnostics — report
  them separately and do not claim full mathematical verification.

**Metrics** (`summary.selection.counterfactual`): `weak_correct_recovery_rate`
(+`_strict`) = P(pick a correct weak peer | Gemma wrong, weak right);
`strong_correct_retention_rate` (+`_strict`) = P(pick Gemma | Gemma correct, weak wrong);
`strong_wrong_rejection_rate`; `selected_correct_response_rate`; and
`accuracy_by_counterfactual_type`. Per record, logs add `counterfactual_type`,
`strict_counterfactual`, `correctness_by_slot`/`_by_peer_id`, `target_slots`/`target_peer_ids`,
`selected_peer_id`/`name`. **Desired result:** δ-Mem improves `weak_correct_recovery_rate`
over `S0+RO`/NoMem while keeping `strong_correct_retention_rate` high (not just over-picking weak).

## Joint-input shared-state selectors (Delta-Mem steering)

Moves beyond per-peer independent scoring: the central model sees **all peer
responses in one joint prompt** and a **shared online state S steers its attention
via the real Delta-Mem mechanism** (`deltamem.core.delta.attach_delta_mem`) — not
prefix tuning. Two variants behind `model_variant`:

| variant | how it selects | loss | eval |
| --- | --- | --- | --- |
| `ar_shared_state_selector` (A, main) | central LM autoregressively generates `Peer j` | LM loss on the canonical target id | score `P("Peer j")` per peer, argmax |
| `joint_bce_shared_state_selector` (B, control) | per-peer head on S-steered joint hidden states | BCEWithLogits over peer correctness | argmax `sigmoid(logit_j)` |

`use_shared_state: false` gives the **no-memory joint** control (LoRA fine-tune, no
S). The shared S is one Delta-Mem online memory over the whole sequence (peer
identity is in the tokens, so its read/write keys are peer- and context-specific);
it does **not** replace the existing per-peer state path (`task=selection`), which
is untouched.

**Weak central agent.** The central agent is config-driven (`central_model`) — pass
`Qwen/Qwen3-0.6B` (weak) or `Qwen/Qwen3-4B`; both work with no code changes.

```bash
# Variant A (AR), shared state, Qwen3-0.6B central agent
PYTHONPATH=. python train_joint_selector.py --config configs/joint_ar_selector.yaml \
  --offline_data data/math_rag/train.jsonl --output_dir outputs/joint_ar
# State modes via init_state x mode (S0/trS x RO/ON):
PYTHONPATH=. python eval_joint_selector.py --config configs/joint_ar_selector.yaml \
  --checkpoint outputs/joint_ar --offline_data data/math_rag/eval.jsonl \
  --init_state trained --mode read_only --output outputs/joint_ar_eval
# Variant B (BCE control)
PYTHONPATH=. python train_joint_selector.py --config configs/joint_bce_selector.yaml ...
# No-memory joint control: add --use_shared_state false (LoRA fine-tunes the agent)
```

**Delta-Mem write protocol (modular + safe).** Selection / candidate scoring is
**always** read-only (`write_enabled=False` inside `score_candidates`); the shared
state `S` is updated only by a **separate write pass**, controlled per split:

| `train_write_policy` / `eval_write_policy` | write pass |
| --- | --- |
| `none` | no online update |
| `selection_tokens` | write the ordinary selection prompt (no correctness labels) |
| `feedback` (default, main) | write a feedback prompt with **explicit per-peer `Feedback: correct/incorrect`** (and optional selected-peer lines) |

`use_feedback_in_write: false` downgrades `feedback`→`selection_tokens`;
`include_selected_peer_in_feedback_prompt` toggles the selected-peer lines;
`debug_write: true` logs the S-norm before/after each write and **asserts the state
is unchanged during candidate scoring**. The AR target `Peer j` is **not** treated as
feedback — it only names the selected peer; correctness for all peers comes from the
feedback prompt (`feedback_state/joint_prompt.build_feedback_prompt`,
`feedback_state/joint_write.run_write_policy`). Supported ablations: `none/none`,
`selection_tokens/*`, `feedback/feedback`, `feedback/none`, and `none/feedback` (the
last prints a train/eval mismatch warning). Recommend `per_device_train_batch_size: 1`
so the online stream is sequential.

**Math + RAG.** Records carry `domain` (`math`/`rag`), `retrieved_context`,
`evidence_quality`, and `counterfactual_type`. The joint collator reuses the
existing canonical peer view, fixed orders, and the **task-aware** correctness
evaluator (math `math_equal`, RAG EM/F1), so math and RAG records flow through the
same pipeline. RAG counterfactuals reuse the existing context-deprivation
(`deprive_context_models` in generation) + the counterfactual classifier; math
counterfactuals reuse the `build_math_counterfactuals.py` pipeline (the
`X% strong_wrong_weak_correct` mix).

**Counterfactual / domain are optional at run time** (no rebuild). Point
`--offline_data` at a combined math+RAG(+counterfactual) JSONL and toggle via config
— independently for train and eval (`feedback_state.data.filter_records`):

**Natural diagnostic counterfactuals are ON by default**; synthetic is a separate switch.

| config key | effect |
| --- | --- |
| `use_counterfactuals_train` / `use_counterfactuals_eval` | synthetic master — `false` → drop **synthetic** counterfactuals for that split, **natural kept** |
| `use_synthetic_counterfactuals_in_train` / `_in_eval` | overrides the master (synthetic on/off) |
| `include_natural_counterfactuals_train` / `_eval` | `false` → explicit opt-out of **natural** diagnostics (default `true`) |
| `domains: [math]` (or `[math, rag]`) | keep only those domains |
| `counterfactual_filter_types: [strong_wrong_weak_correct]` | keep only these counterfactual types |

Defaults include everything (no-op), so existing runs — and natural diagnostics —
are unaffected. The same keys work for the existing `train_feedback_state.py` /
`eval_feedback_state.py` (Setting A).

**Eval outputs** (`eval_joint_selector.py`): `selected_correct_rate`, `accuracy`,
`oracle_ceiling`, `gap_to_ceiling`, `improvement_over_bayesian`; per-domain and
per-counterfactual-type accuracy; `response_override_rate`, `trust_retention_rate`,
`rag_evidence_override_rate`; slot/peer pick distributions; candidate logprobs +
entropy (A) or sigmoid probs (B). Baselines: Bayesian global / domain / dataset and
the oracle ceiling. Writes `predictions.jsonl`, `eval_metrics.json`,
`eval_by_{domain,counterfactual_type,dataset,peer}.csv`, and a 10-example joint-prompt
debug preview. The scientific target: the AR-shared-state selector should beat the
Bayesian priors and the no-memory joint selector and move toward the oracle ceiling.

## Known limitations

- **Setting B per-problem state boundary is not enforced.** `SettingBStepDataset` flattens steps from all problems into one stream; state bleeds across problems. A `ProblemBatchSampler` that resets state at problem boundaries would fix this.
- **Encoder cost.** `feedback_state/model.py` re-uses the 4B backbone with `output_hidden_states=True` to encode each peer and the answer — that's `1 + K + 1` forwards per training example. Swapping to a smaller frozen encoder is a worthwhile follow-up.
- **`generate_setting_b_rollouts.py` loads all generators simultaneously** (planner + K peers). On a single GPU you may need to either shrink the peer set, load one model at a time, or use vLLM.

## Citation

The underlying δ-mem method:

```bibtex
@misc{lei2026deltamemefficientonlinememory,
      title={$\delta$-mem: Efficient Online Memory for Large Language Models},
      author={Jingdi Lei and Di Zhang and Junxian Li and Weida Wang and Kaixuan Fan and Xiang Liu and Qihan Liu and Xiaoteng Ma and Baian Chen and Soujanya Poria},
      year={2026},
      eprint={2605.12357},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.12357},
}
```
