# The memory on other central-model families

Does the Bayesian reliability memory help a central model that is not Qwen3-4B? This runs the whole pipeline for
Meta-Llama-3.1-8B-Instruct, Ministral-8B-Instruct-2410, Qwen2.5-7B-Instruct and phi-4, **each with its own memory**, and compares three ways of answering on two whole test streams.

```
bash run_families.sh                 # everything in training/configs/families.yaml, 8 GPUs, resumable; ~1 h per 8B model, ~2 h for phi-4
python scripts/families_table.py     # the result table (also written to outputs/gen/families/table.md)
```

Start with `bash run_families.sh --smoke --models llama31` (4 minutes, 48 events, every step) to see the pipeline run
before spending GPU-hours. All parameters live in `training/configs/families.yaml`; `--models`, `--steps`, `--gpus`
override it for one run. Long runs: `nohup bash run_families.sh > logs/run_families.out 2>&1 &` (or tmux).

## Setup on your own machine

```bash
git clone https://github.com/Peilin-FF/Helman-Mem-.git sigma-mem && cd sigma-mem
conda create -n sigma python=3.12 && conda activate sigma
pip install -r requirements_qwen3.txt          # torch 2.6.0 (CUDA 12.4), transformers 4.56.2, vLLM 0.8.5 (exact: the memory's attention
                                               # kernels are patched from vLLM 0.8.5's source); requirements_families_freeze.txt is the exact env
bash datasets/unpack.sh                        # the six-peer streams ship in the repo (datasets/*.jsonl.gz) -> data/
```

Then point `models_root` in `training/configs/families.yaml` at the directory that holds the downloaded models
(`Meta-Llama-3.1-8B-Instruct`, `Ministral-8B-Instruct-2410`, `Qwen2.5-7B-Instruct`, `phi-4`, one sub-directory each;
or write absolute paths per model), run the smoke command above, then `bash run_families.sh`. Eight A100-80GB (or
similar) GPUs; the driver uses every GPU listed under `gpus`. `SIGMA_ENV=<name>` if the conda env is not called `sigma`.

## The pipeline

```
 six peers' answers (fixed)                 the family's model M (frozen)                          outputs
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 train6 / indist6 / ood6 ─►  1. ADDRESS ENCODING: M reads question + answers  ─►  features    outputs/context_features/<tag>_*
        streams                (plain-text Yes/No judge prompt, hidden states)                  (peer_hidden, sem, q_mean, margins)
                                          │
                                          ▼
                             2. THE RECORD: PCA-256 addresses (fit on train6);  ─►  prompts     outputs/gen/<tag>/prompts_<stream>_probe.jsonl
                                Bayesian linear record run along each test                       (memory_prob per peer, read before write)
                                stream, read before write, cold start                            outputs/gen/<tag>/record_<stream>.json (quality)
                                          │
                                          ▼
                             3. EVALUATION: M answers each event three ways ─►  evals        outputs/gen/families/<tag>/full_<stream>6_<cond>/
                                peers + memory (tilt) / peers / question only                  eval_metrics.json, generations.jsonl
                                          │
                                          ▼
                             4. TABLE: accuracy per condition and stream,     ─►  table       outputs/gen/families/table.md
                                tilt − peers, record AUC, vs frozen Qwen3-4B
```

No model is trained anywhere in this pipeline: the family's model stays frozen in both roles, and the evaluation is on
the whole in-distribution stream and the **whole** OOD stream. The train stream is touched only to fit the PCA
addresses (label-free; `streams.fit: self` in the YAML fits them on each test stream instead and skips it).

**What changes per family and what does not.** Qwen3-4B played two roles in the original pipeline: the frozen judge
whose hidden states address the record, and the central model that answers. Both roles are taken by the family's model
here; nothing Qwen is involved in a family's run. Everything else is identical to the Qwen3-4B experiment: the six
peers and their fixed answers, the three streams and their order, the record's equations and prior, the tilt and its
kernels, the grading. A family's rows are therefore the counterpart of the frozen "base central model" rows of the
Qwen3-4B table (in-dist 67.1 / 64.8 / 60.6, OOD 74.0 / 69.2 / 67.8 for peers + memory / peers / question only): no RL
training is involved.

### 1. Address encoding (`scripts/encode_context_features.py`)

For every event of a stream the family's model, frozen, runs two kinds of forward pass, without labels:

- **Question features**: the question text alone; hidden states of three layers (at depth ⅓, ⅔ and the last layer),
  mean-pooled and last-token → `q_mean`, `q_last`.
- **Candidate-judge features**: one plain-text prompt per peer (no chat template), containing the question, the passage
  where there is one, all six answers as `Response 0 … Response 5` with the peer under review marked
  `[candidate under review]`, and the instruction *"Should the candidate response under review be selected as the best
  answer for the task? Answer only Yes or No."* followed by `Answer:`. The last-token hidden states of the same three
  layers give one vector per peer (`peer_hidden`, 3 × hidden size, e.g. 12,288 for Llama), their mean over peers
  (`sem`), and the model's own Yes/No log-odds (`margins`, not used by the record).

The encoder is a plain HF forward pass (no vLLM), one event's six prompts at a time, up to 8,192 tokens each. The
train stream is encoded only to fit the addresses; the two test streams are what the record is run on. One shard per
GPU (`--num-shards N --shard-index k`), ~0.5 s per event for an 8B model, ~6 GB of caches per family.

### 2. The record (`scripts/build_generation_prompts.py`, `feedback_state/kalman_memory.py`)

Addresses: `ψ_q` = PCA-256 of `sem`, `ψ_c` = PCA-256 of `peer_hidden`, both fit on train6. For event *t* and peer *i*
the address is `x_{t,i} = [ψ_q(t) placed in peer i's block of six | ψ_c(t,i) | 1]` (6·256 + 256 + 1 = 1,793 numbers):
the question part is written into the slot of the peer it belongs to, so the record can learn "this peer on this kind
of question". The state is the exact posterior of a linear-Gaussian model of signed correctness `s ∈ {−1, +1}`:

    Λ = λI + Σ x xᵀ,   b = Σ s x,   λ = 100          read-out:  μ = xᵀ Λ⁻¹ b,  v = xᵀ Λ⁻¹ x,  p = Φ(μ / √(1 + v))

The record is run along each test stream **read before write**: the estimate `p_i` for every peer is read from the
state built by the *earlier* events only, the model answers, and only then are the event's verified peer labels written
in (one Sherman–Morrison update per peer). Cold start at every stream. The estimates land in the prompt file as
`memory_prob` per peer (with `memory_evidence`, the effective number of similar past cases), so the evaluation never
recomputes anything. `scripts/record_quality.py` scores the record against the labels (AUC, favourite right on events
where the peers disagree) and writes `record_<stream>.json`; for Qwen3-4B the record reaches AUC 0.92.

### 3. Evaluation (`tests/experiments/common/evaluate_memory_generator.py`)

The family's model answers every event of the stream, greedy, thinking off, 768 new tokens, with vLLM:

| condition | prompt | memory |
|---|---|---|
| `tilt` = peers + memory | system text, the question (options or passage), the six answers as `Peer 1 … Peer 6`, the task instruction | the tilt: every attention score onto a token of peer *i*'s block receives `γ · log(p_i / max_j p_j)`, γ = 3, in every layer and head (zero for the favourite; zero everywhere when the record is flat, spread ≤ 0.1). Nothing about the memory is written into the prompt. |
| `peers` | the same prompt | none |
| `solo` = question only | system text, the question, the instruction | none |

The tilt is exact reweighting of the attention: `softmax(s + b) = Norm(A ⊙ c)` with `c_i = (p_i / max p)^γ`, which is
defined for any softmax attention (MHA or GQA, RoPE, any head size), so it applies to all five models. It runs inside
vLLM's Triton attention kernels (`feedback_state/vllm_attn_bias.py`, a per-KV-slot bias; prefix caching is made aware
of it), at engine speed. Per family the only model-specific work is locating the peer blocks' token positions under
that model's tokenizer and chat template (`feedback_state/attn_bias.py`). Answers are graded by the same verifier as
for Qwen3-4B (exact match after answer extraction; code runs against the hidden tests); `eval_metrics.json` holds the
accuracy, the per-task breakdown, and the accuracy curve along the stream.

### 4. The final table

`python scripts/families_table.py [--by_task]` prints one row per model plus the frozen Qwen3-4B with its own record as
the reference. This is what it looks like once the runs are in (the Qwen3-4B row is real, the others are what the
script fills from `outputs/gen/families/<tag>/full_*/eval_metrics.json` and `outputs/gen/<tag>/record_*.json`):

| central model · record | own record AUC / fav (in, OOD) | in-dist: peers + memory | peers | question only | OOD: peers + memory | peers | question only | tilt − peers (in / OOD) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B (frozen) | 0.92 / 91%, 0.92 / 83% | 67.2 | 64.8 | 60.5 | 74.0 | 69.2 | 67.8 | +2.4 / +4.9 |
| Meta-Llama-3.1-8B-Instruct · own record | … | … | … | … | … | … | … | … |
| Ministral-8B-Instruct-2410 · own record | … | … | … | … | … | … | … | … |
| Qwen2.5-7B-Instruct · own record | … | … | … | … | … | … | … | … |
| phi-4 (14B) · own record | … | … | … | … | … | … | … | … |

Columns: the record's quality on each stream (AUC of its per-peer estimate against the peers' verified labels, and how
often its favourite is right on events where the peers disagree); accuracy in percent for the three conditions on the
in-distribution stream (4,319 events, standard error ≈ 0.7) and the whole OOD stream (17,403 events, ≈ 0.35); and
`tilt − peers`, the memory's contribution over the same prompt, which is the number the experiment is about.
`peers − solo` is the value of reading the six answers at all. `--by_task` adds the per-task breakdown under the table.

## Running it

| | |
|---|---|
| config | `training/configs/families.yaml`: GPUs, model tags and paths, run order, steps, streams, encoder / record / evaluation parameters, output roots, smoke settings |
| start | `bash run_families.sh` (→ `scripts/run_families.py`); `--models llama31 phi4`, `--steps eval table`, `--gpus 4,5,6,7`, `--smoke` |
| steps | `features` (one shard per GPU, all GPUs) → `prompts` (one GPU) → `eval` (one evaluation per GPU: 2 streams × 3 conditions) → `table` |
| resume | every task is skipped when its output exists; re-run the same command after an interruption |
| time, 8 GPUs | per 8B family: features ~40 min (~25 min with `streams.fit: self`), prompts ~10 min, evaluation ~15 min; phi-4 about twice; all five ≈ 6–7 h |
| logs | `logs/enc_<tag>_<stream>_<shard>.out`, `logs/prompts_<tag>_<stream>.out`, `logs/fam_<tag>_full_<stream>_<cond>.out`; the driver prints `start / done / FAILED <task>` with the log to read, and `FAMILIES_RUN_COMPLETE` at the end |
| smoke | `--smoke`: 48 events of train6 and indist6 under the tag `<tag>_smoke`, every step, ~4 min per model; `python scripts/families_table.py --smoke` shows accuracy on the slice plus the sanity checks (prompts tilted, bias kernels taken, generations changed by the tilt) |

Model tags: `llama31` Meta-Llama-3.1-8B-Instruct · `ministral` Ministral-8B-Instruct-2410 · `qwen25` Qwen2.5-7B-Instruct ·
`phi4` phi-4 (14B). The base Meta-Llama-3-8B was tried in the smoke run and dropped: without a chat template it cannot
follow the answer format (12.5% on the question alone), so it says nothing about the memory.

Smoke run of 2026-09-11 (48 events, every family, the whole pipeline; numbers mean nothing at this size, the checks do):

| model, own record | peers + memory | peers | question only | prompts tilted | bias kernels | generations changed by the tilt |
|---|---:|---:|---:|---:|---|---:|
| Meta-Llama-3.1-8B-Instruct | 72.9 | 70.8 | 58.3 | 44/48 | yes | 38/48 |
| Ministral-8B-Instruct-2410 | 66.7 | 64.6 | 50.0 | 44/48 | yes | 18/48 |
| phi-4 | 70.8 | 68.8 | 64.6 | 45/48 | yes | 44/48 |
| Qwen2.5-7B-Instruct | 70.8 | 66.7 | 64.6 | 43/48 | yes | 20/48 |

## Notes and troubleshooting

- vLLM claims `gpu_memory_utilization` (0.85) of a card when it starts, so an evaluation needs a free GPU; on a shared
  GPU set 0.35 in the YAML for an 8B model, 0.5 for phi-4. The encoder needs ~40 GB for an 8B model (six 8k prompts
  with all hidden states) — one shard per GPU, never two.
- Ministral's 32k sliding window is switched off inside the engine (the bias kernels are plain causal); nothing changes
  numerically because our prompts are under 4k tokens.
- Every condition hands vLLM the same token ids (tokenised with `add_special_tokens=False`; the chat template already
  carries BOS where the family uses one), so tilt and no-tilt see identical sequences.
- `ADDR=q3_4b bash training/scripts/launchers/launch_families.sh full <tag>` evaluates a family on the prompt files whose
  record was built with the Qwen3-4B judge (results suffixed `_q3addr`): a comparison of the two records, not the
  family's result.
- The streams: `data/mixed_train_big6/train.jsonl` (17,709), `data/indist6/test.jsonl` (4,319: GSM8K, SQuAD, APPS),
  `data/ood6/test.jsonl` (17,403: yes/no, multiple-choice and short-answer tasks never seen in training). The peers:
  gemma-3-4b-it, Phi-4-mini-instruct, Qwen2.5-Coder-7B-Instruct, Meta-Llama-3.1-8B-Instruct,
  DeepSeek-Coder-V2-Lite-Instruct, DeepSeek-R1-Distill-Qwen-7B (answers generated once and graded; fixed).
- Design notes and evidence: `docs/memory_judge_design.md` (section 23 for the families, 19–21 for the mechanism).

Files: `run_families.sh` (entry point) · `training/configs/families.yaml` (parameters) · `scripts/run_families.py`
(driver) · `scripts/encode_context_features.py` (1) · `scripts/build_generation_prompts.py`, `scripts/record_quality.py`,
`feedback_state/kalman_memory.py`, `feedback_state/addresses.py` (2) · `tests/experiments/common/evaluate_memory_generator.py`,
`feedback_state/attn_bias.py`, `feedback_state/vllm_attn_bias.py` (3) · `scripts/families_table.py` (4).
