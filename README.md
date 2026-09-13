<div align="center">
<h2>Kalman Mem: an online reliability record that steers a multi-agent central model</h2>

[Peilin Feng](https://peilin-ff.github.io/),
[Suorong Yang](https://suorongyang.github.io/)<sup>†</sup>,
[Soujanya Poria](https://soujanyaporia.github.io/)<sup>†</sup>

[DeCLaRe Lab](https://declare-lab.github.io/), Nanyang Technological University
</div>

A central model answers a stream of questions after reading the answers of several peer models. After each answer,
every peer's answer is verified, so the system learns over time which peers to trust on which kind of question.
**Kalman Mem** keeps that knowledge in an exact Bayesian record, and it hands the record to the central model
without adding any text to the prompt. It tilts the model's attention toward the peers the record trusts.

## The method

For event *t* and peer *i*, the central model, frozen, reads the question and all the answers. Its hidden states
give an address `x_{t,i}`: a PCA projection of the question features placed in peer *i*'s block, a projection of
the features of peer *i*'s answer, and a constant.

The record is the exact posterior of a linear-Gaussian model of signed correctness `s ∈ {−1, +1}`:

```
Λ = λI + Σ x xᵀ,    b = Σ s x,    λ = 100
read-out:   μ = xᵀ Λ⁻¹ b,   v = xᵀ Λ⁻¹ x,   p = Φ(μ / √(1 + v))
```

It runs along the stream **read before write**. The estimate `p_i` for every peer comes from earlier events only.
Then the model answers, and only then are the event's verified labels written in, one Sherman–Morrison update per
peer. Every stream starts cold.

The estimates enter as an **attention tilt**. Every attention score onto a token of peer *i*'s answer gets
`γ · log(p_i / max_j p_j)` added, with γ = 3, in every layer and head. The favourite peer is untouched, and nothing
is tilted while the record is still flat. This is exact reweighting, `softmax(s + b) = Norm(A ⊙ c)` with
`c_i = (p_i / max p)^γ`, so it applies to any softmax attention. It runs inside vLLM's Triton attention kernels at
engine speed (`feedback_state/vllm_attn_bias.py`).

Frozen Qwen3-4B, thinking off, on whole streams:

| stream | peers + memory | peers | question only | record AUC |
|---|---:|---:|---:|---:|
| in-distribution (4,319 events) | 67.2 | 64.8 | 60.5 | 0.92 |
| OOD (17,403 events) | 74.0 | 69.2 | 67.8 | 0.92 |

## Setup

```bash
git clone https://github.com/Peilin-FF/Helman-Mem-.git sigma-mem && cd sigma-mem
conda create -n sigma python=3.12 && conda activate sigma
pip install -r requirements_qwen3.txt          # torch 2.6.0 (CUDA 12.4), transformers 4.56.2, vLLM 0.8.5 (exact)
bash training/setup_env.sh                     # only for training: hydra, tensordict, flash-attn for the vendored verl
python datasets/download.py                    # the released datasets from Hugging Face -> data/ (streams, misleading answers and streams)
pytest tests/unit                              # CPU only: the rules, the record, the configs and job expansion
```

vLLM must be exactly 0.8.5: the tilt's attention kernels are patched from its source. Set `paths.models_root` in
`configs/base.yaml` to the directory holding the models (one sub-directory each): the central model `Qwen/Qwen3-4B`, and,
to generate new peer answers, the six peers (each registered in `configs/peers/`):

| slot | peer | registered as |
|---|---|---|
| `peer_0` | `google/gemma-3-4b-it` | `gemma3_4b` |
| `peer_1` | `microsoft/Phi-4-mini-instruct` | `phi4_mini` |
| `peer_2` | `Qwen/Qwen2.5-Coder-7B-Instruct` | `qwen25_coder_7b` |
| `peer_3` | `meta-llama/Llama-3.1-8B-Instruct` | `llama31` |
| `peer_4` | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | `deepseek_coder_v2_lite` |
| `peer_5` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `r1_distill_qwen_7b` |

Code answers are graded by executing them (a subprocess with limits, not a sandbox): use a disposable machine.

## Data

| dataset | events | content |
|---|---:|---|
| `train6`: `data/mixed_train_big6/train.jsonl` | 17,709 | GSM8K, SQuAD, APPS |
| `indist6`: `data/indist6/test.jsonl` | 4,319 | GSM8K test, SQuAD dev, APPS test |
| `ood6`: `data/ood6/test.jsonl` | 17,403 | yes/no, multiple-choice and short-answer tasks never seen in training |
| `indist6_misleading`, `ood6_misleading` | | every peer's verified misleading answer to every event |
| `indist6_misleading_p000` … `_p100`, `ood6_misleading_p000` … `_p100` (group `misleading_rates`) | | the streams with 0, 25, 50, 75, 100% of every peer's answers misleading |

Each event carries the question, the gold answer, the six peers' answers and their verified correctness
(`datasets/README.md`), released on Hugging Face as
[Sssunset/kalman-mem-peers](https://huggingface.co/datasets/Sssunset/kalman-mem-peers). The pipeline starts from these
released datasets.

## Running experiments

Every experiment is a YAML file, run by one command:

```bash
bash run.sh configs/experiments/main.yaml --dry-run     # the jobs, and which are already done
bash run.sh configs/experiments/main.yaml --smoke       # 48 events, every step, into outputs/smoke/ (run this first)
bash run.sh configs/experiments/main.yaml               # the whole experiment; resumable, finished work is skipped
bash run.sh configs/experiments/main.yaml --steps evaluate --gpus 4,5,6,7 --set evaluation.max_new_tokens=1024
```

| experiment | question | guide |
|---|---|---|
| `main` | the frozen Qwen3-4B with its record on the in-distribution and OOD streams (the reference rows) | [docs/experiments/main.md](docs/experiments/main.md) |
| `families` | does the record help other central models (Llama-3.1-8B, Ministral-8B, Qwen2.5-7B, phi-4), each with its own record | [docs/experiments/families.md](docs/experiments/families.md) |
| `misleading` | does it survive peers that are misleading on purpose, at any poison ratio | [docs/experiments/misleading.md](docs/experiments/misleading.md) |
| `misleading_families` | the same misleading datasets for Llama-3.1-8B, Ministral-8B, Qwen2.5-7B, phi-4 and Qwen3-14B, each with its own record (a guide for running it elsewhere) | [docs/experiments/misleading_families.md](docs/experiments/misleading_families.md) |
| `train_tilt` | GRPO of the central model with and without the tilt | [docs/experiments/train_tilt.md](docs/experiments/train_tilt.md) |

### The stages

| stage | entry point | does |
|---|---|---|
| peers | `pipeline/peers.py` | a peer model answers every event of a stream, honestly or with verified misleading answers |
| streams | `pipeline/streams.py` | a derived stream: peers added, or answers replaced under a regime |
| features | `pipeline/features.py` | the frozen judge reads question + answers; its hidden states address the record |
| record | `pipeline/record.py` | the Bayesian record along the stream, read before write, and its quality |
| train | `pipeline/train.py` | GRPO of the central model (`training/`) |
| evaluate | `pipeline/evaluate.py` | the central model answers each event under a condition; answers graded |
| table | `pipeline/table.py` | the experiment's result table |

Each entry point is a plain command with explicit paths (`python -m pipeline.<stage> --help`); `pipeline/run.py` derives
the paths from the config and schedules the jobs on the GPUs, one GPU each (a training run takes `gpus_per_run`).

### Registering datasets and models

Datasets, models and peers are registered one YAML file each, and experiments refer to them by file name, so
adding one is adding a file (`python -m pipeline.registry` lists everything and checks every reference):

```
configs/datasets/<name>.yaml   kind stream: {path, peers: [peer names, in peer_0 ... order]}; kind answers: {base, mode};
                               kind misleading: {base, answers, regime};
                               or a group: {group: [names]}. `include: _misleading.yaml` pulls in a template.
configs/models/<name>.yaml     {hf_id, path (under paths.models_root), engine, env_vars, prefix_caching, ...}
configs/peers/<name>.yaml      {model: a registered model, reasoning, ...}: one peer; a new peer is a new file
```

For example a new regime, two peers that always lie, is `configs/datasets/indist6_saboteurs.yaml`:

```yaml
include: _misleading.yaml
base: indist6
regime: {kind: fraction, rate: 1.0, peers: [1, 4]}
```

and `--set "datasets=[indist6_saboteurs]"` runs it (the streams step builds it from the answers already generated).

### Adding an experiment

Copy the closest file in `configs/experiments/` and change what differs. What a file can set:

- `steps`: any of peers, streams, features, record, train, evaluate, table.
- `central`: the models that answer (registered names).
- `datasets`: registered datasets or groups. The steps follow from their kinds: `peers` generates the answers datasets
  the named misleading datasets need, `streams` builds those, and features, record and evaluate run on each.
- `eval_conditions` (conditions are defined once under `conditions:`; a new setting, e.g. another γ, gets a new name,
  because results are stored by condition name).
- `record`: design, dim, lam, order, fit (a dataset name, or `self`).
- `arms`, `overrides`, `tilt_overrides`, `train_data`, `val_data`: training (docs/experiments/train_tilt.md; any key of
  `configs/train/grpo.yaml` can be overridden).
- `table`: rows (`models`, `regimes` or `runs`), `reference` models, `deltas`.

A result is reused only if it was produced with the same settings: an evaluation stores its full condition, and the
runner stops with a clear message instead of silently reusing a result made with other settings.

### Where results go

```
data/<dataset>/                                        registered datasets: released, generated answers (<peer>/), built streams
outputs/features/<model>/<stream>/                     the judge's features
outputs/record/<model>/<stream>/<order>.fit-<fit>.jsonl  the record (+ .quality.json)
outputs/eval/<model or run>/<stream>/<condition>/       generations.jsonl + eval_metrics.json
outputs/train/<run>/                                   training runs; outputs/train/data/ their parquets
outputs/tables/<experiment>.md                         result tables
outputs/runs/<experiment>/                             the resolved config and every command
logs/<experiment>/<job>.log
```

Experiments share this layout, so a result computed once (the main experiment's rows) is read by every other experiment
that needs it.

## Repository layout

```
run.sh               the one command
configs/             base.yaml (paths, defaults, conditions), datasets/, models/, peers/ (the registries), experiments/,
                     train/grpo.yaml
pipeline/            the stages, the runner (run.py), config loading and the output layout
feedback_state/      the method: kalman_memory.py (the record), addresses.py, memory_runtime.py, attn_bias.py +
                     vllm_attn_bias.py (the tilt), judge_prompt.py, memory_generator.py (prompts, grading), tasks.py,
                     adversarial.py (misleading answers and regimes), peer_generation.py
training/            kalman_rl/ (the GRPO code), scripts/train_grpo.sh, verl/ (vendored, patched)
analysis/            studies of the record and the trained models (memsim, attention mass, tilt effect, checks)
data/builders/       code grading and the original stream construction
datasets/            the released six-peer streams and their packaging
docs/                design log (memory_judge_design.md), the method in detail (kalman_mem_notion.md), experiments/
tests/unit/          CPU tests
```

The earlier Σ-Mem method (symmetric memory matrices, joint G, M-Route / M-Vote) is at the git tag `sigma-mem-final`;
the code before this pipeline layout is at `pre-pipeline-restructure`.
