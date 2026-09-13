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
bash datasets/unpack.sh                        # the six-peer streams -> data/
pytest tests/unit                              # no GPU needed
```

vLLM must be exactly 0.8.5, because the attention kernels are patched from its source. The drivers activate the
conda environment named in `KALMAN_ENV`, which defaults to `sigma`.

Download the central model (`Qwen/Qwen3-4B`) and the six peers:

| slot | peer |
|---|---|
| `peer_0` | `google/gemma-3-4b-it` |
| `peer_1` | `microsoft/Phi-4-mini-instruct` |
| `peer_2` | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| `peer_3` | `meta-llama/Llama-3.1-8B-Instruct` |
| `peer_4` | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` |
| `peer_5` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |

The peers' answers are already in the streams, graded. Peer models are needed only to generate new answers, for
example the misleading ones.

## Data

| stream | events | tasks |
|---|---:|---|
| `data/mixed_train_big6/train.jsonl` | 17,709 | GSM8K, SQuAD, APPS |
| `data/indist6/test.jsonl` | 4,319 | GSM8K test, SQuAD dev, APPS test |
| `data/ood6/test.jsonl` | 17,403 | yes/no, multiple-choice and short-answer tasks never seen in training |

Each event carries the question, the gold answer, the six peers' answers and their verified correctness. See
`datasets/README.md`.

## Running the pipeline

The pipeline for the frozen Qwen3-4B has three steps. They are shown for the in-distribution stream on one GPU.
For the OOD stream, use `ood6` and `data/ood6/test.jsonl`.

```bash
export PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
M=models/Qwen3-4B

# 1. addresses: the frozen model reads question + answers (also for train6, where the PCA is fit)
python scripts/encode_context_features.py --input data/mixed_train_big6/train.jsonl \
    --output outputs/context_features/q3_4b_big6_ph/train/shard0.pt --central-model $M \
    --include-context --save-peer-hidden --num-peers 6 --max-length 8192 --dtype bfloat16
python scripts/encode_context_features.py --input data/indist6/test.jsonl \
    --output outputs/context_features/q3_4b_indist6_ph/ood/shard0.pt --central-model $M \
    --include-context --save-peer-hidden --num-peers 6 --max-length 8192 --dtype bfloat16

# 2. the record along the stream, read before write, and its quality against the labels
python scripts/build_generation_prompts.py --model q3_4b --fit-stream train6 --stream indist6 \
    --order shuffled0 --design qc --dim 256 --lam 100 --out outputs/gen/q3_4b/prompts_indist6_probe.jsonl
python scripts/record_quality.py --prompts outputs/gen/q3_4b/prompts_indist6_probe.jsonl \
    --out outputs/gen/q3_4b/record_indist6.json

# 3. the central model answers: peers + memory (tilt), peers alone, question alone
for cond in "tilt:--mode peers --attn_gamma 3" "peers:--mode peers" "solo:--mode solo"; do
  python -m tests.experiments.common.evaluate_memory_generator --central_model $M --engine vllm \
      --thinking off --max_new_tokens 768 --prompts outputs/gen/q3_4b/prompts_indist6_probe.jsonl \
      --records data/indist6/test.jsonl ${cond#*:} --output outputs/gen/q3_4b/indist6_${cond%%:*}
done
```

Encoding shards over GPUs with `--num-shards N --shard-index k`. Each evaluation writes `eval_metrics.json` and
`generations.jsonl`.

Three experiments have their own driver and guide, and each one is resumable across 8 GPUs:

| guide | what it runs |
|---|---|
| [`README_families.md`](README_families.md) | The same pipeline with other central models (Llama-3.1-8B, Ministral-8B, Qwen2.5-7B, phi-4), each with its own record: `bash run_families.sh` |
| [`README_misleading_peers.md`](README_misleading_peers.md) | Generating misleading but relevant peer answers, and building streams at a chosen ratio (a share of each peer's answers, or a number of misleading peers per question) |
| [`README_adversarial.md`](README_adversarial.md) | The whole pipeline on those streams, against the honest rows: `bash run_adversarial.sh` |
| [`training/README.md`](training/README.md) | Multi-GPU GRPO / SFT of the central model with the record and the tilt (vendored verl, `training/kalman_rl`) |

## Repository layout

```
feedback_state/      the method: kalman_memory.py (the record), addresses.py, memory_runtime.py,
                     attn_bias.py + vllm_attn_bias.py (the tilt), memory_generator.py (prompts, grading),
                     tasks.py (task registry and graders), feature_streams.py, adversarial.py
scripts/             encoding, record, tables, drivers (run_families.py, run_adversarial.py), analyses
tests/               experiments/common/evaluate_memory_generator.py (the evaluator), unit/ (pytest)
training/            kalman_rl/ (our trainer code), verl/ (vendored, patched), configs/, scripts/
data/builders/       stream construction and code grading (common/, mixed_train/)
datasets/            the six-peer streams, compressed
docs/                memory_judge_design.md (the design log), kalman_mem_notion.md (the method in detail)
```

The earlier Σ-Mem method is no longer in this repository. It used symmetric memory matrices with joint G,
M-Route and M-Vote, and counterfactual, peer-generalization and feedback-availability experiments. Its code is at
the git tag `sigma-mem-final`.
