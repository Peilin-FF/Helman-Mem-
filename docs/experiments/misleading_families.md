# misleading_families: how robust are other central models to misleading peers?

This guide runs one experiment: five central models answer the ten misleading datasets (0, 25, 50, 75 and 100% of the
six peers' answers misleading, in-distribution and OOD), each with its own Kalman Mem record. It is the same protocol
that produced the Qwen3-4B result (`docs/reports/misleading_peers.md`). You only evaluate; the misleading answers and
datasets are released on Hugging Face, and nothing is generated.

| registered name | model | Hugging Face |
|---|---|---|
| `llama31` | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` (gated: accept the licence on its page first) |
| `ministral` | Ministral-8B-Instruct-2410 | `mistralai/Ministral-8B-Instruct-2410` (gated) |
| `qwen25` | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| `phi4` | phi-4 (14B) | `microsoft/phi-4` |
| `qwen3_14b` | Qwen3-14B, thinking off | `Qwen/Qwen3-14B` |

For every model and dataset, three conditions are measured:

- `tilt`: the six peer answers in the prompt, with the record's attention tilt (peers + memory).
- `peers`: the same prompt, no tilt (peers only).
- `solo`: the question only. It contains no peer answers, so it is run once per stream and shared by all rates.

## 1. Set up (once)

```bash
git clone https://github.com/Peilin-FF/Helman-Mem-.git kalman-mem && cd kalman-mem
conda create -n sigma python=3.12 && conda activate sigma
pip install -r requirements_qwen3.txt           # torch 2.6.0, transformers 4.56.2, vLLM 0.8.5 (exactly this version)
python datasets/download.py                     # the datasets from Hugging Face into data/ (about 1 GB download, 3 GB on disk)
```

It must end with `ok     15 datasets` (the three streams, the peers' misleading answers on two of them, and the ten
misleading datasets). Every file is checked against its sha256; a `MISMATCH` means the download is corrupt or not the
release: run it again, and tell us if it persists. The data is at
https://huggingface.co/datasets/Sssunset/kalman-mem-peers.

If the five models are not on the machine yet, download them into one directory (each in its own sub-directory, named
as below), for example:

```bash
export MODELS=/path/to/models
huggingface-cli login                           # for the two gated models
for m in meta-llama/Llama-3.1-8B-Instruct:Meta-Llama-3.1-8B-Instruct mistralai/Ministral-8B-Instruct-2410:Ministral-8B-Instruct-2410 \
         Qwen/Qwen2.5-7B-Instruct:Qwen2.5-7B-Instruct microsoft/phi-4:phi-4 Qwen/Qwen3-14B:Qwen3-14B; do
  huggingface-cli download "${m%%:*}" --local-dir "$MODELS/${m##*:}"
done
```

Set `paths.models_root` in `configs/base.yaml` to the directory holding the models, or add
`--set paths.models_root=$MODELS` to every command below. Each model's sub-directory name must match its `path:` in
`configs/models/<name>.yaml` (edit `path:` if your copy is named differently, or give an absolute path there).

## 2. Check before using GPUs (CPU, a minute)

```bash
PYTHONPATH=. python -m pipeline.registry                                   # lists datasets and models; must exit without PROBLEM lines
PYTHONPATH=. pytest -q tests/unit                                           # all pass
bash run.sh configs/experiments/misleading_families.yaml --dry-run | tail  # the jobs that would run
```

Code answers are graded by executing them (a subprocess with limits, not a sandbox): use a disposable machine.

## 3. Smoke test (about 20 minutes per model)

48 events of one dataset, every step, written to `outputs/smoke/` so that it never mixes with real results:

```bash
bash run.sh configs/experiments/misleading_families.yaml --smoke --set central=[qwen25] --gpus 0,1
```

It must end with `RUN_COMPLETE misleading_families (smoke)` and print a small table. Do this once for each model before
its full run (replace `qwen25`); it catches a missing model directory or a licence problem in minutes.

## 4. Full run

One model at a time, on the GPUs you have (every command is resumable: run it again after an interruption and finished
work is skipped; a failed job is retried once):

```bash
bash run.sh configs/experiments/misleading_families.yaml --set central=[llama31]   --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/misleading_families.yaml --set central=[ministral] --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/misleading_families.yaml --set central=[qwen25]    --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/misleading_families.yaml --set central=[phi4]      --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/misleading_families.yaml --set central=[qwen3_14b] --gpus 0,1,2,3,4,5,6,7
```

Two models can run side by side on different GPUs (e.g. `--gpus 0,1,2,3` and `--gpus 4,5,6,7`). For long runs use
`tmux` or `nohup bash run.sh ... > logs/<model>.out 2>&1 &`; per-job logs are in `logs/misleading_families/`.

Each run does, per model: the model's features on the ten datasets (one GPU per shard), its record on each, then 22
evaluations (10 datasets × `tilt`, `peers`, plus `solo` on the two base streams), then the table.

| | 8B models (`llama31`, `ministral`, `qwen25`) | 14B models (`phi4`, `qwen3_14b`) |
|---|---|---|
| time on 8 × A100-80GB | about 4 h | about 6–7 h |
| disk (features) | about 25 GB | about 32 GB |
| GPU memory | 40 GB per feature shard; vLLM takes 85% of a card | 80 GB cards |

Times are estimates from the Qwen3-4B run (about 2.5 h on 8 A100s).

## 5. Results

When all five models are done, build one table with every model (seconds, no GPU):

```bash
bash run.sh configs/experiments/misleading_families.yaml --steps table
```

`outputs/tables/misleading_families.md` has one block of rows per model (`llama31 · p000` ... `qwen3_14b · p100`),
with accuracy in `tilt`, `peers` and `solo` for in-distribution and OOD, plus the record's quality. Please send back
this archive (a few MB):

```bash
tar czf misleading_families_results.tgz outputs/tables/misleading_families.md \
  outputs/eval/*/*_misleading_p*/*/eval_metrics.json outputs/eval/*/indist6/solo/eval_metrics.json outputs/eval/*/ood6/solo/eval_metrics.json \
  outputs/record/*/*_misleading_p*/*.json outputs/runs/misleading_families
```

## Troubleshooting

- **`STALE ... stored results used other settings`**: a result from a run with different settings is in the way. Nothing
  is overwritten; tell us before deleting anything.
- **Out of GPU memory in evaluation** (a shared card): lower vLLM's share, e.g. `--set evaluation.gpu_memory_utilization=0.5`.
  For features, give the job a free card.
- **`this needs vLLM 0.8.5`**: the tilt patches vLLM 0.8.5's attention kernels; other versions are refused on purpose.
- **A model directory is not found**: check `paths.models_root` and that the sub-directory name matches `path:` in
  `configs/models/<name>.yaml`.
- **Ministral**: its 32k sliding window is switched off inside the engine; the prompts are under 4k tokens, so nothing changes.
- **Qwen3-14B** runs with thinking off, like Qwen3-4B.

## Result (run 2026-09-14 by our partner)

Accuracy (%) with memory / without memory (`tilt` / `peers`) at each misleading rate, and question only (`solo`), each
model with its own record. Page: https://claude.ai/code/artifact/ad571df8-9290-44e9-81b2-33f154a7a26e (section 4).

OOD (unaffected by the reading-grading change):

| model | 0% | 25% | 50% | 75% | 100% | solo |
|---|---|---|---|---|---|---:|
| Qwen3-14B | 75.6 / 69.8 | 72.2 / 65.6 | 69.6 / 60.4 | 65.7 / 55.2 | 54.8 / 50.6 | 66.5 |
| Qwen3-4B (ours) | 74.0 / 69.2 | 70.3 / 64.0 | 67.9 / 58.1 | 63.5 / 51.4 | 50.2 / 45.8 | 67.8 |
| Qwen2.5-7B | 71.5 / 65.9 | 67.9 / 59.3 | 65.2 / 52.9 | 59.7 / 47.3 | 48.3 / 43.3 | 59.1 |
| Llama-3.1-8B | 71.3 / 68.3 | 67.9 / 61.2 | 65.1 / 54.5 | 60.2 / 48.2 | 49.4 / 45.5 | 67.4 |
| phi-4 | 71.1 / 65.8 | 68.8 / 62.4 | 66.9 / 57.9 | 62.8 / 52.5 | 52.2 / 48.3 | 63.4 |
| Ministral-8B | 70.9 / 64.0 | 66.1 / 54.8 | 62.2 / 41.6 | 53.7 / 27.4 | 26.1 / 18.0 | 58.6 |

In-distribution, **reading graded by exact match** (this run predates commit b349366, which grades reading by token-F1
≥ 0.5; Qwen3-4B is shown under the same rule):

| model | 0% | 25% | 50% | 75% | 100% | solo |
|---|---|---|---|---|---|---:|
| Qwen3-14B | 68.4 / 67.0 | 67.9 / 66.7 | 67.4 / 66.2 | 66.0 / 64.9 | 65.2 / 64.6 | 63.6 |
| Qwen3-4B (ours) | 67.3 / 64.8 | 66.0 / 64.9 | 65.4 / 64.5 | 64.4 / 63.4 | 63.8 / 62.9 | 60.5 |
| Qwen2.5-7B | 66.7 / 62.8 | 65.5 / 61.9 | 64.4 / 61.0 | 62.8 / 59.6 | 61.4 / 58.9 | 63.6 |
| phi-4 | 66.4 / 62.0 | 64.8 / 62.2 | 63.7 / 61.9 | 62.5 / 61.3 | 62.2 / 60.6 | 59.4 |
| Llama-3.1-8B | 63.3 / 57.8 | 62.1 / 57.1 | 59.5 / 55.4 | 55.2 / 52.3 | 52.7 / 50.4 | 57.7 |
| Ministral-8B | 60.1 / 53.5 | 57.7 / 52.9 | 54.3 / 51.1 | 51.6 / 49.0 | 51.3 / 47.3 | 47.0 |

To put the in-distribution rows under the current rule, update the code (`git pull`) and re-grade the stored
generations, CPU only: `for d in outputs/eval/*/indist6*/*/; do python -m pipeline.evaluate --regrade --output $d
--stream data/$(basename $(dirname $d))/test.jsonl; done`, then rebuild the table with `--steps table`.

Reading: with memory every model is above its no-memory accuracy at every rate; a model stays above its question-only
accuracy on OOD up to 25% misleading (Llama-3.1-8B), 50% (Qwen3-4B, Qwen3-14B, phi-4, Ministral-8B) or 75%
(Qwen2.5-7B). Ministral-8B is the most swayed without memory (64.0 → 18.0 on OOD); Qwen3-14B is the highest with memory
at every rate on both streams. The record built from each model's own features separates honest from misleading answers
alike (AUC 0.68–0.73 in-distribution, 0.77–0.87 OOD).
