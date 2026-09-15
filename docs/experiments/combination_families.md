# combination_families: combination for other central models

This guide runs one experiment: five central models, each on the ten misleading datasets (0, 25, 50, 75 and 100% of
the six peers' answers misleading, in-distribution and OOD). It is the protocol that produced the Qwen3-4B and Qwen3-8B
results (`docs/experiments/combination.md`). Peers' answers and datasets are released on Hugging Face; the models only
answer.

| registered name | model | Hugging Face |
|---|---|---|
| `llama31` | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` (gated) |
| `ministral` | Ministral-8B-Instruct-2410 | `mistralai/Ministral-8B-Instruct-2410` (gated) |
| `qwen25` | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| `phi4` | phi-4 (14B) | `microsoft/phi-4` |
| `qwen3_14b` | Qwen3-14B, thinking off | `Qwen/Qwen3-14B` |

For every model and dataset, four results are produced:

- **peers + memory** (`tilt`): the question and the six peers' answers, attention tilted by the model's record.
- **question + peers** (`peers`): the same prompt, no tilt.
- **question alone** (`solo`): the question only. It contains no peer answers, so it is run once per stream and shared by
  all rates.
- **combination**: per question, the final answer is peers + memory or question alone, chosen from the record and the
  model's reading line before the answer is graded.

How it works: every model answers alone first, and its own answer joins the six peers as a seventh answer. One record,
built from the model's own features (each model is its own frozen judge), estimates all seven answers. Peers + memory
reads the six peers with the tilt from that record. Combination then compares T·ρ + (1 − T)(κ − δ) with κ, where T is
the record's top trust among the peers, κ its estimate of the question-alone answer, and ρ, δ the reading line learned
from earlier questions. The full derivation is on the "Mathematical" page
(https://claude.ai/code/artifact/d375f7dc-624f-465e-9e66-b2cf0b015b40).

## 1. Update your copy (once)

If you ran `misleading_families`, you already have the repository, the datasets and the models: update the code and
re-grade the question-alone results.

```bash
cd kalman-mem && git pull
for m in llama31 ministral qwen25 phi4 qwen3_14b; do
  PYTHONPATH=. python -m pipeline.evaluate --regrade --output outputs/eval/$m/indist6/solo --stream data/indist6/test.jsonl
done
```

The re-grade is needed. Your `misleading_families` run graded reading answers by exact match; the code now uses
token-F1 ≥ 0.5. Each model's question-alone answer becomes the seventh answer of the record, so its correctness has to
follow the same rule as the peers'. The command only re-grades stored generations (CPU, a minute) and prints how many
verdicts changed; OOD has no reading questions and needs nothing.

On a new machine, follow section 1 of `docs/experiments/misleading_families.md` (clone, environment, `python
datasets/download.py`, models, `paths.models_root`). Then nothing needs re-grading: question alone is run by this
experiment.

## 2. Check before using GPUs (CPU, a minute)

```bash
PYTHONPATH=. python -m pipeline.registry                                     # must exit without PROBLEM lines
PYTHONPATH=. pytest -q tests/unit                                             # all pass
bash run.sh configs/experiments/combination_families.yaml --dry-run | tail   # the jobs that would run
```

In the dry run, `eval_<model>_indist6_solo` and `eval_<model>_ood6_solo` should be listed as `skip`: your stored
question-alone results are reused. Code answers are graded by executing them (a subprocess with limits, not a sandbox):
use a disposable machine.

## 3. Smoke test (about 20 minutes per model)

48 questions of one dataset, every step, written to `outputs/smoke/`:

```bash
bash run.sh configs/experiments/combination_families.yaml --smoke --set central=[qwen25] --gpus 0,1
```

It must end with `RUN_COMPLETE combination_families (smoke)` and print a table with a combination column and a
"reading line" table. Run it once per model before its full run.

## 4. Full run

One model at a time. Every command is resumable: run it again after an interruption and finished work is skipped; a
failed job is retried once.

```bash
bash run.sh configs/experiments/combination_families.yaml --set central=[llama31]   --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/combination_families.yaml --set central=[ministral] --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/combination_families.yaml --set central=[qwen25]    --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/combination_families.yaml --set central=[phi4]      --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/combination_families.yaml --set central=[qwen3_14b] --gpus 0,1,2,3,4,5,6,7
```

Two models can run side by side on different GPUs (`--gpus 0,1,2,3` and `--gpus 4,5,6,7`). For long runs use `tmux`
or `nohup bash run.sh ... > logs/<model>.out 2>&1 &`; per-job logs are in `logs/combination_families/`.

What each run does, per model:

1. **own**: its question-alone answers (reused from your stored runs) join each of the ten datasets as a seventh
   answer (`data/<dataset>+<model>/`).
2. **features**: the model reads the seven answers of every question (one GPU per shard).
3. **record**: one record per dataset over the seven answers.
4. **evaluate**: 20 evaluations, peers + memory and question + peers on each dataset.
5. **combination**: CPU, seconds.
6. **table**.

| | 8B models (`llama31`, `ministral`, `qwen25`) | 14B models (`phi4`, `qwen3_14b`) |
|---|---|---|
| time on 8 × A100-80GB | about 3 h | about 5 h |
| disk (features) | about 28 GB | about 37 GB |
| GPU memory | 40 GB per feature shard; vLLM takes 85% of a card | 80 GB cards |

Times are from our Qwen3-8B run (3 h on 8 A100s: 2 h of features, then the records and evaluations). The six-answer
features of `misleading_families` (`outputs/features/<model>/<dataset>/`, without `+own`) are not read by this
experiment.

## 5. Results

When all five models are done, build one table (seconds, no GPU):

```bash
bash run.sh configs/experiments/combination_families.yaml --steps table
```

`outputs/tables/combination_families.md` has one block of rows per model (`llama31 · p000` … `qwen3_14b · p100`). It
shows peers + memory, question + peers, question alone and combination for in-distribution and OOD, then the reading
line per task type (the share of questions that took peers + memory, ρ̂, δ̂). Please send back this archive (a few MB):

```bash
tar czf combination_families_results.tgz outputs/tables/combination_families.md \
  outputs/eval/*/*_misleading_p*+own/*/eval_metrics.json \
  outputs/eval/*/indist6/solo/eval_metrics.json outputs/eval/*/ood6/solo/eval_metrics.json \
  outputs/record/*/*_misleading_p*+own/*.json outputs/runs/combination_families
```

## Troubleshooting

- **`STALE ... stored results used other settings`**: a stored result was made with other settings. Nothing is
  overwritten; tell us before deleting anything.
- **`own_<model>_... FAILED`**: the question-alone run is missing. Check `outputs/eval/<model>/indist6/solo/` and
  `ood6/solo/` (`generations.jsonl` with 4,319 and 17,403 lines). After the own step, every
  `data/<dataset>+<model>/manifest.json` must show `"dropped_for_missing_answers": {}`; anything else means an
  incomplete question-alone run, and the record would cover fewer questions.
- **Out of GPU memory**: in evaluation, lower vLLM's share (`--set evaluation.gpu_memory_utilization=0.5`); for
  features, give the job a free card.
- **`this needs vLLM 0.8.5`**: the tilt patches vLLM 0.8.5's attention kernels; other versions are refused on purpose.
- **A model directory is not found**: check `paths.models_root` and the `path:` in `configs/models/<name>.yaml`.

## Our results, for comparison

Accuracy %, each model with its own features, record and answers (`docs/experiments/combination.md`; page
https://claude.ai/code/artifact/8d7155a5-2ec2-4c1c-84bf-a6e717d9c15e).

OOD:

| model · condition | 0% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B · peers + memory | 73.8 | 70.4 | 67.5 | 62.5 | 50.7 |
| Qwen3-4B · question + peers | 69.2 | 64.0 | 58.1 | 51.4 | 45.8 |
| Qwen3-4B · question alone | 67.8 | 67.8 | 67.8 | 67.8 | 67.8 |
| Qwen3-4B · **combination** | **75.0** | **72.4** | **70.8** | **70.2** | **69.3** |
| Qwen3-8B · peers + memory | 74.2 | 70.1 | 66.9 | 61.0 | 46.2 |
| Qwen3-8B · question + peers | 67.9 | 61.6 | 54.3 | 46.7 | 40.0 |
| Qwen3-8B · question alone | 65.6 | 65.6 | 65.6 | 65.6 | 65.6 |
| Qwen3-8B · **combination** | **74.9** | **71.4** | **70.0** | **68.7** | **67.7** |

In-distribution:

| model · condition | 0% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B · peers + memory | 76.7 | 75.5 | 74.3 | 73.1 | 72.4 |
| Qwen3-4B · question + peers | 74.6 | 74.2 | 73.7 | 72.4 | 71.3 |
| Qwen3-4B · question alone | 69.2 | 69.2 | 69.2 | 69.2 | 69.2 |
| Qwen3-4B · **combination** | **76.2** | **75.4** | **74.1** | **72.7** | **72.0** |
| Qwen3-8B · peers + memory | 78.1 | 77.0 | 76.3 | 74.8 | 74.0 |
| Qwen3-8B · question + peers | 75.7 | 75.1 | 74.3 | 73.6 | 72.3 |
| Qwen3-8B · question alone | 72.9 | 72.9 | 72.9 | 72.9 | 72.9 |
| Qwen3-8B · **combination** | **77.4** | **76.6** | **76.1** | **74.6** | **73.7** |

On OOD, combination is above both peers + memory and question alone at every rate for both models. In-distribution it
is within 0.7 points of peers + memory.
