# baselines_families: combination against multi-agent baselines, for six central models

This guide runs one experiment: six central models, each on the ten misleading datasets (0, 25, 50, 75 and 100% of the
six peers' answers misleading, in-distribution and OOD). It is the protocol that produced the Qwen3-4B comparison
(`docs/experiments/baselines.md`). Peers' answers and datasets are released on Hugging Face; the models only answer.

| registered name | model | Hugging Face |
|---|---|---|
| `qwen3_8b` | Qwen3-8B, thinking off | `Qwen/Qwen3-8B` |
| `qwen3_14b` | Qwen3-14B, thinking off | `Qwen/Qwen3-14B` |
| `llama31` | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` (gated) |
| `ministral` | Ministral-8B-Instruct-2410 | `mistralai/Ministral-8B-Instruct-2410` (gated) |
| `phi4` | phi-4 (14B; not phi-4-mini) | `microsoft/phi-4` |
| `qwen25` | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |

For every model and dataset, seven results are recorded:

- **combination**: per question, peers + memory or question alone, chosen from the record and the model's reading line
  before the answer is graded.
- **peers + memory** (`tilt`): the question and the six peers' answers, attention tilted by the model's record.
- **question + peers** (`peers`): the same prompt, no tilt.
- **debate (2 rounds)** (`debate2`): multi-agent debate (Du et al., 2023). The model starts from its question-alone
  answer, reads the six peers' answers and updates its answer, twice. The peers' answers stay as released, so a
  misleading answer stays misleading. Round 1 (`debate1`) is run too, because round 2 continues it, but it is not in
  the table.
- **majority vote (peers + own)** (`vote_all`): the answer most of the six peers and the model's question-alone answer
  give. No model runs; a tie counts as an even split.
- **majority vote (peers)** (`vote_peers`): the answer most of the six peers give.
- **question alone** (`solo`): the question only; shared by all rates of a stream.

One command per model runs whatever is missing, in order: the combination pipeline (question-alone answers, features,
record, peers + memory, question + peers, combination) and then the baselines. Finished work is skipped: if you ran
`combination_families` for a model, its results are reused and only the debate and the votes run. Qwen3-8B was not in
those experiments, and Qwen3-14B's combination was not run, so both run the whole pipeline.

## 1. Update your copy (once)

```bash
cd kalman-mem && git pull
```

If Qwen3-8B or Qwen3-14B is not on the machine yet, download it next to the other models (the sub-directory name must
match `path:` in `configs/models/<name>.yaml`):

```bash
huggingface-cli download Qwen/Qwen3-8B  --local-dir "$MODELS/Qwen3-8B"
huggingface-cli download Qwen/Qwen3-14B --local-dir "$MODELS/Qwen3-14B"
```

On a new machine, follow section 1 of `docs/experiments/misleading_families.md` first (clone, environment,
`python datasets/download.py`, models, `paths.models_root`).

## 2. Check before using GPUs (CPU, a minute)

```bash
PYTHONPATH=. python -m pipeline.registry                                                               # no PROBLEM lines
PYTHONPATH=. pytest -q tests/unit                                                                       # all pass
bash run.sh configs/experiments/baselines_families.yaml --set central=[llama31] --dry-run | grep -c skip  # reused results
bash run.sh configs/experiments/baselines_families.yaml --set central=[llama31] --dry-run | grep dry-run  # what would run
```

For a model whose combination you already ran, the dry run should list only `eval_<model>_<dataset>_debate1_*`,
`debate2_*`, their merges, the `vote_*` jobs and the table; everything else is `skip`. If it wants to run `tilt`, `peers`
or `record` again for such a model, stop and tell us: its stored results are not where the pipeline expects them.
Code answers are graded by executing them (a subprocess with limits, not a sandbox): use a disposable machine.

## 3. Smoke test (about 20 minutes per model)

48 questions of one dataset, every step, written to `outputs/smoke/`:

```bash
bash run.sh configs/experiments/baselines_families.yaml --smoke --set central=[qwen25] --gpus 0,1
```

It must end with `RUN_COMPLETE baselines_families (smoke)` and print a table with the seven columns. Run it once per
model before its full run.

## 4. Full run

One model at a time. Every command is resumable: run it again after an interruption and finished work is skipped; a
failed job is retried once.

```bash
bash run.sh configs/experiments/baselines_families.yaml --set central=[llama31]   --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/baselines_families.yaml --set central=[ministral] --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/baselines_families.yaml --set central=[phi4]      --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/baselines_families.yaml --set central=[qwen25]    --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/baselines_families.yaml --set central=[qwen3_8b]  --gpus 0,1,2,3,4,5,6,7
bash run.sh configs/experiments/baselines_families.yaml --set central=[qwen3_14b] --gpus 0,1,2,3,4,5,6,7
```

Two models can run side by side on different GPUs (`--gpus 0,1,2,3` and `--gpus 4,5,6,7`). For long runs use `tmux`
or `nohup bash run.sh ... > logs/<model>.out 2>&1 &`; per-job logs are in `logs/baselines_families/`.

What each run does, per model:

1. **own**: its question-alone answers join each dataset as a seventh answer (skipped if done).
2. **features**, **record**: the model reads the seven answers of every question; one record per dataset (skipped if done).
3. **evaluate**: peers + memory and question + peers (skipped if done), then debate round 1 on all ten datasets, then
   round 2.
4. **vote**: the two majority votes (CPU, seconds).
5. **combination** (CPU, seconds; skipped if done) and **table**.

| | 7–8B models | 14B models (`phi4`, `qwen3_14b`) |
|---|---|---|
| debate, both rounds, on 8 × A100-80GB | about 2.5 h (Qwen3-4B took 1.5 h) | about 4 h |
| the combination pipeline, when it has not run (`qwen3_8b`, `qwen3_14b`) | about 3 h more | about 5 h more |
| disk for the features, when they have not been computed | about 28 GB | about 37 GB |

The debate times are estimates scaled from Qwen3-4B. Debate prompts are longer than peers + memory's: the peers'
answers are sent in each round, so round 2 prompts are up to twice as long.

## 5. Results

When all six models are done, build one table (seconds, no GPU):

```bash
bash run.sh configs/experiments/baselines_families.yaml --steps table
```

`outputs/tables/baselines_families.md` has one block of rows per model (`qwen3_8b · p000` … `qwen25 · p100`) with the
seven columns for in-distribution and OOD. Please send back this archive (a few MB):

```bash
tar czf baselines_families_results.tgz outputs/tables/baselines_families.md \
  outputs/eval/*/*_misleading_p*+own/{combination,tilt,peers,debate1,debate2,vote_all,vote_peers}/eval_metrics.json \
  outputs/eval/*/indist6/solo/eval_metrics.json outputs/eval/*/ood6/solo/eval_metrics.json \
  outputs/runs/baselines_families
```

## Troubleshooting

- **`STALE ... stored results used other settings`**: a stored result was made with other settings. Nothing is
  overwritten; tell us before deleting anything.
- **`... holds 1 answers per event; debate round 2 needs 2`** or **`event ... has no answer in ...`**: round 2 found an
  incomplete round 1, or round 1 an incomplete question-alone run. Run the same command again (round 1 finishes first);
  if it persists, check that `outputs/eval/<model>/<dataset>+own/debate1/generations.jsonl` has 4,319 lines
  (in-distribution) or 17,403 (OOD).
- **Out of GPU memory**: lower vLLM's share (`--set evaluation.gpu_memory_utilization=0.5`), or give a 14B debate job a
  free card.
- **`this needs vLLM 0.8.5`**: the tilt patches vLLM 0.8.5's attention kernels; other versions are refused on purpose.
- **A model directory is not found**: check `paths.models_root` and the `path:` in `configs/models/<name>.yaml`.

## Our results, for comparison

Qwen3-4B, accuracy % (`docs/experiments/baselines.md`; page https://claude.ai/artifact/LRzXMhXACPWjqmAoqFDHcs).

OOD:

| misleading | combination | peers + memory | question + peers | debate (2 rounds) | majority vote (peers + own) | majority vote (peers) | question alone |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0% | **75.0** | 73.8 | 69.2 | 71.4 | 68.2 | 64.1 | 67.8 |
| 25% | **72.4** | 70.4 | 64.0 | 68.6 | 56.8 | 48.7 | 67.8 |
| 50% | **70.8** | 67.5 | 58.1 | 64.6 | 34.3 | 24.5 | 67.8 |
| 75% | **70.2** | 62.5 | 51.4 | 60.0 | 10.4 | 5.9 | 67.8 |
| 100% | **69.3** | 50.7 | 45.8 | 56.1 | 3.0 | 1.2 | 67.8 |

In-distribution:

| misleading | combination | peers + memory | question + peers | debate (2 rounds) | majority vote (peers + own) | majority vote (peers) | question alone |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0% | 76.2 | **76.7** | 74.6 | 73.5 | 68.0 | 64.3 | 69.2 |
| 25% | 75.4 | **75.5** | 74.2 | 73.2 | 61.8 | 53.0 | 69.2 |
| 50% | 74.1 | **74.3** | 73.7 | 73.0 | 46.9 | 32.5 | 69.2 |
| 75% | 72.7 | **73.1** | 72.4 | 72.3 | 27.3 | 14.6 | 69.2 |
| 100% | 72.0 | **72.4** | 71.3 | 72.1 | 18.8 | 9.4 | 69.2 |
