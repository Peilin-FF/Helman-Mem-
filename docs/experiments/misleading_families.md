# misleading_families: how robust are other central models to misleading peers?

This guide runs one experiment: five central models answer the ten misleading datasets (0, 25, 50, 75 and 100% of the
six peers' answers misleading, in-distribution and OOD), each with its own Kalman Mem record. It is the same protocol
that produced the Qwen3-4B result (`docs/reports/misleading_peers.md`). You only evaluate; the misleading answers and
datasets are released in `datasets/`, and nothing is generated.

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
bash datasets/unpack.sh                         # the streams, the misleading answers, then builds the ten misleading datasets
```

`unpack.sh` must end with ten `ok` lines, one per built dataset, e.g.

```
ok     data/ood6_misleading_p050/test.jsonl (17,403 events, 50.0% misleading)
```

A `MISMATCH` means the data differs from the release: stop and tell us.

Download the five models into one directory (each in its own sub-directory, named as below), for example:

```bash
export MODELS=/path/to/models
huggingface-cli login                           # for the two gated models
for m in meta-llama/Llama-3.1-8B-Instruct:Meta-Llama-3.1-8B-Instruct mistralai/Ministral-8B-Instruct-2410:Ministral-8B-Instruct-2410 \
         Qwen/Qwen2.5-7B-Instruct:Qwen2.5-7B-Instruct microsoft/phi-4:phi-4 Qwen/Qwen3-14B:Qwen3-14B; do
  huggingface-cli download "${m%%:*}" --local-dir "$MODELS/${m##*:}"
done
```

Then set `paths.models_root` in `configs/base.yaml` to that directory, or add `--set paths.models_root=$MODELS` to every
command below. The sub-directory names are the `path:` fields in `configs/models/*.yaml`.

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
