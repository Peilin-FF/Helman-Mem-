# Runbook: the memory on other central-model families

Question: does the Bayesian memory help a central model that is not Qwen3-4B? Five models are on the server:
Meta-Llama-3-8B (base), Meta-Llama-3.1-8B-Instruct, Ministral-8B-Instruct-2410, Qwen2.5-7B-Instruct, phi-4 (14B).
For every model we compare three ways of answering, on two whole test streams:

| condition | what the central model sees |
|---|---|
| peers + memory | the six peer answers in the prompt, and the memory's tilt on the attention (gamma 3) |
| peers | the same prompt, no memory |
| question only | the question alone |

**Rule: each family is tested with its own memory.** The record's address (the features that index the memory) comes
from that family's model, used as a frozen judge, never from the Qwen3-4B judge. So per family the whole pipeline is
rebuilt: features -> record -> prompt files -> evaluation. Everything below runs on the server in the `sigma` conda env
from `/mnt/data/peilin/sigma-mem` with `PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1` (the launchers set these themselves).

Tags (used everywhere): `llama3` Meta-Llama-3-8B (base model, plain prompt layout, weak instruction following) ·
`llama31` Meta-Llama-3.1-8B-Instruct (the meaningful Llama row) · `ministral` Ministral-8B-Instruct-2410 ·
`qwen25` Qwen2.5-7B-Instruct · `phi4` phi-4.

## One command per family

```bash
cd /mnt/data/peilin/sigma-mem
GPUS=0,1,2,3 bash training/scripts/launchers/launch_family_memory.sh all llama31    # features -> prompts -> record quality -> evaluation
python scripts/families_table.py --by_task                                           # the table: outputs/gen/families/table.md
```

`GPUS` is the list of GPUs the run may use (`nvidia-smi` first; the box is shared). `all` runs the four steps below in
order; each step can also be run on its own, and every step skips work that is already on disk, so an interrupted run
is resumed by running the same command again.

| step | command (`bash training/scripts/launchers/launch_family_memory.sh <step> <tag>`) | GPUs | time for an 8B model | writes |
|---|---|---|---|---|
| `features` | the family's model encodes every event of train6, indist6 and ood6 (question features and the six candidate-judge hidden states; plain-text Yes/No judge prompts) | all in `GPUS`, one shard each | ~5 GPU-hours for the three streams (phi-4 ~8); measured 0.5 s/event on the smoke run | `outputs/context_features/<tag>_{big6,indist6,6}_ph/*/shard*.pt` (~6 GB) |
| `prompts` | PCA-256 addresses fit on train6; the record run along indist6 and ood6 read-before-write; its quality against the labels | first GPU of `GPUS` | ~10 min | `outputs/gen/<tag>/prompts_{indist6,ood6}_probe.jsonl`, `record_{indist6,ood6}.json` |
| `eval` | the three conditions on both streams with vLLM (`launch_families.sh full <tag>`) | all in `GPUS`, one evaluation each | ~1 GPU-hour | `outputs/gen/families/<tag>/full_{indist,oodfull}6_{tilt,peers,solo}/eval_metrics.json` |
| `smoke` | the whole pipeline on the first 48 events of train6 and indist6, under the tag `<tag>_smoke` | 4 | ~4 min | same layout under `*_smoke` |

Logs: `logs/enc_<tag>_<stream>_<shard>.out`, `logs/prompts_<tag>_<stream>.out`, `logs/fam_<tag>_full_<stream>_<cond>.out`.
A step prints `FAILED ...` with the log to look at when something breaks; `FAMILY_MEMORY_DONE` when it finished.

## Reading the table

`python scripts/families_table.py` prints one row per model (plus the frozen Qwen3-4B with its own record as the
reference), with the record's quality (AUC of its per-peer estimate against the peers' verified labels, and how often
its favourite is right on events where the peers disagree), the accuracy of the three conditions on both streams, and
`tilt - peers`, which is the memory's contribution over the same prompt. For Qwen3-4B that contribution is +2.2
in-distribution and +4.9 on OOD. `--by_task` adds the per-task breakdown.

## Notes and troubleshooting

- vLLM claims 85% of a GPU when it starts. On a GPU shared with another process: `UTIL=0.35` for the 8B models,
  `UTIL=0.5` for phi-4 (`UTIL` is read by `launch_families.sh`).
- The feature encoder is a plain HF forward pass (no vLLM); it holds the six judge prompts of one event at a time, up to
  8,192 tokens each. One shard per GPU; do not put two shards on one GPU.
- Meta-Llama-3-8B has no chat template: the prompt is laid out as BOS + system text + user text + `Answer:`. Its
  numbers will be low for that reason alone; Meta-Llama-3.1-8B-Instruct is the Llama row that means something.
- Ministral's sliding window (32k) is switched off inside the engine because the memory's attention kernels are plain
  causal; nothing changes numerically since our prompts are under 4k tokens.
- `ADDR=q3_4b bash training/scripts/launchers/launch_families.sh full <tag>` evaluates the same central model on the
  prompt files whose record was built with the Qwen3-4B judge (results get the suffix `_q3addr`). That is a comparison of
  the two records, not the family's result.
- The three streams are `data/mixed_train_big6/train.jsonl` (17,709 events, used only to fit the addresses),
  `data/indist6/test.jsonl` (4,319) and `data/ood6/test.jsonl` (17,403). The peers and their answers are fixed; only the
  central model changes.
- Design and evidence: `docs/memory_judge_design.md` section 23; the mechanism: sections 19-21.
