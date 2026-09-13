# The released datasets (version 4, 2026-09)

Everything behind the reliability-memory results since 2026-09-07 (`docs/memory_judge_design.md` sections 19–23,
`docs/experiments/`), gzipped so that each archive stays under GitHub's size limit. Each is a registered dataset
(`configs/datasets/<name>.yaml`). `bash datasets/unpack.sh` restores them into `data/`, where the code reads them, and
builds the misleading streams; `manifest.json` has the sha256 of each archive, the row counts, each peer's generation
summary, and the content digest of each misleading stream, which the unpacking checks after building.

| dataset | archive | unpacks to | rows | content |
|---|---|---|---:|---|
| `train6` | `mixed_train_big6.train.jsonl.gz` (26 MB) | `data/mixed_train_big6/train.jsonl` | 17,709 events | GSM8K, SQuAD, APPS; six peers. Fits the record's addresses (label-free); RL training stream of the Qwen3-4B runs |
| `indist6` | `indist6.test.jsonl.gz` (90 MB) | `data/indist6/test.jsonl` | 4,319 events | GSM8K test, SQuAD dev, APPS test (with the hidden tests, 240 MB unpacked); six peers |
| `ood6` | `ood6.test.jsonl.gz` (5 MB) | `data/ood6/test.jsonl` | 17,403 events | PIQA, MMLU, OpenBookQA, SciQ, BBH, SuperGLUE (WiC, RTE, CB, COPA, MultiRC, WSC); six peers |
| `indist6_misleading` | `indist6.misleading.jsonl.gz` (9 MB) | `data/indist6_misleading/<peer>/` | 25,914 answers | every peer's verified misleading answer to every in-distribution event |
| `ood6_misleading` | `ood6.misleading.jsonl.gz` (22 MB) | `data/ood6_misleading/<peer>/` | 104,418 answers | the same on the OOD stream |
| `indist6_misleading_p000` … `p100`, `ood6_misleading_p000` … `p100` | none: built | `data/<dataset>/test.jsonl` | | 0, 25, 50, 75, 100% of every peer's answers misleading |

## Streams

One JSON object per line, one event each, in stream order (the order is part of the data: the record is run along it
read-before-write). Fields: `id`, `source`, `task_type` (`math` / `rag` / `code` in-distribution; `mcqa` / `boolqa` /
`shortqa` OOD), `problem`, `answer` (+ `answer_aliases`, `context`, `choices` / `choice_labels`, or the code tests),
`peer_responses` (`peer_0` … `peer_5` → the answer text), `peer_metadata` (generating model and settings; in a
misleading stream also `regime`, `misled`, `adversarial_forced`), `peer_correct` / `correctness_by_peer` (externally
evaluated binary correctness per peer: post-decision feedback).

| key | peer | registered as |
|---|---|---|
| `peer_0` | google/gemma-3-4b-it | `gemma3_4b` |
| `peer_1` | microsoft/Phi-4-mini-instruct | `phi4_mini` |
| `peer_2` | Qwen/Qwen2.5-Coder-7B-Instruct | `qwen25_coder_7b` |
| `peer_3` | meta-llama/Llama-3.1-8B-Instruct | `llama31` |
| `peer_4` | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | `deepseek_coder_v2_lite` |
| `peer_5` | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (the answer after its think block) | `r1_distill_qwen_7b` |

Honest peer answers were generated once with vLLM (temperature 0.2, top-p 0.95; 768 new tokens on the training and
in-distribution streams, 96 on OOD) and graded with the task's verifier (exact match after answer extraction for math,
F1 ≥ 0.5 for reading, the hidden tests for code, option match for OOD).

## Misleading answers

One JSON object per (event, peer): `peer` (the model directory), `id`, `source`, `task_type`, `response`, `target` /
`correct` (the graded correctness), `accepted` (usable: graded wrong and passing every check), `forced` (the conclusion
was rewritten), `attempts`, `reasons` (why an unusable answer was rejected), `soft`. How they are produced and the usable
share per peer: `docs/experiments/misleading.md`. A misleading stream only ever uses accepted answers; an event without
one keeps the honest answer.

## Re-packing

After new answers or a new release stream (on the server, from the live project so the archives land in `datasets/`):
`PYTHONPATH=. python datasets/release_dataset.py --only <datasets or groups>`. Without `--only` every registered
dataset is re-packed; gzip stamps the time, so unchanged archives would still change.
