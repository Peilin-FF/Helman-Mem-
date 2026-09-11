# The six-peer streams (version 3, 2026-09)

The three event streams behind every reliability-memory result since 2026-09-07 (`docs/memory_judge_design.md`
sections 19–23, `README_families.md`), gzipped so that each file stays under GitHub's size limit. `bash datasets/unpack.sh`
restores them into `data/`, where the code reads them; `manifest.json` has the sha256 of each archive, the event counts
and the source benchmarks.

| archive | unpacks to | events | content |
|---|---|---:|---|
| `mixed_train_big6.train.jsonl.gz` (26 MB) | `data/mixed_train_big6/train.jsonl` | 17,709 | GSM8K, SQuAD, APPS; six peers. Fits the record's addresses (label-free); RL training stream of the Qwen3-4B runs |
| `indist6.test.jsonl.gz` (90 MB) | `data/indist6/test.jsonl` | 4,319 | GSM8K test, SQuAD dev, APPS test (with the hidden tests, 240 MB unpacked); six peers |
| `ood6.test.jsonl.gz` (5 MB) | `data/ood6/test.jsonl` | 17,403 | PIQA, MMLU, OpenBookQA, SciQ, BBH, SuperGLUE (WiC, RTE, CB, COPA, MultiRC, WSC); six peers |

One JSON object per line, one event each, in stream order (the order is part of the data: the record is run along it
read-before-write). Fields: `id`, `source`, `task_type` (`math` / `rag` / `code` in-distribution; `mcqa` / `boolqa` /
`shortqa` OOD), `problem`, `answer` (+ `answer_aliases`, `context`, `choices` / `choice_labels`, or the code tests),
`peer_responses` (`peer_0` … `peer_5` → the answer text), `peer_metadata` (generating model and settings),
`peer_correct` / `correctness_by_peer` (externally evaluated binary correctness per peer: post-decision feedback).

| key | peer |
|---|---|
| `peer_0` | google/gemma-3-4b-it |
| `peer_1` | microsoft/Phi-4-mini-instruct |
| `peer_2` | Qwen/Qwen2.5-Coder-7B-Instruct |
| `peer_3` | meta-llama/Llama-3.1-8B-Instruct |
| `peer_4` | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct |
| `peer_5` | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (the answer after its think block) |

Peer answers were generated once with vLLM (temperature 0.2, top-p 0.95; 768 new tokens on the training and
in-distribution streams, 96 on OOD) and graded with the task's verifier (exact match after answer extraction for math,
F1 ≥ 0.5 for reading, the hidden tests for code, option match for OOD). Re-packing after a change to `data/`:
`PYTHONPATH=. python scripts/release_dataset.py`.
