# The released datasets

The data lives on the Hugging Face Hub: [Sssunset/kalman-mem-peers](https://huggingface.co/datasets/Sssunset/kalman-mem-peers)
(dataset card with fields and statistics there). This folder holds the scripts and `manifest.json`, which pins the
repo revision and every file's sha256.

```bash
python datasets/download.py                         # everything, into data/ (the layout the pipeline reads)
python datasets/download.py --only indist6 ood6     # some datasets
```

| dataset (configs/datasets/) | file on the Hub | goes to |
|---|---|---|
| `train6` | `data/train6.jsonl.gz` | `data/mixed_train_big6/train.jsonl` (17,709 events: GSM8K, SQuAD, APPS; six peers) |
| `indist6` | `data/indist6.jsonl.gz` | `data/indist6/test.jsonl` (4,319 events: GSM8K test, SQuAD dev, APPS test with hidden tests) |
| `ood6` | `data/ood6.jsonl.gz` | `data/ood6/test.jsonl` (17,403 events: PIQA, MMLU, OpenBookQA, SciQ, BBH, SuperGLUE) |
| `indist6_misleading`, `ood6_misleading` | `answers/<dataset>.jsonl.gz` | `data/<dataset>/<peer>/`: every peer's misleading answer to every event |
| `indist6_misleading_p000` … `p100`, `ood6_misleading_p000` … `p100` | `data/<dataset>.jsonl.gz` | `data/<dataset>/test.jsonl` + `manifest.json`: 0–100% of every peer's answers misleading |

The peers, in `peer_0` … `peer_5` order, are registered in `configs/peers/`: gemma-3-4b-it, Phi-4-mini-instruct,
Qwen2.5-Coder-7B-Instruct, Llama-3.1-8B-Instruct, DeepSeek-Coder-V2-Lite-Instruct, DeepSeek-R1-Distill-Qwen-7B.

## Releasing a new version

On the machine with the data (logged in with `huggingface-cli login`), from the project root:

```bash
PYTHONPATH=. python datasets/release_dataset.py --repo Sssunset/kalman-mem-peers
```

It packs every dataset with a `release:` file in `configs/datasets/` (gzip without timestamps, so unchanged data gives
identical files), writes the card and manifest, uploads them as one commit and pins that revision in
`datasets/manifest.json`; commit that file. A misleading dataset can also be rebuilt locally from its stream and answers
(`PYTHONPATH=. python -m pipeline.streams build <names>`); `python -m pipeline.streams digest <file>` compares it with
the `digest` in the manifest.
