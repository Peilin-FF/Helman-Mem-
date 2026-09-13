# Generating misleading peer answers

This guide shows how to make the six peers of the Σ-Mem streams answer with **misleading but relevant solutions**,
and how to set **how much** of the stream is misleading. The answers are confident, on topic and in the usual
format, and each one is verified wrong by the grader.

Two steps, kept apart on purpose:

1. **Generate** a misleading answer for every event from every peer. This is the GPU step, and it runs once.
2. **Build streams** that mix those answers with the honest ones at the ratio you choose. This is a CPU step that
   takes minutes, so any number of ratios reuse the same generated answers.

To run the memory on the resulting streams afterwards, see `README_adversarial.md`.

## Setup

```bash
git clone https://github.com/Peilin-FF/Helman-Mem-.git sigma-mem && cd sigma-mem
conda create -n sigma python=3.12 && conda activate sigma
pip install -r requirements_qwen3.txt          # torch 2.6.0, transformers 4.56.2, vLLM 0.8.5
bash datasets/unpack.sh                        # the honest six-peer streams -> data/indist6, data/ood6
pytest tests/unit/test_adversarial.py          # checks the rules and the ratio logic, no GPU needed
```

In `training/configs/adversarial.yaml`, set `models_root` to the directory that holds the six peer models, one
sub-directory each:

| slot | model |
|---|---|
| `peer_0` | gemma-3-4b-it |
| `peer_1` | Phi-4-mini-instruct |
| `peer_2` | Qwen2.5-Coder-7B-Instruct |
| `peer_3` | Meta-Llama-3.1-8B-Instruct |
| `peer_4` | DeepSeek-Coder-V2-Lite-Instruct |
| `peer_5` | DeepSeek-R1-Distill-Qwen-7B |

Also list the GPUs you can use under `gpus:`.

> **Generated programs are executed.** Code answers are graded by running them against hidden tests in a
> subprocess with a timeout and memory limits. That is not a security sandbox, so use a disposable or
> containerised machine. `run_adversarial.sh` sets `FEEDBACK_CODE_EXEC_ALLOW=1` for this.

## Step 1: generate the answers

Check everything on 48 events first. This takes about 5 minutes, one GPU per peer:

```bash
bash run_adversarial.sh --smoke --steps peers
```

Then generate for both whole test streams, with one shard per GPU:

```bash
nohup bash run_adversarial.sh --steps peers > logs/peers.out 2>&1 &
```

That is about 3 to 5 hours on 8 A100s for the in-distribution stream (4,319 events) and the OOD stream (17,403
events). DeepSeek-R1 is the slowest peer because it thinks first. The step resumes: re-run the same command after
an interruption and finished shards are skipped.

To generate for one peer by hand:

```bash
PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
python scripts/adversarial_peers.py \
    --model /path/to/Meta-Llama-3.1-8B-Instruct \
    --records data/indist6/test.jsonl \
    --output outputs/peer_adv/indist6/Meta-Llama-3.1-8B-Instruct \
    --num_shards 8 --shard_index 0
```

Add `--reasoning` for DeepSeek-R1-Distill-Qwen-7B. Prefix the command with `VLLM_USE_V1=0` for
DeepSeek-Coder-V2-Lite-Instruct. The driver does both on its own.

### How one answer is produced

Asking once is not enough. In a first pilot, 44% of the answers a peer gave when told to be wrong were correct
anyway. So every event goes through these steps until it has a usable answer:

| step | what happens |
|---|---|
| ask | The peer gets its normal task prompt plus an instruction to be plausibly wrong. It is told the correct answer only so it can avoid it. |
| grade | The pipeline's own grader scores the answer: word overlap for reading, exact match for math, hidden tests for code. |
| check | The answer is kept only if it is graded wrong, keeps the task's answer format, never hints that it is wrong on purpose, does not refuse, does not name the correct answer, takes a reading answer from the passage, and, for code, is a real program rather than a table of hard-coded outputs. |
| retry | Failed events are asked again, up to 3 attempts at temperature 0.2, then 0.7, then 1.0, each time told what was wrong. On the last attempt, a math peer that keeps getting it right is given a specific wrong number to arrive at. |
| rewrite | Only if all that fails: for math, multiple-choice and yes/no questions, the final answer is replaced and re-graded. These rows are flagged `forced`. |

An event that never gets a usable answer keeps the peer's honest answer in every stream, and it is counted.

### What you get

For each stream and peer, `outputs/peer_adv/<stream>/<peer>/`:

- **`test.shard<k>of<N>.jsonl`** has one row per event: `id`, `response` (the misleading answer), `correct`
  (0 for every usable row), `accepted` (usable or not), `forced`, `attempts`, and `reasons` (why a rejected
  attempt failed).
- **`summary_test.shard<k>of<N>.json`** is the health check: `accepted_pct`, `forced`, `mean_attempts`,
  `by_task` and `unusable_reasons`.

On the smoke run, 77% to 96% of each peer's events had a usable answer, and none of the usable answers was
graded correct. A peer far below that range is usually refusing, which shows up as `refusal` or `leak` in
`unusable_reasons`.

## Step 2: build streams at the ratio you want

There are two independent ways to set the ratio. Both count only usable answers, so the number you ask for is the
number the stream ends up with.

**A share of every peer's answers**, named `p000` to `p100`. For example, `p030` makes 30% of each peer's answers
misleading. Each peer is hit on a different 30% of the events. A higher share poisons the lower share's events
plus more, so a series of shares forms a clean curve. `p000` is the honest stream rebuilt through the same steps.

**A number of misleading peers on every event**, named `k0` to `k6`. For example, `k2` puts exactly 2 of the 6
peers misleading on every question, with a different pair each time. This separates a lying minority (`k1`,
`k2`) from a lying majority (`k4`, `k5`). What is fixed here is the count per event. Each peer's own share
averages k/6 but varies from peer to peer. An event where fewer than k peers have a usable answer gets as many as
it has.

Shares are rounded to whole events, so `p030` on 48 events reaches 14 events, which is 29.2%.

Any such name works directly, with no config entry:

```bash
bash run_adversarial.sh --steps streams --regimes p030             # one share
bash run_adversarial.sh --steps streams --regimes k2 k4            # two per-event counts
bash run_adversarial.sh --steps streams --regimes rates            # every share in the sweep block
bash run_adversarial.sh --steps streams --regimes counts           # every count in the sweep block
bash run_adversarial.sh --steps streams --regimes sweep            # both
```

`rates` and `counts` read this block of `training/configs/adversarial.yaml`, which you can edit freely:

```yaml
sweep:
  rates: [0.0, 0.25, 0.5, 0.75, 1.0]   # -> p000 p025 p050 p075 p100
  counts: [0, 1, 2, 3, 4, 5, 6]        # -> k0 ... k6
  peers: all                           # or a list of slots, e.g. [1, 4]
  exact: true
```

The config also defines named regimes with other shapes:

| regime | who is misleading |
|---|---|
| `all100` | every answer of every peer |
| `saboteurs2` | peers 1 and 4 on every event, the other four honest |
| `all50` | half of every peer's answers |
| `flip` | peers 1 and 4, honest for the first half of the stream and misleading after |
| `targeted` | every peer, exactly on the events it answered correctly |

Add your own under `regimes:` with `kind` set to `fraction`, `count`, `targeted` or `flip`.

To build one stream by hand:

```bash
PYTHONPATH=. python scripts/build_adversarial_stream.py \
    --base data/indist6/test.jsonl \
    --adv outputs/peer_adv/indist6 \
    --regime p030 \
    --out data/indist6_adv_p030/test.jsonl
```

Add `--drop_forced` to use only answers the peers produced themselves, with no rewritten conclusions.

### What you get

`data/<stream>_adv_<regime>/test.jsonl` is the stream. It has the same events, order and six peers as the honest
stream. The selected answers are replaced, and every correctness label is recomputed from the answer actually in the
stream. Each peer entry in `peer_metadata` gains `misled`, `adversarial_forced` and `regime`.

`manifest.json` in the same directory says what was built:

| field | meaning |
|---|---|
| `poison_ratio` | share of all peer answers in the stream that are misleading |
| `events_by_misleading_peers` | how many events have 0, 1, ... 6 misleading peers |
| `peers.<name>.requested_ratio` / `realised_ratio` | what was asked for and what was reached, per peer |
| `peers.<name>.unavailable` | events selected for a peer that had no usable answer there |
| `peers.<name>.accuracy_honest` / `accuracy_in_stream` | the peer's accuracy before and after |

The builder also prints these figures. Here is `p050` on the 48 smoke events:

```
[adv-stream] 144/288 peer answers replaced (50.0% of the stream, 11 forced, 0 selected without a usable answer)
   peer_3 Meta-Llama-3.1-8B-Instruct   ratio  50.0% asked ->  50.0% reached  accuracy  66.7% ->  31.2%
```

And `k2` on the same events:

```
[adv-stream] events by number of misleading peers: 0: 0, 1: 1, 2: 47, 3: 0, 4: 0, 5: 0, 6: 0
```

**The ceiling.** `p100` and `k6` can only reach what the peers could be made to get wrong, which is 77% to 96% per
peer on the smoke set. The manifest shows the shortfall instead of hiding it.

## Troubleshooting

- **`MLACommonMetadataBuilder object has no attribute page_size`** comes from DeepSeek-Coder-V2-Lite-Instruct on
  vLLM 0.8.5 without FlashAttention 3, as on an A100. Run it with `VLLM_USE_V1=0`, which the config already sets
  through `env:` for that peer.
- **Many `forced` rows for one peer** mean it kept answering correctly. These answers can argue for one value and
  end on another. Report them separately, or build with `--drop_forced`.
- **Some misleading reading answers look partly right.** The reading grader counts an answer as correct only when
  its word overlap with the gold answer is at least 0.5. A correct fragment of a long gold answer can therefore
  count as wrong.
- **vLLM takes 85% of a GPU** when it starts. On a shared machine, lower `generation.gpu_memory_utilization`.

Files: `scripts/adversarial_peers.py` (generation) · `feedback_state/adversarial.py` (the prompts, checks and
ratio logic) · `scripts/build_adversarial_stream.py` (streams) · `training/configs/adversarial.yaml` (settings) ·
`run_adversarial.sh` (driver) · `tests/unit/test_adversarial.py` (checks).
