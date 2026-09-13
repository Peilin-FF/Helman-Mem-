# Robustness: when the six peers are misleading

Does the Bayesian reliability memory survive peers that are wrong *on purpose*? The main experiment reads six peers
that answer as well as they can and are wrong by accident. This one replaces their answers with **misleading but
relevant solutions** -- confident, on topic, in the usual format, verified wrong -- and runs the whole pipeline again
with nothing else changed. The question the table answers is whether a record that only ever sees verified feedback
can find the poisoned answers and take their weight away.

```
bash run_adversarial.sh --smoke                 # 48 events, every step, ~20 min: run this first
bash run_adversarial.sh                         # everything in training/configs/adversarial.yaml, 8 GPUs, resumable
python scripts/adversarial_table.py             # the result table (also written to outputs/gen/adversarial/table.md)
```

The honest rows of the same central model are the comparison column, so a regime's row is comparable event for event
with the frozen Qwen3-4B rows of the main table (in-dist 67.2 / 64.8 / 60.5, OOD 74.0 / 69.2 / 67.8 for
peers + memory / peers / question only).

## Setup on your own machine

```bash
git clone https://github.com/Peilin-FF/Helman-Mem-.git sigma-mem && cd sigma-mem
conda create -n sigma python=3.12 && conda activate sigma
pip install -r requirements_qwen3.txt          # torch 2.6.0 (CUDA 12.4), transformers 4.56.2, vLLM 0.8.5 (exact: the
                                               # memory's attention kernels are patched from vLLM 0.8.5's source)
bash datasets/unpack.sh                        # the honest six-peer streams -> data/
pytest tests/unit/test_adversarial.py          # the acceptance test and the regimes, no GPU needed
```

Then point `models_root` in `training/configs/adversarial.yaml` at the directory holding the **central model**
(`Qwen3-4B`) and the **six peer models** (`gemma-3-4b-it`, `Phi-4-mini-instruct`, `Qwen2.5-Coder-7B-Instruct`,
`Meta-Llama-3.1-8B-Instruct`, `DeepSeek-Coder-V2-Lite-Instruct`, `DeepSeek-R1-Distill-Qwen-7B`); `README.md` lists the
Hugging Face ids. Eight A100-80GB or similar. `SIGMA_ENV=<name>` if the conda env is not called `sigma`.

**The peers' programs are executed to grade them.** `run_adversarial.sh` sets `FEEDBACK_CODE_EXEC_ALLOW=1`, which lets
`data/builders/common/code_grading.py` run model-written Python in an isolated subprocess with a timeout and memory
caps. That is not a security sandbox: run this on a disposable or containerised machine, never on a host with secrets
or a network you care about.

## The pipeline

```
 six peer models                                     the central model M (frozen)                       outputs
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 indist6 / ood6 ─►  1. ADVERSARIAL ANSWERS: each peer answers   ─────────────────────────►  answers   outputs/peer_adv/<stream>/<peer>/
   (honest streams)    every event misleadingly; graded and                                           *.jsonl + summary_*.json
                       re-generated until the answer is usable                                        (accepted, forced, attempts)
                                    │
                                    ▼
                    2. THE STREAMS: one copy of each stream per  ────────────────────────►  streams   data/<stream>_adv_<regime>/
                       regime, the selected peers' answers                                            + manifest.json
                       replaced, every label recomputed
                                    │
                                    ▼
                    3. ADDRESS ENCODING: M reads question + the  ─────────────────────────► features  outputs/context_features/
                       (now misleading) answers                                                       <tag>_<stream>_adv_<regime>_ph/
                                    │
                                    ▼
                    4. THE RECORD: PCA-256 addresses, Bayesian   ─────────────────────────► prompts   outputs/gen/<tag>/
                       linear record along the stream, read                                           prompts_<stream>_adv_<regime>_probe.jsonl
                       before write, cold start                                                       record_<stream>_adv_<regime>.json
                                    │
                                    ▼
                    5. EVALUATION: M answers each event four ways ────────────────────────► evals     outputs/gen/adversarial/<tag>/<regime>/
                       peers + memory (tilt) / peers / question only / swapped record                 <stream>_<cond>/eval_metrics.json
                                    │
                                    ▼
                    6. TABLE: accuracy, what the record makes of  ───────────────────────►  table     outputs/gen/adversarial/table.md
                       the misleading answers, the peers
```

Steps 3 to 6 are the honest pipeline, unchanged and unaware of the experiment: an adversarial stream is registered in
`feedback_state/feature_streams.py` like any other stream, so the encoder, the record and the evaluator run on it
without a single conditional. No model is trained anywhere; the central model is frozen in both of its roles.

### 1. The adversarial answers (`scripts/adversarial_peers.py`)

Each peer is asked, with its own chat template and the sampling of the honest run (temperature 0.2, top-p 0.95), for a
solution that is plausible but wrong. The gold answer is given to the peer **only so that it can avoid it**, and the
task decides what "wrong but relevant" means: a natural-looking slip in one step for math, a different span *taken
from the passage* for reading, an argued case for another option for multiple choice, the opposite verdict argued from
the passage for yes/no, a general program with one subtle bug for code.

Asking once is not enough. In a 200-event pilot, half of the answers a peer produced when told to be wrong were
**correct anyway** (44% graded right), some announced the trick, some refused, and some code answers degenerated into a
table of hard-coded outputs. Every event is therefore a small search:

| | |
|---|---|
| generate | attempt 0 at temperature 0.2, the misleading prompt for the task |
| grade | the pipeline's own rule: token-F1 ≥ 0.5 for reading, exact match for math, the hidden tests executed for code |
| accept | `feedback_state.adversarial.accept`: **verified wrong**, no meta-commentary ("deliberately", "an experiment", "as instructed"), no refusal, the gold answer never named as the correct one, the task's answer format present, a reading answer grounded in the passage, a program that parses, reads its input and is not a lookup table, and, for a thinking peer, a closed think block (an answer that ran out of budget mid-thought is not an answer) |
| retry | only the events that failed, at temperature 0.7 then 1.0, told **what** was wrong with the previous attempt |
| target | a peer that keeps solving the problem correctly is given, on its last attempt, the wrong value its derivation has to arrive at -- one of its own intermediate results, so the argument stays coherent |
| force | only if that too failed: for math / multiple-choice / yes-no, the conclusion is rewritten (`force_wrong` replaces the value the grader reads) and re-graded. Such an answer argues for one value and concludes another, so it is counted separately and `--drop_forced` leaves it out. |

The last attempt is also the lenient one: a fault worth another try but not worth discarding the answer for -- currently
only "the wrong span was not lifted from the passage" -- no longer stands in the way there, because the alternative is
the peer's honest answer, which is not adversarial at all. Those answers are counted in `accepted_with_soft_fault`.

`summary_<split>.json` reports the acceptance rate, how many answers were forced, how many attempts each event took and
why the rest were unusable. An answer that never became usable is **not** put into a stream: the peer keeps its honest
answer there and the count appears as `unavailable` in the stream manifest, so the experiment never silently uses an
answer that is not actually misleading.

Answers are generated for **every** event of every stream, once, and are regime-independent: a regime is then only a
choice of which of them to put into the stream, which costs no GPU time and makes the regimes differ by the regime
alone rather than by sampling noise.

### 2. The regimes (`training/configs/adversarial.yaml`)

Who is misleading, and where. `peers` are canonical indices, `peer_1` = Phi-4-mini-instruct, `peer_4` =
DeepSeek-Coder-V2-Lite-Instruct.

| regime | what it is | what it tests |
|---|---|---|
| `all100` | every answer of every peer is misleading | the floor: the peers are pure poison, so reading them has to hurt, and the memory can at best take their weight away again |
| `saboteurs2` | two peers always lie, four answer honestly | the classic: the record should find the two and stop trusting them |
| `all50` | half of every peer's answers are poisoned, a different half per peer | reliability as a property of the question and the answer, not of the peer -- which is what the record is addressed by |
| `flip` | two peers are honest for the first half of the stream, misleading afterwards | the record has to change its mind about a peer it has learned to trust |
| `targeted` | every peer is misleading exactly on the events it answered correctly | the worst case: no useful answer is left for the memory to promote |

### Controlling the poison ratio, 0% to the maximum

The share of answers that are misleading is a dial, and a sweep over it is the dose-response curve of the experiment:

```yaml
sweep:
  rates: [0.0, 0.25, 0.5, 0.75, 1.0]   # -> the regimes p000, p025, p050, p075, p100
  peers: all
  exact: true
```

```bash
bash run_adversarial.sh --regimes sweep     # the whole curve
bash run_adversarial.sh --regimes p050      # one point on it
```

`exact: true` means the number is what the stream **ends up with**, not what was asked for before the peers' refusals
are subtracted: the events are taken in a fixed per-peer order from those that have a usable adversarial answer, until
the requested share of the stream is poisoned. Three things follow.

- `rates: 0.0` is the honest stream rebuilt through the identical steps -- the control the curve starts from, and a
  check that the machinery changes nothing by itself.
- A higher rate poisons the events of the lower one **plus more**, so the points of a sweep are nested rather than
  independent draws, and the curve is not confounded by which events happen to be hit.
- `1.0` is the ceiling: as much as the peers could be made to get wrong. The manifest reports, per peer,
  `requested_ratio` against `realised_ratio` and how many selected events had no usable answer, so a ceiling below
  100% is visible rather than silent. (With `exact: false` the rate is applied to the selection instead, and whatever
  is unavailable is simply lost from it.)

The rate is per peer, so with `peers: all` it is also the share of all peer answers in the stream that are misleading.
`saboteurs2` is the other way to reach a given share: a few peers poisoned completely rather than every peer poisoned a
little, which is a different experiment even at the same ratio.

`run:` in the YAML lists the regimes to build and evaluate (default `all100`, `saboteurs2`, `all50`, with `sweep`
standing for every rate of the sweep block); `--regimes flip` runs another one. Which (peer, event) pairs a regime selects is deterministic -- a hash of the regime, the peer and the
event id -- so the stream can be rebuilt identically at any time, and `flip` is defined in the order the record walks
the stream (`memory.order`, `shuffled0`), not in file order.

### 3 to 5. Features, record, evaluation

Exactly the honest pipeline. The address of event *t* and peer *i* is `[ψ_q(t) in peer i's block | ψ_c(t,i) | 1]`
(PCA-256 of the judge's hidden states), the state is the exact posterior of a linear-Gaussian model of signed
correctness, `Λ = λI + Σ x xᵀ`, `b = Σ s x`, `λ = 100`, read out as `p = Φ(μ / √(1+v))`, run **read before write** with
a cold start on each stream. `streams.fit: self` fits the PCA on each test stream's own features, so the addresses are
label-free and see the adversarial answers, as they would in deployment.

The central model then answers every event four ways:

| condition | prompt | memory |
|---|---|---|
| `tilt` = peers + memory | the question and the six answers as `Peer 1 … Peer 6` | every attention score onto a token of peer *i*'s block receives `γ · log(p_i / max_j p_j)`, γ = 3, in every layer and head. Nothing about the memory is written into the prompt. |
| `peers` | the same prompt | none |
| `solo` = question only | the question alone | none |
| `swap` | the same prompt | the tilt with the record **permuted by rank** (the highest estimate moved onto the least trusted peer): the control that says a gain comes from what the record knows, not from tilting attention at all |

## Reading the table

`outputs/gen/adversarial/table.md` has three parts.

**Accuracy**, per stream and regime: `tilt`, `peers`, `solo`, `swap`, and the three differences. In the honest runs
`peers − solo` is positive (reading the peers helps) and `tilt − peers` is the memory's contribution (+2.4 in-dist,
+4.9 OOD). Under misleading peers `peers − solo` should go **negative** -- that is the attack working -- and the
number the experiment is about is how much of that loss `tilt − peers` gives back, with `swap` showing what a tilt
that ignores the record's content does instead.

**What the record makes of the misleading answers**: its AUC against the peers' verified labels (does it still rank
right above wrong?), the probability it assigns to an honest answer against a misleading one, the AUC of that
separation, and how often the peer it trusts most on an event is a misleading one. This is where a memory that works
shows up directly, independently of what the central model then does with it.

**The peers**: each peer's accuracy before and after, how many of its answers were replaced, how many were forced, the
share of adversarial attempts that were usable, and the mean answer length before and after -- the check that the
misleading answers still *look* like answers rather than like something a reader would discard on sight.

## Running it

| | |
|---|---|
| config | `training/configs/adversarial.yaml`: GPUs, the central model, the six peers, the regimes, generation / encoder / record / evaluation parameters, output roots, smoke settings |
| start | `bash run_adversarial.sh` (→ `scripts/run_adversarial.py`); `--regimes saboteurs2`, `--steps peers`, `--gpus 4,5,6,7`, `--smoke` |
| steps | `peers` (one shard per GPU, all streams, once) → `streams` (CPU) → `features` (one shard per GPU) → `prompts` (one GPU) → `eval` (one evaluation per GPU) → `table` |
| resume | every task is skipped when its output exists; re-run the same command after an interruption |
| time, 8 GPUs | peers ~3-5 h for both streams and all six models (the reasoning peer is the slowest: 4,096-token budget); then per regime: features ~40 min, record ~10 min per stream, evaluation ~30 min. Three regimes ≈ 8-10 h in total. |
| logs | `logs/advpeer_<stream>_<peer>_<shard>.out`, `logs/advstream_<regime>_<stream>.out`, `logs/advenc_*.out`, `logs/advprompts_*.out`, `logs/adv_<tag>_<regime>_<stream>_<cond>.out`; the driver prints `start / done / FAILED <task>` with the log to read, and `ADVERSARIAL_RUN_COMPLETE` at the end |
| smoke | `--smoke`: 48 events of indist6 under one regime, every step, ~20 min, into separate directories (`outputs/peer_adv_smoke/`, `data/indist6_adv_all100_smoke/`, tag `q3_4b_smoke`) so it can never be mistaken for a real run |

Cheaper variants: drop `swap` from `evaluation.conditions` to save a quarter of the evaluation, and run
`--steps peers streams` first if you only want the adversarial answers and the streams (they are what the rest is
built from, and they are the expensive part).

## Notes and troubleshooting

- **The acceptance rate is the health check.** Read `summary_*.json` after the `peers` step, or the last table. A peer
  whose answers are usable far less often than the others is either refusing (visible in `unusable_reasons`) or its
  chat template is not being applied; a stream whose `forced` count is large means the models kept being right and the
  conclusions were rewritten, which is worth reporting separately in any write-up.
- **`--drop_forced`** on `scripts/build_adversarial_stream.py` builds a stream from naturally misleading answers only,
  leaving the honest answer wherever the conclusion would have had to be rewritten. Slower to reach a given poison
  rate, but every answer is then adversarial end to end.
- **DeepSeek-Coder-V2-Lite-Instruct needs `prefix_caching: false`** (it is set in the YAML): its MLA attention crashes
  vLLM 0.8.5's V1 engine with `MLACommonMetadataBuilder object has no attribute page_size` when prefix caching is on.
  Per-peer `trust_remote_code`, `enforce_eager` and `env: {VAR: value}` are available in the same place.
- vLLM claims `gpu_memory_utilization` (0.85) of a card when it starts, so each task needs a free GPU; on a shared box
  lower it in the YAML.
- The code peers are graded by execution, which is why the `peers` step uses CPU threads (`generation.grade_workers`)
  next to the GPU: grading a shard of APPS answers takes minutes, not seconds.
- A regime is identified by name everywhere: the stream directory (`data/ood6_adv_all50/`), the feature cache
  (`outputs/context_features/q3_4b_ood6_adv_all50_ph/`), the prompt file and the evaluation directory. Renaming a
  regime in the YAML starts a new run rather than overwriting an old one.
- Design notes, the pilot that motivated the verified generator, and the smoke numbers: `docs/memory_judge_design.md`,
  section 25.
- The one-shot misleading generator that the pilot used, `scripts/peer_answers.py --mislead_rate p`, is still there for
  a quick look at a single peer. It accepts whatever comes out; it is not what the streams are built from.

Files: `run_adversarial.sh` (entry point) · `training/configs/adversarial.yaml` (parameters) · `scripts/run_adversarial.py`
(driver) · `feedback_state/adversarial.py` (the prompts, the acceptance test, the regimes) · `scripts/adversarial_peers.py`
(1) · `scripts/build_adversarial_stream.py` (2) · `scripts/encode_context_features.py` (3) ·
`scripts/build_generation_prompts.py`, `scripts/record_quality.py` (4) ·
`tests/experiments/common/evaluate_memory_generator.py` (5) · `scripts/adversarial_table.py` (6) ·
`tests/unit/test_adversarial.py` (the checks).
