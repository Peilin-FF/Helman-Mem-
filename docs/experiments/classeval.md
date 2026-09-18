# classeval: a swarm that builds classes, with the record on every method

Question: does the record help a real multi-agent system whose sub-tasks have no fixed types? In the database swarm
(docs/experiments/marble_db_online.md) the sub-steps are five fixed questions ("is X a root cause?"), so a table of peer
accuracy per question is a strong baseline there. Here every sub-step is a different method of a different class, the six
peers work on each of them as agents, the answer the central model commits is what the next sub-steps build on, and the only
reliability signal is whether each peer's method passes its hidden tests.

## The benchmark

ClassEval (Du et al., ICSE 2024; `FudanSELab/ClassEval`, pinned to commit `eaeac44`): 100 Python classes, 410 methods, a
hidden unittest class per method and class-level tests. The classes use the standard library and numpy, pandas, nltk,
gensim, python-docx, openpyxl, bs4, Pillow, PyPDF2 and reportlab.

## The swarm (`classeval.yaml`, the main experiment)

Qwen3-4B builds every class, one method at a time in the benchmark's order (its incremental strategy), the classes in a
seeded order (`shuffled0`) forming one stream of methods. For each method:

1. **The peers work on it as agents.** The six peers of the QA streams (Gemma-3-4B, Phi-4-mini, Qwen2.5-Coder-7B,
   Llama-3.1-8B, DeepSeek-Coder-V2-Lite, R1-Distill-Qwen-7B) each get the class as it stands (the imports, docstring and
   constructor, the methods committed so far with their bodies, the later methods as signature + docstring) and the method to
   write. Each writes the method, its docstring's examples run on the class with that method in place (the **visible
   check**), and on a failure it reads the doctest report or traceback and revises, up to three answers
   (`feedback_state.peer_generation.agentic_answers`, `feedback_state.classeval.visible_check`).
2. **The record is read.** The frozen judge (Qwen3-4B) reads the method's question and the seven answers (the six peers' and
   Qwen3-4B's own question-alone answer); its hidden states address the record (design `qc`, the paper's PCA fit on train6,
   256 components per part, λ = 100), which gives every answer's P(correct) from the methods before this one.
3. **The central model commits a method.** Depending on the condition it writes the method alone, reading the peers, reading
   them with the tilt (γ = 3), or, per method, one of the last two by the reading line (combination). What it writes is
   committed to the class: the next methods see it, and the peers' visible checks run against it.
4. **Feedback, at once.** Every answer is verified by the method's hidden tests with the answer put into the **gold** class
   (`feedback_state.classeval.hidden_test`), so each peer is judged on its own method only, never on what was committed
   before it. The seven labels are written into the record before the next method is read (`lanes: 1`).
5. When a class is finished, its committed version runs all of the class's tests (class pass); each committed method is also
   tested inside the committed class (in context). That is where errors of the swarm propagate.

The conditions are tracks of one run (`pipeline.classeval online`; the engine `feedback_state/online_swarm.py` with ClassEval as
its adapter, `feedback_state/classeval_online.py`, the same engine as the database swarm, docs/experiments/marble_db_online.md),
each with its own committed classes (and, for the tilt and combination, its own record, starting cold):

| track | commits |
|---|---|
| `online_solo` | Qwen3-4B's own method (question alone) |
| `online_peers` | Qwen3-4B's method after reading the six peers' methods (question + peers) |
| `online_tilt` | the same with the record's tilt on its attention (peers + memory) |
| `online_combination` | per method, peers + memory or question alone, by the reading line |

Credit assignment is per sub-step and per answer: a peer's label never depends on the central model's commits, the central
model's commit is judged the same way (methods) and in the class it built (in context, classes).

## Why a table is not enough here, and what measures it

The report (`pipeline.classeval report`) puts the record next to reliability tables, each read before write along the same
stream: per peer; per peer and class (the benchmark's task, which the record is never told); per peer and the method's
libraries; per peer and dependency kind (standalone / uses other methods / uses fields only). With 410 methods of 410
different specifications there is no fixed set of sub-task types, so the tables either pool everything (per peer) or learn
within one class of four methods; the record addresses each method by its content.

## Setup

- Everything of the QA pipeline (`requirements_qwen3.txt`, vLLM 0.8.5), and the ClassEval grading environment in the same env:
  `pip install -r requirements_classeval.txt`. The `questions` step runs `python -m pipeline.classeval setup` (the pinned
  ClassEval_data.json into `datasets/classeval/`, sha256-checked, and nltk's corpora) before building the stream.
- The build runs every gold method through its own hidden tests and drops the methods whose gold fails in this environment
  (listed in `data/classeval_q/manifest.json`: missing libraries, and a few tests tied to the date, the host name or float
  printing); the swarm skips a class with a dropped method. On a laptop without gensim / PyPDF2 / nltk data 394 of 410 methods
  (91 complete classes) survive. It also keeps, per method, the strictest visible check its gold passes: the docstring's
  examples with their outputs (`doctest`, 198 methods), only that no example raises (`run`, 76: ClassEval's examples are often
  illustrative), or that the class loads and defines the method (`load`, 120).
- The train6 judge features of the main experiment (`outputs/features/q3_4b/train6`) fit the record's addresses; the online
  step makes them first if they are missing. `online.fit: warmup` fits them on each record track's first 64 methods instead.
- Programs are executed (`data.builders.common.code_grading.run_python`: a subprocess with limits and a private temp
  directory, not a sandbox): use a disposable machine.

## Run

```bash
bash run.sh configs/experiments/classeval.yaml --smoke --gpus 0,1,2,3,4,5,6    # two classes, every track, under outputs/smoke/
bash run.sh configs/experiments/classeval.yaml --gpus 0,1,2,3,4,5,6            # the swarm: six peer servers + Qwen3-4B (answers and judge)
bash run.sh configs/experiments/classeval.yaml --gpus 0,1,2,3,4,5,6 --set online.lanes=4   # four classes at a time (the record sees an event up to 3 later)
```

The online job serves each peer with vLLM's OpenAI server on its own GPU (ports 8300-8305) and runs Qwen3-4B in-process
(vLLM with the tilt's kernels for its answers, transformers for the judge) on the first GPU. With `lanes: 1` a method waits for
the slowest peer's three turns (R1-Distill thinks up to 4,096 tokens per turn), so the run is long (untimed; many hours for
four tracks); `lanes` trades that for a few methods of delay in the record.

## The teacher-forced ablation (`classeval_offline.yaml`)

The same stream with every method seeing the **gold** methods before it instead of the committed ones, so the peers' answers
are generated once (`pipeline.peers --turns 3`, the agents' visible checks run on the gold class) and every condition is
read from one stream by the pipeline's offline steps (own, features, record, evaluate, combination). The record is still read
before and written after every method; what is not live is the class. Its features can also fit the swarm's addresses
(`--set online.fit=classeval6` with `datasets=[classeval6]`).

## Outputs

```
data/classeval_q/test.jsonl, manifest.json               the methods: question, class so far, the gold method's metadata, visible check
outputs/eval/q3_4b/classeval_q/online_<track>/           generations.jsonl (per method: the committed answer, its label and in-context
                                                         label, the peers' answers, labels, turns and visible checks, the record's
                                                         estimates), units.jsonl (per class: class pass), eval_metrics.json
outputs/eval/q3_4b/classeval_q/vllm_<peer>.log           the peer servers
outputs/tables/classeval.md                              methods per track
outputs/tables/classeval_q3_4b_classeval_q_classeval.md  methods, classes, in context per track; the records against the tables
```

Status (2026-09-18): implemented and tested on CPU (the event construction, the answer surgery, every check and test on the
real benchmark, the agentic loop, the online loop with a real judge forward pass and scripted peers, the report and the job
expansion); not yet run with the models.
