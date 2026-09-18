# swarm: a real agent swarm (MultiAgentBench's database diagnosis)

Question: does the record, the tilt and combination help a real multi-agent system, with agents that query a live
database and coordinate through a planner, not peers answering fixed questions?

## The benchmark

MultiAgentBench (MARBLE; Zhu et al., ACL 2025) has a database environment inspired by D-Bot: a PostgreSQL database shows a
performance anomaly, and five agents, each assigned by a planner to one possible root cause (INSERT_LARGE_DATA,
LOCK_CONTENTION, VACUUM, REDUNDANT_INDEX, FETCH_LARGE_DATA), query its system views (`pg_stat_statements`, `pg_locks`, ...)
and can talk to each other; the planner decides the most likely root causes. The ground truth is the injected anomaly.
The benchmark's rule: a prediction is correct if one of its allowed guesses (2 or 3 per task) is a true root cause. We
report that (`accuracy`) and whether the predicted set is exactly right (`exact_accuracy`).

The tasks are the benchmark's own 100: 10 scenarios (e-commerce, education, finance, ...) × 10 anomaly sets (the five
single anomalies and five pairs), extracted to `datasets/marble_db/tasks.jsonl` from the vendored code
(`third_party/marble`, VENDOR.md there). Everything the agents see, the planner's prompts, the agents' tool loop and the
anomaly workloads are the benchmark's; only the model is ours (Qwen3-4B for planner and agents, thinking off).

## What we add

Per task, after the benchmark's loop:

- **findings**: each agent reports its finding on the cause it investigated, beginning `Root cause investigated: X. Verdict:
  YES|NO`, graded against the ground truth. The five findings are the event's candidate answers.
- **question alone**: the central model investigates alone with the same tool budget (five queries) and diagnoses.
- Then the usual pipeline on that stream: the model's own answer joins the findings, the judge's features, the record,
  **question + peers** (the model reads the five findings), **peers + memory** (with the tilt), and **combination**.

Conditions in the table: `solo` (question alone), `swarm` (the benchmark as-is: the planner's final decision), `verdicts`
(the causes whose agents said YES), `peers`, `tilt`, `combination`.

## Setup

- PostgreSQL in user space, no docker: `conda create -n pg -c conda-forge postgresql=18.6`. The swarm job creates and starts a
  cluster under `swarm.pg_data/<port>` when nothing listens on its port (`KALMAN_PG_BIN` if the env is elsewhere).
- Into the `sigma` env: `pip install psycopg2-binary "litellm>=1.52" "httpx[socks]" beartype colorlog`.
- The job starts vLLM's OpenAI server for the model on its GPU (`--enable-auto-tool-choice --tool-call-parser hermes`).

## Run

```bash
bash run.sh configs/experiments/swarm.yaml --smoke --gpus 0                    # four tasks end to end, under outputs/smoke/ (about 15 minutes)
bash run.sh configs/experiments/swarm.yaml --gpus 1,2,3,4 --set swarm.shards=4   # every task; one server and one cluster per shard
```

A task takes 2–3 minutes (an anomaly workload is about 90 s: the bulk insert, then 60 s of the anomaly; then the swarm's
iterations and the solo investigation), so the 100 tasks take about 75 minutes on four shards. The shards share the task list and each
claims the next free task (`outputs/swarm/q3_4b/marble_db/claims/`), so the work balances. The run resumes: finished
tasks are in `outputs/swarm/q3_4b/marble_db/shard*/events.jsonl`, and a stopped shard gives its unfinished claims back. Check `injection[].returncode` there: a workload that
dies at once leaves the agents a clean database (the vendored trigger needed the `pymysql` import made optional).

The planner assigns two agents per iteration: the benchmark's naive-planning prompt shows a two-agent JSON example and
Qwen3-4B follows it. That is the benchmark as-is; an agent never assigned gives its finding from an empty memory.

## Results: Qwen3-4B (2026-09-18)

100 tasks, four shards, `outputs/tables/swarm.md`. Accuracy is the benchmark's hit rule; set F1 is between the predicted and the
true set; single cause / pair: hits on the 50 tasks of each kind.

| condition | accuracy | exact set | set F1 | single cause | pair | guesses |
|---|---:|---:|---:|---:|---:|---:|
| question alone | 64.0 | 5.0 | 40.7 | 23/50 | 41/50 | 2.06 |
| MARBLE swarm (the planner's decision) | 61.0 | 3.0 | 35.8 | 19/50 | 42/50 | 2.33 |
| agents' verdicts (the YESes) | 69.0 | 3.0 | 40.5 | 24/50 | 45/50 | 2.33 |
| question + peers | 64.0 | 3.0 | 38.9 | 22/50 | 42/50 | 2.18 |
| peers + memory | 63.0 | 3.0 | 38.2 | 22/50 | 41/50 | 2.07 |
| combination | 66.0 | 4.0 | 40.6 | 22/50 | 44/50 | 2.10 |
| always guess the maximum (chance) | 65.0 | – | 37.3 | 40% | 90% | 2.50 |

- Everything sits near chance under the hit rule: a wrong extra guess never costs, so guessing the maximum scores 65. Exact set is
  3–5 for every condition; no condition beats question alone on set F1. Qwen3-4B barely solves this benchmark as planner, agents
  and central model; the single-cause tasks are where the conditions differ at all.
- The planner's decision (61) is below the model alone (64). The literal YESes score 69 because they list more causes, not because
  they are more right (set F1 ties question alone).
- The findings carry little evidence: vacuum agent 76/100 verdicts right; insert 45, lock 36, index 43, fetch 51; insert and
  fetch say YES on 83 and 79 of the 100 tasks. The planner assigns 2.7 agents per iteration. Record AUC 0.69, mostly identity.
- The tilt alone changes nothing (63 vs 64; five tasks flip each way) and cannot under this rule. The record's trust does separate
  where reading helps: mean trust 0.6–0.8 (58 tasks) peers + memory 72.4 vs 65.5 alone; trust 0.4–0.6 (37 tasks) 48.6 vs 56.8.
  Combination follows that split (ρ̂ 0.81, δ̂ 0.23, consult when mean trust ≥ 0.48; consulted 69%) and ends at 66: +2 over
  the model alone, +5 over the benchmark's planner, within noise at 100 tasks (standard error about 5 points).
- Sharper tests: a swarm whose findings carry evidence (Qwen3-8B, running), a longer stream, a metric that charges for wrong
  guesses, compromised agents. Cost: 163 s per task (118 s of anomaly workload, 40 s of star iterations, 6 s alone).

## Results: Qwen3-8B (2026-09-18)

Same experiment (`configs/experiments/swarm_qwen3_8b.yaml`), 100 tasks, `outputs/tables/swarm_qwen3_8b.md`.

| condition | accuracy | exact set | set F1 | single cause | pair | guesses |
|---|---:|---:|---:|---:|---:|---:|
| question alone | 78.0 | 20.0 | 56.7 | 37/50 | 41/50 | 1.96 |
| MARBLE swarm (the planner's decision) | 54.0 | 5.0 | 34.3 | 21/50 | 33/50 | 1.83 |
| agents' verdicts (the YESes) | 64.0 | 22.0 | 47.5 | 27/50 | 37/50 | 1.58 |
| question + peers | 65.0 | 21.0 | 48.6 | 30/50 | 35/50 | 1.64 |
| peers + memory | 64.0 | 21.0 | 47.6 | 29/50 | 35/50 | 1.61 |
| combination | 64.0 | 20.0 | 47.3 | 30/50 | 34/50 | 1.67 |

- The 8B alone is well above chance; the benchmark's coordination destroys it (planner 54, reading the findings 65). The agents
  are careful by saying NO (fetch YES 5, index 13, vacuum 18 of 100; each cause true on 30), and a reader that follows them drops
  the causes they omitted (FETCH named 5 times vs 51 alone). The tilt cannot restore an omitted cause (2 vs 3 flips against
  question + peers).
- Trust here measures the wrong thing: a finding is labelled right when its verdict matches the truth, so a NO on an absent
  cause counts as right though it tells the reader nothing, and the NO-heavy agents earn the highest trust. In the QA streams a
  correct answer is the useful one; here verdict correctness and usefulness come apart. Record AUC 0.71.
- Combination should have acted alone (alone beats reading in every trust bucket: 78 vs 59 at trust 0.4–0.6, 78 vs 67 at
  0.6–0.8) but consulted on 94 tasks: the record's estimate of the model's own ability rose only from 0.52 (first 50 tasks) to
  0.60 (last 50; 0.70 at event 100) against a true 78, because one identity coordinate under prior precision 100 needs more than
  100 events. The same rule with the own ability from a running average of the verified own answers (online) would consult on
  40 tasks and score 67; with the true value, 24 tasks and 72; choosing right everywhere, 90.

## Why the memory gained so little, and what follows

1. The hit rule cannot reward discounting (a wrong guess is free; chance 65); exact set and set F1 can.
2. With the 4B there is nothing to weigh (findings near coin flips, answers from a prior).
3. With the 8B the candidates are per-cause verdicts, so the labels reward NO-heavy agents, and the own-ability estimate learns
   too slowly for a 100-event stream.

Proposed, not run: (a) each agent reports a diagnosis (its most likely causes) instead of a verdict on its assigned cause, so the
candidates answer the same question and their labels measure usefulness; (b) estimate the autonomous ability directly from the
model's verified own answers; (c) a metric that charges wrong guesses as the headline; (d) a longer stream; (e) a compromised agent.

Results page (generated from the events and evaluations, both models, with the diagnosis): https://claude.ai/artifact/YDkUMzyLvTNvrJQYWhwKZt

## Another central model

A swarm stream depends on the model that runs it, so each central model has its own dataset entry, five peer entries and an
experiment that inherits from `swarm.yaml` with its own ports and clusters (they can run beside the Qwen3-4B run):

```
configs/datasets/marble_db_qwen3_8b.yaml      path marble_db_qwen3_8b/test.jsonl, built: swarm
configs/peers/swarm_{insert,lock,vacuum,index,fetch}_qwen3_8b.yaml
configs/experiments/swarm_qwen3_8b.yaml       base: swarm.yaml; central [qwen3_8b]; swarm {port: 8150, pg_port: 5450, shards: 4}
bash run.sh configs/experiments/swarm_qwen3_8b.yaml --gpus 0,5,6,7
```

## A pool of peers on every sub-step (`swarm_steps`)

The original method on the swarm's sub-steps. The benchmark's planner splits a task into five sub-steps, "is X a root cause?", one
per agent. With one model behind every agent there is one source per sub-step, so nothing to choose between. Here every sub-step
is answered by the pool of the QA streams (Gemma-3-4B, Phi-4-mini, Qwen2.5-Coder-7B, Llama-3.1-8B, DeepSeek-Coder-V2-Lite,
R1-Distill-Qwen-7B), the injected anomaly verifies every answer, the record tracks every peer, and Qwen3-4B consults the pool
with the tilt or answers alone. Five of these six cannot make the benchmark's tool call through vLLM, so the pool answers from
the evidence the benchmark's agent gathered for the sub-step (its queries and results, from a finished swarm run), as the QA peers
answer a question they are all shown.

```
pipeline.swarm steps     a finished run's events -> data/marble_db_steps_q/test.jsonl   500 yes/no questions (task type boolqa),
                                                                                         passage = the task + the agent's evidence
peers, streams           the six peers answer (marble_db_steps_answers) -> data/marble_db_steps/test.jsonl
own ... combination      the usual steps: own answer, features, record, question + peers, peers + memory, combination
pipeline.swarm diagnose  per condition, a task's diagnosis = the causes answered yes, scored by the benchmark's rule, the exact
                         set and the set F1 -> outputs/tables/swarm_steps_<model>_<dataset>_diagnosis.{json,md}
bash run.sh configs/experiments/swarm_steps.yaml --gpus 0,1,2,3,4,5,6,7     # needs the swarm and swarm_qwen3_8b runs' events
```

`marble_db_steps` uses the evidence Qwen3-4B's agents gathered, `marble_db_steps_qwen3_8b` Qwen3-8B's.

## A team of different models

The benchmark gives the planner and every agent one `llm`, so its five experts are five profiles over one model; its engine also
reads an optional `llm` per agent. `swarm_team` uses that: Qwen3-4B stays the planner and the central model, and agent1 ... agent5
are the models registered as the dataset's peers (`configs/datasets/marble_db_team.yaml`, `configs/peers/swarm_team_*.yaml`:
Llama-3.1-8B, Ministral-8B, Qwen2.5-7B, Mistral-7B-v0.3, Qwen3-8B). An agent model must make the `query_db` tool call through vLLM,
with the parser named in its model file (`tool_parser:`). Whenever a swarm dataset's peers are not the central model, the swarm step
runs `python -m pipeline.swarm team`: every distinct model is served once, on its own GPU when there are enough, and `swarm.workers`
task loops, each with its own PostgreSQL cluster, share the servers.

```bash
bash run.sh configs/experiments/swarm_team.yaml --smoke --gpus 0,1,2,3,4,5
bash run.sh configs/experiments/swarm_team.yaml --gpus 0,1,2,3,4,5
```

## Outputs

```
outputs/swarm/q3_4b/marble_db/shard<k>/events.jsonl     per task: the injection, every iteration (assignments, results, decision),
                                                        the findings with verdicts and labels, the solo investigation and diagnosis
data/marble_db/test.jsonl                               the stream: the five findings as peer_0 ... peer_4, labelled
outputs/eval/q3_4b/marble_db/solo/                      question alone (read by the own step)
outputs/eval/q3_4b/marble_db+own/{swarm,verdicts}/      the benchmark's decision and the naive aggregation
outputs/eval/q3_4b/marble_db+own/{peers,tilt,combination}/
outputs/tables/swarm.md
```
