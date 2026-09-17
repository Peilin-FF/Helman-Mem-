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

## Another central model

A swarm stream depends on the model that runs it, so each central model has its own dataset entry, five peer entries and an
experiment that inherits from `swarm.yaml` with its own ports and clusters (they can run beside the Qwen3-4B run):

```
configs/datasets/marble_db_qwen3_8b.yaml      path marble_db_qwen3_8b/test.jsonl, built: swarm
configs/peers/swarm_{insert,lock,vacuum,index,fetch}_qwen3_8b.yaml
configs/experiments/swarm_qwen3_8b.yaml       base: swarm.yaml; central [qwen3_8b]; swarm {port: 8150, pg_port: 5450, shards: 4}
bash run.sh configs/experiments/swarm_qwen3_8b.yaml --gpus 0,5,6,7
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
