# marble_db_online: the database swarm online, the same six peers on every root cause

Question: in MultiAgentBench's database diagnosis, run as a swarm with one planner, does the record help when every root cause
the planner assigns is investigated by the same six peers and the central model commits the verdict the planner goes on with?

This is the database counterpart of the ClassEval swarm (docs/experiments/classeval.md), on the same engine
(`feedback_state/online_swarm.py`) with the benchmark as an adapter (`feedback_state/marble_online.py`). It differs from the
earlier swarm experiments (docs/experiments/swarm.md): `swarm_pool` ran seven independent teams, each with its own planner
and one model behind all five agents, and built the record afterwards; here there is one planner per condition, every
assigned cause fans out to all six peers, the central model's verdict is what the planner reads, and the record is read before
and written after every planner iteration.

## One task

1. **The database.** The benchmark's task (one of 100: 10 scenarios × 10 anomaly sets) is prepared once for every track: the
   database reset, the scenario's schema and benign queries, the anomaly workloads (the benchmark's scripts, 60 s each), on a
   user-space PostgreSQL (`feedback_state.swarm`, as upstream).
2. **The planner.** Each track runs the benchmark's star loop with its own planner: MARBLE's EnginePlanner (naive planning) over
   the task's five agent profiles, one per root cause, with Qwen3-4B behind it. A planner iteration is a tick of the swarm.
3. **The events.** Every root cause the planner assigns that the track has not investigated yet is an event, "is X a root
   cause?". All five are investigated on every task: the planner chooses the order (the causes it names, in its order); an
   iteration where it names none takes the next cause in the benchmark's order; once it stops (decide_next_step) or its five
   iterations run out, the causes left go together (`forced` in the rows).
4. **The investigators.** The six peers (Gemma-3-4B, Phi-4-mini, Qwen2.5-Coder-7B, Llama-3.1-8B, DeepSeek-Coder-V2-Lite,
   R1-Distill-Qwen-7B) each investigate the cause as a MARBLE agent: the cause's profile, the planner's assignment, three
   `query_db` actions (one SQL statement each, through upstream's `force_tool` protocol, so every model can query), its own
   memory, then its finding ending 'Final answer: yes|no' (upstream's `finding_prompt`). On the record tracks Qwen3-4B
   investigates too, the same way (its own answer: recorded, never shown).
5. **The record is read.** The frozen judge reads the question and the seven findings; its hidden states address the record
   (design `qc`, the paper's PCA fit on train6, 256 components per part, λ = 100): P(correct) of every finding from the
   iterations before this one.
6. **The verdict.** Qwen3-4B reads the six findings (tilted by the record on the record tracks, γ = 3) and commits the cause's
   verdict, or, on combination, the reading line picks that or its own finding. The planner receives the verdict as that
   agent's result: its progress, its next assignments and its final decision read it.
7. **Feedback, at once.** Every finding and every verdict is graded against the injected anomaly; the seven labels go into the
   record before the next iteration is read.
8. **The diagnosis.** When all five causes are done the planner decides from all five verdicts (the benchmark's
   `summarize_output`), scored by the benchmark's rule (a true cause among the allowed guesses), exactly (the predicted set is
   the true set) and by set F1. The verdicts themselves are a second diagnosis: the causes the track said yes to.

Tracks: `online_solo` (Qwen3-4B's own findings are the verdicts), `online_peers` (it reads the peers), `online_tilt` (with the
tilt), `online_combination` (per cause either, by the reading line). Each has its own planner, verdicts and, for the tilt and
combination, its own record starting cold; the database of a task is shared by all tracks.

## What to expect, and what measures it

- The sub-steps are five fixed questions, so a table of peer accuracy per cause is a strong baseline for reliability: the
  report puts every record next to tables per peer, per peer and cause, per peer and its verdict, per peer, cause and verdict,
  and per peer and scenario, all read before write along the same stream. The record's answer part reads the finding itself,
  so it can separate a peer's reliable YES from its unreliable NO without being told.
- A cause is a root cause in 150 of the 500 events: always saying no is right 70% of the time. The report gives the verdicts'
  balanced accuracy (the mean of the recall on true and on false causes) beside their accuracy, and the diagnoses' exact-set
  accuracy and set F1 beside the benchmark's hit rule (which never charges a wrong extra guess).
- The planner is where the swarm's decisions propagate: which causes it names first and what it concludes. The investigators
  read the planner's assignment, never another track's verdicts.

## Setup

As for the database swarm (docs/experiments/swarm.md): PostgreSQL in user space (`conda create -n pg -c conda-forge
postgresql`), `pip install psycopg2-binary "litellm>=1.52" "httpx[socks]" beartype colorlog` into the sigma env. The
judge's train6 features come from the main experiment (made first if missing).

## Run

```bash
bash run.sh configs/experiments/marble_db_online.yaml --smoke --gpus 0,1,2,3,4,5,6,7     # two tasks, every track, under outputs/smoke/
bash run.sh configs/experiments/marble_db_online.yaml --gpus 0,1,2,3,4,5,6,7             # the 100 tasks
```

The job uses eight GPUs: Qwen3-4B in-process on the first (its readings, with the tilt, and its judge), the six peers' servers
one GPU each (ports 8400-8405), and Qwen3-4B's server (port 8410) for its MARBLE roles (the planner, its own investigations);
with seven GPUs that server shares the first. `online.lanes: L` runs L tasks at once, each on its own PostgreSQL cluster (the
record then sees an iteration up to L - 1 iterations later). Untimed: a task is the injection (about two minutes) plus two to
six iterations of 18-22 parallel investigations.

## Outputs

```
outputs/eval/q3_4b/marble_db_swarm/online_<track>/     generations.jsonl (per task and cause: the verdict and its label, the six
                                                       findings, their verdicts and labels, the own finding, the record's
                                                       estimates, the iteration, whether the planner chose it), units.jsonl
                                                       (per task: the verdicts, the planner's decision, both diagnoses scored,
                                                       the injection), eval_metrics.json
outputs/eval/q3_4b/marble_db_swarm/vllm_<model>.log    the servers
outputs/tables/marble_db_online_q3_4b_marble_db_swarm_marble.md    the verdicts, the diagnoses, the peers, the records against the tables
```

Differences from the benchmark as-is: the planner's final decision is made once, from all five verdicts (the benchmark's engine
re-decides after every iteration from that iteration's results only); a cause is investigated once, with three queries per
investigator (the benchmark's agents act once per iteration they are assigned); every cause is investigated.

Status (2026-09-18): implemented and tested on CPU (the adapter with stand-in planners, investigators and models: the order
and the forced coverage, the verdicts the planner reads, the labels, the record, the diagnoses, failing roles, several lanes,
the report and the job expansion); the MARBLE roles (`MarblePlanner`, `investigate_marble`) follow upstream's `run_swarm` and
`run_solo` but have not been run here (no PostgreSQL, litellm or GPUs on the machine it was written on).
