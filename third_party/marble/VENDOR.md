# MARBLE (MultiAgentBench), vendored

Source: https://github.com/ulab-uiuc/MARBLE, commit 8d60fa17b5596b44458a52d4296061b9fc13d6f2, cloned 2026-09-18 (ACL 2025 main:
"MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents", Zhu et al., https://arxiv.org/abs/2503.01935).
Its licence is in LICENSE. Only what the database diagnosis needs is kept: `marble/` without the werewolf modules, the
minecraft / coding / research utilities, the example configs (the 100 database tasks are extracted to
`datasets/marble_db/tasks.jsonl`, identical under both of the repository's model prefixes), the docker files and the
benchmark data of `db_env_docker/`, and the workload generators the five anomalies do not use.

Patches, each marked `kalman-mem` in the code:

- `environments/db_env_docker/anomaly_trigger/promethues.py`: `restart_decision()` returns (0, 0); the benchmark read CPU and
  memory use from Prometheus for logging only, and no Prometheus runs here.
- `environments/db_env_docker/anomaly_trigger/utils/database.py`, `createdatabase.py`, `dropdatabase.py`: the PostgreSQL port
  comes from `KALMAN_PG_PORT` (default 5432), so several clusters can run side by side.
- `environments/db_env_docker/anomaly_trigger/anomaly.py`: `vacuum()` opened its own connection on a hard-coded port 5432, so
  with several clusters every shard's `VACUUM FULL` ran on the first one and the shards deadlocked; it now uses `DB_CONFIG`'s port.
- `environments/db_env_docker/anomaly_trigger/utils/database.py`: `pymysql` is an optional import (only its MySQL branch
  uses it; without the patch the anomaly workloads died on import and the agents investigated a clean database).
- `environments/__init__.py`: the environments other than the base and database ones are optional imports.
- `agent/__init__.py`, `evaluator/__init__.py`: the werewolf imports are removed; `engine/__init__.py` imports nothing (the
  `Engine` class is not used, and `evaluator.py` upstream has a mis-indented `except` that does not parse).

Not patched but bypassed: `DBEnvironment.__init__` (docker compose, Prometheus) is replaced by `feedback_state.swarm.make_env`,
which registers the same `query_db` action on a user-space PostgreSQL and cuts a query result at 6,000 characters; the engine's
star loop is reproduced in `feedback_state.swarm.run_swarm` with the benchmark's planner and agents, without the LLM-judged
planning and KPI scores. The LLM calls go through `litellm` as in the benchmark, routed to a local vLLM server by
`feedback_state.swarm.patch_llm`.
