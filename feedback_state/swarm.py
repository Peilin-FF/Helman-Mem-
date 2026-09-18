"""A real agent swarm: MultiAgentBench's database diagnosis (Zhu et al., 2025), run with a local model, as a stream of events.

The benchmark: a PostgreSQL database shows a performance anomaly. A planner assigns five agents, each investigating one
possible root cause (INSERT_LARGE_DATA, LOCK_CONTENTION, VACUUM, REDUNDANT_INDEX, FETCH_LARGE_DATA); the agents query
the database's system views and the planner decides the most likely root causes. Ground truth is the injected anomaly,
and a prediction is correct when one of its allowed guesses is a true root cause (the benchmark's rule; we also keep
whether the predicted set is exactly right). The vendored benchmark code is third_party/marble (VENDOR.md there).

What this module adds, per task, on the benchmark's own agents (star coordination, the planner's prompts, the agents'
tool loop, all as in the benchmark):

  swarm      the benchmark as-is: the planner's final decision after its iterations (a condition, `swarm`)
  findings   after the loop, every agent reports its finding on the cause it investigated, with a YES/NO verdict; the
             verdict is graded against the ground truth. The five findings are the event's candidate answers, so the
             record estimates each agent's finding, the central model reads them with the tilt, and combination applies.
  solo       the central model investigates alone with the same tool budget: question alone
  verdicts   the naive aggregation of the findings (the causes with a YES verdict): a condition, `verdicts`

The LLM is served by vLLM's OpenAI-compatible server (tool calling on); PostgreSQL runs in user space from a conda env
(no docker, no root). The anomaly workloads are the benchmark's scripts, run against that server.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
MARBLE = REPO / "third_party" / "marble"
TRIGGER = MARBLE / "marble" / "environments" / "db_env_docker" / "anomaly_trigger"
LABELS = ["INSERT_LARGE_DATA", "LOCK_CONTENTION", "VACUUM", "REDUNDANT_INDEX", "FETCH_LARGE_DATA"]
DB_USER, DB_PASSWORD, DB_NAME = "test", "Test123_456", "sysbench"
RESULT_CHARS = 6000          # a tool result longer than this is cut (the benchmark returns it whole; a long one overflows the context)
MEMORY_CHARS = 8000          # the part of an agent's memory shown when it reports its finding
SOLO_PROFILE = ("You are the central database expert. Investigate the database's performance issue yourself, considering every "
                "possible root cause: INSERT_LARGE_DATA (recommended table pg_stat_statements, search for INSERTs), LOCK_CONTENTION "
                "(pg_locks), VACUUM (pg_stat_all_tables; VACUUM queries in pg_stat_statements), REDUNDANT_INDEX (pg_stat_user_indexes, "
                "pg_indexes) and FETCH_LARGE_DATA (SELECTs in pg_stat_statements). Use one query at a time and build up evidence.")


def marble_on_path() -> None:
    if str(MARBLE) not in sys.path:
        sys.path.insert(0, str(MARBLE))


def cause_of(profile: str) -> str:
    """The root cause an agent's profile assigns it ('agentK will explore the possibility of X as a root cause')."""
    m = re.search(r"possibility of ([A-Z_]+) as a root cause", profile)
    if not m or m.group(1) not in LABELS:
        raise ValueError(f"no root cause in the profile: {profile[:80]!r}")
    return m.group(1)


# --- grading -------------------------------------------------------------------------------------------------------------
def predicted_causes(text: str) -> list[str]:
    """The root causes a text names, in order of first appearance (after 'Final answer:' when that line exists), no repeats."""
    text = str(text or "")
    tail = text.rsplit("Final answer:", 1)[1] if "Final answer:" in text else text
    found = []
    for m in re.finditer(r"[A-Z][A-Z_ ]+[A-Z]", tail.upper()):
        name = m.group(0).replace(" ", "_")
        for label in LABELS:
            if label in name and label not in found:
                found.append(label)
    return found


def hit(text: str, root_causes: list[str], allowed: int) -> bool:
    """The benchmark's rule: one of the first `allowed` predicted causes is a true root cause."""
    return any(c in root_causes for c in predicted_causes(text)[: int(allowed)])


def exact(text: str, root_causes: list[str], allowed: int) -> bool:
    return set(predicted_causes(text)[: int(allowed)]) == set(root_causes)


def verdict_of(finding: str) -> str | None:
    """A finding's verdict: its 'Verdict: YES|NO' line, or the last 'Final answer: yes|no' (after a think block, if any)."""
    text = str(finding or "").rsplit("</think>", 1)[-1]
    m = re.search(r"Verdict:\s*\**\s*(YES|NO)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    finals = re.findall(r"final\s+answer\s*(?:is|:)?\s*\**\s*(yes|no)\b", text, flags=re.IGNORECASE)
    return finals[-1].upper() if finals else None


def finding_correct(finding: str, cause: str, root_causes: list[str]) -> int:
    v = verdict_of(finding)
    return int(v is not None and (v == "YES") == (cause in root_causes))


def verdict_answer(findings: list[dict]) -> str:
    """The naive aggregation: the causes whose agents said YES, in agent order."""
    yes = [f["cause"] for f in findings if verdict_of(f["text"]) == "YES"]
    return "Final answer: " + (", ".join(yes) if yes else "NONE")


# --- PostgreSQL in user space -----------------------------------------------------------------------------------------------
def pg_bin() -> Path:
    return Path(os.environ.get("KALMAN_PG_BIN") or Path.home() / "miniconda3" / "envs" / "pg" / "bin")


def pg_ready(port: int) -> bool:
    return subprocess.run([str(pg_bin() / "pg_isready"), "-h", "localhost", "-p", str(port)], capture_output=True).returncode == 0


def ensure_postgres(port: int, data_dir: Path) -> None:
    """Start (initialising if needed) a PostgreSQL cluster owned by this user, with pg_stat_statements loaded and the
    benchmark's role and database. Nothing outside data_dir and the conda env is touched."""
    if pg_ready(port):
        return
    b = pg_bin()
    if not (b / "initdb").exists():
        raise SystemExit(f"no PostgreSQL under {b}: conda create -n pg -c conda-forge postgresql (or set KALMAN_PG_BIN)")
    data_dir = Path(data_dir)
    if not (data_dir / "PG_VERSION").exists():
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        pw = data_dir.parent / f".pw{port}"
        pw.write_text(DB_PASSWORD)
        subprocess.run([str(b / "initdb"), "-D", str(data_dir), "-U", DB_USER, "--auth=md5", f"--pwfile={pw}"], check=True, capture_output=True)
        pw.unlink()
        with (data_dir / "postgresql.conf").open("a") as f:
            f.write(f"\nport = {port}\nlisten_addresses = 'localhost'\nunix_socket_directories = '{data_dir}'\n"
                    "shared_preload_libraries = 'pg_stat_statements'\npg_stat_statements.track = all\nmax_connections = 200\n"
                    "logging_collector = on\nlog_min_error_statement = info\n")
    subprocess.run([str(b / "pg_ctl"), "-D", str(data_dir), "-l", str(data_dir / "server.log"), "-o", f"-p {port}", "start"],
                   check=True, capture_output=True)
    for _ in range(60):
        if pg_ready(port):
            break
        time.sleep(1)
    else:
        raise SystemExit(f"PostgreSQL on port {port} did not come up; see {data_dir / 'server.log'}")
    subprocess.run([str(b / "psql"), "-h", "localhost", "-p", str(port), "-U", DB_USER, "-d", "postgres", "-c", f"CREATE DATABASE {DB_NAME}"],
                   env={**os.environ, "PGPASSWORD": DB_PASSWORD}, capture_output=True)   # exists already: fine


class MarbleDB:
    """The benchmark's database on our server: reset, initialise, inject, query."""

    def __init__(self, port: int):
        self.port = int(port)

    def connect(self, dbname: str = DB_NAME):
        import psycopg2

        conn = psycopg2.connect(user=DB_USER, password=DB_PASSWORD, database=dbname, host="localhost", port=str(self.port))
        conn.autocommit = True
        return conn

    def reset(self) -> None:
        conn = self.connect("postgres")
        cur = conn.cursor()
        for db in ("tmp", DB_NAME):
            cur.execute(f"DROP DATABASE IF EXISTS {db} WITH (FORCE)")
        cur.execute(f"CREATE DATABASE {DB_NAME}")
        conn.close()
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
        cur.execute("SELECT pg_stat_statements_reset()")
        conn.close()

    def initialise(self, init_sql: str) -> None:
        """As the benchmark: the scenario's schema, sample rows and benign queries, one statement at a time."""
        marble_on_path()
        from marble.environments.db_env import split_sql_statements

        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SET client_min_messages TO WARNING;")
        for statement in split_sql_statements(init_sql):
            cur.execute(statement)
        cur.execute("RESET client_min_messages;")
        conn.close()

    def inject(self, anomalies: list[dict], duration: int) -> list[dict]:
        """The benchmark's anomaly workloads (its scripts, its arguments), against this server."""
        logs = []
        for a in anomalies:
            cmd = [sys.executable, "main.py", "--anomaly", a["anomaly"], "--threads", str(a["threads"]), "--ncolumn", str(a["ncolumn"]),
                   "--colsize", str(a["colsize"]), "--duration", str(int(duration))]
            t0 = time.time()
            try:
                p = subprocess.run(cmd, cwd=TRIGGER, env={**os.environ, "KALMAN_PG_PORT": str(self.port)}, capture_output=True, text=True,
                                   timeout=int(duration) * 4 + 600)
                rc, tail = p.returncode, (p.stdout + p.stderr)[-600:]
            except subprocess.TimeoutExpired as e:   # a workload that hangs must not take the shard with it; the event records it
                rc, tail = -1, f"timeout after {int(duration) * 4 + 600}s: " + ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else str(e.stdout or ""))[-500:]
            logs.append({"anomaly": a["anomaly"], "seconds": round(time.time() - t0, 1), "returncode": rc, "tail": tail})
        return logs

    def query(self, sql: str) -> dict:
        """The benchmark's query_db handler, with the result cut at RESULT_CHARS."""
        marble_on_path()
        from marble.environments.db_env import split_sql_statements

        try:
            conn = self.connect()
            cur = conn.cursor()
            statements = split_sql_statements(sql)
            for s in statements:
                cur.execute(s)
            result = cur.fetchall()
            conn.close()
            shown = str(result)
            if len(shown) > RESULT_CHARS:
                shown = shown[:RESULT_CHARS] + f" ... [cut: {len(result)} rows, {len(str(result))} characters]"
            return {"status": "success", "function_name": "query_db",
                    "explanation": f"Your query on the database was successful{'' if len(result) else ' but no data was returned'}. "
                                   f"\nYour query is: {statements} \nResult: {shown}"}
        except Exception as e:   # the benchmark reports every error to the agent
            return {"status": "error", "function_name": "query_db", "explanation": f"An error occurred while you tried to query the database: {e}"}


# --- the LLM behind the benchmark's agents ---------------------------------------------------------------------------------
def route_for(model: str, routes: dict | None, default: str) -> str:
    """The server of a model: routes maps a served name (with or without litellm's 'openai/' prefix) to its base URL."""
    return (routes or {}).get(str(model or "").split("/", 1)[-1], default)


TOOL_PROTOCOL = ("\n\nTo act you must query the database: reply with your reasoning in a sentence or two, then exactly one PostgreSQL statement "
                 "inside a ```sql code block, and nothing after it.")


def extract_sql(text: str) -> str | None:
    """The SQL statement a model wrote: the last ```sql block after a think block (any fenced block, else a trailing statement)."""
    text = str(text or "").rsplit("</think>", 1)[-1]
    blocks = re.findall(r"```(?:sql|postgresql|postgres|psql)\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL) \
        or re.findall(r"```\w*\s*\n(.*?)```", text, flags=re.DOTALL)
    sql = blocks[-1].strip() if blocks else None
    if not sql:
        m = re.findall(r"(?:^|\n)\s*((?:SELECT|WITH|EXPLAIN|SHOW)\b[^;]*;)", text, flags=re.IGNORECASE)
        sql = m[-1].strip() if m else None
    return sql or None


def patch_llm(api_base: str, api_key: str = "EMPTY", thinking: bool = False, timeout: int = 300, routes: dict | None = None,
              force_tool: str | None = None, reasoning_models: tuple = ()) -> None:
    """Route the benchmark's litellm calls to local OpenAI-compatible servers, thinking off: every model to api_base, or, with
    routes (served name -> base URL), each model to its own server (a team whose agents are different models)."""
    for k in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"] = api_base
    import litellm

    litellm.suppress_debug_info = True
    original = litellm.completion

    def completion(*args, **kwargs):
        kwargs.setdefault("api_base", route_for(kwargs.get("model", args[0] if args else ""), routes, api_base))
        kwargs.setdefault("api_key", api_key)
        kwargs.setdefault("timeout", timeout)
        named = [t for t in kwargs.get("tools") or [] if t.get("function", {}).get("name") == force_tool] if force_tool else []
        if named:
            # An agent's action, for any model: no function-calling API (several peers never call the tool when left to choose, and a
            # server-side forced call breaks on others). The model is asked for one SQL statement in a code block; we parse it and
            # hand the benchmark's agent a regular tool call. The benchmark's agent code is untouched.
            params = named[0]["function"].get("parameters", {})
            arg = (params.get("required") or list(params.get("properties", {"sql": 0})))[0]
            kwargs.pop("tools", None); kwargs.pop("tool_choice", None)
            msgs = [dict(m) for m in kwargs.get("messages") or []]
            if msgs:
                msgs[-1]["content"] = str(msgs[-1].get("content") or "") + TOOL_PROTOCOL
                kwargs["messages"] = msgs
            model = str(kwargs.get("model", args[0] if args else "")).split("/", 1)[-1]
            if model in reasoning_models:
                kwargs["max_tokens"] = max(int(kwargs.get("max_tokens") or 0), 3072)     # room to think before the statement
        if not thinking:
            extra = dict(kwargs.get("extra_body") or {})
            ct = dict(extra.get("chat_template_kwargs") or {})
            ct.setdefault("enable_thinking", False)
            extra["chat_template_kwargs"] = ct
            kwargs["extra_body"] = extra
        resp = original(*args, **kwargs)
        if named:
            from litellm.types.utils import ChatCompletionMessageToolCall, Function

            msg = resp.choices[0].message
            sql = extract_sql(msg.content)
            if sql:
                msg.tool_calls = [ChatCompletionMessageToolCall(id=f"call_{int(time.time() * 1000) % 10 ** 9}", type="function",
                                                                function=Function(name=force_tool, arguments=json.dumps({arg: sql})))]
                msg.content = str(msg.content or "").rsplit("</think>", 1)[-1].split("```", 1)[0].strip()
        return resp

    litellm.completion = completion


def start_vllm(model_path: str, served_name: str, port: int, log: Path, gpu_memory_utilization: float = 0.6, max_model_len: int = 16384,
               tool_parser: str = "hermes", gpu: str | None = None, env_vars: dict | None = None, prefix_caching: bool = True,
               trust_remote_code: bool = False):
    """Start vLLM's OpenAI server for one model (tool calling on, with the model's parser), on `gpu` if given, else the visible GPU.
    env_vars, prefix_caching and trust_remote_code are the model's registered settings (configs/models/)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["vllm", "serve", model_path, "--served-model-name", served_name, "--port", str(port), "--max-model-len", str(max_model_len),
           "--gpu-memory-utilization", str(gpu_memory_utilization), "--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
    cmd += [] if prefix_caching else ["--no-enable-prefix-caching"]
    cmd += ["--trust-remote-code"] if trust_remote_code else []
    env = {k: v for k, v in os.environ.items() if k.lower() not in ("all_proxy", "https_proxy", "http_proxy")}
    env.update({str(k): str(v) for k, v in (env_vars or {}).items()})
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, env=env)


def wait_vllm(proc, served_name: str, port: int, log: Path, minutes: int = 20) -> None:
    import urllib.request

    for _ in range(minutes * 12):
        if proc.poll() is not None:
            raise SystemExit(f"vLLM exited with {proc.returncode}; see {log}")
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=3) as r:
                if served_name in r.read().decode():
                    return
        except Exception:
            pass
        time.sleep(5)
    proc.terminate()
    raise SystemExit(f"vLLM on port {port} did not come up in {minutes} minutes; see {log}")


def serve_vllm(model_path: str, served_name: str, port: int, log: Path, gpu_memory_utilization: float = 0.6, max_model_len: int = 16384,
               tool_parser: str = "hermes", gpu: str | None = None):
    """Start vLLM's OpenAI server for the swarm's model and wait for it; returns the process."""
    proc = start_vllm(model_path, served_name, port, log, gpu_memory_utilization, max_model_len, tool_parser, gpu)
    wait_vllm(proc, served_name, port, log)
    return proc


# --- the benchmark's swarm, per task -----------------------------------------------------------------------------------------
def marble_config(task: dict, llm: str, relationships: list | None = None, agents: list | None = None):
    marble_on_path()
    from marble.configs.config import Config

    return Config({"coordinate_mode": "star", "relationships": task["relationships"] if relationships is None else relationships,
                   "llm": llm, "environment": {"type": "DB", "name": "DB Simulation Environment", "max_iterations": task["max_iterations"]},
                   "agents": task["agents"] if agents is None else agents, "memory": {"type": "SharedMemory"},
                   "task": {"content": task["task"], "output_format": task["output_format"], "labels": task["labels"],
                            "root_causes": task["root_causes"], "number_of_labels_pred": task["number_of_labels_pred"]},
                   "engine_planner": {"initial_progress": "Starting the simulation."}})


def make_env(db: MarbleDB):
    marble_on_path()
    from marble.environments.base_env import BaseEnvironment
    from marble.environments.db_env import DBEnvironment

    class SwarmEnv(DBEnvironment):
        """The benchmark's database environment on our server: its query_db action and description, no docker, no Prometheus."""

        def __init__(self, database: MarbleDB):
            BaseEnvironment.__init__(self, "DB Environment", {"max_iterations": 10 ** 9})
            self.db = database
            self.register_actions()

        def query_db_handler(self, sql: str) -> dict:
            return self.db.query(sql)

    return SwarmEnv(db)


def summarize_results(agents_results: list[dict]) -> str:
    """The engine's _summarize_results: every result cut at 1,000 characters."""
    summary = "Agents' Results Summary:\n"
    for result in agents_results:
        summary += f"- {result}"[:1000] + "\n"
    return summary


def run_swarm(task: dict, llm: str, env, log, agent_llms: dict | None = None, answer_line: bool = False, reasoning: bool = False,
              tag: str = "") -> dict:
    """The benchmark's star coordination on one task: the planner assigns, the agents act, the planner summarises and
    decides, for up to max_iterations. Returns the planner's final decision, the iterations, and each agent's finding.
    llm is the planner's model; agent_llms (agent id -> model) gives an agent its own, as the benchmark's per-agent `llm` does."""
    marble_on_path()
    from marble.agent.base_agent import BaseAgent
    from marble.engine.engine_planner import EnginePlanner
    from marble.graph.agent_graph import AgentGraph
    from marble.llms.model_prompting import model_prompting
    from marble.memory.shared_memory import SharedMemory

    config = marble_config(task, llm)
    agents = [BaseAgent(config=a, env=env, model=(agent_llms or {}).get(a["agent_id"], llm)) for a in task["agents"]]
    graph = AgentGraph(agents, config)
    for a in agents:
        a.set_agent_graph(graph)
    planner = EnginePlanner(graph, SharedMemory(), config.engine_planner, task["task"], model=llm)
    iterations, final, errors = [], "", []
    for it in range(int(task["max_iterations"])):
        try:
            assignment = planner.assign_tasks(planning_method="naive")
        except Exception as e:   # the benchmark's planner parses the model's JSON; a bad reply costs the iteration, not the task
            errors.append(f"iteration {it + 1} assign: {type(e).__name__}: {str(e)[:300]}")
            log(f"    planner failed to assign: {type(e).__name__}: {str(e)[:120]}")
            assignment = {"tasks": {}, "continue": True}
        tasks = assignment.get("tasks", {}) or {}
        results = []
        for agent_id, t in tasks.items():
            try:
                result, _ = graph.get_agent(agent_id).act(str(t))
                results.append({agent_id: result})
            except Exception as e:   # the engine logs the error and moves on
                errors.append(f"iteration {it + 1} {agent_id}: {type(e).__name__}: {str(e)[:300]}")
                log(f"    {agent_id} failed: {type(e).__name__}: {str(e)[:120]}")
        summary = summarize_results(results)
        try:
            final = planner.summarize_output(summary, task["task"], task["output_format"]).content or final
            planner.update_progress(summary)
            cont = planner.decide_next_step(results)
        except Exception as e:
            errors.append(f"iteration {it + 1} planner: {type(e).__name__}: {str(e)[:300]}")
            log(f"    planner failed: {type(e).__name__}: {str(e)[:120]}")
            cont = True
        iterations.append({"iteration": it + 1, "assignments": tasks, "results": results, "decision": final, "continue": bool(cont)})
        log(f"    {tag}iteration {it + 1}: {len(tasks)} assigned, {len(results)} acted, continue={cont}")
        if not cont:
            break
    findings = []
    for a in agents:
        cause = cause_of(a.profile)
        memory = a.memory.get_memory_str()
        if len(memory) > MEMORY_CHARS:
            memory = memory[: MEMORY_CHARS // 2] + " ... " + memory[-MEMORY_CHARS // 2:]
        if answer_line:   # a pool of peers on the same sub-step: the finding answers the sub-step's yes/no question
            prompt = (f"You are {a.agent_id}: {a.profile}\nYour investigation so far (your tool calls and their results): {memory}\n\n"
                      f"Question: is {cause} a root cause of this database's performance issue? Give the evidence from your investigation in at "
                      f"most six sentences, then end with exactly 'Final answer: yes' or 'Final answer: no'.")
        else:
            prompt = (f"You are {a.agent_id}: {a.profile}\nYour investigation so far (your tool calls and their results): {memory}\n\n"
                      f"Report your finding to the planner. Begin with exactly one line of the form 'Root cause investigated: {cause}. Verdict: YES' "
                      f"or 'Root cause investigated: {cause}. Verdict: NO' (YES if {cause} is a root cause of the performance issue, NO if it is not), "
                      f"then give the evidence in at most six sentences.")
        try:
            text = model_prompting(llm_model=a.llm, messages=[{"role": "user", "content": prompt}], return_num=1,
                                   max_token_num=2048 if reasoning else 400, temperature=0.0)[0].content or ""   # the agent reports with its own model
        except Exception as e:
            errors.append(f"finding {a.agent_id}: {type(e).__name__}: {str(e)[:300]}")
            text = f"Root cause investigated: {cause}. Verdict: NO\n(The agent failed to report: {type(e).__name__}.)"
        findings.append({"agent_id": a.agent_id, "cause": cause, "model": str(a.llm).split("/", 1)[-1], "text": text, "verdict": verdict_of(text),
                         "correct": finding_correct(text, cause, task["root_causes"])})
    return {"final": final, "iterations": iterations, "findings": findings, "errors": errors}


def run_pool(task: dict, llm: str, teams: dict, make_team_env, log) -> dict:
    """A pool of peers on every sub-step: each team is the benchmark's star loop with the planner's model `llm` and one model
    behind all five agents (teams: name -> {"llm", "reasoning"}); the teams investigate the same injected database at the same
    time, so every sub-step ("is X a root cause?") gets one finding per team, each from that model's own queries."""
    from concurrent.futures import ThreadPoolExecutor

    def one(item):
        name, spec = item
        try:
            return name, run_swarm(task, llm, make_team_env(), log, agent_llms={a["agent_id"]: spec["llm"] for a in task["agents"]},
                                   answer_line=True, reasoning=bool(spec.get("reasoning")), tag=f"[{name}] ")
        except Exception as e:   # one team failing must not lose the others' work on this task
            log(f"    [{name}] team failed: {type(e).__name__}: {str(e)[:160]}")
            return name, {"final": "", "iterations": [], "findings": [], "errors": [f"team: {type(e).__name__}: {str(e)[:300]}"]}

    with ThreadPoolExecutor(max_workers=len(teams)) as ex:
        return dict(ex.map(one, teams.items()))


def run_solo(task: dict, llm: str, env, iterations: int, log) -> dict:
    """The central model alone with the same tool budget: one action per iteration, then its diagnosis."""
    marble_on_path()
    from feedback_state.memory_generator import INSTRUCTIONS
    from marble.agent.base_agent import BaseAgent
    from marble.graph.agent_graph import AgentGraph
    from marble.llms.model_prompting import model_prompting

    agent = BaseAgent(config={"agent_id": "planner", "profile": SOLO_PROFILE}, env=env, model=llm)
    graph = AgentGraph([agent], marble_config(task, llm, relationships=[], agents=[{"agent_id": "planner", "profile": SOLO_PROFILE}]))
    agent.set_agent_graph(graph)
    acts, errors = [], []
    for it in range(int(iterations)):
        t = (f"{task['task']}\nThis is investigation step {it + 1} of {iterations}: query the database (one query) for evidence you do not "
             f"have yet, using your memory of earlier results.")
        try:
            result, _ = agent.act(t)
            acts.append(result)
        except Exception as e:
            errors.append(f"step {it + 1}: {type(e).__name__}: {str(e)[:300]}")
            log(f"    solo step {it + 1} failed: {type(e).__name__}: {str(e)[:120]}")
    memory = agent.memory.get_memory_str()
    if len(memory) > MEMORY_CHARS:
        memory = memory[: MEMORY_CHARS // 2] + " ... " + memory[-MEMORY_CHARS // 2:]
    prompt = (f"{task['task']}\nYour investigation (your queries and their results): {memory}\n\n"
              f"Possible root causes: {', '.join(task['labels'])}.\nInstruction: {INSTRUCTIONS['dbdiag']}")
    try:
        final = model_prompting(llm_model=llm, messages=[{"role": "user", "content": prompt}], return_num=1, max_token_num=600,
                                temperature=0.0)[0].content or ""
    except Exception as e:
        errors.append(f"diagnosis: {type(e).__name__}: {str(e)[:300]}")
        final = ""
    return {"final": final, "acts": acts, "errors": errors}


# --- the stream and the evaluations built from the events -------------------------------------------------------------------
def problem_text(task: dict) -> str:
    return f"{task['task'].strip()}\n\nPossible root causes: {', '.join(task['labels'])}."


def stream_record(event: dict, task: dict, model_name: str) -> dict:
    """One event of the stream: the task as the question, the five findings as the candidate answers."""
    rec = {"id": task["id"], "task_type": "dbdiag", "source": task["scenario"], "problem": problem_text(task),
           "answer": list(task["root_causes"]), "number_of_labels_pred": int(task["number_of_labels_pred"]), "labels": list(task["labels"]),
           "peer_responses": {}, "peer_correct": {}, "correctness_by_peer": {}, "peer_metadata": {}}
    for k, f in enumerate(event["swarm"]["findings"]):
        key = f"peer_{k}"
        rec["peer_responses"][key] = f["text"]
        rec["peer_correct"][key] = float(f["correct"])
        rec["correctness_by_peer"][key] = int(f["correct"])
        rec["peer_metadata"][key] = {"model": f.get("model") or model_name, "agent_id": f["agent_id"], "cause": f["cause"], "verdict": f["verdict"],
                                     "received_context": True, "num_samples": 1}
    return rec


def graded(text: str, task: dict) -> dict:
    k = int(task["number_of_labels_pred"])
    return {"correct": int(hit(text, task["root_causes"], k)), "exact": int(exact(text, task["root_causes"], k)),
            "predicted": predicted_causes(text)[:k]}
