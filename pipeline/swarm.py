"""Swarm: run MultiAgentBench's database diagnosis with a local model, and turn it into a stream with evaluations.

    python -m pipeline.swarm run   --tasks datasets/marble_db/tasks.jsonl --model /models/Qwen3-4B --served-name Qwen3-4B --port 8123 \
                                   --pg-port 5432 --pg-data /mnt/data/peilin/pg/5432 --out outputs/swarm/q3_4b/marble_db [--shard k/N] [--limit N]
    python -m pipeline.swarm merge --out outputs/swarm/q3_4b/marble_db --stream data/marble_db/test.jsonl \
                                   --solo outputs/eval/q3_4b/marble_db/solo --swarm outputs/eval/q3_4b/marble_db+own/swarm \
                                   --verdicts outputs/eval/q3_4b/marble_db+own/verdicts --model /models/Qwen3-4B

    python -m pipeline.swarm team  ... --agents '{"agent1": {"name": ..., "path": ..., "parser": ...}, ...}' --workers 5
                                   a team whose agents are different models: every distinct model served once, the workers share them

    python -m pipeline.swarm steps    --events outputs/swarm/q3_4b/marble_db --tasks ... --out data/marble_db_steps_q/test.jsonl
                                      a finished run's sub-steps as yes/no questions for a pool of peers (500 events from 100 tasks)
    python -m pipeline.swarm diagnose --tasks ... --events ... --stream data/marble_db_steps/test.jsonl --eval tilt=<dir> ... --out <json>
                                      each task's diagnosis from its five sub-step verdicts, scored by the benchmark's rule

`run` starts vLLM's OpenAI server for the model on the visible GPU and a user-space PostgreSQL if none listens on --pg-port,
then for every task it claims (the shards share the task list, in file order, and each takes the next free one through
<out>/claims/, so the work balances; a stopped shard releases its unfinished claims when it restarts): reset the database, run the scenario's schema and benign queries,
inject the anomaly workloads, run the benchmark's swarm (feedback_state.swarm.run_swarm), collect the agents' findings, and
run the central model alone (run_solo). Every task is one line of <out>/[shard<k>/]events.jsonl; a task already there is
skipped, so a run resumes. `merge` reads the events and writes the stream (the five findings as the candidate answers), the
question-alone evaluation the `own` step reads, and the `swarm` and `verdicts` evaluations, all in the pipeline's layout.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

from pipeline.config import shown


def load_tasks(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open()]


def done_ids(events: Path) -> set[str]:
    if not events.exists():
        return set()
    return {json.loads(line)["id"] for line in events.open() if line.strip()}


def all_done_ids(root: Path) -> set[str]:
    """The tasks any shard of this run has finished (every events.jsonl under root)."""
    ids = set()
    for f in [root / "events.jsonl"] + [Path(p) for p in glob.glob(str(root / "shard*" / "events.jsonl"))]:
        ids |= done_ids(f)
    return ids


def claim(claims: Path, task_id: str, who: str) -> bool:
    """Take a task for this shard: an atomic create of <claims>/<task>.claim. Shards share the task list and claim the next
    free task, so the work balances however long the tasks take."""
    try:
        with (claims / f"{task_id}.claim").open("x") as f:
            f.write(who)
        return True
    except FileExistsError:
        return False


def release_stale_claims(claims: Path, who: str, done: set[str]) -> list[str]:
    """A claim of this shard without an event is from a run that was stopped mid-task: give the task back."""
    released = []
    for c in sorted(claims.glob("*.claim")):
        if c.read_text().strip() == who and c.stem not in done:
            c.unlink()
            released.append(c.stem)
    return released


def cmd_run(args) -> None:
    from feedback_state.swarm import MarbleDB, ensure_postgres, graded, make_env, patch_llm, run_solo, run_swarm, serve_vllm

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    who = f"shard{args.shard.split('/')[0]}" if args.shard else "single"
    out = args.out / (who if args.shard else "")
    out.mkdir(parents=True, exist_ok=True)
    events = out / "events.jsonl"
    log_file = (out / "run.log").open("a")

    def log(msg: str) -> None:
        line = f"[swarm {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n"); log_file.flush()

    claims = args.out / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    mine, done = done_ids(events), all_done_ids(args.out)
    released = release_stale_claims(claims, who, mine)
    todo = [t for t in tasks if t["id"] not in done]
    log(f"{len(tasks)} tasks in this run, {len(done)} done ({len(mine)} by {who}), {len(todo)} to claim"
        + (f"; released {released}" if released else ""))
    if not todo:
        (out / "complete.json").write_text(json.dumps({"tasks": len(tasks)}))
        return
    ensure_postgres(args.pg_port, args.pg_data)
    db = MarbleDB(args.pg_port, read_only=bool(args.teams))   # teams share each injected database, so their agents only read it
    proc = None
    routes = json.loads(args.routes) if args.routes else None        # a team: served name -> base URL, servers started by `team`
    if routes:
        api_base = routes[args.served_name]
    elif not args.api_base:
        proc = serve_vllm(args.model, args.served_name, args.port, out / "vllm.log", gpu_memory_utilization=args.gpu_memory_utilization,
                          max_model_len=args.max_model_len, tool_parser=args.tool_parser)
        api_base = f"http://localhost:{args.port}/v1"
    else:
        api_base = args.api_base
    llm = f"openai/{args.served_name}"
    agent_llms = {a: f"openai/{name}" for a, name in json.loads(args.agent_models).items()} if args.agent_models else None
    teams = {n: {"llm": f"openai/{n}", "reasoning": bool(v.get("reasoning"))} for n, v in json.loads(args.teams).items()} if args.teams else None
    patch_llm(api_base, thinking=False, routes=routes, force_tool=args.force_tool,
              reasoning_models=tuple(n for n, v in (teams or {}).items() if v["reasoning"]))
    env = make_env(db)
    try:
        ran = 0
        for task in todo:
            if not claim(claims, task["id"], who):
                continue                                   # another shard has it
            ran += 1
            t0 = time.time()
            log(f"task {task['id']} ({ran} by {who}; {len(all_done_ids(args.out))}/{len(tasks)} done): root causes {task['root_causes']}")
            db.reset()
            db.initialise(task["init_sql"])
            inject = db.inject(task["anomalies"], args.anomaly_duration)
            log(f"    injected {[x['anomaly'] for x in task['anomalies']]} in {sum(x['seconds'] for x in inject):.0f}s")
            t1 = time.time()
            if teams:   # a pool of peers on every sub-step: one team per model, all on this injected database at once
                from feedback_state.swarm import run_pool

                pool = run_pool(task, llm, teams, lambda: make_env(db), log)
                for res in pool.values():
                    res.update(graded(res["final"], task))
                t2 = time.time()
                row = {"id": task["id"], "scenario": task["scenario"], "root_causes": task["root_causes"], "labels": task["labels"],
                       "number_of_labels_pred": task["number_of_labels_pred"], "anomalies": [x["anomaly"] for x in task["anomalies"]],
                       "injection": inject, "teams": pool, "model": args.served_name, "seconds": {"inject": round(t1 - t0, 1), "pool": round(t2 - t1, 1)}}
                with events.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                log("    " + " | ".join(f"{n}: hit {r['correct']} findings {sum(x['correct'] for x in r['findings'])}/5" for n, r in pool.items())
                    + f" | {t2 - t0:.0f}s")
                continue
            swarm = run_swarm(task, llm, env, log, agent_llms)
            swarm.update(graded(swarm["final"], task))
            t2 = time.time()
            solo = run_solo(task, llm, env, args.iterations, log)
            solo.update(graded(solo["final"], task))
            t3 = time.time()
            row = {"id": task["id"], "scenario": task["scenario"], "root_causes": task["root_causes"], "labels": task["labels"],
                   "number_of_labels_pred": task["number_of_labels_pred"], "anomalies": [x["anomaly"] for x in task["anomalies"]],
                   "injection": inject, "swarm": swarm, "solo": solo, "model": args.served_name,
                   "seconds": {"inject": round(t1 - t0, 1), "swarm": round(t2 - t1, 1), "solo": round(t3 - t2, 1)}}
            with events.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"    swarm hit {swarm['correct']} exact {swarm['exact']} {swarm['predicted']} | findings "
                f"{[x['verdict'] for x in swarm['findings']]} right {sum(x['correct'] for x in swarm['findings'])}/5 | "
                f"solo hit {solo['correct']} exact {solo['exact']} {solo['predicted']} | {t3 - t0:.0f}s")
        (out / "complete.json").write_text(json.dumps({"tasks": len(tasks), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(30)
            except Exception:
                proc.kill()


def cmd_team(args) -> None:
    """A team whose agents are different models: serve every distinct model once (the planner's and the agents'), each with its
    tool-call parser, spread over the visible GPUs; run the workers (each `run --shard k/N` with its own PostgreSQL cluster)
    against those servers; stop the servers."""
    import subprocess
    import sys

    from feedback_state.swarm import start_vllm, wait_vllm

    agents = json.loads(args.agents) if args.agents else {}           # agent id -> {"name", "path", "parser"}: one model per role
    teams = json.loads(args.teams) if args.teams else {}              # name -> {"path", ...}: a pool, one team per model
    models = {args.served_name: {"path": args.model, "parser": args.tool_parser}}
    for a in list(agents.values()) + [dict(v, name=n) for n, v in teams.items()]:
        models.setdefault(a["name"], {"path": a["path"], "parser": a.get("parser", "hermes"), "env_vars": a.get("env_vars"),
                                      "prefix_caching": a.get("prefix_caching", True), "trust_remote_code": bool(a.get("trust_remote_code"))})
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g != ""] or ["0"]
    per_gpu = {g: 0 for g in gpus}
    for i, name in enumerate(models):
        models[name]["gpu"] = gpus[i % len(gpus)]; per_gpu[models[name]["gpu"]] += 1
    args.out.mkdir(parents=True, exist_ok=True)
    procs, routes = [], {}
    try:
        for i, (name, m) in enumerate(models.items()):
            port = args.port + i
            share = min(float(args.gpu_memory_utilization), 0.88 / per_gpu[m["gpu"]])
            m["port"], m["log"] = port, args.out / f"vllm_{name}.log"
            m["proc"] = start_vllm(m["path"], name, port, m["log"], gpu_memory_utilization=round(share, 2), max_model_len=args.max_model_len,
                                   tool_parser=m["parser"], gpu=m["gpu"], env_vars=m.get("env_vars"), prefix_caching=m.get("prefix_caching", True),
                                   trust_remote_code=m.get("trust_remote_code", False))
            procs.append(m["proc"]); routes[name] = f"http://localhost:{port}/v1"
            print(f"[swarm team] serving {name} (parser {m['parser']}) on GPU {m['gpu']} port {port}", flush=True)
        for name, m in models.items():
            wait_vllm(m["proc"], name, m["port"], m["log"])
        print(f"[swarm team] {len(models)} servers up; " + (f"teams {list(teams)}" if teams else f"agents {({a: v['name'] for a, v in agents.items()})}"), flush=True)
        common = [sys.executable, "-m", "pipeline.swarm", "run", "--tasks", str(args.tasks), "--model", str(args.model), "--served-name", args.served_name,
                  "--iterations", str(args.iterations), "--anomaly-duration", str(args.anomaly_duration), "--out", str(args.out),
                  "--routes", json.dumps(routes)]
        common += ["--teams", json.dumps({n: {"reasoning": bool(v.get("reasoning"))} for n, v in teams.items()})] if teams else \
                  ["--agent-models", json.dumps({a: v["name"] for a, v in agents.items()})]
        common += ["--force-tool", args.force_tool] if args.force_tool else []
        if args.limit:
            common += ["--limit", str(args.limit)]
        workers = []
        for k in range(args.workers):
            cmd = common + ["--pg-port", str(args.pg_port + k), "--pg-data", str(Path(args.pg_data) / str(args.pg_port + k))]
            cmd += ["--shard", f"{k}/{args.workers}"] if args.workers > 1 else []
            wlog = (args.out / f"worker{k}.log").open("a")
            workers.append(subprocess.Popen(cmd, stdout=wlog, stderr=subprocess.STDOUT))
        codes = [w.wait() for w in workers]
        print(f"[swarm team] workers exited {codes}", flush=True)
        if any(codes):
            raise SystemExit(f"a worker failed ({codes}); see {args.out}/worker*.log")
    finally:
        for pr in procs:
            pr.terminate()
        for pr in procs:
            try:
                pr.wait(30)
            except Exception:
                pr.kill()


def evidence_of(event: dict, agent_id: str, chars: int = 6000) -> str:
    """What one agent gathered on a task: its actions' results over the iterations (the model's text and the tool's result)."""
    parts = [f"[iteration {it['iteration']}] {res[agent_id]}" for it in event["swarm"]["iterations"] for res in it["results"] if agent_id in res]
    text = "\n".join(parts) or "(the agent ran no query on this task)"
    return text if len(text) <= chars else text[: chars // 2] + "\n ... \n" + text[-chars // 2:]


def load_events(root: Path) -> dict[str, dict]:
    rows = {}
    for f in sorted(glob.glob(str(root / "shard*" / "events.jsonl"))) or [str(root / "events.jsonl")]:
        if Path(f).exists():
            for line in open(f):
                if line.strip():
                    r = json.loads(line); rows[r["id"]] = r
    return rows


def cmd_steps(args) -> None:
    """The sub-steps of a finished swarm run as a stream of yes/no questions: one event per task and root cause, 'is X a root
    cause?', with the task and what the agent assigned to X gathered as the passage. A pool of peers answers each (pipeline.peers)
    and the verifier is the injected anomaly, so the record tracks every peer on the same question, as in the QA streams."""
    from feedback_state.swarm import cause_of

    events = load_events(args.events)
    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    missing = [t["id"] for t in tasks if t["id"] not in events]
    if missing:
        raise SystemExit(f"{len(missing)} tasks have no event under {args.events} (e.g. {missing[:3]}): run the swarm experiment first")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    n = yes = 0
    with tmp.open("w") as f:
        for t in tasks:
            ev = events[t["id"]]
            for a in t["agents"]:
                cause = cause_of(a["profile"])
                rec = {"id": f"{t['id']}::{cause}", "task_type": "boolqa", "source": t["scenario"], "task_id": t["id"], "cause": cause,
                       "agent_id": a["agent_id"],
                       # everything in the question itself: the central model's prompt shows a passage only for the reading task type
                       "problem": (f"{t['task'].strip()}\n\nThe agent assigned to {cause} investigated the database ({a['profile'].strip()}) "
                                   f"Its queries and their results:\n{evidence_of(ev, a['agent_id'])}\n\n"
                                   f"Question: is {cause} a root cause of this database's performance issue?"), "context": "",
                       "answer": "yes" if cause in t["root_causes"] else "no", "root_causes": list(t["root_causes"]),
                       "number_of_labels_pred": int(t["number_of_labels_pred"]), "labels": list(t["labels"]), "evidence_model": ev.get("model"),
                       "peer_responses": {}, "peer_correct": {}, "correctness_by_peer": {}, "peer_metadata": {}}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1; yes += rec["answer"] == "yes"
    tmp.replace(args.out)
    (args.out.parent / "manifest.json").write_text(json.dumps({"kind": "steps", "events": n, "yes": yes, "tasks": len(tasks), "source": shown(args.events)}, indent=1))
    print(f"[swarm] {n} sub-step questions ({yes} yes) from {len(tasks)} tasks of {shown(args.events)} -> {shown(args.out)}")


def cmd_diagnose(args) -> None:
    """A task's diagnosis from its five sub-step verdicts: the causes answered yes, in agent order, scored by the benchmark's rule,
    the exact set and the set F1, for every condition; beside the pool's majority, the benchmark's planner and the model alone."""
    from feedback_state.swarm import LABELS, graded, verdict_of
    from feedback_state.tasks import boolqa_extract_answer

    tasks = {t["id"]: t for t in load_tasks(args.tasks)}
    events = load_events(args.events)
    stream = [json.loads(line) for line in args.stream.open()]
    ids = [t for t in tasks if any(r["task_id"] == t for r in stream)]

    def f1(pred, truth):
        pred, truth = set(pred), set(truth)
        return 0.0 if not pred else 2 * len(pred & truth) / (len(pred) + len(truth))

    def score(name, note, answers: dict[str, list[str]], steps_right=None):
        rows = [graded("Final answer: " + (", ".join(answers.get(t, [])) or "NONE"), tasks[t]) for t in ids]
        out = {"condition": name, "note": note, "tasks": len(ids), "accuracy": sum(r["correct"] for r in rows) / len(ids),
               "exact_accuracy": sum(r["exact"] for r in rows) / len(ids),
               "set_f1": sum(f1(r["predicted"], tasks[t]["root_causes"]) for r, t in zip(rows, ids)) / len(ids),
               "guesses": sum(len(answers.get(t, [])) for t in ids) / len(ids)}
        if steps_right is not None:
            out["substep_accuracy"] = steps_right
        return out

    def from_verdicts(verdict: dict[str, str]) -> dict[str, list[str]]:
        return {t: [c for c in LABELS if verdict.get(f"{t}::{c}") == "yes"] for t in ids}

    results = []
    for spec in args.eval or []:
        name, directory = spec.split("=", 1)
        gens = [json.loads(line) for line in (Path(directory) / "generations.jsonl").open()]
        verdict = {g["id"]: boolqa_extract_answer(g.get("generation", "")) for g in gens}
        right = sum(g["correct"] for g in gens) / max(1, len(gens))
        results.append(score(name, f"the central model's sub-step verdicts ({shown(Path(directory))})", from_verdicts(verdict), right))
    gold = {r["id"]: r["answer"] for r in stream}
    maj = {}
    for r in stream:
        votes = [(verdict_of(x) or boolqa_extract_answer(x) or "").lower() for x in r["peer_responses"].values()]
        maj[r["id"]] = "yes" if votes.count("yes") > votes.count("no") else "no"
    results.append(score("pool majority", "per sub-step, the majority of the peers' verdicts (ties: no)", from_verdicts(maj),
                         sum(maj[i] == gold[i] for i in gold) / len(gold)))
    for k in sorted(stream[0]["peer_responses"], key=lambda x: int(x.split("_")[1])):
        one = {r["id"]: (verdict_of(r["peer_responses"][k]) or boolqa_extract_answer(r["peer_responses"][k]) or "").lower() for r in stream}
        results.append(score(f"{k}: {stream[0]['peer_metadata'][k]['model']}", "one peer's verdicts", from_verdicts(one),
                             sum(one[i] == gold[i] for i in gold) / len(gold)))
    if all("teams" in events[t] for t in ids):   # a pool run: every team is a swarm of its own (the planner's model, one model behind the agents)
        for team in events[ids[0]]["teams"]:
            results.append(score(f"MARBLE swarm, agents = {team}", "the benchmark's planner with that model behind its five agents",
                                 {t: graded(events[t]["teams"].get(team, {}).get("final", ""), tasks[t])["predicted"] for t in ids}))
    else:
        results.append(score("MARBLE swarm", "the benchmark's planner", {t: graded(events[t]["swarm"]["final"], tasks[t])["predicted"] for t in ids}))
        results.append(score("question alone (five queries)", "the central model investigating by itself", {t: graded(events[t]["solo"]["final"], tasks[t])["predicted"] for t in ids}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"stream": shown(args.stream), "events": shown(args.events), "results": results}, indent=1))
    lines = [f"# {args.title}: task-level diagnosis from the sub-step verdicts", "",
             "| condition | accuracy | exact set | set F1 | guesses | sub-step accuracy |", "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        ss = f"{100 * r['substep_accuracy']:.1f}" if "substep_accuracy" in r else "-"
        lines.append(f"| {r['condition']} | {100 * r['accuracy']:.1f} | {100 * r['exact_accuracy']:.1f} | {100 * r['set_f1']:.1f} | {r['guesses']:.2f} | {ss} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_merge_pool(args) -> None:
    """A pool run as a stream: one event per task and root cause ("is X a root cause?"), the peers' findings (each from its own
    investigation) as peer_0 ... in --peers order, labelled against the injected anomaly; the central model's own finding as the
    question-alone evaluation the `own` step reads; and how each team did as a swarm (teams.json)."""
    from feedback_state.swarm import LABELS, verdict_of
    from pipeline.evaluate import summarise
    from pipeline.streams import peer_text

    events = load_events(args.out)
    tasks = [t for t in load_tasks(args.tasks) if t["id"] in events]
    missing = [t["id"] for t in load_tasks(args.tasks) if t["id"] not in events]
    if not tasks:
        raise SystemExit(f"no events under {args.out}")
    if missing and not args.partial:
        raise SystemExit(f"{len(missing)} tasks have no event (e.g. {missing[:3]}); pass --partial to merge what is there")
    peers = args.peers.split(",")

    def finding(ev, team, cause):
        f = next((x for x in ev["teams"].get(team, {}).get("findings", []) if x["cause"] == cause), None)
        text = peer_text(f["text"]) if f else "(this peer's team did not finish the task)\nFinal answer: no"
        v = verdict_of(text)
        return text, v, f

    args.stream.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.stream.with_name(args.stream.name + ".tmp")
    own_rows, n = [], 0
    with tmp.open("w") as out:
        for t in tasks:
            ev = events[t["id"]]
            for cause in LABELS:
                gold = cause in t["root_causes"]
                rec = {"id": f"{t['id']}::{cause}", "task_type": "boolqa", "source": t["scenario"], "task_id": t["id"], "cause": cause,
                       "problem": f"{t['task'].strip()}\n\nQuestion: is {cause} a root cause of this database's performance issue?", "context": "",
                       "answer": "yes" if gold else "no", "root_causes": list(t["root_causes"]), "number_of_labels_pred": int(t["number_of_labels_pred"]),
                       "labels": list(t["labels"]), "peer_responses": {}, "peer_correct": {}, "correctness_by_peer": {}, "peer_metadata": {}}
                for k, team in enumerate(peers):
                    text, v, f = finding(ev, team, cause)
                    right = int(v is not None and (v == "YES") == gold)
                    rec["peer_responses"][f"peer_{k}"] = text
                    rec["peer_correct"][f"peer_{k}"] = float(right); rec["correctness_by_peer"][f"peer_{k}"] = right
                    rec["peer_metadata"][f"peer_{k}"] = {"model": team, "cause": cause, "verdict": v, "investigated": f is not None,
                                                         "received_context": True, "num_samples": 1}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                text, v, _ = finding(ev, args.served_name, cause)
                own_rows.append({"pos": n, "id": rec["id"], "task_type": "boolqa", "source": t["scenario"], "correct": int(v is not None and (v == "YES") == gold),
                                 "peer_correct": [rec["correctness_by_peer"][f"peer_{k}"] for k in range(len(peers))], "memory_prob": None, "generation": text})
                n += 1
    tmp.replace(args.stream)
    (args.stream.parent / "manifest.json").write_text(json.dumps({"kind": "pool", "events": n, "tasks": len(tasks), "missing": missing, "peers": peers,
                                                                   "central": args.served_name, "source": shown(args.out)}, indent=1))
    metrics = {"condition": "solo", "mode": "solo", "gamma": 0.0, "swap_record": False, "bias_form": None, "max_new_tokens": int(args.max_new_tokens),
               "engine": "vllm-openai", "central_model": args.model, "checkpoint": None, "record": None, "stream": shown(args.stream), "every": 1,
               "max_examples": None, "shard": None, "note": "the central model's own finding on every sub-step, from its own investigation"}
    metrics.update(summarise(own_rows, args.windows))
    args.solo.mkdir(parents=True, exist_ok=True)
    with (args.solo / "generations.jsonl").open("w") as f:
        for r in own_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.solo / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    teams = {}
    for team in [args.served_name] + peers:
        rs = [events[t["id"]]["teams"].get(team) for t in tasks]
        rs = [r for r in rs if r]
        fs = [x for r in rs for x in r["findings"]]
        teams[team] = {"tasks": len(rs), "swarm_accuracy": sum(r.get("correct", 0) for r in rs) / max(1, len(rs)),
                       "swarm_exact": sum(r.get("exact", 0) for r in rs) / max(1, len(rs)),
                       "findings_right": sum(x["correct"] for x in fs) / max(1, len(fs)), "yes_rate": sum(x["verdict"] == "YES" for x in fs) / max(1, len(fs)),
                       "no_verdict": sum(x["verdict"] is None for x in fs), "errors": sum(len(r.get("errors", [])) for r in rs)}
    (args.stream.parent / "teams.json").write_text(json.dumps(teams, indent=1))
    print(f"[swarm] pool stream {shown(args.stream)}: {n} events from {len(tasks)} tasks; own answer right {100 * metrics['accuracy']:.1f}%")
    for team, v in teams.items():
        print(f"[swarm]   {team:34s} swarm {100 * v['swarm_accuracy']:5.1f} | findings right {100 * v['findings_right']:5.1f} | says yes {100 * v['yes_rate']:5.1f}% | no verdict {v['no_verdict']}")


def cmd_merge(args) -> None:
    from feedback_state.swarm import stream_record, verdict_answer
    from pipeline.evaluate import summarise

    files = sorted(glob.glob(str(args.out / "shard*" / "events.jsonl"))) or [str(args.out / "events.jsonl")]
    rows = {}
    for f in files:
        if Path(f).exists():
            for line in open(f):
                if line.strip():
                    r = json.loads(line); rows[r["id"]] = r
    tasks = {t["id"]: t for t in load_tasks(args.tasks)}
    ids = [t for t in tasks if t in rows]
    missing = [t for t in tasks if t not in rows]
    if not ids:
        raise SystemExit(f"no events under {args.out}")
    if missing and not args.partial:
        raise SystemExit(f"{len(missing)} of {len(tasks)} tasks have no event (e.g. {missing[:3]}); pass --partial to merge what is there")
    args.stream.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.stream.with_name(args.stream.name + ".tmp")
    with tmp.open("w") as f:
        for t in ids:
            f.write(json.dumps(stream_record(rows[t], tasks[t], args.served_name), ensure_ascii=False) + "\n")
    tmp.replace(args.stream)
    (args.stream.parent / "manifest.json").write_text(json.dumps({"kind": "swarm", "events": len(ids), "missing": missing, "model": args.served_name,
                                                                   "sources": [shown(Path(f)) for f in files]}, indent=1))
    findings_right = [x["correct"] for t in ids for x in rows[t]["swarm"]["findings"]]
    print(f"[swarm] stream {shown(args.stream)}: {len(ids)} events; findings right {100 * sum(findings_right) / max(1, len(findings_right)):.1f}%")

    def write_eval(directory: Path, condition: str, mode: str, texts: dict[str, str], extra: dict | None = None) -> None:
        out_rows = []
        for pos, t in enumerate(ids):
            task = tasks[t]
            from feedback_state.swarm import graded

            g = graded(texts[t], task)
            out_rows.append({"pos": pos, "id": t, "task_type": "dbdiag", "source": task["scenario"], "correct": g["correct"], "exact": g["exact"],
                             "predicted": g["predicted"], "peer_correct": [x["correct"] for x in rows[t]["swarm"]["findings"]],
                             "memory_prob": None, "generation": texts[t]})
        metrics = {"condition": condition, "mode": mode, "gamma": 0.0, "swap_record": False, "bias_form": None,
                   "max_new_tokens": int(args.max_new_tokens), "engine": "vllm-openai", "central_model": args.model, "checkpoint": None,
                   "record": None, "stream": shown(args.stream), "every": 1, "max_examples": None, "shard": None}
        metrics.update(summarise(out_rows, args.windows))
        metrics["exact_accuracy"] = sum(r["exact"] for r in out_rows) / len(out_rows)
        metrics.update(extra or {})
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "generations.jsonl").open("w") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (directory / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
        print(f"[swarm] {condition}: accuracy {100 * metrics['accuracy']:.1f} (exact {100 * metrics['exact_accuracy']:.1f}) -> {shown(directory)}")

    write_eval(args.solo, "solo", "solo", {t: rows[t]["solo"]["final"] for t in ids})
    write_eval(args.swarm, "swarm", "swarm", {t: rows[t]["swarm"]["final"] for t in ids},
               {"iterations_mean": sum(len(rows[t]["swarm"]["iterations"]) for t in ids) / len(ids)})
    write_eval(args.verdicts, "verdicts", "swarm", {t: verdict_answer(rows[t]["swarm"]["findings"]) for t in ids})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tasks", type=Path, required=True)
    r.add_argument("--model", required=True, help="the model's directory (served by vLLM here unless --api-base)")
    r.add_argument("--served-name", required=True)
    r.add_argument("--api-base", default=None, help="an OpenAI-compatible server already serving the model")
    r.add_argument("--port", type=int, default=8123)
    r.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    r.add_argument("--max-model-len", type=int, default=16384)
    r.add_argument("--pg-port", type=int, default=5432)
    r.add_argument("--pg-data", type=Path, required=True, help="the PostgreSQL cluster's directory (created if missing)")
    r.add_argument("--iterations", type=int, default=5, help="the central model's tool calls when it investigates alone")
    r.add_argument("--anomaly-duration", type=int, default=60, help="seconds of each anomaly workload (the benchmark's default)")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--shard", default=None, help="k/N: tasks k, k+N, ...")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--tool-parser", default="hermes", help="vLLM's tool-call parser for the model")
    r.add_argument("--routes", default=None, help="JSON, served name -> base URL: the servers of a team (started by `team`)")
    r.add_argument("--agent-models", default=None, help="JSON, agent id -> served name: an agent's own model (default: the planner's)")
    r.add_argument("--teams", default=None, help="JSON, served name -> {reasoning}: a pool of peers, one team (all five agents) per model")
    r.add_argument("--force-tool", default=None, help="an agent's action is this tool, asked for as one SQL statement in a code block and parsed here: every model can query (e.g. query_db)")
    t = sub.add_parser("team")
    for a, kw in (("--tasks", dict(type=Path, required=True)), ("--model", dict(required=True)), ("--served-name", dict(required=True)),
                  ("--tool-parser", dict(default="hermes")), ("--agents", dict(default=None, help="JSON, agent id -> {name, path, parser}")),
                  ("--teams", dict(default=None, help="JSON, served name -> {path, parser, reasoning, env_vars, prefix_caching, trust_remote_code}")),
                  ("--force-tool", dict(default=None)),
                  ("--workers", dict(type=int, default=4)), ("--port", dict(type=int, default=8170)), ("--pg-port", dict(type=int, default=5460)),
                  ("--pg-data", dict(type=Path, required=True)), ("--iterations", dict(type=int, default=5)),
                  ("--anomaly-duration", dict(type=int, default=60)), ("--gpu-memory-utilization", dict(type=float, default=0.6)),
                  ("--max-model-len", dict(type=int, default=16384)), ("--limit", dict(type=int, default=None)), ("--out", dict(type=Path, required=True))):
        t.add_argument(a, **kw)
    mp = sub.add_parser("merge-pool")
    for a, kw in (("--tasks", dict(type=Path, required=True)), ("--out", dict(type=Path, required=True)), ("--stream", dict(type=Path, required=True)),
                  ("--solo", dict(type=Path, required=True)), ("--peers", dict(required=True, help="the pool's served names, in peer_0, peer_1, ... order")),
                  ("--model", dict(required=True)), ("--served-name", dict(required=True)), ("--max-new-tokens", dict(type=int, default=768)),
                  ("--windows", dict(type=int, default=10))):
        mp.add_argument(a, **kw)
    mp.add_argument("--partial", action="store_true")
    st = sub.add_parser("steps")
    st.add_argument("--events", type=Path, required=True, help="a finished swarm run: the directory holding shard*/events.jsonl")
    st.add_argument("--tasks", type=Path, required=True)
    st.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    st.add_argument("--out", type=Path, required=True)
    dg = sub.add_parser("diagnose")
    dg.add_argument("--tasks", type=Path, required=True)
    dg.add_argument("--events", type=Path, required=True)
    dg.add_argument("--stream", type=Path, required=True)
    dg.add_argument("--eval", action="append", default=None, help="condition=directory (repeatable)")
    dg.add_argument("--title", default="swarm steps")
    dg.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--tasks", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--stream", type=Path, required=True)
    m.add_argument("--solo", type=Path, required=True)
    m.add_argument("--swarm", type=Path, required=True)
    m.add_argument("--verdicts", type=Path, required=True)
    m.add_argument("--model", required=True)
    m.add_argument("--served-name", required=True)
    m.add_argument("--max-new-tokens", type=int, default=768)
    m.add_argument("--windows", type=int, default=10)
    m.add_argument("--partial", action="store_true")
    args = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    {"run": cmd_run, "team": cmd_team, "merge": cmd_merge, "steps": cmd_steps, "diagnose": cmd_diagnose, "merge-pool": cmd_merge_pool}[args.cmd](args)


if __name__ == "__main__":
    main()
