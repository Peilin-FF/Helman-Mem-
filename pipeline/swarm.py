"""Swarm: run MultiAgentBench's database diagnosis with a local model, and turn it into a stream with evaluations.

    python -m pipeline.swarm run   --tasks datasets/marble_db/tasks.jsonl --model /models/Qwen3-4B --served-name Qwen3-4B --port 8123 \
                                   --pg-port 5432 --pg-data /mnt/data/peilin/pg/5432 --out outputs/swarm/q3_4b/marble_db [--shard k/N] [--limit N]
    python -m pipeline.swarm merge --out outputs/swarm/q3_4b/marble_db --stream data/marble_db/test.jsonl \
                                   --solo outputs/eval/q3_4b/marble_db/solo --swarm outputs/eval/q3_4b/marble_db+own/swarm \
                                   --verdicts outputs/eval/q3_4b/marble_db+own/verdicts --model /models/Qwen3-4B

    python -m pipeline.swarm team  ... --agents '{"agent1": {"name": ..., "path": ..., "parser": ...}, ...}' --workers 5
                                   a team whose agents are different models: every distinct model served once, the workers share them

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
    db = MarbleDB(args.pg_port)
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
    patch_llm(api_base, thinking=False, routes=routes)
    llm = f"openai/{args.served_name}"
    agent_llms = {a: f"openai/{name}" for a, name in json.loads(args.agent_models).items()} if args.agent_models else None
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

    agents = json.loads(args.agents)                                  # agent id -> {"name", "path", "parser"}
    models = {args.served_name: {"path": args.model, "parser": args.tool_parser}}
    for a in agents.values():
        models.setdefault(a["name"], {"path": a["path"], "parser": a.get("parser", "hermes")})
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
                                   tool_parser=m["parser"], gpu=m["gpu"])
            procs.append(m["proc"]); routes[name] = f"http://localhost:{port}/v1"
            print(f"[swarm team] serving {name} (parser {m['parser']}) on GPU {m['gpu']} port {port}", flush=True)
        for name, m in models.items():
            wait_vllm(m["proc"], name, m["port"], m["log"])
        print(f"[swarm team] {len(models)} servers up; agents {({a: v['name'] for a, v in agents.items()})}", flush=True)
        common = [sys.executable, "-m", "pipeline.swarm", "run", "--tasks", str(args.tasks), "--model", str(args.model), "--served-name", args.served_name,
                  "--iterations", str(args.iterations), "--anomaly-duration", str(args.anomaly_duration), "--out", str(args.out),
                  "--routes", json.dumps(routes), "--agent-models", json.dumps({a: v["name"] for a, v in agents.items()})]
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
    t = sub.add_parser("team")
    for a, kw in (("--tasks", dict(type=Path, required=True)), ("--model", dict(required=True)), ("--served-name", dict(required=True)),
                  ("--tool-parser", dict(default="hermes")), ("--agents", dict(required=True, help="JSON, agent id -> {name, path, parser}")),
                  ("--workers", dict(type=int, default=4)), ("--port", dict(type=int, default=8170)), ("--pg-port", dict(type=int, default=5460)),
                  ("--pg-data", dict(type=Path, required=True)), ("--iterations", dict(type=int, default=5)),
                  ("--anomaly-duration", dict(type=int, default=60)), ("--gpu-memory-utilization", dict(type=float, default=0.6)),
                  ("--max-model-len", dict(type=int, default=16384)), ("--limit", dict(type=int, default=None)), ("--out", dict(type=Path, required=True))):
        t.add_argument(a, **kw)
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
    {"run": cmd_run, "team": cmd_team, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
