"""The online swarm on MultiAgentBench's database diagnosis (docs/experiments/marble_db_online.md).

    online   one planner (the central model, MARBLE's EnginePlanner) per track; every root cause it assigns is investigated by the
             six peers (MARBLE agents on their own models, querying the injected database) and, on record tracks, by the central
             model itself; the central model reads the findings (tilted on record tracks) and commits the cause's verdict, which
             the planner reads; every finding and verdict is labelled against the injected anomaly at once and written into the
             record before the next iteration is read (feedback_state.marble_online, feedback_state.online_swarm)
    report   per track: the verdicts' accuracy (balanced too), the planner's diagnosis and the verdicts' own (the benchmark's
             hit rule, exact set, set F1), the peers, and every record against reliability tables (per peer; per peer and cause;
             per peer and its verdict; per peer, cause and verdict; per peer and scenario)

    PYTHONPATH=. python -m pipeline.marble_online online --tasks datasets/marble_db/tasks.jsonl --model /models/Qwen3-4B \
        --served-name Qwen3-4B --peers '[{"name": ..., "path": ..., "reasoning": false}, ...]' \
        --conditions '{"online_tilt": {"kind": "tilt", "gamma": 3.0}}' --fit-stream ... --fit-features ... --fit-peers 6 \
        --pg-data /mnt/data/peilin/pg --out-root outputs/eval/q3_4b/marble_db_swarm

The job serves the peers (vLLM's OpenAI server, one GPU each) and the central model for its MARBLE roles (the planner and its
own investigations, over litellm), runs the central model in-process for its readings (vLLM with the tilt) and its judge
(transformers), and a user-space PostgreSQL cluster per lane (feedback_state.swarm.ensure_postgres).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np

from pipeline.config import shown


def cmd_online(args) -> None:
    from feedback_state.marble_online import MarbleDBAdapter, MarblePlanner, investigate_marble
    from feedback_state.online_central import central_engine, make_tracks, projections, save_tracks, serve, stop
    from feedback_state.online_swarm import run_online
    from feedback_state.swarm import MarbleDB, ensure_postgres, make_env, patch_llm
    from pipeline.swarm import load_tasks

    tasks = load_tasks(args.tasks)
    tasks = tasks[: args.limit] if args.limit else tasks
    peers = json.loads(args.peers)
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g != ""] or ["0"]
    # GPU 0: the central model in-process (its readings) and its judge; the peers one GPU each after it; the central model's
    # server (its MARBLE roles) on the next GPU, or on GPU 0 beside the in-process engine when there is none
    peer_gpus = gpus[1:1 + len(peers)] or gpus[:1]
    central_gpu = gpus[1 + len(peers)] if len(gpus) > 1 + len(peers) else gpus[0]
    reserved = {gpus[0]: args.gpu_memory_utilization + 0.12}
    servers = []
    try:
        servers += serve(peers, peer_gpus, args.port, args.out_root, args.max_model_len, reserved)
        servers += serve([{"name": args.served_name, "path": args.model}], [central_gpu], args.central_port, args.out_root, args.max_model_len,
                         reserved if central_gpu == gpus[0] else {})
        routes = {s["name"]: f"http://localhost:{s['port']}/v1" for s in servers}
        patch_llm(routes[args.served_name], thinking=False, routes=routes, force_tool="query_db",
                  reasoning_models=tuple(p["name"] for p in peers if p.get("reasoning")))
        dbs = []
        for lane in range(args.lanes):
            port = args.pg_port + lane
            ensure_postgres(port, args.pg_data / str(port))
            dbs.append(MarbleDB(port))
        central_fn, features_fn = central_engine(args.model, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
                                                 max_new_tokens=args.max_new_tokens, bias_form=args.bias_form,
                                                 judge_max_length=args.judge_max_length, num_answers=len(peers) + 1)
        proj_q, proj_c = projections(args.fit_stream, args.fit_features, args.fit_peers or len(peers), args.dim)
        reasoning = {p["name"]: bool(p.get("reasoning")) for p in peers}

        def prepare(lane, task):
            t0 = time.time()
            db = dbs[lane]
            db.reset()
            db.initialise(task["init_sql"])
            injected = db.inject(task["anomalies"], args.anomaly_duration)
            return {"anomalies": injected, "seconds": round(time.time() - t0, 1)}

        adapter = MarbleDBAdapter(tasks, peers=[p["name"] for p in peers], own_model=args.served_name, prepare=prepare,
                                  planner_for=lambda track, lane, task: MarblePlanner(task, f"openai/{args.served_name}", make_env(dbs[lane])),
                                  investigate=lambda lane, model, task, agent, assignment: investigate_marble(
                                      make_env(dbs[lane]), f"openai/{model}", task, agent, assignment, args.queries, reasoning.get(model, False)),
                                  lanes=args.lanes, workers=args.workers)
        tracks = make_tracks(args.conditions, proj_q, proj_c, len(peers) + 1, design=args.design, lam=args.lam, dim=args.dim,
                             warmup=args.warmup, prior=args.prior, line_lam=args.line_lam, seed=args.seed)
        t0 = time.time()
        run_online(adapter, tracks, central_fn=central_fn, features_fn=features_fn, workers=args.workers)
        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        save_tracks(tracks, args.out_root, adapter, central_model=args.model, stream=shown(args.tasks), peers=[p["name"] for p in peers],
                    seconds=round(time.time() - t0), config=config)
    finally:
        stop(servers)


CELLS = {
    "per peer and cause": lambda r, s: r["cause"],
    "per peer and its verdict": lambda r, s: r["peer_verdicts"][r["peer_order"][s]],
    "per peer, cause and its verdict": lambda r, s: (r["cause"], r["peer_verdicts"][r["peer_order"][s]]),
    "per peer and scenario": lambda r, s: r["source"],
}


def cmd_report(args) -> None:
    from feedback_state.reliability_tables import compare, markdown

    res = {"tasks": shown(args.tasks), "conditions": {}, "peers": {}, "reliability": {}}
    pooled = collections.defaultdict(lambda: {"right": [], "yes": [], "no_verdict": []})
    names = None
    for spec in args.eval or []:
        name, directory = spec.split("=", 1)
        d = Path(directory)
        if not (d / "eval_metrics.json").exists():
            print(f"[marble] no {d / 'eval_metrics.json'}: {name} skipped")
            continue
        m = json.load(open(d / "eval_metrics.json"))
        rows = [json.loads(line) for line in (d / "generations.jsonl").open()]
        res["conditions"][name] = {k: m.get(k) for k in ("kind", "accuracy", "balanced_accuracy", "yes_recall", "no_recall", "tasks", "planner",
                                                           "verdict_diagnosis", "forced_share", "own_accuracy", "combination")}
        names = names or m.get("peers")
        for r in rows:
            for p, (ok, v) in enumerate(zip(r.get("peer_correct_by_peer", []), r.get("peer_verdicts", []))):
                pooled[p]["right"].append(ok); pooled[p]["yes"].append(v == "YES"); pooled[p]["no_verdict"].append(v is None)
        if rows and "memory_prob" in rows[0] and "peer_verdicts" in rows[0]:
            res["reliability"][name] = compare(rows, CELLS)
    for p, v in sorted(pooled.items()):
        res["peers"][(names or [])[p] if names and p < len(names) else f"peer_{p}"] = {
            "right": float(np.mean(v["right"])), "says_yes": float(np.mean(v["yes"])), "no_verdict": float(np.mean(v["no_verdict"])), "n": len(v["right"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    pct = lambda x: "-" if x is None else f"{100 * x:.1f}"
    md = [f"# {args.title or 'MultiAgentBench database, online swarm'}", "",
          "Verdicts: each cause's committed verdict against the injected anomaly (balanced: the mean of its recall on true and on "
          "false causes). Diagnosis: a task's predicted causes, by the benchmark's rule (a true cause among the allowed guesses), "
          "exactly, and by set F1; the planner's final decision, and the causes the track said yes to.", "",
          "| track | verdicts right | balanced | planner: hit | exact | set F1 | verdicts: hit | exact | set F1 | tasks |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, v in res["conditions"].items():
        p, q = v.get("planner") or {}, v.get("verdict_diagnosis") or {}
        md.append(f"| {name} | {pct(v.get('accuracy'))} | {pct(v.get('balanced_accuracy'))} | {pct(p.get('accuracy'))} | {pct(p.get('exact'))} | "
                  f"{pct(p.get('set_f1'))} | {pct(q.get('accuracy'))} | {pct(q.get('exact'))} | {pct(q.get('set_f1'))} | {v.get('tasks')} |")
    if res["peers"]:
        md += ["", "The peers' findings over every track (right: the verdict matches the injected anomaly):", "",
               "| peer | right | says yes | no verdict |", "|---|---:|---:|---:|"]
        md += [f"| {k} | {pct(v['right'])} | {pct(v['says_yes'])} | {pct(v['no_verdict'])} |" for k, v in res["peers"].items()]
    for name, rel in res["reliability"].items():
        md += markdown(f"the record of {name}", rel)
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("online")
    o.add_argument("--tasks", type=Path, required=True, help="the benchmark's tasks (datasets/marble_db/tasks.jsonl)")
    o.add_argument("--model", required=True, help="the central model's directory")
    o.add_argument("--served-name", required=True, help="the central model's name on its server (its MARBLE roles)")
    o.add_argument("--peers", required=True, help="JSON list of {name, path, reasoning, env_vars, prefix_caching, trust_remote_code}")
    o.add_argument("--conditions", required=True, help='JSON {name: {kind: solo|peers|tilt|combination, gamma}}')
    o.add_argument("--out-root", type=Path, required=True, help="each condition's results go to <out-root>/<condition>/")
    o.add_argument("--warmup", type=int, default=64, help="without --fit-stream: the record's addresses fit on each record track's first N events")
    o.add_argument("--fit-stream", type=Path, default=None, help="the stream whose judge features fit the addresses (label-free)")
    o.add_argument("--fit-features", type=Path, default=None)
    o.add_argument("--fit-peers", type=int, default=None, help="answers per event in --fit-stream")
    o.add_argument("--design", default="qc")
    o.add_argument("--dim", type=int, default=256)
    o.add_argument("--lam", type=float, default=100.0)
    o.add_argument("--prior", default="0.5,0.0")
    o.add_argument("--line-lam", type=float, default=1.0)
    o.add_argument("--bias-form", default="logratio")
    o.add_argument("--lanes", type=int, default=1, help="tasks at once, each on its own PostgreSQL cluster (1: every iteration's labels are "
                   "in the record before the next iteration is read)")
    o.add_argument("--queries", type=int, default=3, help="an investigator's queries per cause")
    o.add_argument("--port", type=int, default=8400, help="the peers' servers: port ... port + 5")
    o.add_argument("--central-port", type=int, default=8410)
    o.add_argument("--pg-port", type=int, default=5490, help="+ lane")
    o.add_argument("--pg-data", type=Path, required=True, help="the PostgreSQL clusters go to <pg-data>/<port>")
    o.add_argument("--anomaly-duration", type=int, default=60, help="seconds of each anomaly workload (the benchmark's default)")
    o.add_argument("--gpu-memory-utilization", type=float, default=0.45, help="the central model's in-process engine (the judge shares its GPU)")
    o.add_argument("--max-model-len", type=int, default=16384)
    o.add_argument("--max-new-tokens", type=int, default=768, help="the central model's readings")
    o.add_argument("--judge-max-length", type=int, default=12288)
    o.add_argument("--workers", type=int, default=48, help="investigations at once")
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    r = sub.add_parser("report")
    r.add_argument("--tasks", type=Path, required=True)
    r.add_argument("--eval", action="append", default=None, help="name=evaluation directory (repeatable)")
    r.add_argument("--title", default=None)
    r.add_argument("--out", type=Path, required=True, help="<name>.json (and <name>.md)")
    args = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    {"online": cmd_online, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
