"""ClassEval with a pool of peers on every method (docs/experiments/classeval.md).

    setup     the grading environment's data: the pinned ClassEval_data.json and nltk's corpora
    build     ClassEval -> the question stream: one event per method, classes in a seeded order, methods in the benchmark's
              order. Every gold method is run through its own hidden tests here, and methods whose gold fails in this
              environment are dropped (listed in manifest.json); each event keeps the strictest visible check its gold method
              passes (doctest, run, load). The online run takes its class order, visible checks and drops from it; offline
              (the teacher-forced ablation) its events are answered as they are, each with the gold methods before it
    report    per condition and per peer: method accuracy, class pass (every method of a class right), and the record against
              reliability tables (per peer; per peer and class; per peer and library; per peer and dependency kind), each read
              before write along the stream
    online    the swarm (the main experiment): the classes built for real, the central model committing every method, the
              peers working against the committed class, every method verified as soon as it is done and its labels written
              into the record before the next method is read (feedback_state.classeval_online); one run holds every online
              condition as a track, and serves the peers with vLLM, one GPU each

    PYTHONPATH=. FEEDBACK_CODE_EXEC_ALLOW=1 python -m pipeline.classeval build --out data/classeval_q/test.jsonl
    PYTHONPATH=. python -m pipeline.classeval report --stream data/classeval6+q3_4b/test.jsonl \
        --record outputs/record/q3_4b/classeval6+own/fixed.fit-self.qc-d64-lam100.jsonl \
        --eval solo=outputs/eval/q3_4b/classeval6/solo --eval tilt=outputs/eval/q3_4b/classeval6+own/tilt --out outputs/tables/classeval_q3_4b.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pipeline.config import shown

NLTK_DATA = ("punkt", "punkt_tab", "wordnet", "omw-1.4", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "stopwords")


def class_order(n: int, order: str) -> list[int]:
    """'fixed' = the benchmark's order; 'shuffledK' = numpy.random.default_rng(K).permutation(n) of the classes."""
    return list(range(n)) if order == "fixed" else [int(i) for i in np.random.default_rng(int(order.replace("shuffled", ""))).permutation(n)]


# --- setup / build ----------------------------------------------------------------------------------------------------------
def cmd_setup(args) -> None:
    from feedback_state.classeval import fetch

    print(f"[classeval] data: {fetch(args.source)}")
    import nltk

    for name in NLTK_DATA:
        ok = nltk.download(name, quiet=True)
        print(f"[classeval] nltk {name}: {'ok' if ok else 'FAILED'}")


def cmd_build(args) -> None:
    from feedback_state.classeval import events_of, fetch, load_classes, validate

    classes = load_classes(fetch(args.source))
    order = class_order(len(classes), args.class_order)
    events = [ev for ci in order for ev in events_of(classes[ci])]
    if args.limit:
        events = events[: args.limit]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        checks = list(ex.map(validate, events))
    kept, dropped = [], []
    for ev, chk in zip(events, checks):
        if chk["hidden_ok"]:
            ev["classeval"]["visible"] = chk["visible"]
            kept.append(ev)
        else:
            dropped.append({"id": ev["id"], "error": chk["hidden_error"][-300:]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    with tmp.open("w") as f:
        for ev in kept:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    visible = collections.Counter(ev["classeval"]["visible"] for ev in kept)
    classes_kept = len({ev["classeval"]["task_id"] for ev in kept})
    complete = len({ev["classeval"]["task_id"] for ev in kept} - {d["id"].split("/")[0] for d in dropped})
    manifest = {"kind": "classeval", "source": shown(args.source), "class_order": args.class_order, "events_in": len(events),
                "events": len(kept), "classes": classes_kept, "classes_complete": complete, "visible": dict(visible),
                "dropped": dropped, "seconds": round(time.time() - t0, 1)}
    (args.out.parent / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[classeval] {len(kept)}/{len(events)} events ({classes_kept} classes, {complete} complete), visible checks {dict(visible)}, "
          f"dropped {len(dropped)} whose gold fails here -> {args.out}", flush=True)
    for d in dropped:
        print(f"[classeval]   dropped {d['id']}: {d['error'].strip().splitlines()[-1] if d['error'].strip() else ''}")


# --- report -------------------------------------------------------------------------------------------------------------------
def reliability(rows: list[dict], info: dict[str, dict]) -> dict:
    """The record against reliability tables (feedback_state.reliability_tables), on the six peers: per peer; per peer and class
    (the benchmark's task, which the record is never told); per peer and the method's libraries; per peer and its dependency
    kind (standalone / other methods / fields only)."""
    from feedback_state.reliability_tables import compare

    def dep(r):
        return info[r["id"]].get("dependencies") or {}

    return compare(rows, {"per peer and class": lambda r, s: info[r["id"]]["task_id"],
                          "per peer and library": lambda r, s: ",".join(sorted(dep(r).get("lib_dependencies") or [])) or "none",
                          "per peer and dependency kind": lambda r, s: "standalone" if dep(r).get("Standalone")
                          else "methods" if dep(r).get("method_dependencies") else "fields"})


def cmd_report(args) -> None:
    from feedback_state.data import JsonlDataset
    from feedback_state.reliability_tables import markdown

    stream = {str(r["id"]): r for r in JsonlDataset(args.stream).records}
    info = {i: r["classeval"] for i, r in stream.items()}
    by_class = collections.defaultdict(list)
    for i, c in info.items():
        by_class[c["task_id"]].append(i)
    complete = {t: ids for t, ids in by_class.items() if len(ids) == info[ids[0]]["steps"]}

    def class_pass(correct: dict[str, int]) -> float:
        done = [t for t, ids in complete.items() if all(i in correct for i in ids)]
        return float(np.mean([all(correct[i] for i in complete[t]) for t in done])) if done else float("nan")

    res = {"stream": shown(args.stream), "events": len(stream), "classes": len(by_class), "classes_complete": len(complete), "conditions": {}, "peers": {}}
    for spec in args.eval or []:
        name, directory = spec.split("=", 1)
        path = Path(directory) / "generations.jsonl"
        if not path.exists():
            print(f"[classeval] no {path}: {name} skipped")
            continue
        rows = [json.loads(line) for line in path.open()]
        gens = {str(g["id"]): int(g["correct"]) for g in rows}
        res["conditions"][name] = {"methods": float(np.mean(list(gens.values()))), "n": len(gens), "class_pass": class_pass(gens)}
        m = json.load(open(Path(directory) / "eval_metrics.json")) if (Path(directory) / "eval_metrics.json").exists() else {}
        if m.get("mode") == "online":   # built for real: the committed class's own tests, and each method inside it
            res["conditions"][name].update(class_pass=m["class_pass"], in_context=m["in_context_accuracy"], online=True)
            if rows and "memory_prob" in rows[0]:
                res.setdefault("reliability_online", {})[name] = reliability(rows, info)
    keys = sorted(next(iter(stream.values()))["peer_responses"], key=lambda k: int(k.split("_")[1]))
    for k in keys:
        labels = {i: int(round(float((r.get("correctness_by_peer") or r.get("peer_correct") or {}).get(k, 0)))) for i, r in stream.items()}
        model = str((next(iter(stream.values())).get("peer_metadata") or {}).get(k, {}).get("model", k)).split("/")[-1]
        res["peers"][k] = {"model": model, "methods": float(np.mean(list(labels.values()))), "class_pass": class_pass(labels)}
    if args.record and args.record.exists():
        rows = [json.loads(l) for l in args.record.open()]
        res["reliability"] = reliability(rows, info)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    md = [f"# {args.title or 'ClassEval'}", "", f"{res['events']} methods in {res['classes']} classes ({res['classes_complete']} complete). "
          "Methods: each answer on its own method in the gold class. Classes: offline, every method of the class right; online, the "
          "committed class passing all its tests. In context (online): the committed method inside the committed class.", "",
          "| answer | methods | classes | in context |", "|---|---:|---:|---:|"]
    for name, v in res["conditions"].items():
        md.append(f"| {name} | {100 * v['methods']:.1f} | {100 * v['class_pass']:.1f} | {100 * v['in_context']:.1f} |" if v.get("online")
                  else f"| {name} | {100 * v['methods']:.1f} | {100 * v['class_pass']:.1f} | - |")
    for k, v in res["peers"].items():
        md.append(f"| {k} {v['model']} | {100 * v['methods']:.1f} | {100 * v['class_pass']:.1f} | - |")
    for title, rel in [("the offline record", res.get("reliability"))] + [(f"the record of {n}", r) for n, r in res.get("reliability_online", {}).items()]:
        if rel:
            md += markdown(title, rel)
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# --- online -----------------------------------------------------------------------------------------------------------------
def chat(base: str, model: str, messages: list[dict], *, max_tokens: int, temperature: float, top_p: float, seed: int,
         timeout: float = 600.0, retries: int = 2) -> str:
    """One chat completion from a vLLM OpenAI server (thinking off where the template knows the switch); '' on failure."""
    import urllib.error
    import urllib.request

    body = json.dumps({"model": model, "messages": messages, "max_tokens": int(max_tokens), "temperature": float(temperature),
                       "top_p": float(top_p), "seed": int(seed), "chat_template_kwargs": {"enable_thinking": False}}).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(f"{base}/chat/completions", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as e:   # e.g. the conversation outgrew the context: no retry helps
            print(f"[online] {model}: HTTP {e.code} {e.read().decode(errors='replace')[:200]}", flush=True)
            return ""
        except Exception as e:
            if attempt == retries:
                print(f"[online] {model}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                return ""
            time.sleep(5)
    return ""


def cmd_online(args) -> None:
    from feedback_state import classeval as ce
    from feedback_state.classeval_online import ClassEvalAdapter
    from feedback_state.data import JsonlDataset
    from feedback_state.memory_generator import strip_thinking
    from feedback_state.online_central import central_engine, make_tracks, projections, save_tracks, serve, stop
    from feedback_state.online_swarm import run_online
    from feedback_state.peer_generation import agentic_answers

    peers = json.loads(args.peers)
    questions = JsonlDataset(args.questions).records
    visible = {str(q["id"]): q["classeval"].get("visible", "load") for q in questions}
    kept = collections.defaultdict(set)
    for q in questions:
        kept[q["classeval"]["task_id"]].add(q["classeval"]["method_name"])
    source = {c["task_id"]: c for c in ce.load_classes(ce.fetch(args.source))}
    order = list(dict.fromkeys(q["classeval"]["task_id"] for q in questions))
    classes = [source[t] for t in order if kept[t] == {m["method_name"] for m in source[t]["methods_info"]}]
    skipped = [t for t in order if source[t] not in classes]
    if args.limit_classes:
        classes = classes[: args.limit_classes]
    print(f"[online] {len(classes)} classes ({sum(len(c['methods_info']) for c in classes)} methods); skipped {len(skipped)} with a method "
          f"whose gold fails here: {skipped}", flush=True)

    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g != ""] or ["0"]
    # the peers one GPU each after the first, which holds the central model's engine and its judge (about 0.12 more)
    servers = serve(peers, gpus[1:] or gpus[:1], args.port, args.out_root, args.max_model_len, {gpus[0]: args.gpu_memory_utilization + 0.12})
    try:
        central_fn, features_fn = central_engine(args.model, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
                                                 max_new_tokens=args.max_new_tokens, bias_form=args.bias_form,
                                                 judge_max_length=args.judge_max_length, num_answers=len(peers) + 1)
        proj_q, proj_c = projections(args.fit_stream, args.fit_features, args.fit_peers or len(peers) + 1, args.dim)
        turn_seed = [0]

        def peers_fn(events):
            def one(server):
                p = next(x for x in peers if x["name"] == server["name"])
                base, reasoning = f"http://localhost:{server['port']}/v1", bool(p.get("reasoning"))

                def generate(items):
                    turn_seed[0] += 1
                    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                        return list(ex.map(lambda it: chat(base, p["name"], it[1], max_tokens=4096 if reasoning else args.peer_max_tokens,
                                                           temperature=args.temperature, top_p=args.top_p, seed=turn_seed[0]), items))

                return agentic_answers(events, generate, ce.visible_check, args.turns, (lambda t: strip_thinking(t)) if reasoning else (lambda t: t),
                                       first_prompt=ce.peer_prompt, workers=args.workers)

            with ThreadPoolExecutor(max_workers=len(servers)) as ex:
                per_peer = list(ex.map(one, servers))
            return [[per_peer[p][e] for p in range(len(servers))] for e in range(len(events))]

        tracks = make_tracks(args.conditions, proj_q, proj_c, len(peers) + 1, design=args.design, lam=args.lam, dim=args.dim,
                             warmup=args.warmup, prior=args.prior, line_lam=args.line_lam, seed=args.seed)
        adapter = ClassEvalAdapter(classes, lanes=args.lanes, peers_fn=peers_fn, central_fn=central_fn,
                                   grade_fn=lambda ev, text, overrides: ce.hidden_test(ev, text, overrides).passed,
                                   class_test_fn=lambda ev, committed: ce.class_test(ev, committed).passed, visible=visible, num_peers=len(peers))
        t0 = time.time()
        run_online(adapter, tracks, central_fn=central_fn, features_fn=features_fn, workers=args.workers)
        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        save_tracks(tracks, args.out_root, adapter, central_model=args.model, stream=shown(args.questions), peers=[p["name"] for p in peers],
                    seconds=round(time.time() - t0), skipped_classes=skipped, config=config)
    finally:
        stop(servers)


def main(argv=None) -> None:
    from feedback_state.classeval import DEFAULT_SOURCE

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    b = sub.add_parser("build")
    b.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE), help="ClassEval_data.json (downloaded, pinned, if missing)")
    b.add_argument("--class-order", default="shuffled0", help="fixed | shuffledK: the order of the classes in the stream")
    b.add_argument("--limit", type=int, default=None, help="only the first N events (a smoke run)")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--stream", type=Path, required=True, help="the stream the record read (with the own answer if it had one)")
    r.add_argument("--record", type=Path, default=None)
    r.add_argument("--eval", action="append", default=None, help="name=evaluation directory (repeatable)")
    r.add_argument("--title", default=None)
    r.add_argument("--out", type=Path, required=True, help="<name>.json (and <name>.md)")
    o = sub.add_parser("online")
    o.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    o.add_argument("--questions", type=Path, required=True, help="the built question stream: class order, visible checks, dropped methods")
    o.add_argument("--model", required=True, help="the central model's directory (answers and judge)")
    o.add_argument("--peers", required=True, help="JSON list of {name, path, reasoning, env_vars, prefix_caching, trust_remote_code}")
    o.add_argument("--conditions", required=True, help='JSON {name: {kind: solo|peers|tilt|combination, gamma}}')
    o.add_argument("--warmup", type=int, default=64, help="the record's addresses are fit (label-free) on the judge features of the first "
                   "N methods of each record track; their labels are written once it is fit, their reads are the cold record")
    o.add_argument("--fit-stream", type=Path, default=None, help="instead of the warmup: fit the addresses on this stream's features")
    o.add_argument("--fit-features", type=Path, default=None)
    o.add_argument("--fit-peers", type=int, default=None, help="answers per event in --fit-stream (default: the online run's)")
    o.add_argument("--out-root", type=Path, required=True, help="each condition's results go to <out-root>/<condition>/")
    o.add_argument("--design", default="qc")
    o.add_argument("--dim", type=int, default=64)
    o.add_argument("--lam", type=float, default=100.0)
    o.add_argument("--prior", default="0.5,0.0")
    o.add_argument("--line-lam", type=float, default=1.0)
    o.add_argument("--bias-form", default="logratio")
    o.add_argument("--lanes", type=int, default=1, help="classes built at once per track (1: every method's labels are written before "
                   "the next method is read; L: the record sees an event up to L - 1 events later, about L times faster)")
    o.add_argument("--turns", type=int, default=3, help="a peer's answers per method, revising on the visible check")
    o.add_argument("--temperature", type=float, default=0.2)
    o.add_argument("--top-p", type=float, default=0.95)
    o.add_argument("--peer-max-tokens", type=int, default=1024)
    o.add_argument("--max-new-tokens", type=int, default=1024, help="the central model's answers")
    o.add_argument("--max-model-len", type=int, default=16384)
    o.add_argument("--judge-max-length", type=int, default=12288)
    o.add_argument("--gpu-memory-utilization", type=float, default=0.5, help="the central model's vLLM engine (the judge shares its GPU)")
    o.add_argument("--port", type=int, default=8300)
    o.add_argument("--concurrency", type=int, default=32, help="requests in flight per peer server")
    o.add_argument("--workers", type=int, default=16, help="programs run at once (visible checks and tests)")
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--limit-classes", type=int, default=None)
    args = ap.parse_args(argv)
    {"setup": cmd_setup, "build": cmd_build, "report": cmd_report, "online": cmd_online}[args.cmd](args)


if __name__ == "__main__":
    main()
