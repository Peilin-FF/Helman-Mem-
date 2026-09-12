"""Run the family pipeline from training/configs/families.yaml (README_families.md, docs section 23).

    bash run_families.sh [--models tag ...] [--steps features prompts eval table] [--gpus 0,1,2,3] [--smoke]

Per model, in the YAML's run order: features (the family's model as the frozen judge on the train and test streams, one
shard per GPU), prompts (PCA addresses fit on the train stream, the record along each test stream read-before-write,
then its quality against the labels), eval (the conditions on every test stream with vLLM, one evaluation per GPU), and
the comparison table at the end.  Every task is skipped when its output is already on disk, so an interrupted run is
resumed by running the same command again.  Exit code 1 if any task failed (the log to read is printed).
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

TEST_DIR = {"indist6": "indist", "ood6": "oodfull"}   # output-directory names, the ones families_table.py reads
MODE = {"tilt": ("peers", "--attn_gamma {gamma}"), "peers": ("peers", ""), "solo": ("solo", "")}


def stamp() -> str:
    return time.strftime("%H:%M:%S")


class Pool:
    """Runs shell commands on the given GPUs, one command per GPU at a time (CUDA_VISIBLE_DEVICES set per command)."""

    def __init__(self, gpus: list[int]):
        self.q: queue.Queue = queue.Queue()
        for g in gpus:
            self.q.put(g)
        self.n = len(gpus)
        self.failed: list[str] = []

    def run_all(self, tasks: list[tuple[str, str, Path, Path | None]]) -> None:
        def one(t):
            name, cmd, log, done = t
            if done is not None and done.exists():
                print(f"[{stamp()}] skip  {name} (already on disk)", flush=True)
                return
            g = self.q.get()
            try:
                print(f"[{stamp()}] start {name} on GPU {g}", flush=True)
                log.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                with open(log, "w") as f:
                    rc = subprocess.call(cmd, shell=True, executable="/bin/bash", stdout=f, stderr=subprocess.STDOUT, env=env)
                ok = rc == 0 and (done is None or done.exists())
                print(f"[{stamp()}] {'done ' if ok else 'FAILED'} {name}" + ("" if ok else f" (see {log})"), flush=True)
                if not ok:
                    self.failed.append(name)
            finally:
                self.q.put(g)

        with ThreadPoolExecutor(max_workers=self.n) as ex:
            list(ex.map(one, tasks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/configs/families.yaml")
    ap.add_argument("--models", nargs="*", help="tags to run (default: the YAML's run list)")
    ap.add_argument("--steps", nargs="*", help="features prompts eval table (default: the YAML's steps)")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: the YAML's gpus)")
    ap.add_argument("--smoke", action="store_true", help="the whole pipeline on a few events under the tag <tag><suffix>")
    ap.add_argument("--smoke-suffix", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    os.environ["SIGMA_FEATURE_ROOT"] = str(cfg["outputs"]["features"])   # read by feedback_state.feature_streams in the subprocesses
    from feedback_state.feature_streams import DATA_ROOT, FEATURE_ROOT, MODELS, STREAMS

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else [int(g) for g in cfg["gpus"]]
    models = args.models or list(cfg["run"])
    steps = args.steps or list(cfg["steps"])
    fe, mem, ev, out, sm = cfg["features"], cfg["memory"], cfg["evaluation"], cfg["outputs"], cfg.get("smoke", {})
    train, tests = str(cfg["streams"].get("fit", "train6")), list(cfg["streams"]["test"])
    fit_self = train == "self"   # PCA addresses fit on each test stream's own features (label-free): the train stream is not encoded
    suffix = args.smoke_suffix or sm.get("suffix", "_smoke")
    if args.smoke:
        tests = [t for t in tests if t == "indist6"] or tests[:1]
    what = "smoke" if args.smoke else "full"
    for m in models:
        if m not in cfg["models"]:
            sys.exit(f"unknown model tag {m!r}; the YAML knows {list(cfg['models'])}")
        if (m + (suffix if args.smoke else "")) not in MODELS and m not in MODELS:
            print(f"note: tag {m} is not in feedback_state.feature_streams.MODELS (only a label; the caches are named by the tag)")
    pool = Pool(gpus)
    L = Path("logs"); L.mkdir(exist_ok=True)
    print(f"[{stamp()}] {what}: models {models}, steps {steps}, GPUs {gpus}, streams train={train} test={tests}", flush=True)

    root = Path(cfg.get("models_root", "."))
    for base in models:
        spec = cfg["models"][base]
        spec = {"path": spec} if isinstance(spec, str) else dict(spec)
        path = Path(spec["path"])
        path = path if path.is_absolute() else root / path
        if not (path / "config.json").exists():
            sys.exit(f"model directory {path} has no config.json (models_root / models in the YAML)")
        engine = str(spec.get("engine", ev.get("engine", "vllm")))
        shards_eval = 1 if args.smoke else int(spec.get("shards", 1 if engine == "vllm" else len(gpus)))
        env = spec.get("env")   # a conda env for models the default env cannot load (Qwen3.5 needs transformers 5)
        pre = f'eval "$(conda shell.bash hook)" && conda activate {env} && ' if env else ""
        tag = base + (suffix if args.smoke else "")
        print(f"===== {tag}: {path}  engine {engine}{f' (env {env})' if env else ''}  {stamp()}", flush=True)
        if "features" in steps:
            for st in ([] if fit_self else [train]) + tests:
                jsonl, cache = STREAMS[st]
                cache_dir, inp = FEATURE_ROOT / cache.format(m=tag), DATA_ROOT / jsonl
                shards = 1 if args.smoke else len(gpus)
                tasks = []
                for k in range(shards):
                    outp = cache_dir / f"shard{k}.pt"
                    cmd = (f"{pre}python scripts/encode_context_features.py --input {inp} --output {outp} --central-model {path} --include-context --save-peer-hidden "
                           f"--num-peers {fe['num_peers']} --max-length {fe['max_length']} --dtype {fe['dtype']} --progress-every {fe.get('progress_every', 500)} "
                           f"--num-shards {shards} --shard-index {k}" + (f" --max-examples {int(sm.get('events', 48))}" if args.smoke else ""))
                    tasks.append((f"features {tag} {st} shard {k}/{shards}", cmd, L / f"enc_{tag}_{st}_{k}.out", outp))
                pool.run_all(tasks)
        if "prompts" in steps:
            gen = Path(out["prompts"]) / tag
            gen.mkdir(parents=True, exist_ok=True)
            dim = int(sm.get("dim", 32)) if args.smoke else int(mem["dim"])
            for st in tests:
                pf = gen / f"prompts_{st}_probe.jsonl"
                cmd = (f"{pre}python scripts/build_generation_prompts.py --model {tag} --fit-stream {st if fit_self else train} --stream {st} --order {mem['order']} "
                       f"--design {mem['design']} --dim {dim} --lam {mem['lam']} --out {pf} "
                       f"&& python scripts/record_quality.py --prompts {pf} --out {gen / f'record_{st}.json'}")
                pool.run_all([(f"prompts {tag} {st} (record + quality)", cmd, L / f"prompts_{tag}_{st}.out", pf)])
        if "eval" in steps:
            tasks, merges = [], []
            for st in tests:
                pf, rec = Path(out["prompts"]) / tag / f"prompts_{st}_probe.jsonl", DATA_ROOT / STREAMS[st][0]
                if not pf.exists():
                    print(f"[{stamp()}] FAILED eval {tag} {st}: no prompt file {pf} (run the prompts step first)", flush=True)
                    pool.failed.append(f"eval {tag} {st}")
                    continue
                for cond in ev["conditions"]:
                    mode, extra = MODE[cond]
                    od = Path(out["evals"]) / tag / f"{what}_{TEST_DIR[st]}6_{cond}"
                    common = (f"{pre}python -m tests.experiments.common.evaluate_memory_generator --central_model {path} --engine {engine} "
                              f"--thinking {ev['thinking']} --max_new_tokens {ev['max_new_tokens']} --prompts {pf} --records {rec} --mode {mode} "
                              f"{extra.format(gamma=ev['gamma'])}")
                    if engine == "vllm":
                        tasks.append((f"eval {tag} {st} {cond}", f"{common} --gpu_memory_utilization {ev['gpu_memory_utilization']} --output {od}",
                                      L / f"fam_{tag}_{what}_{TEST_DIR[st]}_{cond}.out", od / "eval_metrics.json"))
                    else:   # HF engine (the tilt through the attention hook): one shard per GPU, joined afterwards
                        if (od / "eval_metrics.json").exists():
                            print(f"[{stamp()}] skip  eval {tag} {st} {cond} (already on disk)", flush=True)
                            continue
                        for k in range(shards_eval):
                            tasks.append((f"eval {tag} {st} {cond} shard {k}/{shards_eval}",
                                          f"{common} --batch_size {int(spec.get('batch_size', 8))} --shard {k}/{shards_eval} --output {od / f'shard{k}'}",
                                          L / f"fam_{tag}_{what}_{TEST_DIR[st]}_{cond}_shard{k}.out", od / f"shard{k}" / "eval_metrics.json"))
                        merges.append((f"merge {tag} {st} {cond}", f"python scripts/merge_eval_shards.py --out {od}", L / f"fam_{tag}_{what}_{TEST_DIR[st]}_{cond}.out", od / "eval_metrics.json"))
            pool.run_all(tasks)
            for name, cmd, log, done in merges:   # CPU work, after every shard of the model is in
                if done.exists():
                    continue
                with open(log, "w") as f:
                    rc = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
                print(f"[{stamp()}] {'done ' if rc == 0 and done.exists() else 'FAILED'} {name}" + ("" if rc == 0 else f" (see {log})"), flush=True)
                if rc != 0:
                    pool.failed.append(name)
    if "table" in steps or "eval" in steps:
        subprocess.call(f"python scripts/families_table.py {'--smoke' if args.smoke else '--by_task'}", shell=True)
    if pool.failed:
        print(f"[{stamp()}] FAILED tasks: {pool.failed}", flush=True)
        sys.exit(1)
    print(f"[{stamp()}] FAMILIES_RUN_COMPLETE ({what})", flush=True)


if __name__ == "__main__":
    main()
