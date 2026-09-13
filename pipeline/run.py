"""Run an experiment: expand its YAML into jobs, schedule them on the GPUs, skip what is already done.

    bash run.sh configs/experiments/main.yaml                     every step of the experiment
    bash run.sh configs/experiments/main.yaml --steps evaluate    one step
    bash run.sh configs/experiments/misleading.yaml --smoke       48 events of one dataset, every step, into outputs/smoke/
    bash run.sh configs/experiments/main.yaml --dry-run           print the jobs and whether each is done
    bash run.sh configs/experiments/main.yaml --set evaluation.max_new_tokens=1024 --gpus 4,5,6,7

Steps run in the order peers -> streams -> features -> record -> train -> evaluate -> table; the jobs of a step run in
parallel, one GPU each (or as many as a training job asks for). A job is skipped when its output exists; a sharded
output counts only once all its shards finished (a complete.json is written then); an evaluation is skipped only if the
stored settings match the requested ones, and a mismatch stops the job instead of silently reusing the old result.
The resolved config and every command go to outputs/runs/<experiment>/, logs to logs/<experiment>/.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pipeline.config import load, shown
from pipeline.layout import Layout

ORDER = ["peers", "streams", "features", "record", "train", "evaluate", "table"]


def stamp() -> str:
    return time.strftime("%H:%M:%S")


@dataclass
class Job:
    step: str
    name: str
    cmd: str
    gpus: int = 1
    done: Path | None = None              # the output whose existence means the job is finished
    group: Path | None = None             # a sharded output directory: complete.json there means every shard is done
    group_size: int = 0
    check: object = None                  # callable() -> str | None: a reason the stored result cannot be reused
    env: dict = field(default_factory=dict)
    wave: int = 0                         # within a step, waves run in order; the jobs of a wave run in parallel


# --- expansion ------------------------------------------------------------------------------------------------------------
class Plan:
    def __init__(self, cfg: dict, cfg_path: str, smoke: bool, gpus: list[int]):
        self.cfg, self.cfg_path, self.smoke, self.gpus = cfg, cfg_path, smoke, gpus
        self.L = Layout(cfg, smoke)
        self.exp = cfg["name"]
        sm = cfg.get("smoke", {})
        self.events = int(sm.get("events", 48)) if smoke else None
        asked = self.L.registry.expand(cfg.get("datasets", []))
        self.datasets = self.L.registry.expand(sm["datasets"]) if smoke and "datasets" in sm else asked[:1] if smoke else asked
        self.shards = 1 if smoke else len(gpus)

    def eval_datasets(self) -> list[str]:
        """The datasets the central model is run on (answers datasets only feed the streams)."""
        return [d for d in self.datasets if self.L.stream(d)["kind"] != "answers"]

    def answer_datasets(self) -> list[str]:
        """The answers datasets to generate: those named, and those the named misleading streams are built from."""
        out = []
        for d in self.datasets:
            spec = self.L.stream(d)
            a = d if spec["kind"] == "answers" else spec.get("answers") if spec["kind"] == "misleading" else None
            if a and a not in out:
                out.append(a)
        return out

    def fit(self) -> str:
        return str(self.cfg.get("record", {}).get("fit", "self"))

    def model_env(self, spec: dict) -> str:
        return f'eval "$(conda shell.bash hook)" && conda activate {spec["conda_env"]} && ' if spec.get("conda_env") else ""

    # the steps
    def peers(self) -> list[Job]:
        gen = self.cfg.get("generation", {})
        mis = gen.get("misleading", {})
        jobs = []
        for a in self.answer_datasets():
            answers = self.L.stream(a)
            stream, mode = self.L.stream(answers["base"]), answers["mode"]
            for p in self.L.peer_models(stream["peer_names"]):
                out = answers["path"] / p["name"]
                for k in range(self.shards):
                    cmd = (f"python -m pipeline.peers --mode {mode} --model {p['path']} --stream {stream['path']} --output {out} "
                           f"--shards {self.shards} --shard {k} --max-model-len {gen.get('max_model_len', 8192)} "
                           f"--gpu-memory-utilization {gen.get('gpu_memory_utilization', 0.85)} --grade-workers {gen.get('grade_workers', 8)}")
                    if mode == "misleading":
                        cmd += f" --max-attempts {mis.get('max_attempts', 3)} --temperatures {mis.get('temperatures', '0.2,0.7,1.0')}"
                        cmd += "" if mis.get("force", True) else " --no-force"
                    cmd += " --reasoning" if p.get("reasoning") else ""
                    cmd += " --no-prefix-caching" if p.get("prefix_caching") is False else ""
                    cmd += " --trust-remote-code" if p.get("trust_remote_code") else ""
                    cmd += f" --max-examples {self.events}" if self.events else ""
                    jobs.append(Job("peers", f"peers_{a}_{p['name']}_{k}", cmd, done=out / f"shard{k}of{self.shards}.jsonl",
                                    group=out, group_size=self.shards, env={str(x): str(y) for x, y in (p.get("env_vars") or {}).items()}))
        return jobs

    def streams(self) -> list[Job]:
        jobs = []
        for d in self.eval_datasets():
            spec = self.L.stream(d)
            if spec["kind"] != "misleading":
                continue
            jobs.append(Job("streams", f"stream_{d}", streams_command(self.L, d, self.events), gpus=0, done=spec["path"]))
        return jobs

    def feature_streams(self) -> list[str]:
        return ([self.fit()] if self.fit() != "self" else []) + self.eval_datasets()

    def features(self) -> list[Job]:
        fe = self.cfg.get("features", {})
        jobs = []
        for m in self.cfg.get("central", []):
            spec = self.L.model(m)
            for s in self.feature_streams():
                stream, out = self.L.stream(s), self.L.features_dir(m, s)
                for k in range(self.shards):
                    cmd = (f"{self.model_env(spec)}python -m pipeline.features --stream {stream['path']} --model {spec['path']} "
                           f"--output {out}/shard{k}of{self.shards}.pt --shards {self.shards} --shard {k} --peers {stream['peers']} "
                           f"--max-length {fe.get('max_length', 8192)} --dtype {fe.get('dtype', 'bfloat16')}"
                           + (f" --max-examples {self.events}" if self.events else ""))
                    jobs.append(Job("features", f"features_{m}_{s}_{k}", cmd, done=out / f"shard{k}of{self.shards}.pt", group=out, group_size=self.shards))
        return jobs

    def record(self) -> list[Job]:
        rec = self.cfg.get("record", {})
        dim = int(self.cfg.get("smoke", {}).get("dim", 32)) if self.smoke else int(rec.get("dim", 256))
        jobs = []
        for m in self.cfg.get("central", []):
            spec = self.L.model(m)
            for s in self.eval_datasets():
                stream, out = self.L.stream(s), self.L.record_file(m, s)
                fit = ""
                if self.fit() != "self":
                    fit = f" --fit-stream {self.L.stream(self.fit())['path']} --fit-features {self.L.features_dir(m, self.fit())}"
                cmd = (f"{self.model_env(spec)}python -m pipeline.record --stream {stream['path']} --features {self.L.features_dir(m, s)}{fit} "
                       f"--peers {stream['peers']} --order {rec.get('order', 'shuffled0')} --design {rec.get('design', 'qc')} "
                       f"--dim {dim} --lam {rec.get('lam', 100.0)} --out {out}")
                jobs.append(Job("record", f"record_{m}_{s}", cmd, done=out))
        return jobs

    def eval_jobs(self, model_key: str, model_spec: dict, record_model: str, streams: list[str], checkpoint: Path | None = None) -> list[Job]:
        ev, conds = self.cfg.get("evaluation", {}), self.cfg.get("conditions", {})
        jobs, seen = [], set()
        for d in streams:
            for c in self.cfg.get("eval_conditions", list(conds)):
                if c not in conds:
                    raise SystemExit(f"condition {c!r} is not defined under conditions:")
                cd = conds[c]
                s = self.L.eval_dataset(d, cd)            # question only: the base stream's result, shared by its variants
                out = self.L.eval_dir(model_key, s, c)
                if out in seen:
                    continue
                seen.add(out)
                # the record fixes which events are evaluated, in which order; any record of the same events will do for solo
                stream, record = self.L.stream(s), self.L.record_file(record_model, d)
                engine = model_spec.get("engine", ev.get("engine", "vllm"))
                args = (f"--model {model_spec['path']} --record {record} --stream {stream['path']} --condition {c} --mode {cd.get('mode', 'peers')} "
                        f"--gamma {float(cd.get('gamma', 0.0))}{' --swap' if cd.get('swap') else ''} --bias-form {ev.get('bias_form', 'logratio')} "
                        f"--engine {engine} --max-new-tokens {ev.get('max_new_tokens', 768)} "
                        f"--gpu-memory-utilization {ev.get('gpu_memory_utilization', 0.85)}"
                        + (f" --checkpoint {checkpoint}" if checkpoint else ""))
                want = {"mode": cd.get("mode", "peers"), "gamma": float(cd.get("gamma", 0.0)), "swap_record": bool(cd.get("swap", False)),
                        "max_new_tokens": int(ev.get("max_new_tokens", 768)), "checkpoint": str(checkpoint) if checkpoint else None}
                if cd.get("mode", "peers") != "solo":           # the answers never see the record without peers
                    want["record"] = str(record)
                check = lambda out=out, want=want: stale(out / "eval_metrics.json", want)
                pre = self.model_env(model_spec)
                if engine == "vllm":
                    jobs.append(Job("evaluate", f"eval_{model_key}_{s}_{c}", f"{pre}python -m pipeline.evaluate {args} --output {out}",
                                    done=out / "eval_metrics.json", check=check))
                else:   # HF engine: one shard per GPU, merged when all are in
                    n = 1 if self.smoke else int(model_spec.get("shards", len(self.gpus)))
                    for k in range(n):
                        jobs.append(Job("evaluate", f"eval_{model_key}_{s}_{c}_{k}",
                                        f"{pre}python -m pipeline.evaluate {args} --batch-size {int(model_spec.get('batch_size', 8))} --shard {k}/{n} --output {out}/shard{k}",
                                        done=out / f"shard{k}" / "eval_metrics.json", check=check))
                    jobs.append(Job("evaluate", f"merge_{model_key}_{s}_{c}", f"python -m pipeline.evaluate --merge --output {out}",
                                    gpus=0, done=out / "eval_metrics.json", check=check, wave=1))
        return jobs

    def evaluate(self) -> list[Job]:
        if self.cfg.get("arms"):
            from pipeline.train import evaluate_runs

            return evaluate_runs(self)
        jobs = []
        for m in self.cfg.get("central", []):
            jobs += self.eval_jobs(m, self.L.model(m), m, self.eval_datasets())
        return jobs

    def train(self) -> list[Job]:
        from pipeline.train import train_jobs

        return train_jobs(self)

    def table(self) -> list[Job]:
        resolved = self.L.run_dir(self.exp) / "resolved.yaml"
        return [Job("table", "table", f"python -m pipeline.table --resolved {resolved}" + (" --smoke" if self.smoke else ""), gpus=0)]


def streams_command(L: Layout, name: str, limit: int | None = None) -> str:
    """The pipeline.streams command that builds a registered misleading stream."""
    spec = L.stream(name)
    regime = spec["regime"]
    if isinstance(regime, str):
        from feedback_state.adversarial import adhoc_spec

        label, regime = regime, adhoc_spec(regime)
    else:
        label = name
    answers = L.stream(spec["answers"])
    dirs = ",".join(p["name"] for p in L.peer_models(spec["peer_names"]))
    return (f"python -m pipeline.streams replace --base {L.stream(spec['base'])['path']} --answers {answers['path']} --peer-dirs {dirs} "
            f"--regime {label} --regime-spec {shlex.quote(json.dumps(regime))} --order {spec.get('order', 'shuffled0')} --out {spec['path']}"
            + (" --drop-forced" if spec.get("drop_forced") else "") + (f" --limit {limit}" if limit else ""))


def stale(metrics: Path, want: dict) -> str | None:
    if not metrics.exists():
        return None
    have = json.load(open(metrics))
    diff = {k: (have.get(k), v) for k, v in want.items() if k in have and have.get(k) != v}
    for k in ("record", "checkpoint", "stream"):   # paths: equal if they name the same place in the project, from any snapshot
        if k in diff and None not in diff[k] and shown(diff[k][0]) == shown(diff[k][1]):
            diff.pop(k)
    return None if not diff else "stored results used other settings: " + ", ".join(f"{k} {a!r} (asked {b!r})" for k, (a, b) in diff.items())


# --- scheduling -----------------------------------------------------------------------------------------------------------
class Scheduler:
    def __init__(self, gpus: list[int], exp: str, layout: Layout, dry: bool, retries: int = 1):
        self.retries = retries             # a failed job runs again this many times (a killed process, a busy GPU) before it counts as failed
        self.free = list(gpus)
        self.total = len(gpus)
        self.cv = threading.Condition()
        self.exp, self.L, self.dry = exp, layout, dry
        self.failed: list[str] = []
        self.commands = layout.run_dir(exp) / "commands.log"

    def finished(self, job: Job) -> bool:
        if job.group is not None and (job.group / "complete.json").exists():
            return True
        return job.done is not None and job.done.exists() and job.group is None

    def run_step(self, jobs: list[Job]) -> None:
        todo = []
        for j in jobs:
            reason = j.check() if j.check else None
            if reason:
                print(f"[{stamp()}] STALE {j.name}: {reason}; rename the condition or remove {j.done.parent}", flush=True)
                self.failed.append(j.name)
            elif self.finished(j) or (j.done is not None and j.done.exists() and j.group is not None):
                print(f"[{stamp()}] skip  {j.name}", flush=True)
            else:
                todo.append(j)
        if self.dry:
            for j in todo:
                print(f"[dry-run] {j.name} ({j.gpus} GPU{'s' if j.gpus != 1 else ''}):\n    {j.cmd}", flush=True)
            return
        with ThreadPoolExecutor(max_workers=max(1, len(todo))) as ex:
            list(ex.map(self.run, todo))
        groups: dict[Path, list[Job]] = {}
        for j in jobs:
            if j.group is not None:
                groups.setdefault(j.group, []).append(j)
        for g, members in groups.items():
            if all(m.done.exists() for m in members) and not (g / "complete.json").exists() and not any(m.name in self.failed for m in members):
                (g / "complete.json").write_text(json.dumps({"shards": len(members), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}))

    def run(self, job: Job) -> None:
        need = min(job.gpus, self.total)
        with self.cv:
            while len(self.free) < need:
                self.cv.wait()
            got = [self.free.pop(0) for _ in range(need)]
        try:
            log = self.L.log(self.exp, job.name)
            log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, **job.env)
            if need:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, got))
            print(f"[{stamp()}] start {job.name}" + (f" on GPU {','.join(map(str, got))}" if need else ""), flush=True)
            with open(self.commands, "a") as c:
                c.write(f"# {stamp()} {job.name}\n{' '.join(f'{k}={v}' for k, v in job.env.items())} {job.cmd}\n")
            for attempt in range(1 + self.retries):
                with open(log, "w" if attempt == 0 else "a") as f:
                    if attempt:
                        f.write(f"\n# ---- attempt {attempt + 1} ----\n")
                        f.flush()
                    rc = subprocess.call(job.cmd, shell=True, executable="/bin/bash", stdout=f, stderr=subprocess.STDOUT, env=env)
                ok = rc == 0 and (job.done is None or job.done.exists())
                if ok or attempt == self.retries:
                    break
                print(f"[{stamp()}] retry {job.name} (exit {rc}{', killed by a signal' if rc > 128 or rc < 0 else ''})", flush=True)
            print(f"[{stamp()}] {'done ' if ok else 'FAILED'} {job.name}" + ("" if ok else f" (exit {rc}; see {log})"), flush=True)
            if not ok:
                self.failed.append(job.name)
        finally:
            with self.cv:
                self.free.extend(got)
                self.cv.notify_all()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--steps", nargs="*", help=f"a subset of {ORDER} (default: the config's steps)")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: the config's gpus)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="dotted.key=value (YAML value)")
    args = ap.parse_args(argv)
    cfg = load(args.config, args.overrides)
    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else [int(g) for g in cfg.get("gpus", [0])]
    steps = args.steps or cfg.get("steps", [])
    unknown = [s for s in steps if s not in ORDER]
    if unknown:
        sys.exit(f"unknown step(s) {unknown}; steps are {ORDER}")
    plan = Plan(cfg, args.config, args.smoke, gpus)
    plan.dry = args.dry_run
    run_dir = plan.L.run_dir(plan.exp)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved.yaml").write_text(yaml.safe_dump({**cfg, "_smoke": args.smoke, "_gpus": gpus}, sort_keys=False))
    sched = Scheduler(gpus, plan.exp, plan.L, args.dry_run, retries=int(cfg.get("retries", 1)))
    print(f"[{stamp()}] {plan.exp}{' (smoke)' if args.smoke else ''}: steps {[s for s in ORDER if s in steps]}, GPUs {gpus}, "
          f"datasets {plan.datasets}", flush=True)
    for step in ORDER:
        if step not in steps:
            continue
        jobs = getattr(plan, step)()
        for wave in sorted({j.wave for j in jobs}):   # e.g. shard merges after their shards, training phases in order
            sched.run_step([j for j in jobs if j.wave == wave])
    if sched.failed:
        print(f"[{stamp()}] FAILED: {sched.failed}", flush=True)
        sys.exit(1)
    print(f"[{stamp()}] RUN_COMPLETE {plan.exp}{' (smoke)' if args.smoke else ''}", flush=True)


if __name__ == "__main__":
    main()
