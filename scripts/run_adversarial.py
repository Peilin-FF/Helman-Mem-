"""Run the robustness pipeline from training/configs/adversarial.yaml (README_adversarial.md).

    bash run_adversarial.sh [--regimes all100 ...] [--steps peers streams features prompts eval table] [--gpus 0,1] [--smoke]

Steps, in order.  Each task is skipped when its output is already on disk, so an interrupted run is resumed by running
the same command again; the exit code is 1 if any task failed, with the log to read printed next to it.

  peers     every peer answers every event of every test stream adversarially (scripts/adversarial_peers.py): generate,
            grade, re-generate what is still correct or unusable.  Regime-independent and done once: the regimes below
            are then only a choice of which of those answers to put into the stream.
  streams   one copy of each test stream per regime (scripts/build_adversarial_stream.py), labels recomputed
  features  the central model, frozen, re-reads each adversarial stream (the peers' answers are part of the address)
  prompts   the Bayesian record along each adversarial stream, read before write, and its quality against the labels
  eval      the central model answers each stream in every condition: peers + memory, peers, question only, swapped record
  table     scripts/adversarial_table.py, beside the honest rows of the same central model
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.run_families import MODE, TEST_DIR, Pool, stamp

MODE = dict(MODE, swap=("peers", "--attn_gamma {gamma} --swap_record"))   # the control: the record permuted by rank


def shell(name: str, cmd: str, log: Path, done: Path | None, failed: list[str]) -> None:
    """Run one CPU task (stream building, the table) in the foreground, skipping it when its output exists."""
    if done is not None and done.exists():
        print(f"[{stamp()}] skip  {name} (already on disk)", flush=True)
        return
    print(f"[{stamp()}] start {name}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.call(cmd, shell=True, executable="/bin/bash", stdout=f, stderr=subprocess.STDOUT)
    ok = rc == 0 and (done is None or done.exists())
    print(f"[{stamp()}] {'done ' if ok else 'FAILED'} {name}" + ("" if ok else f" (see {log})"), flush=True)
    if not ok:
        failed.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/configs/adversarial.yaml")
    ap.add_argument("--regimes", nargs="*", help="regimes to run (default: the YAML's run list)")
    ap.add_argument("--steps", nargs="*", help="peers streams features prompts eval table (default: the YAML's steps)")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: the YAML's gpus)")
    ap.add_argument("--smoke", action="store_true", help="the whole pipeline on a few events of one stream (~15 min)")
    ap.add_argument("--smoke-suffix", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    os.environ["SIGMA_FEATURE_ROOT"] = str(cfg["outputs"]["features"])   # read by feedback_state.feature_streams in the subprocesses
    from feedback_state.feature_streams import DATA_ROOT, FEATURE_ROOT, STREAMS

    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else [int(g) for g in cfg["gpus"]]
    steps = args.steps or list(cfg["steps"])
    gen, fe, mem, ev, out, sm = cfg["generation"], cfg["features"], cfg["memory"], cfg["evaluation"], cfg["outputs"], cfg.get("smoke", {})
    suffix = args.smoke_suffix or sm.get("suffix", "_smoke")
    from feedback_state.adversarial import expand_run, regimes_from_config

    asked = args.regimes or (list(sm.get("regimes", cfg["run"])) if args.smoke else list(cfg["run"]))
    regimes = expand_run(asked, cfg)     # "rates" = p000 ... p100, "counts" = k0 ... k6, "sweep" = both
    known = regimes_from_config(cfg, regimes)   # p030 (30% of every peer's answers) and k2 (2 peers per event) need no config entry
    unknown = [r for r in regimes if r not in known]
    if unknown:
        sys.exit(f"unknown regime(s) {unknown}; the config knows {sorted(known)}, "
                 "and pNNN (a share of every peer's answers, p000-p100) or kN (N misleading peers per event) need no entry")
    tests = list(cfg["streams"]["test"])
    if args.smoke:
        tests = [t for t in tests if t == "indist6"] or tests[:1]
    fit_cfg = str(cfg["streams"].get("fit", "self"))
    root = Path(cfg.get("models_root", "."))
    central = dict(cfg["central"])
    tag = str(central["tag"]) + (suffix if args.smoke else "")
    cpath = Path(central["path"])
    cpath = cpath if cpath.is_absolute() else root / cpath
    if not (cpath / "config.json").exists():
        sys.exit(f"central model directory {cpath} has no config.json (models_root / central in the YAML)")
    engine = str(central.get("engine", ev.get("engine", "vllm")))
    env = central.get("env")
    pre = f'eval "$(conda shell.bash hook)" && conda activate {env} && ' if env else ""
    shards = 1 if args.smoke else len(gpus)
    peers_root = Path(sm.get("peers_root", "outputs/peer_adv_smoke") if args.smoke else gen.get("root", "outputs/peer_adv"))
    pool = Pool(gpus)
    L = Path("logs"); L.mkdir(exist_ok=True)
    what = "smoke" if args.smoke else "full"
    print(f"[{stamp()}] {what}: central {tag} ({cpath}), regimes {regimes}, steps {steps}, GPUs {gpus}, streams {tests}", flush=True)

    # ---- 1. the peers' adversarial answers (regime-independent, generated once per stream) -------------------------
    if "peers" in steps:
        tasks = []
        for st in tests:
            jsonl = DATA_ROOT / STREAMS[st][0]
            split = Path(STREAMS[st][0]).stem
            for peer in cfg["peers"]:
                name = str(peer["name"])
                mpath = Path(peer.get("path", name))
                mpath = mpath if mpath.is_absolute() else root / mpath
                odir = peers_root / st / name
                extra = ("" if peer.get("prefix_caching", True) else " --no_prefix_caching")
                extra += (" --trust_remote_code" if peer.get("trust_remote_code") else "")
                extra += (" --enforce_eager" if peer.get("enforce_eager") else "")
                penv = "".join(f"{k_}={v} " for k_, v in (peer.get("env") or {}).items())
                for k in range(shards):
                    done = odir / (f"{split}.jsonl" if shards == 1 else f"{split}.shard{k}of{shards}.jsonl")
                    cmd = (f"{pre}{penv}python scripts/adversarial_peers.py --model {mpath} --records {jsonl} --output {odir} "
                           f"--num_shards {shards} --shard_index {k} --max_attempts {gen['max_attempts']} "
                           f"--temperatures {gen['temperatures']} --grade_workers {gen.get('grade_workers', 8)} "
                           f"--max_model_len {gen.get('max_model_len', 8192)} "
                           f"--gpu_memory_utilization {gen.get('gpu_memory_utilization', 0.85)}"
                           + extra
                           + (" --reasoning" if peer.get("reasoning") else "")
                           + ("" if gen.get("force", True) else " --no_force")
                           + (f" --max_examples {int(sm.get('events', 48))}" if args.smoke else ""))
                    tasks.append((f"peers {st} {name} shard {k}/{shards}", cmd, L / f"advpeer_{st}_{name}_{k}.out", done))
        pool.run_all(tasks)

    # ---- 2 to 5. one pipeline per regime ---------------------------------------------------------------------------
    for regime in regimes:
        rtag = regime + (suffix if args.smoke else "")
        streams = {st: f"{st}_adv_{rtag}" for st in tests}
        print(f"===== regime {rtag}: {known[regime].describe()}  {stamp()}", flush=True)
        if "streams" in steps:
            for st in tests:
                base, split = DATA_ROOT / STREAMS[st][0], Path(STREAMS[st][0]).stem
                target = DATA_ROOT / STREAMS[streams[st]][0]
                cmd = (f"python scripts/build_adversarial_stream.py --base {base} --adv {peers_root / st} "
                       f"--config {args.config} --regime {regime} --order {mem['order']} --split {split} --out {target}"
                       + (f" --limit {int(sm.get('events', 48))}" if args.smoke else ""))
                shell(f"stream {rtag} {st}", cmd, L / f"advstream_{rtag}_{st}.out", target, pool.failed)
        if "features" in steps:
            fit = [] if fit_cfg == "self" else [fit_cfg]
            for st in fit + [streams[s] for s in tests]:
                cache, inp = FEATURE_ROOT / STREAMS[st][1].format(m=tag), DATA_ROOT / STREAMS[st][0]
                tasks = []
                for k in range(shards):
                    outp = cache / f"shard{k}.pt"
                    cmd = (f"{pre}python scripts/encode_context_features.py --input {inp} --output {outp} --central-model {cpath} "
                           f"--include-context --save-peer-hidden --num-peers {fe['num_peers']} --max-length {fe['max_length']} "
                           f"--dtype {fe['dtype']} --progress-every {fe.get('progress_every', 500)} "
                           f"--num-shards {shards} --shard-index {k}" + (f" --max-examples {int(sm.get('events', 48))}" if args.smoke else ""))
                    tasks.append((f"features {tag} {st} shard {k}/{shards}", cmd, L / f"advenc_{tag}_{st}_{k}.out", outp))
                pool.run_all(tasks)
        if "prompts" in steps:
            gdir = Path(out["prompts"]) / tag
            gdir.mkdir(parents=True, exist_ok=True)
            dim = int(sm.get("dim", 32)) if args.smoke else int(mem["dim"])
            for st in tests:
                name = streams[st]
                pf = gdir / f"prompts_{name}_probe.jsonl"
                cmd = (f"{pre}python scripts/build_generation_prompts.py --model {tag} "
                       f"--fit-stream {name if fit_cfg == 'self' else fit_cfg} --stream {name} --order {mem['order']} "
                       f"--design {mem['design']} --dim {dim} --lam {mem['lam']} --out {pf} "
                       f"&& python scripts/record_quality.py --prompts {pf} --out {gdir / f'record_{name}.json'}")
                pool.run_all([(f"prompts {tag} {name} (record + quality)", cmd, L / f"advprompts_{tag}_{name}.out", pf)])
        if "eval" in steps:
            tasks, merges = [], []
            for st in tests:
                name = streams[st]
                pf = Path(out["prompts"]) / tag / f"prompts_{name}_probe.jsonl"
                rec = DATA_ROOT / STREAMS[name][0]
                if not pf.exists():
                    print(f"[{stamp()}] FAILED eval {tag} {name}: no prompt file {pf} (run the prompts step first)", flush=True)
                    pool.failed.append(f"eval {tag} {name}")
                    continue
                for cond in ev["conditions"]:
                    mode, extra = MODE[cond]
                    od = Path(out["evals"]) / tag / rtag / f"{TEST_DIR[st]}6_{cond}"
                    common = (f"{pre}python -m tests.experiments.common.evaluate_memory_generator --central_model {cpath} "
                              f"--engine {engine} --thinking {ev['thinking']} --max_new_tokens {ev['max_new_tokens']} "
                              f"--prompts {pf} --records {rec} --mode {mode} {extra.format(gamma=ev['gamma'])}")
                    if engine == "vllm":
                        tasks.append((f"eval {tag} {rtag} {st} {cond}",
                                      f"{common} --gpu_memory_utilization {ev['gpu_memory_utilization']} --output {od}",
                                      L / f"adv_{tag}_{rtag}_{TEST_DIR[st]}_{cond}.out", od / "eval_metrics.json"))
                    else:   # HF engine: one shard per GPU, joined afterwards
                        if (od / "eval_metrics.json").exists():
                            print(f"[{stamp()}] skip  eval {tag} {rtag} {st} {cond} (already on disk)", flush=True)
                            continue
                        n_sh = 1 if args.smoke else int(central.get("shards", len(gpus)))
                        for k in range(n_sh):
                            tasks.append((f"eval {tag} {rtag} {st} {cond} shard {k}/{n_sh}",
                                          f"{common} --batch_size {int(central.get('batch_size', 8))} --shard {k}/{n_sh} --output {od / f'shard{k}'}",
                                          L / f"adv_{tag}_{rtag}_{TEST_DIR[st]}_{cond}_shard{k}.out", od / f"shard{k}" / "eval_metrics.json"))
                        merges.append((f"merge {tag} {rtag} {st} {cond}", f"python scripts/merge_eval_shards.py --out {od}",
                                       L / f"adv_{tag}_{rtag}_{TEST_DIR[st]}_{cond}.out", od / "eval_metrics.json"))
            pool.run_all(tasks)
            for name, cmd, log, done in merges:
                shell(name, cmd, log, done, pool.failed)

    if "table" in steps or "eval" in steps:
        subprocess.call(f"python scripts/adversarial_table.py --config {args.config} --tag {tag} "
                        f"--peers-root {peers_root} "
                        f"--regimes {' '.join(r + (suffix if args.smoke else '') for r in regimes)}"
                        + (" --smoke" if args.smoke else ""), shell=True)
    if pool.failed:
        print(f"[{stamp()}] FAILED tasks: {pool.failed}", flush=True)
        sys.exit(1)
    print(f"[{stamp()}] ADVERSARIAL_RUN_COMPLETE ({what})", flush=True)


if __name__ == "__main__":
    main()
