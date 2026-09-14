"""Train: the jobs of a training experiment (configs/experiments/train_tilt.yaml), used by pipeline.run.

For every arm, in waves (each wave waits for the previous one):
  0  the record files the data needs (the train stream in fixed order, the validation stream), if missing
  1  the arm's train and validation parquets (training/kalman_rl/build_rl_data.py)
  2  GRPO from the initial model: training/scripts/train_grpo.sh with the arm's overrides -> outputs/train/<arm>/
A combination arm (prompt: combination, configs/experiments/train_combination.yaml) needs no wave 0: the experiment's own,
features and record steps make the training stream's seven-answer record and its saved addresses, and the trainer runs
combination online from them (feedback_state.combination).
A run is done when outputs/train/<arm>/complete.json exists (written when training exits cleanly).
The evaluate step then evaluates every arm's last checkpoint (``evaluate_runs``).
"""
from __future__ import annotations

import glob
from pathlib import Path

from pipeline.config import deep_merge
from pipeline.layout import Layout


def arms(plan) -> dict:
    all_arms = plan.cfg.get("arms", {})
    keep = plan.cfg.get("smoke", {}).get("arms") if plan.smoke else None
    return {k: v for k, v in all_arms.items() if keep is None or k in keep}


def last_checkpoint(run_dir: Path) -> Path | None:
    steps = sorted(glob.glob(str(run_dir / "hf" / "global_step_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    return Path(steps[-1]) if steps else None


def train_jobs(plan) -> list:
    from pipeline.run import Job

    cfg, L, judge = plan.cfg, plan.L, plan.cfg["judge"]
    td, vd = cfg["train_data"], cfg["val_data"]
    rec = cfg.get("record", {})
    dim = int(cfg.get("smoke", {}).get("dim", 32)) if plan.smoke else int(rec.get("dim", 256))
    jobs, rec_files = [], {}
    combination = [a for a, s in arms(plan).items() if s.get("prompt") == "combination"]
    if combination and not L.own:
        raise SystemExit("a combination arm needs own_answer: true (the record estimates the central model's own answer)")
    if combination and td.get("order") != rec.get("order", "shuffled0"):
        raise SystemExit("a combination arm trains in the record's order: set train_data.order to record.order")
    for key, spec in (() if len(combination) == len(arms(plan)) else (("train", td), ("val", vd))):   # wave 0: the records the data needs
        lay = Layout(deep_merge(cfg, {"record": {"order": spec["order"]}}), plan.smoke)
        stream = lay.stream(spec["dataset"])
        out = lay.record_file(judge, spec["dataset"])
        rec_files[key] = (out, stream)
        fit = rec.get("fit", "train6")
        fit_args = "" if fit == "self" else f" --fit-stream {lay.stream(fit)['path']} --fit-features {lay.features_dir(judge, fit)}"
        jobs.append(Job("train", f"record_{judge}_{spec['dataset']}_{spec['order']}",
                        f"python -m pipeline.record --stream {stream['path']} --features {lay.features_dir(judge, spec['dataset'])}{fit_args} "
                        f"--peers {stream['peers']} --order {spec['order']} --dim {dim} --lam {rec.get('lam', 100.0)} --out {out}"
                        + (f" --limit {plan.events}" if plan.events and key == "train" else ""), done=out, wave=0))
    data_dir = L.outputs / "train" / "data"
    gpus = int(cfg.get("smoke", {}).get("gpus_per_run", 2)) if plan.smoke else int(cfg.get("gpus_per_run", len(plan.gpus)))
    init = L.model(cfg["init"])["path"]
    smoke = list(cfg.get("smoke", {}).get("overrides", [])) if plan.smoke else []   # last: a smoke run always wins
    for arm, spec in arms(plan).items():
        prompt, sf = spec.get("prompt", "peers"), float(spec.get("solo_fraction", 0.0))
        args = list(cfg.get("overrides", [])) + (list(cfg.get("tilt_overrides", [])) if spec.get("tilt") else [])
        out = L.train_dir(arm)
        if prompt == "combination":
            record, stream = L.record_file(judge, td["dataset"]), L.own_stream(judge, td["dataset"])
            stem = f"{judge}_{td['dataset']}_combination"
            train_pq, val_pq = data_dir / f"{stem}.parquet", data_dir / f"{stem}_val_limit{int(vd.get('limit', 64))}.parquet"
            build = f"python -m training.kalman_rl.build_rl_data --record {record} --stream {stream} --prompt combination"
            jobs.append(Job("train", f"data_{stem}", f"{build} --out {train_pq}", gpus=0, done=train_pq, wave=1))
            jobs.append(Job("train", f"data_val_{arm}", f"{build} --limit {int(vd.get('limit', 64))} --out {val_pq}", gpus=0, done=val_pq, wave=1))
            cb = cfg.get("combination", {})
            prior = ",".join(str(float(x)) for x in cb.get("prior", [0.5, 0.0]))
            args += [f"data.combination.addresses={L.addresses_file(record)}", f"data.combination.prior=[{prior}]", f"data.combination.lam={float(cb.get('lam', 1.0))}"]
            cmd = (f"EXP={arm} MODEL={init} OUT={out} TRAIN={train_pq} VAL={val_pq} bash training/scripts/train_grpo.sh {' '.join(args + smoke)} "
                   f"&& echo '{{\"run\": \"{arm}\"}}' > {out}/complete.json")
            jobs.append(Job("train", f"train_{arm}", cmd, gpus=gpus, done=out / "complete.json", wave=2))
            continue
        (rec_t, st_t), (rec_v, st_v) = rec_files["train"], rec_files["val"]
        stem = f"{judge}_{td['dataset']}_{prompt}" + (f"_solo{int(round(100 * sf))}" if prompt == "peers" and sf else "")
        train_pq = data_dir / f"{stem}.parquet"
        val_pq = data_dir / f"{judge}_{vd['dataset']}_{prompt}_val_every{vd.get('every', 1)}_limit{vd.get('limit', 'all')}.parquet"
        jobs.append(Job("train", f"data_{stem}", f"python -m training.kalman_rl.build_rl_data --record {rec_t} --stream {st_t['path']} "
                        f"--prompt {prompt} --solo-fraction {sf} --out {train_pq}", gpus=0, done=train_pq, wave=1))
        jobs.append(Job("train", f"data_val_{arm}", f"python -m training.kalman_rl.build_rl_data --record {rec_v} --stream {st_v['path']} "
                        f"--prompt {prompt} --every {vd.get('every', 1)}" + (f" --limit {vd['limit']}" if vd.get("limit") else "")
                        + f" --out {val_pq}", gpus=0, done=val_pq, wave=1))
        cmd = (f"EXP={arm} MODEL={init} OUT={out} TRAIN={train_pq} VAL={val_pq} bash training/scripts/train_grpo.sh {' '.join(args + smoke)} "
               f"&& echo '{{\"run\": \"{arm}\"}}' > {out}/complete.json")
        jobs.append(Job("train", f"train_{arm}", cmd, gpus=gpus, done=out / "complete.json", wave=2))
    return jobs


def evaluate_runs(plan) -> list:
    """The evaluation jobs of every arm's last checkpoint (hf/global_step_N with the largest N)."""
    cfg, L = plan.cfg, plan.L
    jobs = []
    for arm in arms(plan):
        ck = last_checkpoint(L.train_dir(arm))
        if ck is None:
            if getattr(plan, "dry", False):
                print(f"[dry-run] evaluate {arm}: its checkpoint appears when the train step has run")
                continue
            raise SystemExit(f"{arm}: no checkpoint under {L.train_dir(arm)}/hf (run the train step first)")
        jobs += plan.eval_jobs(arm, L.model(cfg["init"]), cfg["judge"], plan.eval_datasets(), checkpoint=ck)
    return jobs
