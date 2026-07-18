"""Run selective-feedback Sigma-Mem OOD evaluations for Qwen3.5 models.

This launcher regenerates the score streams needed for the feedback-sparsity
ablation.  Center scores are memory-free and run once per model.  Sigma scores
must be regenerated for each feedback mask because masked feedback changes the
future runtime memory state and therefore the residual steering.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from eval_ood_feedback_sparsity import FEEDBACK_PERCENTS, SEEDS
from eval_ood_memory_routing import CANONICAL_STREAM_SIZE, RUN_PROFILES


PROFILES = ("q35_4b", "q35_9b")
DEFAULT_GPUS = ("0", "1", "2", "5", "6", "7")
SIGMA_ENV_PYTHON = Path("/home/peilin/miniconda3/envs/sigma3_5/bin/python")
DEFAULT_DATA = Path(
    "data/peer_generalization/hf_generalization_all_canonical_peers/"
    "hf_generalization_all_peer012.labeled.jsonl"
)


@dataclass(frozen=True)
class Job:
    name: str
    profile: str
    output: Path
    log: Path
    command: list[str]


def _metrics_complete(output: Path) -> bool:
    metrics_path = output / "eval_metrics.json"
    selections_path = output / "selections.jsonl"
    if not metrics_path.is_file() or not selections_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return False
    if int(metrics.get("num_samples", -1)) != CANONICAL_STREAM_SIZE:
        return False
    return True


def _make_eval_command(
    *,
    profile_name: str,
    output: Path,
    offline_data: Path,
    ablate_memory: bool,
    feedback_percent: int,
    feedback_seed: int,
) -> list[str]:
    profile = RUN_PROFILES[profile_name]
    command = [
        str(SIGMA_ENV_PYTHON),
        "-u",
        "eval_symmetric_memory.py",
        "--config",
        "configs/symmetric_memory_candidate_yesno.yaml",
        "--central_model",
        profile.central_model,
        "--offline_data",
        str(offline_data),
        "--legacy_prompt_protocol",
        "on",
        "--feedback_percent",
        str(feedback_percent),
        "--feedback_seed",
        str(feedback_seed),
        "--output",
        str(output),
    ]
    if ablate_memory:
        command.append("--ablate_memory")
    else:
        command.extend(["--checkpoint", str(profile.checkpoint)])
    return command


def _jobs(root: Path, offline_data: Path) -> list[Job]:
    jobs: list[Job] = []
    for profile_name in PROFILES:
        center_output = root / profile_name / "center"
        jobs.append(
            Job(
                name=f"{profile_name}:center",
                profile=profile_name,
                output=center_output,
                log=center_output / "run.log",
                command=_make_eval_command(
                    profile_name=profile_name,
                    output=center_output,
                    offline_data=offline_data,
                    ablate_memory=True,
                    feedback_percent=100,
                    feedback_seed=0,
                ),
            )
        )
        for percent in FEEDBACK_PERCENTS:
            seeds = (0,) if int(percent) == 100 else SEEDS
            for seed in seeds:
                sigma_output = root / profile_name / f"sigma_p{percent}_s{seed}"
                jobs.append(
                    Job(
                        name=f"{profile_name}:sigma:p{percent}:s{seed}",
                        profile=profile_name,
                        output=sigma_output,
                        log=sigma_output / "run.log",
                        command=_make_eval_command(
                            profile_name=profile_name,
                            output=sigma_output,
                            offline_data=offline_data,
                            ablate_memory=False,
                            feedback_percent=int(percent),
                            feedback_seed=int(seed),
                        ),
                    )
                )
    return jobs


def _launch(job: Job, gpu: str) -> subprocess.Popen:
    job.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_handle = job.log.open("w")
    print(f"[sigma-sparsity/launch] gpu={gpu} {job.name} -> {job.log}", flush=True)
    return subprocess.Popen(
        job.command,
        cwd=Path.cwd(),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/ood_sigma_feedback_sparsity_with_piqa"))
    parser.add_argument("--offline-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--gpus", nargs="+", default=list(DEFAULT_GPUS))
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SIGMA_ENV_PYTHON.is_file():
        raise FileNotFoundError(SIGMA_ENV_PYTHON)
    args.output.mkdir(parents=True, exist_ok=True)

    pending = _jobs(args.output, args.offline_data)
    if args.force:
        runnable = pending
    else:
        runnable = [job for job in pending if not _metrics_complete(job.output)]
    skipped = len(pending) - len(runnable)
    print(
        f"[sigma-sparsity] jobs={len(pending)} runnable={len(runnable)} skipped={skipped}",
        flush=True,
    )
    if not runnable:
        print("[sigma-sparsity] all score streams already complete", flush=True)
        return

    if int(args.slots_per_gpu) < 1:
        raise ValueError("--slots-per-gpu must be positive")
    available_gpus = [
        str(gpu)
        for gpu in args.gpus
        for _ in range(int(args.slots_per_gpu))
    ]
    if not available_gpus:
        raise ValueError("at least one GPU id is required")
    running: dict[subprocess.Popen, tuple[Job, str]] = {}
    failures: list[tuple[Job, int]] = []

    while runnable or running:
        while runnable and len(running) < len(available_gpus):
            used_counts: dict[str, int] = {}
            for _, gpu in running.values():
                used_counts[gpu] = used_counts.get(gpu, 0) + 1
            free_gpu = next(
                gpu
                for gpu in available_gpus
                if used_counts.get(gpu, 0)
                < sum(1 for candidate in available_gpus if candidate == gpu)
            )
            job = runnable.pop(0)
            proc = _launch(job, free_gpu)
            running[proc] = (job, free_gpu)

        time.sleep(args.poll_seconds)
        for proc in list(running):
            ret = proc.poll()
            if ret is None:
                continue
            job, gpu = running.pop(proc)
            if ret != 0 or not _metrics_complete(job.output):
                failures.append((job, int(ret)))
                print(
                    f"[sigma-sparsity/fail] gpu={gpu} {job.name} ret={ret} log={job.log}",
                    flush=True,
                )
            else:
                print(f"[sigma-sparsity/done] gpu={gpu} {job.name}", flush=True)

    if failures:
        details = ", ".join(f"{job.name}:{ret}" for job, ret in failures[:8])
        raise RuntimeError(f"{len(failures)} Sigma sparsity jobs failed: {details}")
    print("[sigma-sparsity] all score streams complete", flush=True)


if __name__ == "__main__":
    main()
