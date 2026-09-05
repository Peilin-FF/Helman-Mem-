"""Run the five-model, three-seed shuffled-OOD full-Sigma accuracy table on GPUs.

Each worker owns a deterministic, disjoint queue.  Every subprocess writes an
individual log and publishes its normal completion marker, so a stopped worker
can be restarted safely.  Only the released Sigma-Mem checkpoints are evaluated;
the Base arm is order-independent and is taken from the fixed-order evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MINE = Path("/mnt/data/peilin/sigma-mem-mine")
MODEL_ROOT = Path("/mnt/data/peilin/HF_MODEL")
SIGMA_PYTHON = Path("/home/peilin/miniconda3/envs/sigma/bin/python")
SIGMA35_PYTHON = Path("/home/peilin/miniconda3/envs/sigma3_5/bin/python")
OUTPUT_ROOT = ROOT / "outputs/main_accuracy_table"


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    marker: Path
    log: Path
    weight: int


MODELS = {
    "q3_0_6b": (MODEL_ROOT / "Qwen3-0.6B", SIGMA_PYTHON, 7),
    "q3_4b": (MODEL_ROOT / "Qwen3-4B", SIGMA_PYTHON, 15),
    "q3_8b": (MODEL_ROOT / "Qwen3-8B", SIGMA_PYTHON, 25),
    "q35_4b": (MODEL_ROOT / "Qwen3.5-4B", SIGMA35_PYTHON, 20),
    "q35_9b": (MODEL_ROOT / "Qwen3.5-9B", SIGMA35_PYTHON, 35),
}

METHODS = {
    "original_sigma": (
        ROOT / "configs/symmetric_memory_candidate_yesno.yaml",
        {
            "q3_0_6b": MINE / "outputs/sigma_candidate_yesno_q3_0.6b/proto",
            "q3_4b": MINE / "outputs/sigma_candidate_yesno_q3_4b/proto",
            "q3_8b": MINE / "outputs/sigma_candidate_yesno_q3_8b/proto",
            "q35_4b": MINE / "outputs/sigma_candidate_yesno_q35_4b/proto",
            "q35_9b": MINE / "outputs/sigma_candidate_yesno_q35_9b/proto",
        },
    ),
}


def evaluation_job(method: str, model_name: str, seed: int) -> Job:
    config, checkpoints = METHODS[method]
    model, python, weight = MODELS[model_name]
    output = OUTPUT_ROOT / f"full_sigma/{method}/{model_name}/seed{seed}"
    command = (
        str(python),
        "-u",
        "-m",
        "tests.experiments.common.evaluate_sigma",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoints[model_name]),
        "--central_model",
        str(model),
        "--offline_data",
        str(OUTPUT_ROOT / f"data/ood_shuffled_seed{seed}.jsonl"),
        "--output",
        str(output),
        "--graph_posterior",
        "ising",
        "--max_length",
        "8192",
        "--legacy_prompt_protocol",
        "off",
    )
    return Job(
        name=f"eval_{method}_{model_name}_seed{seed}",
        command=command,
        marker=output / "eval_metrics.json",
        log=OUTPUT_ROOT / f"logs/{method}_{model_name}_seed{seed}.log",
        weight=weight,
    )


def build_queues(num_workers: int = 8) -> tuple[list[list[Job]], list[int]]:
    queues: list[list[Job]] = [[] for _ in range(num_workers)]
    loads = [0 for _ in range(num_workers)]
    remaining = []
    for model_name in MODELS:
        for method in METHODS:
            for seed in range(3):
                remaining.append(evaluation_job(method, model_name, seed))
    remaining.sort(key=lambda job: (-job.weight, job.name))
    for job in remaining:
        worker = min(range(num_workers), key=lambda index: (loads[index], index))
        queues[worker].append(job)
        loads[worker] += job.weight
    return queues, loads


def run_job(job: Job, gpu: int) -> None:
    if job.marker.is_file():
        print(f"[worker {gpu}] skip complete {job.name}: {job.marker}", flush=True)
        return
    job.log.parent.mkdir(parents=True, exist_ok=True)
    job.marker.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(ROOT)
    print(f"[worker {gpu}] start {job.name}", flush=True)
    print(f"[worker {gpu}] task log {job.log}", flush=True)
    with job.log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            job.command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not job.marker.is_file():
        print(
            f"[worker {gpu}] FAILED {job.name} exit={completed.returncode}; "
            f"see {job.log}",
            flush=True,
        )
        raise SystemExit(completed.returncode or 1)
    print(f"[worker {gpu}] complete {job.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=range(8))
    parser.add_argument("--show-plan", action="store_true")
    args = parser.parse_args()
    queues, loads = build_queues()
    if args.show_plan:
        payload = {
            str(index): {
                "estimated_minutes": loads[index],
                "jobs": [job.name for job in queue],
            }
            for index, queue in enumerate(queues)
        }
        print(json.dumps(payload, indent=2))
        return
    if args.worker is None:
        parser.error("--worker is required unless --show-plan is used")
    gpu = int(args.worker) % 4
    for job in queues[args.worker]:
        run_job(job, gpu)
    print(f"[worker {args.worker}] all jobs complete", flush=True)


if __name__ == "__main__":
    main()
