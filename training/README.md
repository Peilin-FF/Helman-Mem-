# training/: GRPO of the central model on vendored verl

Run training through the pipeline: `bash run.sh configs/experiments/train_tilt.yaml` (docs/experiments/train_tilt.md).
This directory holds the pieces that experiment uses.

```
training/
  kalman_rl/
    build_rl_data.py   a record file -> the verl parquet (prompt, reward, the record's estimates and peer spans for the tilt)
    dataset.py         KalmanRLDataset: prompts rendered like the evaluations (thinking off) and the tilt tensor per row
    reward.py          the task verifier; a parallel reward manager (code graded in Ray workers, stdin on /dev/null)
    trainer.py         KalmanRayPPOTrainer: verl GRPO, metrics.jsonl, bf16 HF checkpoints under hf/global_step_N
    main_grpo.py       hydra entry point; config configs/train/grpo.yaml
  scripts/
    train_grpo.sh      one run: environment, a per-run Ray temp dir, the log, process-tree cleanup on kill
    prune_checkpoints.sh   remove the weights of intermediate checkpoints
  setup_env.sh         the packages verl needs on top of requirements_qwen3.txt (hydra, tensordict, flash-attn, ...)
  verl/                vendored verl 0.3.1 with the tilt patches (VENDOR.md)
```

Outputs of a run (`outputs/train/<arm>/`): `train.log`, `metrics.jsonl` (one line per step: rewards and
accuracy per task, validation accuracy, timings), `hf/global_step_N/` (bf16, loadable by `pipeline.evaluate --checkpoint`),
`hydra/` (the resolved config), `complete.json`.

Full-parameter training only: verl's LoRA path does not work with the vLLM rollout here, and the objective is to change
the model itself.
