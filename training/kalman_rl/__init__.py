"""GRPO of the central model on the vendored verl (Ray + FSDP + vLLM), driven by pipeline.train.

  build_rl_data   a record file -> the verl parquet (prompt, reward, the record's estimates and peer spans for the tilt)
  dataset         KalmanRLDataset: our prompt rendering and the tilt tensor per row
  reward          the task verifier; a parallel reward manager (code graded in Ray workers)
  trainer         KalmanRayPPOTrainer: verl GRPO, metrics.jsonl, bf16 HF checkpoints under hf/
  main_grpo       hydra entry point (config: configs/train/grpo.yaml)
"""
