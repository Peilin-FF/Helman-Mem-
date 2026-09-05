"""Multi-GPU training of the central model on top of the vendored verl (Ray + FSDP + vLLM).

Modules
  build_rl_data   memory-annotated prompt stream  ->  verl parquet (question-only prompt, hinted prompt, reward info)
  build_sft_data  verified generations            ->  SFT parquet (prompt, response)
  dataset         SigmaRLDataset / SigmaSFTDataset (prompts rendered without the thinking block, like our evaluations)
  reward          verifier reward or memory pseudo-reward; parallel reward manager
  trainer         SigmaRayPPOTrainer: plain GRPO plus the memory hint pass for zero-reward groups
  main_grpo       hydra entry point (config: training/configs/grpo_sigma.yaml)

Strict outcome layer (contributed separately; guards and a future tensor-memory path, not used by train_grpo.sh):
  outcome_protocol  public prompts (question + all peers, no notes), CausalPeerEpisode (read -> answer -> grade -> write)
  outcome_batch     DataProto guards requiring peer_evidence tensors from the rollout (no backend provides them yet)
  outcome_reward    OutcomeRewardManager: post-answer verification only (reward_model.reward_manager=outcome)
  audit_outcome_data  token-budget audit of question + all peers (no filtering)
"""
