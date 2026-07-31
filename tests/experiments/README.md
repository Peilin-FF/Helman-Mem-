# Experiment layout

- `common/evaluate_sigma.py`: shared Base, Sigma without G, and Sigma with joint G evaluator.
- `counterfactual/`: CF@0/50/70/90 evaluation, direct M-Route, and the Appendix Beta B1 baseline.
- `peer_generalization/`: four- and five-peer evaluation with unseen peers.
- `selection_mechanisms/`: OOD Majority, M-Route, and M-Vote evaluation.
- `feedback_availability/`: 5%--100% OOD feedback ablation for M-Route, M-Vote, and Sigma with joint G.

Central-model locations are configured in `configs/experiments/central_models.json`.
Learned memory dynamics are read from each checkpoint at runtime.
