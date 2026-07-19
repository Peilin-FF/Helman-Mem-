# Sigma-Mem

Sigma-Mem is an event-level memory mechanism for selecting reliable peer
responses. It records task-conditioned competence in one symmetric matrix
`M_p` per peer and peer-to-peer correctness relationships in a symmetric graph
`G`. The center model remains frozen; historical evidence is injected only as a
residual shift in the upper decoder blocks.

## Method

For event `t`, the center model produces a normalized competence direction
`phi(x_t)`. After external correctness feedback is revealed, each peer memory is
updated by

```text
M_p <- gamma M_p + eta c_p phi(x_t) phi(x_t)^T,
```

where `c_p` is `+1` for a correct response and `-1` otherwise. At decision time,
the readout `M_p phi(x_t)` is projected to the center-model hidden dimension and
added to the residual stream while that peer is evaluated. The graph `G` records
centered pairwise correctness patterns and can be used by the joint posterior
readout.

Correctness feedback for the current event is never available before selection.
The direct memory-routing diagnostics also compute reliability from the question
and memory state before reading peer answer strings.

## Main Entry Points

- `train_symmetric_memory.py`: train the Sigma-Mem projection and memory dynamics.
- `eval_symmetric_memory.py`: evaluate residual-steered Sigma-Mem and joint `G`.
- `eval_cf_memory_routing.py`: direct `M` routing on counterfactual streams.
- `eval_ood_memory_routing.py`: training-free OOD routing and weighted voting.
- `eval_ood_feedback_sparsity.py`: sparse-feedback diagnostic for direct routing.
- `run_ood_sigma_feedback_sparsity.py`: sparse-feedback Sigma-Mem evaluation.
- `summarize_ood_sigma_feedback_sparsity.py`: aggregate sparse-feedback runs.

Core implementation modules live in `feedback_state/`:

- `symmetric_memory.py`: first-order `M`, competence readout, and residual steering.
- `joint_prompt.py`: peer comparison and candidate-judging prompts.
- `joint_data.py`: candidate prompt tokenization.
- `joint_models.py`: frozen-center candidate scoring.
- `ood_routing.py`: direct M-Route, M-Vote, and Majority aggregation.
- `prompt_protocol.py`: reproducible prompt and context formatting.
- `checkpoint_manifest.py`: checkpoint and dataset provenance validation.

## Installation

Use the environment matching the center model family:

```bash
pip install -r requirements_qwen3.txt
# or
pip install -r requirements_qwen3.5.txt
```

The shared dependency set is in `requirements.txt`.

## Training

The candidate Yes/No configuration matches the published Sigma-Mem checkpoints:

```bash
PYTHONPATH=. python3 train_symmetric_memory.py \
  --config configs/symmetric_memory_candidate_yesno.yaml \
  --central_model Qwen/Qwen3-0.6B \
  --output_dir outputs/sigma_candidate_yesno_q3_0.6b/proto
```

Training data must be a JSONL stream with peer responses and externally evaluated
`correctness_by_peer` labels. New checkpoints contain `sym_memory.pt`,
`train_config.json`, and a provenance manifest.

## Evaluation

```bash
PYTHONPATH=. python3 eval_symmetric_memory.py \
  --config configs/symmetric_memory_candidate_yesno.yaml \
  --checkpoint outputs/sigma_candidate_yesno_q3_0.6b/proto \
  --offline_data data/CF_unified/p50.jsonl \
  --output outputs/eval_sigma_candidate_yesno_q3_0.6b/p50
```

Every evaluation writes `eval_metrics.json` as its completion marker. Published
checkpoint and result directories should be validated with:

```bash
python3 scripts/validate_published_sigma_checkpoint.py --help
python3 scripts/validate_sigma_eval_output.py --help
```

## Tests

```bash
python3 -m pytest -q
```

The tests cover memory updates, prompt construction, routing isolation,
provenance, scoring, and result validation.
