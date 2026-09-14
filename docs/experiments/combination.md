# Combination: peers + memory or question alone, chosen per event

Question: with Qwen3-4B fixed as the central model, can it keep accuracy up as the peers turn misleading, by choosing per
event between peers + memory and question alone?

## Setup

- **Seven answers per event.** The six peers of the stream, plus Qwen3-4B's question-alone answer. One record estimates
  all seven alike, addressed by question and answer; Qwen3-4B's own answer is recorded but never shown in the prompt.
- **Peers + memory.** Qwen3-4B answers with the question and the six peers in the prompt, attention tilted (γ = 3) by
  that same record.
- **The reading line.** On each event, T is the record's top estimate among the six peers and κ its estimate of the
  question-alone answer. Peers + memory is worth A(T) = T·ρ + (1 − T)(κ − δ), question alone κ. With u = [T, T − 1] and
  z = y − (1 − T)κ (y: whether peers + memory was right), (ρ, δ) = P⁻¹q with P = I + Σuuᵀ and q = (0.5, 0) + Σu·z: a 2×2
  state per task type, separate from the record's Λ, read before write.
- **Combination.** The final answer is peers + memory when A(T) ≥ κ, otherwise question alone. The lines cross at
  T* = δ / (ρ + δ − κ): with δ > 0 and ρ > κ peers + memory wins for T ≥ T*; the table says where it wins in every case.
- **Datasets.** `misleading_rates`: in-distribution and OOD at 0, 25, 50, 75 and 100% misleading answers.

The derivation of (ρ, δ) = P⁻¹q is on the "Mathematical" page (https://claude.ai/code/artifact/d375f7dc-624f-465e-9e66-b2cf0b015b40);
the mechanism and results are on "Answer Combination by the Record" (https://claude.ai/code/artifact/8d7155a5-2ec2-4c1c-84bf-a6e717d9c15e).

## Run

```bash
bash run.sh configs/experiments/combination.yaml --smoke     # 48 events of indist6_misleading_p050, into outputs/smoke/
bash run.sh configs/experiments/combination.yaml             # every rate; finished work is skipped
```

Steps: `own` (Qwen3-4B's question-alone answers, shared with the base streams, join each stream as
`data/<dataset>+q3_4b/`), `features` and `record` on the seven answers, `evaluate` (peers + memory), `combination`, `table`.

## Outputs

```
outputs/features/q3_4b/<dataset>+own/
outputs/record/q3_4b/<dataset>+own/shuffled0.fit-self.jsonl       rows carry own_prob and own_correct
outputs/eval/q3_4b/<dataset>+own/tilt/                            peers + memory
outputs/eval/q3_4b/<dataset>+own/combination/                     the choice per event and eval_metrics.json: accuracy,
                                                                  share_peers_memory, peers_memory_accuracy,
                                                                  question_alone_accuracy, reading_line (ρ, δ per task
                                                                  type), by_trust
outputs/eval/q3_4b/<base stream>/solo/                            question alone
outputs/tables/combination.md
```

## Results (2026-09-15)

Qwen3-4B, accuracy % (job 20260915-021113; `outputs/tables/combination.md`, per event and per trust band in
`outputs/eval/q3_4b/<dataset>+own/combination/`). "Took peers + memory" is the share of events where combination chose it.

| misleading | in-dist: peers + memory | question alone | combination | took peers + memory | OOD: peers + memory | question alone | combination | took peers + memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 76.7 | 69.2 | 76.2 | 86% | 73.8 | 67.8 | 75.0 | 86% |
| 25% | 75.5 | 69.2 | 75.4 | 85% | 70.4 | 67.8 | 72.4 | 78% |
| 50% | 74.3 | 69.2 | 74.1 | 85% | 67.5 | 67.8 | 70.8 | 69% |
| 75% | 73.1 | 69.2 | 72.7 | 86% | 62.5 | 67.8 | 70.2 | 53% |
| 100% | 72.4 | 69.2 | 72.0 | 85% | 50.7 | 67.8 | 69.3 | 15% |

- OOD: combination is above peers + memory and question alone at every rate. From 0% to 100% it falls 5.7 points;
  peers + memory falls 23.1.
- In-distribution: peers + memory stays above question alone at every rate, and combination is within 0.5 points of it.
- The straight line over-states peers + memory at high trust: ρ̂ is above 1 for most task types (up to 1.48).
  In-distribution at low trust combination keeps question alone where peers + memory was a little better (0%, T 0.2–0.4:
  peers + memory right on 15.8%, question alone on 10.0%, combination took peers + memory on 11% of those events).
