# Combination: peers + memory or question alone, chosen per event

Question: with a fixed central model (Qwen3-4B, then Qwen3-8B), can it keep accuracy up as the peers turn misleading, by
choosing per event between peers + memory and question alone?

## Setup

- **Seven answers per event.** The six peers of the stream, plus the central model's question-alone answer. One record
  estimates all seven alike, addressed by question and answer; the central model's own answer is recorded but never shown
  in the prompt. Each central model is its own judge: its features, record and answers are its own.
- **Peers + memory.** The central model answers with the question and the six peers in the prompt, attention tilted
  (γ = 3) by that same record. **Question + peers** is the same prompt without the tilt.
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

Steps: `own` (each central model's question-alone answers, shared with the base streams, join each stream as
`data/<dataset>+<model>/`), `features` and `record` on the seven answers, `evaluate` (peers + memory, question + peers),
`combination`, `table`.

## Outputs

```
outputs/features/<model>/<dataset>+own/
outputs/record/<model>/<dataset>+own/shuffled0.fit-self.jsonl     rows carry own_prob and own_correct
outputs/eval/<model>/<dataset>+own/tilt/                          peers + memory
outputs/eval/<model>/<dataset>+own/peers/                         question + peers
outputs/eval/<model>/<dataset>+own/combination/                   the choice per event and eval_metrics.json: accuracy,
                                                                  share_peers_memory, peers_memory_accuracy,
                                                                  question_alone_accuracy, reading_line (ρ, δ per task
                                                                  type), by_trust
outputs/eval/<model>/<base stream>/solo/                          question alone
outputs/tables/combination.md
```

## Results (2026-09-15)

Qwen3-4B, accuracy % (job 20260915-021113; `outputs/tables/combination.md`, per event and per trust band in
`outputs/eval/q3_4b/<dataset>+own/combination/`). "Took peers + memory" is the share of events where combination chose it.

| misleading | in-dist: peers + memory | question + peers | question alone | combination | took peers + memory | OOD: peers + memory | question + peers | question alone | combination | took peers + memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 76.7 | 74.6 | 69.2 | 76.2 | 86% | 73.8 | 69.2 | 67.8 | 75.0 | 86% |
| 25% | 75.5 | 74.2 | 69.2 | 75.4 | 85% | 70.4 | 64.0 | 67.8 | 72.4 | 78% |
| 50% | 74.3 | 73.7 | 69.2 | 74.1 | 85% | 67.5 | 58.1 | 67.8 | 70.8 | 69% |
| 75% | 73.1 | 72.4 | 69.2 | 72.7 | 86% | 62.5 | 51.4 | 67.8 | 70.2 | 53% |
| 100% | 72.4 | 71.3 | 69.2 | 72.0 | 85% | 50.7 | 45.8 | 67.8 | 69.3 | 15% |

- OOD: combination is above peers + memory and question alone at every rate. From 0% to 100% it falls 5.7 points;
  peers + memory falls 23.1.
- In-distribution: peers + memory stays above question alone at every rate, and combination is within 0.5 points of it.
- The memory's tilt carries the reading: question + peers falls to 45.8 on OOD at 100%; peers + memory is 4.6 points
  above it with honest peers and 9.4–11.1 points at 50–75%.
- The straight line over-states peers + memory at high trust: ρ̂ is above 1 for most task types (up to 1.48).
  In-distribution at low trust combination keeps question alone where peers + memory was a little better (0%, T 0.2–0.4:
  peers + memory right on 15.8%, question alone on 10.0%, combination took peers + memory on 11% of those events).
