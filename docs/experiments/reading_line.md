# Reading line: the central model reads the peers or keeps its own answer

Question: with Qwen3-4B fixed as the central model, can it keep accuracy up as the peers turn misleading, by deciding per
event whether to read the peers or to answer alone?

## Setup

- **Seven answers per event.** The six peers of the stream, plus Qwen3-4B's own question-only answer. One record
  estimates all seven alike, addressed by question and answer; the own answer is recorded but never shown in the prompt.
- **Reading.** Qwen3-4B reads the six peers with the tilt (γ = 3) from that same record.
- **The reading line.** On each event, T is the record's top estimate among the six peers and κ its estimate of the own
  answer. Reading is worth A(T) = T·ρ + (1 − T)(κ − δ), the own answer κ. With u = [T, T − 1] and z = y − (1 − T)κ
  (y: whether the reading was right), (ρ, δ) = P⁻¹q with P = I + Σuuᵀ and q = (0.5, 0) + Σu·z: a 2×2 state per task type,
  separate from the record's Λ, read before write.
- **Decision.** The final answer is the reading when A(T) ≥ κ, otherwise the own answer. The lines cross at
  T* = δ / (ρ + δ − κ): with δ > 0 and ρ > κ the reading wins for T ≥ T*; the table says where it wins in every case.
- **Datasets.** `misleading_rates`: in-distribution and OOD at 0, 25, 50, 75 and 100% misleading answers.

The derivation of (ρ, δ) = P⁻¹q is on the "Mathematical" page.

## Run

```bash
bash run.sh configs/experiments/reading_line.yaml --smoke     # 48 events of indist6_misleading_p050, into outputs/smoke/
bash run.sh configs/experiments/reading_line.yaml             # every rate; finished work is skipped
```

Steps: `own` (Qwen3-4B's question-only answers, shared with the base streams, join each stream as `data/<dataset>+q3_4b/`),
`features` and `record` on the seven answers, `evaluate` (the tilt reading), `decide`, `table`.

## Outputs

```
outputs/features/q3_4b/<dataset>+own/
outputs/record/q3_4b/<dataset>+own/shuffled0.fit-self.jsonl       rows carry own_prob and own_correct
outputs/eval/q3_4b/<dataset>+own/tilt/                            the reading
outputs/eval/q3_4b/<dataset>+own/decide/                          the decision per event and eval_metrics.json:
                                                                  accuracy, read_share, always_read, own_answer,
                                                                  reading_line (ρ, δ, T* per task type), by_trust
outputs/eval/q3_4b/<base stream>/solo/                            the own answer (question only)
outputs/tables/reading_line.md
```
