# misleading: the six peers answer with misleading but relevant solutions

```bash
bash datasets/unpack.sh                                                          # the released answers and streams -> data/
bash run.sh configs/experiments/misleading.yaml --smoke                          # 48 events of indist6_misleading_p050, every step
bash run.sh configs/experiments/misleading.yaml                                  # the sweep: datasets [misleading_rates]
bash run.sh configs/experiments/misleading.yaml --set "datasets=[ood6_misleading_p100]"   # one dataset
```

Does the record survive peers that are wrong on purpose? Every peer answers every event with a confident, on-topic,
verified-wrong solution once: the answers datasets `indist6_misleading` and `ood6_misleading` (`peers` step, 3-5 h on 8
GPUs; released in `datasets/`, so it is skipped after unpacking). Each misleading dataset (`configs/datasets/`) names a
base stream and a regime that chooses which of those answers replace the honest ones (`streams` step, seconds), and the
main pipeline runs unchanged on the result.

## How a misleading answer is produced (`pipeline/peers.py --mode misleading`)

Asking once is not enough: in a pilot, 44% of the answers a peer gave when told to be wrong were correct anyway, some
announced the trick, some refused, some programs were tables of hard-coded outputs. So every event is a small search
(rules in `feedback_state/adversarial.py`):

| step | what happens |
|---|---|
| ask | the task's peer prompt plus the instruction to be plausibly wrong; the gold answer is given only so it can be avoided |
| grade | the streams' own rule: token-F1 ≥ 0.5 for reading, exact match for math, hidden tests for code |
| check | kept only if graded wrong, in the task's answer format, free of meta-commentary, refusals and repetition loops, not naming the gold, a reading answer taken from the passage, a real program, a closed think block |
| retry | the failures again at temperature 0.7 then 1.0, told what was wrong; on the last attempt a math peer that keeps reaching the gold is given a wrong value (one of its own intermediate results) to arrive at |
| rewrite | only if all that fails, for math / multiple choice / yes-no: the conclusion is replaced and re-graded, flagged `forced`; never for an answer with no argument besides its final line (at least 20 characters) or a looping one, which would turn an empty reply into a bare wrong label |

Budgets: misleading answers get the honest budgets (math 512, reading 256, code 768 tokens) except on the short-answer
OOD tasks, where 96 tokens cut most arguments off before the final line; there they get 256 and are asked to argue in
one or two sentences. They still come out longer than the honest answers (on one OOD shard of gemma-3-4b-it: 330 against
170 characters on yes/no, 429 against 160 on multiple choice, 379 against 137 on short answer), a difference the central
model could in principle pick up on.

An event with no usable answer keeps the honest answer in every stream. On the 48-event smoke, 77-96% of each peer's
events had a usable answer and none of the usable answers graded correct. `data/<stream>_misleading/<peer>/summary.shard*.json`
(and `datasets/manifest.json`) reports each peer's acceptance rate, forced count, attempts and why the rest were unusable.

The weakest peer here is DeepSeek-Coder-V2-Lite-Instruct: told to argue for a wrong option it often returns an empty
reply, a bare label or a loop ("Final anti anti anti ..."), where its honest answers on the same engine have none of
these. Those replies are rejected, so it ends with the lowest usable share; its remaining events keep the honest answer.

## Regimes and the ratio

A misleading dataset is a file in `configs/datasets/` that includes the template `_misleading.yaml` and sets `base` and
`regime`. The regime is one of two short forms, where the number is what the stream ends up with (only usable answers
count; shares are rounded to whole events):

- `pNNN`: that share of every peer's answers is misleading, a different set of events per peer, nested across rates
  (p050 poisons p025's events plus more). `p000` is the honest stream rebuilt through the same steps.
- `kN`: exactly N of the six peers are misleading on every event, a different set each time (a lying minority k1-k2
  against a lying majority k4-k5).

or a mapping: `{kind: fraction, rate: 1.0, peers: [1, 4]}` (Phi-4-mini and DeepSeek-Coder always lie, four peers honest),
`{kind: flip, at: 0.5, peers: [1, 4]}` (they turn misleading halfway through the record's order), `{kind: targeted, rate:
1.0}` (every peer misleading exactly where it was right). `drop_forced: true` leaves out answers whose conclusion was
rewritten. The released ones are the rates `p000`, `p025`, `p050`, `p075`, `p100` on both streams (group
`misleading_rates`). Each built stream has a `manifest.json` with the requested and realised ratio per peer, events by
number of misleading peers, forced and unavailable counts, and accuracy before and after.

A peer cannot go beyond its usable share, so the top rates reach less than asked. The generation of 2026-09-13:

| peer | usable, indist6 | usable, ood6 | rewritten (of usable), indist6 / ood6 |
|---|---:|---:|---:|
| gemma-3-4b-it | 79.8% | 92.5% | 19.6% / 5.6% |
| Phi-4-mini-instruct | 90.3% | 95.6% | 7.0% / 4.8% |
| Qwen2.5-Coder-7B-Instruct | 89.2% | 98.3% | 2.4% / 4.0% |
| Meta-Llama-3.1-8B-Instruct | 95.5% | 98.2% | 0.7% / 0.5% |
| DeepSeek-Coder-V2-Lite-Instruct | 63.8% | 75.8% | 2.6% / 36.9% |
| DeepSeek-R1-Distill-Qwen-7B | 67.3% | 91.1% | 15.8% / 10.6% |

So p025 and p050 are exact on both streams and p075 on ood6. p075 on indist6 reaches 71.9% overall because the two
DeepSeek peers stop at their usable share. p100 reaches 81.0% on indist6 and 91.9% on ood6, the most these answers allow.

## Result (2026-09-13): frozen Qwen3-4B on the rate datasets

Accuracy (%) of the central model with the six answers in the prompt, with (`tilt`) and without (`peers`) the record's
attention tilt, and on the question alone (`solo`); the record is fit on each stream itself. A question-only prompt holds
no peer answers, so `solo` is the same for every misleading share: it is read from the base stream's result
(`Layout.eval_dataset`), not re-run.

| misleading share asked (reached: indist6 / ood6) | indist6 tilt | indist6 peers | indist6 solo | ood6 tilt | ood6 peers | ood6 solo |
|---|---:|---:|---:|---:|---:|---:|
| honest (main experiment) | 67.2 | 64.8 | 60.5 | 74.0 | 69.2 | 67.8 |
| 0% | 67.3 | 64.8 | 60.5 | 74.0 | 69.2 | 67.8 |
| 25% | 66.5 | 64.2 | 60.5 | 70.5 | 64.4 | 67.8 |
| 50% | 66.1 | 64.1 | 60.5 | 68.5 | 59.2 | 67.8 |
| 75% (71.9 / 75.0) | 64.8 | 62.8 | 60.5 | 63.5 | 52.2 | 67.8 |
| 100% (81.0 / 91.9) | 63.4 | 61.5 | 60.5 | 53.2 | 47.0 | 67.8 |

The record gives misleading answers a lower estimate than honest ones (AUC 0.70-0.72 in-distribution, 0.79-0.89 OOD),
without being told which answers are misleading. Full table: `outputs/tables/misleading.md`. The full report, with
cases, the analysis of the datasets and a known defect in DeepSeek-Coder-V2-Lite's answers: `docs/reports/misleading_peers.md`.

## Reading the table

`outputs/tables/misleading.md`: accuracy per regime under `tilt` (peers + memory), `peers` (peers only) and `solo`
(question only, shared with the base stream), against the honest rows (`swap` is not run by
default: add it to `eval_conditions`); what the record makes of the misleading answers (mean estimate on honest vs misleading answers, the AUC of that
separation, how often its favourite is misleading); and the peers (ratio asked and reached, accuracy before and after,
forced count, usable share).

## Notes

- The record is fit on each misleading stream itself (`record.fit: self`), label-free, as it would be in deployment.
- DeepSeek-Coder-V2-Lite-Instruct runs with `VLLM_USE_V1=0` (set in `configs/models/deepseek_coder_v2_lite.yaml`): its MLA attention has no working
  chunked-prefill path in vLLM 0.8.5's V1 engine on A100.
- Some misleading reading answers are partly right: the grader counts a correct fragment of a long gold answer as wrong
  when its word overlap is under 0.5.
- Time on 8 A100s: the answers 3-5 h for both streams (once); then per dataset features ~40 min, record ~10 min,
  evaluation ~30 min.

Before 2026-09-13 this experiment ran as `bash run_adversarial.sh` (README_adversarial.md, README_misleading_peers.md).
The same work is now `bash run.sh configs/experiments/misleading.yaml`; a regime became a dataset file,
`scripts/adversarial_peers.py` became `pipeline/peers.py --mode misleading`, `scripts/build_adversarial_stream.py` became
`pipeline/streams.py replace`, and results moved from `outputs/peer_adv/` and `outputs/gen/adversarial/` to
`data/<stream>_misleading/<peer>/`, `data/<stream>_misleading_<regime>/` and `outputs/eval/<model>/<stream>_misleading_<regime>/`
(for a few hours on 2026-09-13 they were `outputs/peers/<stream>/misleading/` and `data/<stream>_adv_<regime>/`).
