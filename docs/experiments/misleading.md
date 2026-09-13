# misleading: the six peers answer with misleading but relevant solutions

```bash
bash run.sh configs/experiments/misleading.yaml --smoke                          # 48 events, one regime, every step, ~20 min
bash run.sh configs/experiments/misleading.yaml --steps peers                    # only the misleading answers (3-5 h, done once)
bash run.sh configs/experiments/misleading.yaml                                  # all regimes in regimes_run
bash run.sh configs/experiments/misleading.yaml --set regimes_run=[rates]        # the poison-ratio sweep p000 ... p100
```

Does the record survive peers that are wrong on purpose? Every peer answers every event with a confident, on-topic,
verified-wrong solution once (`peers` step). A regime then chooses which of those answers replace the honest ones
(`streams` step), and the main pipeline runs unchanged on each derived stream `<stream>_adv_<regime>`.

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
events had a usable answer and none of the usable answers graded correct. `summary.shard*.json` of each peer reports the
acceptance rate, forced count, attempts and why the rest were unusable.

The weakest peer here is DeepSeek-Coder-V2-Lite-Instruct: told to argue for a wrong option it often returns an empty
reply, a bare label or a loop ("Final anti anti anti ..."), where its honest answers on the same engine have none of
these. Those replies are rejected, so it ends with the lowest usable share; its remaining events keep the honest answer.

## Regimes and the ratio

Any name of these two forms works without a config entry, and the number is what the stream ends up with (only usable
answers count; shares are rounded to whole events):

- `pNNN`: that share of every peer's answers is misleading, a different set of events per peer, nested across rates
  (p050 poisons p025's events plus more). `p000` is the honest stream rebuilt through the same steps.
- `kN`: exactly N of the six peers are misleading on every event, a different set each time (a lying minority k1-k2
  against a lying majority k4-k5).

Named regimes in the config: `all100` (every answer), `saboteurs2` (peers 1 and 4 always lie), `all50`, `flip` (peers 1
and 4 turn misleading halfway through the record's order), `targeted` (every peer misleading exactly where it was right).
`regimes_run` also accepts the keywords `rates`, `counts` and `sweep` (both) for the `sweep:` block. Each derived stream
has a `manifest.json` with the requested and realised ratio per peer, events by number of misleading peers, forced and
unavailable counts, and accuracy before and after.

## Reading the table

`outputs/tables/misleading.md`: accuracy per regime (tilt, peers, solo, swap and their differences) against the honest
rows; what the record makes of the misleading answers (mean estimate on honest vs misleading answers, the AUC of that
separation, how often its favourite is misleading); and the peers (ratio asked and reached, accuracy before and after,
forced count, usable share). Under misleading peers `peers − solo` should turn negative; the result is how much of that
loss `tilt − peers` recovers, with `swap` at or below `peers`.

## Notes

- The record is fit on each adversarial stream itself (`record.fit: self`), label-free, as it would be in deployment.
- DeepSeek-Coder-V2-Lite-Instruct runs with `VLLM_USE_V1=0` (set in `configs/base.yaml`): its MLA attention has no working
  chunked-prefill path in vLLM 0.8.5's V1 engine on A100.
- Some misleading reading answers are partly right: the grader counts a correct fragment of a long gold answer as wrong
  when its word overlap is under 0.5.
- Time on 8 A100s: the answers 3-5 h for both streams (once); then per regime features ~40 min, record ~10 min per stream,
  evaluation ~30 min.

Before 2026-09-13 this experiment ran as `bash run_adversarial.sh` (README_adversarial.md, README_misleading_peers.md).
The same work is now `bash run.sh configs/experiments/misleading.yaml`; `--regimes X` became `--set regimes_run=[X]`,
`scripts/adversarial_peers.py` became `pipeline/peers.py --mode misleading`, `scripts/build_adversarial_stream.py` became
`pipeline/streams.py replace`, and results moved from `outputs/peer_adv/` and `outputs/gen/adversarial/` to
`outputs/peers/<stream>/misleading/` and `outputs/eval/<model>/<stream>_adv_<regime>/`.
