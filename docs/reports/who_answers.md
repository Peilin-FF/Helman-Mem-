# Who should answer? Readers, peers and the record

2026-09-14. Question: in a multi-agent system with the record, does the final answer need a strong central model, or
is a strong peer chosen by the record enough, and could the central role rotate to whichever agent the record trusts
most? Frozen models, thinking off, the honest streams (`indist6`, 4,319 events; `ood6`, 17,403), the six peers of the
released streams. Every answer is graded by its stream's rule. Numbers: `outputs/analysis/rotating_central.json`,
`outputs/analysis/reader_vs_answerer.json`, `outputs/eval/{q3_4b,qwen3_8b,llama31,phi4_mini}/`. The last section asks
it again with every model answering first, on the honest and the misleading streams. Page:
https://claude.ai/code/artifact/8aba607b-dcd8-4735-8bb1-d642f84c886c.

## Setup

Four readers, each with its own record fit on `train6` (the main-experiment protocol): Qwen3-4B, Qwen3-8B,
Llama-3.1-8B-Instruct (peer_3 of the streams, promoted) and Phi-4-mini-instruct (peer_1, the most reliable peer
in-distribution). Each was evaluated alone (`solo`), reading the six peers (`peers`) and reading them with the tilt
(`tilt`, γ = 3). For the per-event rules, every agent (each reader in each condition, each peer answering alone) gets a
record entry with the same equations and one shared address, PCA-256 of Qwen3-4B's question features plus a constant,
read before write along the stream and updated with the agent's verified correctness. A rule decides per event from the
entries alone.

## Result

| rule | in-dist | OOD |
|---|---:|---:|
| fixed reader + memory: Qwen3-8B / Qwen3-4B / Llama-3.1-8B / Phi-4-mini | 78.0 / 76.6 / 72.9 / 52.0 | 73.8 / 74.0 / 71.1 / 67.2 |
| the same readers, reading without memory | 75.8 / 74.6 / 67.2 / 42.8 | 67.8 / 69.2 / 68.3 / 59.7 |
| the same readers alone | 72.9 / 69.2 / 65.8 / 48.5 | 65.6 / 67.8 / 67.4 / 54.7 |
| the peers alone: gemma / Phi-4-mini / Qwen2.5-Coder / Llama / DeepSeek-Coder / R1 | 30.8 / 60.5 / 53.9 / 58.1 / 38.8 / 43.8 | 57.5 / 52.9 / 54.1 / 58.9 / 36.7 / 60.6 |
| rotating central, chosen by its record as a reader (its tilt answer) | 77.1 | 74.8 |
| rotating central, chosen by its record as an answerer (its tilt answer) | 76.8 | 73.7 |
| most reliable peer's own answer, no reader | 73.2 | 72.0 |
| most reliable agent over readers and peers, one answer per event | 76.4 | 75.1 |
| ceiling: some reader right / some agent right | 82.0 / 84.7 | 81.1 / 86.0 |

By task, readers with memory (in-dist math / reading / code; OOD yes-no / multiple choice / short answer):
Qwen3-8B 93.4 / 87.2 / 39.0 and 84.0 / 86.1 / 54.9; Qwen3-4B 92.9 / 86.7 / 35.1 and 84.9 / 85.8 / 55.1; Llama 89.4 /
84.9 / 27.0 and 82.1 / 82.1 / 52.6; Phi-4-mini 73.2 / 52.0 / 24.0 and 84.3 / 83.0 / 39.6.

Who the rules choose (share of events): rotating by the reader record picks Qwen3-8B 43%, Qwen3-4B 30%, Llama 24%,
Phi-4-mini 4% in-distribution, and 28 / 28 / 25 / 18% on OOD; the agent rule on OOD picks a peer on 40% of events
(R1 18%, Llama 10%).

With misleading peers (50% of every peer's answers, from the misleading report): the most reliable peer's own answer
falls to 57.9 / 63.0 while Qwen3-4B with memory keeps 74.3 / 67.9.

## Reading

1. **Reliability as an answerer does not transfer to reading.** Phi-4-mini is the most reliable peer in-distribution
   (60.5, and 83.5 on reading) and scores 52.0 as a central with memory. Llama-3.1-8B (58.1 as a peer) reaches 72.9,
   four points under Qwen3-4B. Choosing the central by its answerer record is worse than by its reader record on both
   streams. A rotating design must track each agent in the role it will play.
2. **Rotation among readers is worth about a point.** Chosen by the reader record it lands within one point of the best
   fixed reader (−0.9 / +0.8). Some reader is right on 82 / 81% of events, but which one is not predictable from the
   kind of question: the readers' verdicts are too correlated.
3. **Reader strength matters at the bottom, little at the top.** With peers and memory, Qwen3-8B against Qwen3-4B is
   +1.4 / −0.2; Llama is 4 / 3 points lower; Phi-4-mini 25 / 7 lower. The memory adds +2 to +9 to every reader, most to
   the weakest (Phi-4-mini +9.2 / +7.5 over reading without it).
4. **The peers and the record carry most of the value.** With no reader, the most reliable agent's own answer reaches
   76.4 / 75.1, the best rule on OOD and 1–2 points from the best reader with memory; the most reliable peer alone
   reaches 73.2 / 72.0, fourteen points above the best fixed peer.
5. **Where the reader is indispensable.** With misleading peers, selection collapses and the reader with memory holds:
   the reader's own competence is the defence against peers that are wrong on purpose.

Caveats. A promoted peer answers under the central prompt, not the peer prompt it was reliable under: Phi-4-mini's
reading accuracy is 83.5 as a peer and 46.5 alone as a central on the same passages, so part of point 1 is the prompt.
The agent rule's peer entries use the shared question-only address; the streams' own record, which also sees the
answer, selects the peer a little better (74.7 / 72.4 against 73.2 / 72.0). Every reader ran once (vLLM greedy decoding
varies by about 0.1 point between runs).

## Consequence for the design

Before any rotation: with honest peers, "strong peers plus a record that knows whom to trust" matters more than a
strong central model, and a rule that lets the record pick one agent's own answer is within two points of the best
reader. The central model earns its place when peers mislead. If the central role rotates, the chooser is the reader
record (each candidate's reliability when reading, on this kind of question), not the answerer record, and in this pool
the expected gain over the best fixed reader is about one point.

## Everyone answers first, then trust and capability choose

The same four candidates and six peers, on the honest streams and at 50% and 100% misleading peers
(`*_misleading_p050`, `*_misleading_p100`). Every model first answers alone, and one record covers all ten answers with
one address format, question and answer, as for a peer: the ten answers are the ten peer slots of a stream
`<dataset>_all10` (addresses fit on the stream's own features, cold start, read before write). No model rates itself.
Per event, with T the record's top estimate among the six peers (the evidence a candidate reads) and κ_c its estimate of
candidate c's own answer:

    read:  A_c(T) = T·ρ_c + (1 − T)·(κ_c − δ_c)        own answer:  κ_c

ρ_c is c's reading accuracy when the evidence is trustworthy, δ_c how far untrustworthy evidence pulls it below its own
answer. Both are learned per candidate and task type from verified reading outcomes, read before write, as the Bayesian
linear regression of z = y − (1 − T)·κ_c on [T, −(1 − T)] (prior mean (0.5, 0), precision 1). The rule takes the
highest line at the event's T, so the switch points are the lines' intersections. The calibrated form puts the reading
outcome in a record entry with address [probit T, probit κ_c, 1] and the record's own equations (λ = 1). Numbers:
`outputs/analysis/consistent/envelope.json` (the analysis script is not in the repo).

| accuracy % | in-dist | OOD | in-dist 50% | OOD 50% | in-dist 100% | OOD 100% |
|---|---:|---:|---:|---:|---:|---:|
| best fixed central model with memory | 78.0 | 74.0 | 75.5 | 67.9 | 73.4 | 50.2 |
| best single model answering alone | 72.9 | 67.8 | 72.9 | 67.8 | 72.9 | 67.8 |
| most reliable model's own answer, by the record (no reading) | 76.8 | 77.7 | 74.7 | 76.0 | 74.1 | 79.2 |
| answer combination over the ten answers | 77.2 | 75.6 | 75.0 | 75.7 | 74.3 | 74.8 |
| candidates only: read or own answer, by the lines | 77.2 | 76.7 | 74.8 | 75.1 | 72.7 | 79.0 |
| all ten own answers or any candidate reading, by record entries | 77.0 | 77.7 | 74.8 | 75.8 | 73.6 | 79.3 |
| ceiling: some candidate right, reading or own | 85.3 | 87.7 | 84.8 | 87.4 | 84.2 | 86.5 |
| candidates only: share of events answered by reading | 86% | 81% | 83% | 60% | 82% | 10% |

By trust band on OOD (best candidate reading / best candidate's own answer / candidates-only rule): honest, T 0.2–0.4
23.5 / 28.4 / 38.1 and T 0.6–0.8 76.8 / 66.5 / 77.3; at 100% misleading, T 0.2–0.4 46.2 / 65.3 / 76.8 and T 0.6–0.8
94.8 / 88.3 / 94.5. Own answers beat reading below a trust of about 0.4–0.5 and reading wins above it. In-distribution
reading ties or beats the own answer at every trust level (math and code derivations can be checked), so the lines
barely cross.

1. **The choice shifts towards capability as trust falls.** On OOD the candidates-only rule answers by reading on 81% of
   events with honest peers, 60% at 50% misleading and 10% at 100%, handing the rest to the candidates' own answers
   (at 100%: Llama-3.1-8B 32%, Qwen3-4B 23%, Qwen3-8B 22%, Phi-4-mini 14%).
2. **It beats every fixed central model on OOD at every misleading rate** (77.7 / 75.8 / 79.3 against 74.0 / 67.9 /
   50.2) and is within a point of the best fixed reader in-distribution (Qwen3-8B with memory, 78.0 against 77.0 honest).
   Against the answer combination: +2.1 / +0.1 / +4.5 on OOD, −0.2 / −0.2 / −0.7 in-distribution.
3. **The gain is in choosing whose answer, not in reading.** Taking the most reliable of the ten own answers, with no
   model reading at all, already scores 76.8 / 77.7 / 74.7 / 76.0 / 74.1 / 79.2; adding the candidates' readings as
   options changes that by −0.5 to +0.2. At high trust, reading the trusted peer and taking its answer are nearly the
   same; at low trust the own answers win.
4. **The choice needs consistent addresses.** With question-only addresses the most reliable agent's own answer scored
   76.4 / 75.1 on the honest streams from eighteen agents; addressed by question and answer, ten own answers give
   76.8 / 77.7.

Caveats. Every model answers every event (ten generations), and the analysis sees every candidate's reading outcome; a
deployed system reads with the chosen candidate only, so its reading estimates need exploration (sampling from their
posterior), while the own-answer estimates stay fully observed. The straight-line fit gives ρ above 1 for most
candidates and tasks (up to 1.13 on honest OOD, 1.51 at 100% misleading), so the linear form over-promises at high trust;
the record entry stays inside 0–1 and orders the outcomes (Qwen3-4B on OOD: estimates 0.17 / 0.45 / 0.83 against
accuracy 0.05 / 0.45 / 0.97). Correction: the question features in every address (`sem`) are the judge's state averaged
over the peers' prompts, so they already contain the peers' answers; "question-only" above means the address has no
separate answer block.

Consequence. For a system in which every model answers first, the record over all ten answers is the chooser, and the
central model's reading is worth adding only where the evidence can be checked (in-distribution math and code) or
trusted; when trust is low, the most capable model's own answer is the final answer.
