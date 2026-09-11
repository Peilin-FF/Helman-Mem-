# Σ-Mem v2: a Kalman competence memory and a judge trained to read it

Status: design + first measurements (2026-09-04). Code: `feedback_state/kalman_memory.py`,
`feedback_state/kernel_memory.py`, `feedback_state/addresses.py`, `feedback_state/memory_runtime.py`,
`feedback_state/memory_judge.py`, `feedback_state/train_memory_judge.py`,
`tests/experiments/common/evaluate_memory_judge.py`, `scripts/memsim.py`.

## 0. Target

Train a center model that natively stores, updates and uses a memory of peer reliability such that
accuracy rises with the number of stream steps on the training stream, the in-distribution test stream
and the OOD stream, and rises during training, with every improvement explained mathematically.
The model must not memorise the traits of the training peers/datasets; the skill that transfers is
*how to interact with the memory*.

## 1. What was cleared and why

Removed (moved to `~/.rproj-local-trash/sigma-mem-abandoned-20260904`, deleted on the server):

* `posterior_memory.py`, `whitened_memory.py`, the `m_route_*` scripts, `build_whitened_memory_artifact.py`:
  the previous order-invariant memory with a quadratic read-out `u^T M_p u`. It is superseded by the
  linear Kalman read-out (same sufficient statistics, correct posterior mean, +0.3–0.6 pt).
* `train_posterior_steerer.py`, `train_memory_aggregator.py`, `evaluate_posterior_sigma.py`,
  `evaluate_memory_aggregator.py` and their launchers: steering a frozen judge with a scalar lost
  ~1 pt against the memory alone; the aggregator never produced a usable checkpoint.
* All `launch_*.sh` schedulers (they respawned jobs on the shared machine).

Kept: the paper's `symmetric_memory.py` / `train_symmetric_memory.py` / `evaluate_sigma.py` as the published
baseline, `encode_context_features.py` (feature caches), `lora.py`, `ood_routing.py` (answer grading).

## 2. Setting and notation

Stream of events t = 1..T. Event t: question x_t, peers p = 1..P with answers a_{p,t}. The judge must select
one candidate before seeing any label; afterwards the correctness y_{p,t} ∈ {0,1} of every peer is
revealed (s_{p,t} = 2y − 1 ∈ {−1,+1}). The frozen center model gives, for every candidate, the last-token
hidden state of the candidate-judge prompt h_{p,t} (3 layers, 3H dims) and its own Yes/No log-odds z_{p,t};
the mean over candidates is the question representation h_t.

Addresses (unsupervised, fitted once on the training features, no labels): ψ_q(x) = PCA_d(h) and
ψ_c(x, p) = PCA_d(h_p), each standardised and scaled; design `qc` stacks a peer-blocked question part
and a shared candidate part, ψ_p = [e_p ⊗ ψ_q ; ψ_c(x,p) ; 1] (peer block = peer identity, never the
prompt slot).

## 3. The memory: exact Bayesian competence posterior (Kalman / RLS)

Model per candidate row: s = w^T ψ + ε, w ~ N(0, I/λ), ε ~ N(0,1). Sufficient statistics are additive:

    Λ_t = λI + Σ_{τ≤t} ψ_τ ψ_τ^T,      b_t = Σ_{τ≤t} s_τ ψ_τ,

posterior w | D_t ~ N(Λ_t^{-1} b_t, Λ_t^{-1}). Read-out for a query ψ:

    μ(ψ) = ψ^T Λ^{-1} b,   v(ψ) = ψ^T Λ^{-1} ψ,   P(s>0 | D) = Φ( μ / sqrt(1 + v) ),   ℓ = logit P.

Write (Sherman–Morrison): k = Pψ/(1+ψ^T Pψ), P ← P − k ψ^T P, b ← b + sψ. Equivalent form on the
mean m = Pb: m ← m + k (s − m^T ψ). **This is the delta rule with the Kalman gain.** DeltaNet /
Gated DeltaNet / KDA apply S ← S(αI − βkk^T) + βvk^T, one gradient step of the same least-squares
objective with a fixed or gated step β; the Kalman memory is its closed-form optimum with a
direction-dependent step (large where evidence is scarce, small where it is abundant). Forgetting
ρ < 1 (Gated DeltaNet's α, RLS forgetting factor) is available: Λ ← ρΛ + (1−ρ)λI + ψψ^T.

Properties used later:

* **Order invariance** (ρ = 1): Λ and b are sums, so the state after any permutation of the same events
  is identical. Only the causal restriction "decide with the first t events" remains, which is exactly
  the learning curve.
* **Uncertainty**: v(ψ) is the posterior variance along the query; the effective evidence count is
  n(ψ) = (ψ^T ψ / λ) / v(ψ) − 1 (0 at cold start). The probit predictive shrinks uncertain reads to 1/2,
  so an unseen region cannot dominate the decision.
* **Why the curve rises**: online ridge regression has cumulative regret O(d log T) against the best
  fixed linear predictor in hindsight (Azoury–Warmuth / Vovk), so the average excess loss over the
  first T events decays like d log T / T and the plug-in selection converges to the hindsight-optimal
  one. The measured curves confirm both the rise and the saturation (§6).
* **Spectral control** (as in the paper): every write adds a symmetric rank-one term, so by Weyl each
  eigenvalue of Λ moves by at most ‖ψ‖².

The delta rule with a fixed step (`DeltaMemory`) is kept as the ablation "one gradient step vs exact
posterior": it loses 3 pt in-distribution where data is scarce (4,319 events) and 0.3 pt OOD.

Growing-capacity variants (`kernel_memory.py`): episodic Nadaraya–Watson (attention over stored
feedback), random-Fourier-feature kernel memory (GP with an RBF kernel, fixed state) and a hybrid
(Kalman + episodic residual). They exist because a d-dimensional linear memory must saturate once
T ≫ d, while a kernel memory keeps contracting. Measured on frozen Qwen3-4B features they add ≤ 0.15 pt:
the frozen features do not contain instance-level information a richer memory could use (§6).

## 4. Interaction: how the judge reads the memory

For candidate p the memory delivers e_p = [ℓ_p, ν_p, max_{p'≠p} ℓ_{p'}, mean_{p'≠p} ℓ_{p'}] with
ν_p = log(1+n_p)/log(101) (saturating evidence count). The judge scores

    score_p = z_p(θ; e_p injected into the residual stream) + κ · ℓ_p .

* The additive term is the Bayes posterior log-odds under conditionally independent sources: the memory
  supplies the prior "peer p on questions like this", the judge supplies the likelihood of the content.
  Because the memory's term is present during training, the gradient reaching θ only carries what the
  memory cannot explain — the judge is trained to be the *residual* verifier, not to re-learn peer
  traits (the residual/boosting decomposition).
* The residual-stream injection (the paper's `ActivationSteerer`, rank 4, upper half of the layers)
  lets the judge condition its reading of the answers on the reliabilities of *all* candidates, e.g.
  weigh agreement between a reliable and an unreliable peer.
* Selection accuracy is optimised directly: L_t = −log Σ_{p correct} softmax(score)_p is the negative
  log-probability that the selected candidate is correct (events with 0 or P correct candidates carry
  no signal and are skipped). The earlier "first-correct cross-entropy" is wrong when several
  candidates are correct.

## 5. Training objective = area above the learning curve

Episodes are windows of the training stream; each episode starts from an empty memory and, per event,
randomly permutes the candidate slots shown to the judge (the memory is indexed by peer identity, the
judge sees anonymous positions):

    L(θ) = E_episode  Σ_t  −log P_θ( selected correct | x_t, a_t, S_{t−1} ),   S_t = KalmanUpdate(S_{t−1}, ψ_t, s_t).

The memory has no trainable parameters, so the loss can only fall by reading it better; peer
permutation removes position-as-identity; the cold start every episode means the adapters never see a
warm memory they could substitute with stored traits. Accuracy inside an episode rises with t (memory
fills), and across optimiser steps (interaction improves) — both are logged.

Trait overfitting is measurable: a ridge fitted *offline* on the training stream transfers to OOD at
57.9% (judge alone 56.0%), while the *online* memory on the same feature class reaches 62.3% on the
OOD stream itself. What is learned offline about these peers does not carry over; the online skill does.

## 6. Measurements so far (Qwen3-4B, frozen features, `scripts/memsim.py`)

OOD stream (17,403 events; judge 55.98, majority 57.21, per-source oracle route 61.2–61.3,
per-event oracle 69.6):

| arm | fixed | shuffled | windows (shuffled) |
|---|---|---|---|
| Kalman `q` (d=512) | 62.27 | 62.15 | 62.6 60.9 61.5 62.2 62.6 63.1 61.5 62.4 61.8 62.8 |
| Kalman `qc` (d=512) | 62.29 | 62.32 | 62.9 61.0 61.7 62.6 62.9 62.9 61.6 62.5 62.1 63.0 |
| delta rule `qc` (d=256) | 62.02 | 61.71 | |
| kNN `qc` / hybrid `qc` (d=256) | 62.31 / 62.43 | 61.94 / – | |
| RFF-2048 `qc` | 61.36 | | |
| hindsight ridge `qc` | in-sample 62.9, 5-fold CV 62.4, fitted on train 57.8 | | |

In-distribution stream (4,319 events; judge 57.33, majority 60.25, per-source oracle 63.7, oracle 75.2):

| arm | fixed | shuffled | windows (fixed) |
|---|---|---|---|
| Kalman `q` (d=512) | 67.77 | 67.19 | 61.0 64.8 70.4 66.2 67.1 67.1 69.7 71.3 69.4 70.6 |
| Kalman `qc` (d=512) | 68.72 | 68.19 | 61.7 66.0 71.1 67.1 67.6 69.2 71.1 72.2 69.7 71.5 |
| delta rule `qc` (d=256) | 66.47 | 66.40 | |
| hindsight ridge `qc` | in-sample 71.2, CV 69.2, fitted on train 69.0 | | |

Reading: the online memory reaches the hindsight ceiling of its feature class on both streams
(62.3 vs 62.4 CV; 68.7 vs 69.2 CV); the rise is fast (cumulative 53% after 250 events → 62.5% after
1,000 on shuffled OOD) and then flat because the class is exhausted. Voting on answer agreement
(Nitzan–Paroush weights logit p) changes nothing once the memory is present. The judge's own margin
carries no information beyond its hidden state (`cm` = `c`). Therefore further gains — and a curve that
keeps rising — must come from the judge learning to verify content *given* the memory (§4–5), which
is what `train_memory_judge.py` trains; the control run with `--memory off` isolates the interaction.

## 7. Evaluation protocol

`evaluate_memory_judge.py`: cold-start memory, decide-then-update, fixed and shuffled orders, on
`data/mixed_train_big` (as a stream), `data/indist`, `data/ood`; reports total accuracy, 10 windows,
cumulative accuracy at 250/500/1k/2k/4k/8k/16k events, and from the same pass the judge-only and
memory-only selections. Center models for exploration: Qwen3-4B and Qwen3.5-4B.

## 8. Two more learners on the same objective

**Meta-learned address** (`train_address.py`). The PCA address is unsupervised; the Kalman recursion is
differentiable (Sherman–Morrison), so the projection W can be trained on cold-start episodes of the
training stream with the same online selection loss, gradients flowing through the whole recursion.
W is then shaped for *online learnability* rather than for offline classification — the slow-weight /
fast-weight split of DeltaNet, TTT and Mesa layers, with the fast weights solved in closed form. Whether
such an address transfers OOD is an empirical question answered by `memsim.py --proj-ckpt`.

**Test-time adaptation of the judge** (`evaluate_memory_judge.py --online_lr`). After every event's
feedback the judge takes one optimiser step on its own selection loss (LoRA, steerer, κ). The stream
then has two memories on two timescales: the Kalman state (closed-form, per event, order-invariant)
and the adapter weights (gradient, slow). The Kalman memory captures what is linear in the frozen
representation; the adapter can keep improving the nonlinear part (content verification on the new
domain), which is the only route to a curve that keeps rising after the linear memory has saturated.
Decision-then-update is preserved: the step is taken after the decision of the event that supplied the
labels.

Measured (Qwen3-4B, design `qc`, d = 64, 400 episodes of 512 events on the training stream, memory route):

| stream (order) | PCA-64 | meta-learned-64 | PCA-256 |
|---|---|---|---|
| OOD (fixed / shuffled) | 61.86 / 61.88 | 61.64 / 61.78 | 62.29 / 62.32 |
| in-distribution (fixed / shuffled) | 68.05 / 67.86 | **70.22 / 70.04** | 68.86 / 68.63 |
| in-distribution, cumulative after 250 events (fixed) | 61.6 | **69.2** | 61.7 |
| training stream (fixed / shuffled) | – | **79.05 / 79.07** | 76.84 / 76.71 |

The address learned for online learnability learns faster and higher in-domain (cold-start accuracy
after 250 events 69.2% vs 61.6%) and is neutral OOD: what it encodes about math/code/RAG questions
does not describe the OOD tasks, but it does not hurt either.

## 9. Training curves of the judge (Qwen3-4B, one pass over the 17,709-event training stream)

Accuracy of the selection in consecutive tenths of the pass (memory reset every 4,096 events, slots permuted
per event; `mem` = the memory's own argmax on the same events):

| run | tenths of the training pass | final |
|---|---|---|
| judge trained **with** memory (fused) | 73.1 77.6 74.8 76.6 78.7 78.9 79.9 80.0 78.7 79.8 | 77.8 |
| judge trained **without** memory (control) | 69.0 76.2 76.7 78.4 79.8 79.1 80.4 80.3 78.8 80.4 | 77.9 |
| memory alone on the same events | 74.2 77.8 73.9 75.5 75.8 76.1 76.3 75.8 74.9 75.6 | 75.6 |

Both judges rise across the pass (the interaction / verification skill improves with optimiser steps); the
memory lifts the cold start (73.1 vs 69.0 in the first tenth) and the trained judge overtakes the memory
alone in-domain. Whether the *with-memory* judge transfers better is measured on the held-out streams (§10).

## 10. Held-out streams: judge + memory (Qwen3-4B)

In-distribution stream (4,319 events, shuffled order, cold-start memory, decide-then-update):

| arm | total | first half → second half | cumulative @250 / @1000 |
|---|---|---|---|
| frozen judge | 57.33 | 57.5 → 57.2 | 60.8 / 57.5 |
| frozen judge + memory prior (κ=1, untrained fusion) | 65.83 | 65.4 → 66.2 | 65.6 / 63.9 |
| memory alone (Kalman `qc`, d=256) | 68.4–68.5 | 68.2 → 68.8 | 68.0 / 66.4 |
| judge trained **without** memory | 70.99 | 71.4 → 70.6 | 70.4 / 70.6 |
| judge trained **with** memory (fused) | 70.94 | 71.5 → 70.4 | 70.8 / 71.1 |
| per-event oracle | 75.23 | | |

Reading: in-domain the LoRA judge learns to verify math/RAG/code answers (57 → 71) and the memory adds
nothing on top of it, because what the memory knows in-domain (which peer is reliable on which kind of
question) is also learnable from the answers' content and style. The decisive test is OOD (§10.2, pending),
where the judge's in-domain verification does not transfer but the memory relearns the peers online.

## 11. Capacity and forgetting (Qwen3-4B, Kalman `qc`, shuffled order)

| address dim d | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| OOD | 61.10 | 61.55 | 61.90 | 61.98 | 62.32 | 62.25 | 62.29 |
| in-distribution | 65.73 | 67.15 | 67.77 | 68.14 | 68.19 | 68.21 | 68.19 |

The ceiling is reached around d = 256 and does not move beyond it: with T ≫ d the O(d log T) regret term
is negligible and the class itself is the limit (§6). Forgetting (ρ = 0.999, the Gated-DeltaNet decay):
OOD fixed order 62.36 vs 62.29, shuffled 61.45 vs 62.32; in-distribution 67.6 vs 68.6 — the peers'
competence per question type is stationary, so forgetting only throws evidence away. The order-invariant
memory (ρ = 1) is the right default; ρ < 1 is reserved for counterfactual reliability shifts.

### 10.2 OOD stream (17,403 events, shuffled order)

| arm | total | first half → second half | cumulative @250 / @1000 / @4000 |
|---|---|---|---|
| frozen judge | 55.98 | 55.8 → 56.2 | 52.8 / 57.4 / 55.7 |
| judge trained **without** memory | 59.60 | 59.6 → 59.6 | 52.8 / 59.9 / 59.5 |
| memory alone (Kalman `qc`, d=256) | 62.16–62.32 | 62.2 → 62.1 | 51.2 / 62.1 / 61.8 |
| judge trained **with** memory (fused) | 61.75 | 61.8 → 61.7 | 52.8 / 62.3 / 61.6 |
| memory + trained judge's score as an online regressor (`qcm`) | 62.08 | 62.2 → 62.0 | 51.6 / 62.2 / 61.8 |
| per-source oracle route | 61.32 | | |
| per-event oracle | 69.56 | | |

Reading: OOD the memory is the transferable component (+6.3 over the frozen judge, +2.6 over the trained
judge); the trained judge's verdict carries no information the memory does not already have (adding it
as an online-learned regressor changes nothing), so the fused system sits at the memory's level. Together
with §10.1: the interaction trained in-domain yields max(judge, memory) in each regime, not a synergy,
because both read the same frozen representation — the memory extracts its linear, transferable part
online, and the judge's extra verification skill is domain-specific. This bounds what selection can
achieve and motivates the generation stage (§12).

### 11.1 Qwen3.5-4B replicates the memory results

Memory route (Kalman `qc`), shuffled unless noted: OOD 62.40 (judge 59.54, per-source oracle 61.32, hindsight CV 62.56,
ridge fitted on train → OOD 57.96); in-distribution 67.2 (`q`, judge 60.01); training stream 77.0 (judge 61.7,
hindsight CV 77.3). Meta-learned address d = 64 vs PCA-64: in-distribution 70.36 / 70.11 vs 68.88 / 68.12
(fixed / shuffled), training stream 79.07 / 79.23 vs 76.25 / 76.35, OOD 61.72 / 61.70 vs 61.76 / 61.85.
Judge trained with memory, one pass: 77.60 (tenths 73.7 → 80.4); in-distribution evaluation running.

### 10.3 Test-time adaptation: the curve keeps rising OOD

Judge trained with memory, then evaluated on shuffled OOD while taking one AdamW step (lr 5e-5, every 2 events)
on its own selection loss after each event's feedback (`--online_lr 5e-5 --online_accum 2`):

| arm (OOD shuffled, Qwen3-4B) | total | first half → second half | windows |
|---|---|---|---|
| memory alone | 62.16 | 62.2 → 62.1 | 63.0 61.2 61.6 62.6 62.6 63.1 61.2 62.2 61.5 62.6 |
| judge + memory, frozen adapters | 61.75 | 61.8 → 61.7 | 62.8 60.6 61.3 62.6 62.0 62.5 60.7 61.7 61.2 62.3 |
| judge + memory, **adapting online** | **63.03** | **62.5 → 63.6** | 62.9 61.6 61.9 63.1 63.0 63.7 62.0 **64.3 64.0 63.8** |

In-distribution the same procedure gives 71.57 vs 70.94 (flat). Two memories on two timescales: the Kalman
state absorbs the linear, order-invariant part of peer competence within ~1,000 events; the adapters
keep learning the content-verification part from the same feedback, which is what lifts the last three
tenths of the OOD stream above the linear memory's plateau. Controls running: the same adaptation
without the memory, a second shuffle seed, and the Qwen3.5-4B replica.

Qwen3.5-4B, judge trained with memory (frozen adapters): in-distribution 71.54 (memory 68.70, frozen judge
59.99), OOD 62.40 (memory 62.27, judge-with-steering 62.00, frozen judge 59.54).

Qwen3.5-4B, same protocol on shuffled OOD: judge + memory adapting online **63.82** (63.3 → 64.4; windows
62.9 62.6 63.0 63.6 64.3 64.6 63.1 65.1 64.5 64.4) vs memory alone 62.27, judge + memory with frozen adapters
62.40, frozen judge 59.54. In-distribution controls: judge trained without memory 71.82, with memory 71.54.

**Controls (Qwen3-4B, OOD shuffled, same adaptation schedule).** Judge trained *without* memory, adapting
online with the memory off: **63.08** (62.5 → 63.6; last tenths 64.4 64.0 64.0). Second shuffle seed,
judge + memory adapting: 63.35 (62.5 → 64.2). So the rising OOD curve is produced by the adapters
learning from feedback; on top of an adapting judge the explicit Kalman state adds nothing (63.03 vs 63.08).
Reading: the LoRA weights updated online *are* a memory — the slow, gradient-updated one — and given
thousands of feedback events they absorb what the closed-form state knows. The Kalman memory remains the
cheap component (no backward pass, one rank-one update per event, order-invariant, with calibrated
evidence counts) and the one that is exact after a handful of events; the adapting judge is the one
that keeps rising.

## 12. Generation stage: the central model answers (Qwen3-4B, in-distribution stream, shuffled)

The memory is written only with the peers' labels, so its trajectory is independent of the generations:
prompts with the exact decide-then-update state are precomputed (`scripts/build_generation_prompts.py`),
training is batched SFT on (prompt, correct target) pairs (`train_memory_generator.py`), evaluation is
batched greedy decoding + graders (`evaluate_memory_generator.py`; code via sandboxed test execution).

| arm (accuracy of the generated answer) | events | total | math | rag | code |
|---|---|---|---|---|---|
| frozen model, question only | 4,319 | 60.06 | 89.5 | 59.6 | 22.3 |
| frozen model, question only (first 2,000) | 2,000 | 61.3 | | | |
| frozen model, peers with memory annotations (first 2,000) | 2,000 | **63.30** | 90.6 | 66.5 | 21.7 |
| LoRA SFT on peers-only prompts (targets: shortest correct peer response) | 4,319 | 62.56 | 74.9 | 75.7 | 20.0 |
| frozen model, peers **without** annotations (first 2,000) | 2,000 | 62.95 | 90.6 | 65.7 | 21.7 |
| LoRA SFT on memory-annotated prompts (same targets) | 4,319 | **63.02** | 78.4 | 74.0 | 20.9 |
| Qwen3.5-4B, LoRA SFT on memory-annotated prompts | 4,319 | 63.56 | 77.2 | 75.6 | 21.5 |
| best peer per task (reference) | | | 67 | 83 | 19 |
| selection oracle (some peer right) | 4,319 | 75.23 | | | |

The frozen center model already beats every peer on math (89.5 vs ≤ 67) and code (22 vs ≤ 19) and is far
below the best peer on reading comprehension (59.6 vs 83); the memory annotations move it towards
copying the reliable peer there (+6.9 on rag) without hurting math. The peers-only SFT learns to copy peers (rag 59.6 → 75.7) but loses its own math ability
(89.5 → 74.9) because the targets are peer solutions: the next target rule should be "own solution when
it is correct, else a correct peer's" (self-distillation), so the model learns to defer only when the memory
says a peer is more reliable than itself. Memory-annotated SFT, the peers-without-annotations control and
the OOD runs are in progress.

### 12.1 Why the fine-tuned generators lose math (measured on the same 4,319 in-distribution events)

| arm | math acc | mean output (chars) | math acc when every peer is wrong | answer matches a peer | matched peer was wrong | rag output (chars) |
|---|---|---|---|---|---|---|
| frozen, question only | 89.5 | 364 | 54.0 | 86.7% | 3.8% | 233 |
| frozen, peers with memory annotations (2k) | 90.6 | 358 | 50.8 | 90.3% | 5.0% | 458 |
| SFT on peers-only prompts | 74.9 | 173 | 27.3 | 91.8% | 20.2% | 23 |
| SFT on memory-annotated prompts | 78.4 | 182 | 30.4 | 92.8% | 18.1% | 23 |

Code, every peer wrong: frozen 8.2% correct, both SFT models 0.0%. The targets (the shortest correct peer
response; for code always a peer's program) taught three things at once: shorter outputs (reasoning halved),
copying (a wrong peer's answer is copied five times more often), and never solving alone when the peers fail.
Supplying peer answers to the *frozen* model costs nothing (math 89.5 → 90.6); the loss is entirely the
imitation objective, i.e. distilling a stronger solver (89.5% on math) into a weaker one's traces.

### 12.2 Training-free alternative: the central model as a fourth voter (`scripts/self_vote_sim.py`)

The memory tracks the reliability of the central model's own answer alongside the three peers (four heads
on the question address); the decision is the Nitzan–Paroush vote over answer groups of {own, peers} with
weights logit P(source correct | history). No fine-tuning, own reasoning untouched.

| in-distribution, shuffled (Qwen3-4B) | total | math | rag | code |
|---|---|---|---|---|
| frozen model alone | 60.06 | 89.5 | 59.6 | 22.3 |
| memory selection among peers | 67.59 | 78.1 | 84.2 | 20.5 |
| plain majority over {own, peers} | 62.91 | 89.9 | 65.4 | 22.3 |
| **memory-weighted vote over {own, peers}** | **70.48** | 87.5 | 83.2 | 22.5 |
| best fine-tuned generator (memory-annotated SFT) | 63.02 | 78.4 | 74.0 | 20.9 |
| oracle over {own, peers} / over peers only | 79.28 / 75.23 | 94.4 / 87.8 | 90.8 / 89.3 | 36.2 / 30.5 |

Training stream (17,709 events, fixed order), same protocol: frozen model alone 67.86; memory selection among
peers 76.11; plain majority over {own, peers} 70.70; **memory-weighted vote over {own, peers} 80.28**
(79.7 → 80.9 across the stream; rag 78.5, math 92.1, code 38.2 vs peers-only oracle 41.2 and own 24.3);
oracle over {own, peers} 87.15 vs peers-only 82.60.

OOD (Qwen3-4B, first 4,000 shuffled events; frozen model answering directly): model alone **67.07** (boolean 84.4,
MCQA 81.4, short-answer 41.2) — five points above the best peer selection (62.3) and within 2.8 of the peer-selection
oracle (69.85); memory selection among peers on the same events 61.18; memory-weighted self-vote **68.17**; oracle over
own + peers 79.17. Fine-tuned generator (memory notes) answering from the question only, in-distribution: 46.82
(math 21.8 vs 89.5 frozen): the imitation fine-tune removed the model's own solving.

## 13. Making the judge need the memory: reliability-shift episodes (removed 2026-09-05)

*Setup removed after the results below: on held-out streams the shift-trained judges are indistinguishable with and without the memory. The code (`feedback_state/reliability_shift.py`, `--cf_episodes`, `--cf_profile`) is gone; the table stays for the record.*

Diagnosis of §10: with fixed peers on a stationary stream, reliability is a function of answer style
and question type, the adapters learn that function directly, and the memory is redundant *during
training* — so the judge never learns to read it, and OOD (where the learned function is wrong) it has
no habit of consulting the memory that is right (62.2 vs the fused 61.75).

Remedy (`feedback_state/reliability_shift.py`, `train_memory_judge.py --cf_episodes on`): every training
episode (1,024 events, cold-start memory) draws a hidden corruption rate per peer from {0, 0, 0.3, 0.6,
0.9}; at that rate the peer's *correct* answers are replaced by wrong ones that keep its style (math: its
own solution with the final number changed; RAG/code: its own response to another question). Within an
episode reliability is no longer a function of content or style; the only channel to it is the feedback
the memory accumulates. A judge trained on these episodes must read the memory to know whom to trust,
and must keep verifying content to catch the corruptions it can. The memory uses the question-address
design `q` (per-peer heads) because the cached candidate features belong to the original answers.

This is the in-context-learning / meta-learning condition in its usual form: a quantity that is
predictable from the weights never gets learned in-context; a quantity that changes from episode to
episode must be read from the state. It is also what TTT layers, DeltaNet and KDA rely on — their
fast-weight state only carries information the slow weights cannot store — and the reason the paper's
counterfactual streams exist (there the strong peer's correct answer is swapped with a weak peer's
wrong one; here the corruption keeps the style so that it cannot be detected from the swap itself).

Test protocol: natural streams (in-distribution and OOD, shuffled) exactly as before, plus shifted
in-distribution streams with a fixed profile for the whole stream (peer 1 — the RAG expert — corrupted at
0.9; peer 0 — the math expert — corrupted at 0.9), for: the judge trained with memory on shift episodes,
the same without memory (control), the same with the meta-learned address, and the earlier
natural-trained checkpoints and the frozen judge.

### 13.1 Results of reliability-shift training (Qwen3-4B, in-distribution, shuffled)

| judge | natural stream | shifted stream (peer 1 corrupted 0.9) |
|---|---|---|
| frozen | 57.33 | 53.16 |
| frozen + memory prior (κ = 1) | 65.83 | 59.46 |
| memory alone | 68.5 | 61.4 |
| natural-trained, no memory | 70.99 | 63.07 |
| natural-trained, with memory | 70.94 | 63.42 |
| shift-trained, no memory | **71.38** | 63.09 |
| shift-trained, with memory | 70.97 | 63.23 |
| shift-trained, with memory, meta-learned address | 71.13 | 63.37 |

Second profile (peer 0 — the math expert — corrupted 0.9): frozen 54.69, frozen + prior 61.80, natural-trained no memory
65.59 / with memory 65.73, shift-trained no memory 66.66 / with memory + learned address 66.36 (memory alone 63.1).
Frozen judge with the memory written into the prompt as text (natural stream): 66.64 (its own score with the notes 64.67,
from 57.33 without), i.e. the text channel is read, but the total still sits below the memory alone (67.4).
OOD, shuffled: shift-trained judge with memory + learned address 61.28 (memory alone with that address 61.22, judge-with-steering 60.79).


Inside the shift episodes the memory lifts the coldest tenth (47.3 vs 44.2) and nothing else (65.2 vs 64.9 over
the pass). On held-out streams the shift-trained judges are indistinguishable with and without memory, and
shift training itself is a small regulariser for verification (+0.4 natural). Reading: the corruptions (a
changed final number, an off-topic swap) are verifiable from content, and so are most natural errors on these
benchmarks; the judge learns verification, which beats the memory's base rate even under a 90% shift
(63 vs 61). Peer-reliability information is only decisive where verification fails — the frozen judge
(+11 / +6 from the memory) and the OOD stream with frozen adapters (+2.6) — not for a trained verifier on
in-domain data. This is a property of the task family, not of the memory: with three fixed peers whose
answers can be checked, "who is usually right" adds little to "is this right".

## 14. Reversible addresses (`addresses.AddressMap`, `train_address.py --arch`)

What is already reversible: the Kalman state (Λ, b) is a sufficient statistic (lossless under the model, every
write has a closed-form downdate, order-invariant); the read-out Λ⁻¹ is the invertible change of basis that
separates overlapping question directions (raw cos² ≈ 0.5 → whitened ≈ 0.002); the DeltaNet/KDA transition
(I − βkkᵀ) at β = 2 is a Householder reflection, i.e. an orthogonal, lossless rewrite. The only lossy step is
the PCA address, and the dimension sweep (§11) shows the discarded directions carry nothing usable.

Two reversible learned maps, trained with the online-learnability objective of §8 (d = 64, 400 episodes):
`orthogonal` (Q = qr(W), norm-preserving, exactly decodable: x̂ = Q z s) and `flow` (two affine-coupling
layers on the PCA coordinates, a bijection with closed-form inverse). Memory route, Qwen3-4B:

| address (d = 64) | OOD fixed / shuffled | in-dist fixed / shuffled | train fixed / shuffled |
|---|---|---|---|
| PCA | 61.86 / 61.88 | 68.05 / 67.86 | 75.04 / 75.24 |
| learned linear | 61.64 / 61.78 | **70.22 / 70.04** | **79.05 / 79.07** |
| learned orthogonal (decodable) | 61.68 / 61.87 | 69.25 / 68.84 | 77.36 / 77.53 |
| learned flow (invertible, nonlinear) | 61.56 / 61.97 | **70.22 / 70.25** | **80.51 / 80.50** |

The orthonormal constraint keeps about half of the linear map's in-domain gain while making every read-out
decodable back into hidden-state space. The invertible flow is the best address measured: it matches the linear map
in-distribution and adds 1.4 points on the training stream (+5.3 over PCA) while losing nothing (closed-form inverse).
OOD all addresses coincide within 0.4 — the information limit of the frozen features, not the transform, sets the
ceiling there.

### 13.2 Memory written into the prompt as text (`--memory_text on`)

The rank-4 residual injection may simply be too weak a read channel, so the memory's evidence was also written
into each candidate's judge prompt as a note ("estimated probability correct 0.82, 120 similar past cases") with
one sentence explaining it. Frozen judge reading the notes (natural in-distribution stream): 66.64, its own score
with the notes 64.67 (from 57.33 without them) — the channel is read. Trained judges: natural episodes 77.90 on
the pass (72.9 → 80.4; identical to the no-memory judge's 77.90), shift episodes 65.36 (no-memory 64.88,
steering-only memory 65.23). Held-out evaluations of both are running.

Qwen3.5-4B, shift-trained judge with memory, OOD shuffled: 61.63 (memory alone 62.3).

Held-out (Qwen3-4B, shuffled): memory-as-text judge trained on natural episodes: in-distribution 71.17, shifted
(peer 1 at 0.9) 63.05; trained on shift episodes: 71.43 and **63.63** (the best of all judges on the shifted stream,
+0.5 over the shift-trained no-memory judge at 63.09). OOD evaluations running.

## 15. Where this leaves the selection judge

Every read channel (residual steering, logit prior, prompt text), every training distribution (natural, hidden
reliability shifts), every address (PCA, learned linear, orthogonal, invertible flow) gives the same answer on
Qwen3-4B and Qwen3.5-4B: a judge whose adapters are trained to verify content ties a judge that also reads the
memory, within ±0.5 points, on natural in-distribution streams (71.0–71.4) and on streams with a hidden 90% shift
(63.1–63.6). The memory is decisive exactly where verification is unavailable: the frozen judge (+8.5 in-domain,
+6.2 OOD), the frozen judge under a shift (+6.3), and OOD with frozen adapters (+2.2). Test-time adaptation makes
the OOD curve rise (63.0–63.8 from 59.6–60.1) with or without the explicit state, because the adapters then are
the memory. The configuration that beats both the frozen model and every model trained without the memory is the
one that adds the central model's own answer to the vote (70.5 in-distribution, 80.3 on the training stream).

## 16. Repairing the generator: evidence-gated adapters + on-policy targets

Diagnosis (§12.1 and the question-only evaluation): the memory-annotated fine-tune answering from the question alone
scores 46.82 (math 21.8 vs 89.5 frozen). Cause: LoRA (rank 16, all attention projections, lr 1e-4) trained by
imitation of peer solutions, always with peers in the prompt. Two structural fixes (`feedback_state/lora.py`
`set_lora_active`, `train_memory_generator.py --gated on`, `build_generation_prompts.py --target onpolicy`):

1. **Evidence gate.** The adapters are multiplied by a gate that is 1 only when peer answers / memory evidence are
   in the prompt, 0 otherwise, so π_θ(· | question) ≡ π_base(· | question) exactly. The no-peer behaviour is the base
   model by construction, like the empty memory reading as 1/2 and the steering vector being zero.
2. **On-policy targets (rejection-sampling fine-tuning).** Targets are the base model's own generations filtered by
   the graders: the with-peers+notes generation when it is correct, else the solo generation when it is correct
   (learn to ignore unreliable peers), else the event is dropped. The cross-entropy gradient on self-samples is a
   policy-improvement step under the correctness reward with an implicit KL anchor to the base policy; the model
   learns *when* to adopt a peer's answer, in its own words and at its own reasoning length. Next step if headroom
   remains: GRPO with an explicit KL term on the same prompts.

Other results that landed: Qwen3.5-4B self-vote on the in-distribution stream **69.16** (68.7 → 69.6; model alone
61.77, memory selection 67.35, oracle own+peers 78.95); Qwen3.5-4B shift-trained no-memory judge 71.45 natural /
63.09 shifted (with memory 70.71 / 63.07); memory-as-text judges OOD 60.40 (shift-trained) and 60.54 (natural
episodes) vs memory alone 62.2 — the text channel does not transfer OOD either.

## 17. Macro objective: memory-weighted distillation into the parameters

Reframing (2026-09-04, user): process annotation is expensive; a stream of peer solutions with cheap verdicts is a
multi-teacher corpus. The goal is to internalize what the peers know into the central model's parameters by
plain gradient descent (no adapters), so that the model answers **better when separated from the peers**. The
memory is the teacher-selection signal: it tracks each peer's reliability per question type online, plus the
student's own, and decides whose solution is worth learning from, when the student should trust itself, and which
wrong solutions are informative contrasts. Nothing is imitated at test time because peers are absent at test time.

Recipe (`build_generation_prompts.py --target onpolicy_peer`, `train_memory_generator.py --lora_rank 0 --mode solo`):
for each event, the training target under the **question-only** prompt is (1) the student's own peer-and-memory
informed solution if it is correct (knowledge transferred, style and reasoning length preserved), else (2) the
student's own solo solution if correct (learn to trust itself), else (3) the solution of the most reliable
verified-correct teacher according to the memory (imitate only where the student cannot solve even with hints),
else the event is dropped. Full-parameter AdamW (bf16, lr 1e-5, betas 0.9/0.95), one pass in stream order,
checkpoints every 2,000 micro-steps; the metric is the student's **solo** accuracy on the in-distribution and OOD
streams, and its trend across checkpoints is the trend with training steps.

On-policy target pool (training stream, base model): with peers + memory notes 71.46 (math 94.6, reading 59.9, code 28.5)
vs alone 67.86 (math 92.2, reading 55.4, code 24.3); peers-only oracle 82.60.

### 17.1 RLVR with memory-guided hint distillation (`feedback_state/train_rlvr.py`)

*Note (2026-09-05): the hint here is a peer solution known to be correct before the model answers — the labels-before regime, i.e. the classical baseline of §18. The method itself (labels only after the answer) is implemented in `training/`.*

GRPO-style policy gradient on question-only prompts: G = 4 sampled solutions per prompt (temperature 1), reward
from the task verifier (math equivalence, QA normalisation, sandboxed tests for code), advantage r − mean_group r
(no std normalisation, as in Dr. GRPO), one on-policy update per rollout batch (ratio 1, no clipping), full
parameters, lr 1e-6, optional KL to a reference copy (β = 0 by default). The memory enters where plain RLVR is
blind: a group whose four samples all fail carries no gradient, so for those prompts the student gets ONE hinted
rollout in which the solution of the most reliable verified-correct peer (reliability from the memory) is shown
as a hint with the instruction to solve in its own words; a correct hinted rollout is distilled with a
log-likelihood term under the question-only prompt. Runs: `rlvr_hint` (memory-guided) and `rlvr_plain` (control),
1,500 training prompts (40% math, 40% RAG, 20% code), checkpoints every 500 prompts, evaluated alone on the
in-distribution stream and 4,000 OOD events.

### 17.2 First internalization result (question-only evaluation, in-distribution, 4,319 events)

| model, answering alone | total | math | reading | code |
|---|---|---|---|---|
| frozen Qwen3-4B | 60.06 | 89.5 | 59.6 | 22.3 |
| LoRA fine-tuned on peer solutions (peers-only prompts) | 46.31 | 21.5 | 78.2 | 15.3 |
| LoRA fine-tuned on peer solutions (memory-annotated prompts) | 46.82 | 21.8 | 78.8 | 16.0 |
| LoRA fine-tuned, self-distilled targets (own correct solution, else a correct peer's) | **64.34** | 87.8 | 69.8 | 22.4 |
| full-parameter fine-tune on peer solutions (pure imitation control, no adapters) | 53.37 | 46.3 | 77.5 | 14.4 |

The full-parameter imitation control collapses the same way (math 89.5 → 46.3), so the cause is the objective, not the adapters. The target rule alone separates collapse from internalization: with the model's own correct solutions as
targets the solo accuracy rises 4.3 points over the frozen model (reading +10.2) while math stays within 1.7,
whereas imitation of peers loses 13.7 points overall and 68 on math. The full-parameter memory-weighted
distillation (`mwd_full`), the pure-imitation full-parameter control (`imitate_full`) and the two RLVR runs
(`rlvr_hint`, `rlvr_plain`) extend this with learning curves over training.

### 17.3 RLVR after 1,000 prompts (question-only, first 1,000 in-distribution events; frozen on the same events 60.40:
math 91.1, reading 59.6, code 19.4)

| run | total | math | reading | code |
|---|---|---|---|---|
| GRPO + memory-guided hint distillation (`rlvr_hint`) | 61.90 | 91.1 | 62.6 | 20.3 |
| GRPO, verifier reward only (`rlvr_plain`) | 61.90 | 91.7 | 62.4 | 19.8 |

Training rewards (mean over the pass, temperature-1 samples): 0.633 for both; 482 of 1,500 groups had no correct
sample, the reliable-peer hint was tried on the 264 with a verified-correct peer and produced a correct own-words
solution 71 times. Verifier-reward RL preserves math exactly and adds +1.5 overall at this budget; the hinted
distillation term has not yet separated from the control. Final checkpoints and OOD pending.

Final RLVR checkpoints (1,500 prompts), answering alone on the full in-distribution stream: memory-guided
`rlvr_hint` **61.03** (math 89.8, reading 61.3, code 22.7) vs frozen 60.06 (89.5 / 59.6 / 22.3); plain control `rlvr_plain` 61.24 (89.5 / 61.9 / 22.7) — at this budget
the hinted distillation term does not separate from plain verifier RL (71 hinted samples). OOD (first 4,000, alone): `rlvr_hint` 67.17 (boolean 84.0, MCQA 81.3, short-answer 41.9)
vs frozen 67.07 — in-domain verifier RL neither helps nor hurts the OOD tasks. Checkpoint curve on the first 1,000 events: frozen 60.4 → 500 prompts 61.5 → 1,000 prompts 61.9.

### 17.4 On-policy is a measurable property: the likelihood filter

Mean token loss of candidate targets under the **question-only** prompt, base Qwen3-4B (12 samples each): the
model's own solo solutions 0.018; its own peer-informed solutions 1.92 (they refer to the peers, which are absent
in the solo prompt); teacher (peer) solutions 9.37, several above 20. The first distillation run trained on all
three and started at loss 3.4 — it would have learned to write peer-referencing, foreign-style text without
peers, the collapse mechanism again. `scripts/filter_targets.py` keeps a target only if its mean token loss under
the solo prompt is ≤ τ (= 1.0) and it does not mention the peers, i.e. only samples the current policy could
plausibly have produced (importance weight ≈ 1). This is the KL anchor made explicit and data-level. The
generation logs now keep full texts (the earlier last-1,500-character truncation cut long targets mid-sentence).

Filter statistics (τ = 1.0): kept 4,027 of 15,585 candidates — all 755 own solo solutions, 3,057 of 12,655 own
peer-informed solutions (7,789 dropped because they name the peers, the rest for likelihood), 215 of 2,175 teacher
solutions. The filtered distillation run (`mwd_full`, 4,027 examples) starts at loss 0.5 instead of 3.4. Since
most peer-informed solutions mention the peers, the better source is the hinted own-words solution
(`scripts/generate_hinted.py`: the most reliable verified-correct peer's solution shown as a hint, "solve in your
own words"), generated for every training event with a correct peer; correct outputs are self-contained,
on-policy and peer-informed by construction.

`mwd_full` (4,027 filtered targets, full parameters, lr 1e-5) checkpoint at 4,000 micro-steps, answering alone on the first
1,000 in-distribution events: 61.30 (math 85.0, reading 66.7, code 18.6) vs frozen 60.40 (91.1 / 59.6 / 19.4) — less drift
than imitation, more than the self-distilled adapters; the second round (`mwd2_full`, hinted own-words targets) uses lr 3e-6.

### 17.5 Does the memory matter for internalization? (status)

Clean pair so far: RLVR with memory-guided hints vs plain RLVR, 1,500 prompts — 61.03 vs 61.24 in-distribution,
67.17 vs 67.40 OOD (alone): no effect at this budget. The self-distilled adapter result (64.34) used memory-annotated
prompts and memory-ranked teachers but has no no-memory counterpart yet, so its gain is not attributable. Measured
memory effects remain inference-time: selection OOD +2.2 (frozen adapters), memory-weighted vote +7.6 over plain
majority. Control queued: the hinted-data distillation with a **random** verified-correct peer as the hint
(`mwd2r_full`) against the memory-ranked one (`mwd2_full`), same filter, same lr 3e-6.

### 17.6 Basic RL only; the memory's function under verification scarcity

Per the user: no AEPO/ARPO-style refinements — the RL is REINFORCE with a group-mean baseline (GRPO's advantage,
no critic, no clipping, no KL), and the memory is the only addition. Its clearest function appears when the
verifier is scarce: `--verified_fraction 0.3 --pseudo_reward memory` gives the verifier reward on 30% of the
prompts and, for the rest, reward 1 iff the sample's answer falls in the peers' answer group with the largest
reliability-weighted vote (total log-odds > 0), i.e. the memory turns unverified peer answers into a reward.
Plain RL (`--pseudo_reward none`) gets no signal from unverified prompts. Pair queued (`rlvr_v30_memory`,
`rlvr_v30_none`, 1,500 prompts, alone on the in-distribution stream).

## 18. The experiment, restated (2026-09-05): labels only after the answer

**Setting.** A stream of events. At event t the central model receives the question and the peers' solutions
(three of them on our streams). Nothing tells it which solution is right. It has (i) the memory — the Kalman
posterior of each peer's reliability on this question address, built from the feedback of events < t — and
(ii) its own judgment of the solutions. It decides what to trust and what to learn from, and answers. Only after
the answer is the correctness revealed: of its own answer (the reward) and of the peers' solutions (the memory
update). The evaluation is the model's own answer — along the stream, and afterwards alone on the held-out
in-distribution and OOD streams — not a vote over more candidates.

**Contrast with the classical regime.** Previous methods train from solutions annotated as correct (up to
step-level annotation) before the model learns: the label precedes the learning. Here the label follows the
answer, so the cost of annotation is replaced by the feedback loop, and the memory is what makes the loop
usable — it turns past post-hoc labels into a prior over which present, unlabeled solutions to trust.

**Implementation** (`training/`, verl 0.3.1 from the ARPO repository, plain GRPO — group-mean baseline, PPO
clip 0.2, no critic, no KL, full parameters — with one mechanism on top, `training/sigma_rl/trainer.py`):

- The stream is visited in order (`data.shuffle: False`); the prompt file carries the memory's estimate for every
  peer at every event, computed by the exact recursion of §3 from the labels of earlier events only.
- Per event, n *solo* samples from the question-only prompt (exploration; what is evaluated later) and one
  *guided* sample from the guided prompt: the question with the peers' solutions and the memory's notes
  (`--guided memory`), or only the solution of the peer the memory trusts most (`hint_memory`). The guided
  sample is the model's stream-time answer. All samples are graded after the fact by the verifier.
- The guided answer is trained under the guided prompt (learning to use peers + memory) and re-labelled under
  the question-only prompt (internalisation: its `input_ids` are rebuilt from the solo prompt, so the model is
  pushed toward its own successful solution of the question and never sees the peers at loss time), in the same
  GRPO group as the solo samples. With n solo failures and a correct guided answer the group mean is
  μ = 1/(n+1) and the guided answer's advantage (1−μ)/σ, e.g. +2.67 for n = 8 (+1.79 for n = 4, the value logged
  by the smoke run); nothing is filtered by labels before the update.
- Where the verifier is unavailable after the answer (`--verified_fraction p`), the reward of a solo sample is the
  memory's reliability-weighted vote over the peers' answers (§17.6).

**Baselines from the same code.** *Labels before the answer* (the classical regime): the guided prompt shows a
*verified-correct* solution — the most reliable one by the memory (`--guided hint_label`) or a random one
(`hint_random`) — and only verified-correct guided answers enter the update (`memory.guided_filter=verified`);
this is rejection-sampling distillation with oracle labels, i.e. what §17.1–17.5 did on one GPU. Its purest
form, SFT on the annotated solutions themselves (`training/scripts/train_sft.sh` on `build_sft_data.py
--use_targets on`), is the imitation baseline that collapsed the model's own solving (§12). *Labels after, no
memory*: the peers' solutions without notes (`--guided peers`). *No peers*: plain RLVR (`--guided none`).

**What is measured.** `outputs/rl/<EXP>/metrics.jsonl` per step: `reward/acc_guided` (the stream-time answer, per
task), `reward/acc_solo` (the question-only samples), `memory/*` counts; `val-core/<task>/acc` on 512
in-distribution prompts every 20 steps; and the full question-only evaluation of every kept bf16 checkpoint
(`training/scripts/eval_hf.sh`) on the whole in-distribution stream (4,319 events) and the whole OOD stream (17,403
events; the earlier OOD figures such as 67.07 were on its first 4,000 events and are being recomputed on the full stream). The claims to test: (1) the stream-time
accuracy rises along the stream; (2) the question-only accuracy of the checkpoints rises during training and
beats the frozen model; (3) the labels-after model with memory beats labels-after without memory and reaches
or beats the labels-before baseline, which sees information ours never sees.

**Verified end to end** (2 shared A100s, batch 8 × 4 samples): Qwen3-0.6B and Qwen3-4B with parameter + optimizer
offload (48 s per step at that size; peak 59 GB allocated). At the default 64 × 8 batch on four GPUs a step is
~3–5 min, one epoch of the training stream (~230 steps) about half a day. Two pitfalls are baked into the code:
vLLM 0.8.5 counts other users' memory on the device against `gpu_memory_utilization` (default 0.6; offload when
the box is crowded), and sandboxed code grading runs in Ray task workers with stdin on `/dev/null` (forked from
the many-threaded trainer actor it can hang before exec). Qwen3.5-4B cannot use this stack until a vLLM that
supports it (torch 2.8+) is installed in a separate env.

**Addendum (review of the contributed strict layer).** A second implementation of the protocol was added to
`training/sigma_rl/` (`outcome_protocol`, `outcome_batch`, `outcome_reward`, `audit_outcome_data`). It is label-private by construction — public prompts without reliability notes, a live episode wrapper
around `MemoryRuntime` that grades before it writes, a reward manager that refuses pseudo-rewards — but its
trajectory guards require the rollout to return the memory as a tensor (`peer_evidence`), which no backend does, so
it is a contract for a future tensor-memory path rather than a trainer. Its prohibitions of memory notes as text,
of the question-only re-labelling and of mixed GRPO groups are design choices, not consequences of the protocol
(the label still follows the answer in all three), and the running pipeline keeps them; the strict manager can be
selected with `reward_model.reward_manager=outcome`. Adopted from it: the 8192-token budget (audit: question + all
unclipped peers ≤ 3963 tokens on the training stream) and explicit reporting of any guided prompt over budget.

### 18.1 First result of the labels-after regime (Qwen3-4B, `q3_4b_grpo_memory`, 2026-09-05)

One epoch of the training stream (17,709 events, 276 steps of 64 prompts × 4 solo samples + 1 guided answer, 3.7 h on
four A100s). Validation on 512 fixed in-distribution prompts, question only, greedy: 61.7 → 67.0 (step 20) → 71.3 (step 100,
peak) → 66–70 for the rest of the epoch → 67.8 (step 276); reading 62.8 → 76–82, math 86–90 (start 89.5), code 21.8 → 16–21
late. On the stream the question-only samples rise from 0.68 to ≈0.75 and close the gap to the guided answer (0.012 → 0.002).

Final checkpoint, question only, on the **whole** test streams:

| stream | frozen Qwen3-4B | memory-trained (step 276) |
|---|---|---|
| in-distribution (4,319) | 59.92 (math 89.2, reading 59.1, code 23.0) | **67.05** (math 89.9, reading 75.4, code 20.2) |
| OOD (17,403) | 67.61 (boolqa 83.5, mcqa 82.3, shortqa 41.9) | **68.61** (boolqa 77.0, mcqa 81.2, shortqa 51.2) |

(All evaluation numbers from here on are decoded with vLLM, `evaluate_memory_generator --engine vllm`; the transformers-generate
numbers quoted earlier agree within half a point and were deleted.) The no-memory control (`q3_4b_grpo_peers`, same protocol without
the notes): **67.79** in-distribution (math 90.2, reading 78.6, code 16.6) and **69.55** OOD (boolqa 82.6, mcqa 81.7, shortqa 48.5) —
level with or above the memory run on both streams; its step-70 checkpoint reaches 69.85 in-distribution, above its own final.

The trained model beats the frozen one on both streams with math preserved; the in-distribution gain is reading comprehension
(the task where the peers are stronger than the model), the OOD gain is BIG-Bench Hard (+8.8) against a loss on the yes/no task
(−6.5). Attribution to the memory awaits the no-memory (`peers`), classical (`hint_label`) and plain-RLVR runs, chained next.

Frozen references on the whole OOD stream: alone 67.57; with the peers' solutions and the memory's notes in the prompt 63.26 (boolqa
83.8, mcqa 83.3, shortqa 29.3) — on BIG-Bench Hard, where every peer is far below the model (28.7 / 10.7 / 12.8 vs 42.0), the
untrained model follows the peers and loses 13 points; the memory-trained model alone reaches 50.8 there.

### 18.2 The five regimes on the whole test streams (Qwen3-4B, final checkpoints, alone, vLLM, 2026-09-06)

| regime | labels before | memory | in-distribution (math / reading / code) | OOD (boolqa / mcqa / shortqa) | best in-distribution checkpoint |
|---|---|---|---|---|---|
| frozen model | – | – | 59.92 (89.2 / 59.1 / 23.0) | 67.61 (83.5 / 82.3 / 41.9) | – |
| ours: peers + memory notes | no | yes | 67.05 (89.9 / 75.4 / 20.2) | 68.61 (77.0 / 81.2 / 51.2) | 67.15 @210 |
| no memory: peers only | no | no | 67.79 (90.2 / 78.6 / 16.6) | 69.55 (82.6 / 81.7 / 48.5) | 70.83 @140 |
| classical: all peers labelled | yes | no | 70.71 (88.9 / 85.2 / 17.8) | 65.30 (82.6 / 81.0 / 37.7) | 70.71 @276 |
| **plain GRPO, question only** | no | no | **70.36** (91.8 / 78.6 / 25.6) | **73.81** (85.3 / 84.5 / 55.3) | 70.36 @276 (monotone 66.4 → 69.6 → 70.0 → 70.4) |

Reading: on these streams the peers' solutions in the prompt, with or without the memory's notes and with or without their
labels, are not what improves the central model's own answers; the verifier reward on its own samples is. Peer-conditioned runs
learn more reading (75–85 vs 78.6) at the cost of code and, OOD, of the peers' style (all-peers-labelled: 65.30, BIG-Bench Hard
below frozen). Plain GRPO is the only run that improves every task and rises monotonically over the epoch. The memory's notes do
not separate ours from the no-memory control (67.05 vs 67.79, 68.61 vs 69.55). (The memory-ranked verified-peer variant, 67.84 /
69.72 with math down to 82.6, was dropped from the comparison at the user's request; its evaluations remain on disk.) One seed each; the no-memory run had a
length-degeneracy collapse at steps 229–248 (recovered). Thinking-mode runs (memory, plain; 70 steps, 4096-token responses) are in
progress with thinking-mode frozen references.

## 19. Steering the central model with the record (training-free)

Two methods are retained (2026-09-06); everything else that was tried (record in words, sorted / filtered / favourite-only
prompts, second pass, final-answer reranking) was removed from the code after the study, with its numbers kept on the
steering page (`artifacts/helman_mem_steering.html`, generator `scripts/steer_page.py`).

**Method 1, fusion at the decision** (`scripts/fusion_decision.py`). The model answers alone. Where the peers' answers differ
and the record is not flat, the memory picks the answer with the largest summed log-odds of its supporters,
`score(a) = Σ_{i: v_i=a} [logit(p_i) + log(K−1)]` (Nitzan–Paroush; K = distinct answers on the table), the model's own answer
counting as a supporter with weight `logit(q)`, q = its running accuracy on the task (self-record, read-before-write).
Optional Beta-posterior weights `ψ(α) − ψ(β)`. Probe slices, frozen Qwen3-4B: OOD 71.5 (alone 71.1), in-dist 80.5 (alone 70.8);
with thinking 81.5 (81.6) and 84.7 (72.7). Control: `--swap_record`.

**Method 2, fusion in the attention** (`scripts/steer_attention.py`). Peers in the prompt, no reliability text; every head's
attention over peer i's tokens receives `γ · log(p_i / max_j p_j)` (CrAM's Norm(A ⊙ c)). γ = 3: OOD 67.5 (swapped 62.9),
in-dist 74.8 (swapped 71.7); the strongest in-model steering, still below alone on OOD because the peers are in the prompt.

Analysis of any run: `scripts/memory_use_probe.py analyze` (follow rates, selection events, swapped controls, fusion variants).

## 20. Five-peer streams (2026-09-06)

Peers: gemma-3-4b-it, Phi-4-mini-instruct, Qwen2.5-Coder-7B-Instruct, Meta-Llama-3.1-8B-Instruct, DeepSeek-Coder-V2-Lite-Instruct.
New peers' answers generated with `scripts/peer_answers.py` (same task prompt, sampling and grading as the original peers; OOD budget 96
tokens), merged by `scripts/merge_peers.py` into `data/{mixed_train_big5,indist5,ood5}`; judge features `outputs/context_features/
q3_4b_{big5,indist5,5}_ph`; prompt files `outputs/gen/q3_4b/prompts_{train5_fixed,indist5_shuffled0,ood5_shuffled0}.jsonl`; RL data
`outputs/rl/data/q3_4b_5peer/`. Prompt length: max 4,177 tokens (code), so `max_prompt_length` 8192 stands.

Record quality (`scripts/record_quality.py`), 3 → 5 peers: OOD AUC 0.921 → 0.928, favourite right on mixed events 74.2 → 79.6%, mixed
events 29 → 50% of the stream; in-dist AUC 0.912 → 0.921, favourite 88.7 → 89.8%; train AUC 0.906 → 0.926, favourite 91.1 → 92.2%.
Where the favourite goes against the majority label it is right 94–98% of the time.

## 21. The memory steers the attention, not the prompt (2026-09-07)

Decision: the record never enters the central model's prompt as text.  The prompt is the plain peers prompt
(question, the six solutions as `Peer 1 … Peer 6`, the task instruction).  The record's estimate p_i for peer i
(Kalman state over the frozen judge's features, read before the event is written) is added to the attention scores
onto that peer's tokens in every layer and head:

    b_i = γ · log(p_i / max_j p_j),  γ = 3;  b = 0 for the favourite and for flat records (spread ≤ 0.1)
    softmax(s + b) = Norm(A ⊙ c),  c_i = (p_i / max_j p_j)^γ            (§19 Method 2, CrAM-style tilt)

The tilt is applied in every forward of the policy: rollouts, old log-probs, the reference log-probs of the KL term
(so the penalty compares the same attention) and the update.  γ = 0 recovers the base model exactly.

Engine.  Serving engines take no per-token attention bias; HF generation with the mask hook decodes at 100–200 tokens/s
per GPU (151–324 ms per step at batch 32), which made thinking-mode training infeasible (17 h+ per arm).  vLLM 0.8.5's
Triton attention backend is Python source: `feedback_state/vllm_attn_bias.py` rewrites its prefill kernel
(`prefix_prefill._fwd_kernel`) and paged decode kernel (`kernel_paged_attention_2d`) at install time with one float per
KV slot (a bias cache next to the KV cache, written with the token's K/V) added to the scaled score; requests get their
bias through an in-process registry keyed by the prompt token ids; prefix-cache block hashes include the bias prefix so
KV blocks are never shared across tilts (K/V of tokens after a tilted block depend on the tilt).  Check
(`scripts/check_vllm_tilt.py`, 16 six-peer prompts, 48 greedy tokens): mean |Δ log p| vLLM-tilt vs HF-hook-tilt 0.0099,
vs HF-plain 0.34; vLLM-plain vs HF-plain 0.0135, vs HF-tilt 0.46; argmax agreement 99.1%.  Throughput on one GPU,
thinking on: FlashAttention 4,192 tok/s, patched Triton 1,995 tok/s, HF+hook 100–200.  Rejected engines: TensorRT-LLM,
DeepSpeed (compiled kernels, no verl weight sync), SGLang (same patch effort, not integrated), Megatron (training only).

Trainer wiring.  `data.attn_gamma` / `data.attn_bias_form` (dataset builds `attn_bias` over the padded prompt from
extra_info `peer_spans` + `memory_prob`), `actor_rollout_ref.rollout.attn_bias` (vLLM registry), `actor_rollout_ref.model.attn_bias`
(HF hooks on actor and ref; needs `use_remove_padding=False attn_implementation=sdpa`; one shared steer per process —
verl colocates actor and ref).  Evaluator: `--attn_gamma 3 [--swap_record] [--every k]`.

Pilot (before any long run): every 7th event of the six-peer stream (2,530 events, 25% question-only), group of 8,
thinking on (4,096 tokens), 40 steps, KL 0.001; arms pilot_tilt (γ = 3) and pilot_peers (control); tests with thinking on
in four conditions (peers + tilt, peers, alone, swapped tilt) on in-dist (whole stream) and OOD (every 4th event), the frozen
model under the same conditions as the reference.  Driver: `/mnt/data/peilin/launch_pilot.sh`.

### 21.1 Pilot results (2026-09-07, thinking on; in-dist whole stream 4,319, OOD every 4th event 4,351)

Tilt arm (40 steps, KL 0.001, step 39): in-dist peers+tilt 67.5 · alone 66.0 · peers 65.6 · swapped tilt 62.2; OOD 77.6 · 79.5 · 74.7 · 73.5.
Frozen model: in-dist alone 60.0 · peers 60.9; OOD alone 79.1 · peers 72.0 (tilt / swapped references pending at the time of writing).
Reading: the tilt is a real channel (+2.0 / +2.9 over the same prompt without it; the swapped record costs 5.4 / 4.1); in-dist the
steered model beats alone, on OOD the weak 96-token peers cost 4.8 and the tilt recovers 2.9 (it can only damp, not add -> a
self-record term b_i = γ log(p_i / max(p_max, p_self)) is the next mechanism change).  Training under the tilt added little beyond
the tilt itself so far (frozen+tilt 66.2 vs trained 66.6 on the validation events; best 68.4 at step 20); the big training effect
is own ability with thinking on (alone 60.0 -> 66.0, mostly reading format) from the 25% question-only events.
Dynamics: no collapse (entropy 0.25 -> 0.21, grad norm 0.1-0.5, truncation 2-15%); KL drifted to 0.03-0.09 (control 0.19) after
step 18 with coefficient 0.001 -> run 2 uses 0.01, checkpoints every 10 steps, validation every 20.
Cost: tilt-arm step 547 s (gen 148, old log-probs 66, ref 64, update 269) because the SDPA path pads every sequence to
4,608 + 4,096 tokens (~5.8x the real tokens); dp_actor now trims each micro-batch to its real span (4.5x faster update,
|Δ log p| mean 0.009 vs the padded forward).  Control arm (rmpad + flash) 190-250 s/step on 4 GPUs.
Lesson: monitor scripts must run inside the conda env (a silent `python: not found` hid the dynamics for 6 h).

### 21.2 Full-stream comparison, thinking off (2026-09-08): ours vs the no-memory control

Identical schedule (40 steps, then a full epoch from that checkpoint with the reference reset, KL 0.01, n = 8, 768-token answers).
in-dist / OOD (every 4th): frozen peers+tilt 67.1 / 73.9, peers 64.8 / 69.5, alone 60.6 / 68.5;
ours (run3b, step 276) 75.9 / 75.0, 75.1 / 72.3, 73.4 / 70.5, swapped 73.7 / 67.1;
control (ctrl3b, step 276) 74.9 / 75.1, 74.4 / 72.9, 71.4 / 73.0, swapped 70.5 / 67.9.
Verdict: the tilt is a test-time channel every model reads (+0.5 to +4.4 over the same prompt; swapped record -2.2 to -7.9 on all three);
training under the tilt adds ~1 point in-dist deployed (1.5 SE) and nothing on OOD. Training gains are peer reading and own ability
(ours better alone in-dist 73.4 vs 71.4; control better alone OOD 73.0 vs 70.5). Validation: ours 71.3 -> 77.7 peak -> 76.6; control 66.6 -> 76.2.
Next: gamma sweep + log-odds form on both checkpoints (evaluation only); self-record term; a record-aware training signal
(events where the record and the majority disagree). Notion project: Research Studio / Kalman Mem.

### 21.3 Third arm: question-only training (2026-09-09)

Same schedule and settings as run 3b and the control, with the question alone in every training prompt (`full_solo.parquet`,
17,709 events, no peer block; validation on 512 held-out events under the question-only prompt). Launcher
`training/scripts/launchers/launch_solo.sh`; runs `solo3_q` (40 steps) then `solo3b_q` (277 steps, reference reset, seed 2);
about 40 s per step. Final checkpoint (step 276), thinking off, in-dist / OOD: peers + tilt 73.0 / 74.8, peers 72.5 / 71.3,
alone 71.5 / 72.7, swapped tilt 69.5 / 67.6. Validation alone 61.7 -> 66.8 (step 40) -> 74.0 peak (step 240) -> 72.5.

Reading. (1) Own ability: 71.5 in-dist and 72.7 OOD, level with the control (71.4 / 73.0) and below ours in-dist (73.4):
peer prompts in training cost nothing in own ability. (2) Reading peers is learned: this model gains +1.0 from six solutions
in-dist (ours +1.7, control +3.0) and loses 1.4 from them on OOD; deployed it reaches 73.0 in-dist against 75.9 / 74.9 for
the peer-trained arms. (3) The tilt is read by a model that never saw a peer in training: +0.4 in-dist, +3.5 OOD over the same
prompt; the swapped record costs 3.5 / 7.2. The method was renamed from "Kalman record" to "Bayesian linear record" on
2026-09-08 (the write is the Kalman measurement update of a static state, i.e. RLS; "Kalman" is kept for the gain only).

### 21.4 Numerical stability of the record's covariance (2026-09-09)

With rho = 1 the record keeps P = Lambda^{-1} explicitly and updates it by the Sherman-Morrison short
form `P <- P - k x^T P`, which is symmetric only at the exact gain, so asymmetry could in principle
accumulate over the ~106,000 rank-one writes of one pass over the six-peer stream. Measured
(`scripts/check_kalman_numerics.py`, D = 1793, float64, against the Joseph form and against
Lambda inverted exactly at checkpoints):

| writes  | asymmetry | min eigenvalue | rel. err. of P | rel. err. of m |
|---------|-----------|----------------|----------------|----------------|
| 20,000  | 7.2e-16   | 4.10e-06       | 2.5e-14        | 3.0e-14        |
| 60,000  | 6.7e-16   | 1.39e-06       | 4.8e-14        | 5.2e-14        |
| 106,254 | 6.3e-16   | 7.86e-07       | 6.3e-14        | 6.9e-14        |

Asymmetry stays at machine epsilon and drifts slightly DOWN, not up; the state stays positive
definite; the error against the exact inverse grows sublinearly and reaches only 1e-13 after a full
pass. The Joseph form agrees to two digits and buys nothing. No change to the implementation is
needed, and no re-symmetrisation is required. The shrinking smallest eigenvalue is the record
accumulating evidence, not a numerical defect.

### 21.5 Thinking-mode run, stopped early (2026-09-09 to 2026-09-10)

Tilt arm with Qwen3 thinking ON, 8,192-token answers, prompts rebuilt so every central-model system
prompt ends with "Do not reason more than the question needs: think briefly, then give the final
answer." Otherwise identical to run 3/3b (gamma 3, KL 0.01, group of 8, 64 events per step, 8 GPUs);
one sequence per micro-batch, because at 12.8k tokens the fp32 logits of a micro-batch of four
exceed 80 GB in the log-prob pass.

Phase 1 (40 steps from the frozen model) worked: validation on the 512 held-out in-distribution
events rose 69.5 -> 73.6 overall (rag 72.7 -> 79.7, code 29.4 -> 33.6, math flat at 94.4).
Phase 2 then plateaued for 200 steps -- eleven validations spanning 72.7 to 75.2 with no slope. A
code spike to 38.7 at step 160 fell back to 34.5 at step 180. Training health was fine throughout
(reward ~0.78, entropy ~0.23, KL ~0.005, ~590 s/step); truncation at the budget stayed near 5% and
never trended up, so the reminder plus the doubled budget did solve the pilot's problem of code
answers exhausting 4,096 tokens while still thinking.

Stopped by the user at phase-2 step 201 without running the end-of-run tests, so this arm has no
four-condition numbers. Checkpoints at phase-2 steps 40/80/120/160/200 are on disk. The finding that
carries over is the shape: with thinking on, as with thinking off, essentially all of the gain
arrives in the first 40 steps and the full epoch adds nothing.

### 21.6 Full OOD stream, all four thinking-off models (2026-09-10)

The OOD numbers of 21.2/21.3 used every 4th event (4,351), which also gave the record only a quarter of
the stream's feedback, since it is built read-before-write along the test stream. Re-run on the whole
OOD stream (17,403 events; `training/scripts/launchers/launch_oodfull.sh`, 16 evaluations, 8 at a time,
~13 min per round). Accuracy in percent, standard error about 0.35.

| model | peers + memory | peers | question only | swapped record |
|-------|----------------|-------|---------------|----------------|
| Frozen Qwen3-4B        | 74.04 | 69.16 | 67.80 | 64.16 |
| Ours (tilt arm)        | 75.23 | 72.48 | 70.71 | 66.65 |
| Control (no memory)    | 75.47 | 73.27 | 72.67 | 67.74 |
| Question-only arm      | 75.10 | 71.23 | 72.86 | 67.75 |

Findings.
1. The subsample was representative: every full-stream figure is within 0.7 of its every-4th counterpart.
   Quadrupling the record's feedback did NOT widen the tilt's advantage (frozen +4.4 -> +4.9, ours
   +2.7 -> +2.8, control +2.2 -> +2.2, question-only +3.5 -> +3.9). The record saturates early.
2. The tilt is read by every model, and the swapped record costs 7.4 to 9.9 points everywhere.
3. The whole OOD effect lives in the short-answer tasks. boolqa and mcqa sit at 83-86 in every condition
   for every model; shortqa runs from 31.3 (frozen, swapped) to 58.9 (control, deployed).
4. With n = 17,403 the three trained arms are statistically indistinguishable deployed (75.10, 75.23,
   75.47; SE of a difference about 0.5), which confirms 21.2 on four times the data: training under the
   tilt does not beat training without it.
5. New, and only visible at this sample size: on OOD the memory is worth about four times what the RL
   training is. The frozen model deployed reaches 74.04 against 75.10-75.47 for the trained arms, so
   training adds about 1.3 points while the tilt adds 4.9.

**Note (2026-09-10).** The every-4th OOD evaluations that sections 21.1-21.3 quote were deleted at the
user's request once the full-stream results of 21.6 existed: OOD is now always the whole 17,403-event
stream. The launchers no longer subsample (`--every 4` removed from launch_run3b.sh, launch_ctrl.sh,
launch_solo.sh), `scripts/pipeline6_page.py` reads `eval_oodfull6_<cond>`, and its dead every-4th
results table was removed. The OOD figures in 21.1-21.3 are kept as the historical record of what was
measured at the time; the numbers to quote are those of 21.6, which differ from them by less than 0.7.

### 21.7 A training effect on tilt-conversion, hidden by the aggregate (2026-09-11)

Deployed accuracy says training under the tilt bought nothing (in-dist 75.9 vs the control's 74.9,
about 1.4 SE; full OOD 75.23 vs 75.47). That comparison mixes events where the record said something
with events where it said nothing. `scripts/tilt_training_effect.py` separates them.

The tilt is identically zero when the record is flat (spread <= 0.1), so on those events every model
sees a bit-identical prompt: they are a placebo bin. The quantity of interest is each model's **lift**,
acc(peers + tilt) - acc(peers, no tilt) on the same events, and then whether ours converts more of the
same record than the control does. Both differences are paired event by event, so the errors are on the
per-event difference, not on two independent binomials (an unpaired error is ~7x too wide here).

Full OOD stream, ours minus control, difference of lifts:

| bin (record spread) | n | delta lift | se | |
|---------------------|---|-----------|----|-|
| flat, tilt = 0 (placebo) | 1,702  | -0.18 | 0.18 | noise |
| 0.10 - 0.20              | 3,011  | -0.43 | 0.44 | noise |
| 0.20 - 0.30              | 2,326  | +0.69 | 0.70 | noise |
| > 0.30                   | 10,364 | **+0.93** | 0.29 | 3.2 sigma |
| tilt active (> 0.10)     | 15,701 | +0.63 | 0.24 | 2.6 sigma |
| all events               | 17,403 | +0.55 | 0.21 | 2.6 sigma |

Reading. (1) The placebo bin is clean: where the tilt is exactly zero the two models differ by nothing.
(2) The effect grows with the record's spread and is strongest, and individually significant even after
correcting for the twelve bins tested, exactly where the tilt is largest. (3) The mechanism is visible
in the components: on spread > 0.30 ours scores 78.1 with the tilt and 74.2 without, the control 78.4
and 75.3. Ours is *worse* at reading peers unaided and *better* at converting the record. Training under
the tilt did not make the model better overall, it made it tilt-dependent, and that dependency is what
the aggregate cancels out. (4) In-distribution the same pattern appears with the same signs (+0.39 on
tilt-active events, 0.00 on the placebo bin) but the stream is four times smaller and nothing clears
noise. (5) A separate lean diagnostic (accuracy split by whether the record's favourite peer was in fact
right) does NOT show ours leaning harder than the control, so this is not simply "ours obeys the memory
more"; it is a better conversion of the same tilt into accuracy.

So the honest claim is narrow and supported: **training under the tilt measurably changes how the model
uses the tilt, by about +0.9 points of extra lift where the record is most confident, but it does not
raise deployed accuracy because it also costs the model some unaided peer-reading ability.**

### 21.8 Attention on the peer blocks: the tilt is read identically by every model (2026-09-11)

Hypothesis tested (the user's): the tilt re-weights the peers' tokens, which could teach the central
model to tell reliable peers from unreliable ones on its own, and that would be why training under it
helps. If true, at gamma = 0 the tilt-trained model should attend to the record's favourite (or to the
correct peers) more than the frozen model or the no-memory control do.

Measurement (`scripts/attention_mass.py`, figures `scripts/attention_mass_plots.py`): one eager
forward per prompt, a hook per layer reduces the softmaxed weights to the mass on each peer block for
the last prompt token (the position that emits the first answer token), at gamma 0 and 3, for the four
thinking-off models, 300 prompts per stream with a non-flat record and disagreeing peers. Block mass
scales with block length, so the numbers below are per-token densities (the raw-mass table, in
`outputs/address/attention_mass/summary.json`, has the same model-to-model pattern).

Spearman between the per-token attention on each peer and the record's p_i / the peer's correctness;
"favourite" = attention on the record's favourite over the mean peer (1 = no preference);
"correct share" = share of peer attention on correct peers over the fraction of peers that are correct.

| stream | model | gamma | corr(record) | corr(correct) | favourite | correct share |
|--------|-------|-------|--------------|---------------|-----------|---------------|
| indist | frozen        | 0 | +0.05 | +0.01 | 1.01 | 1.00 |
| indist | ours          | 0 | +0.05 | +0.02 | 1.02 | 1.00 |
| indist | control       | 0 | +0.05 | +0.02 | 1.01 | 1.00 |
| indist | question-only | 0 | +0.08 | +0.04 | 1.03 | 1.01 |
| indist | frozen        | 3 | +0.84 | +0.60 | 2.14 | 1.56 |
| indist | ours          | 3 | +0.86 | +0.62 | 2.12 | 1.56 |
| indist | control       | 3 | +0.85 | +0.61 | 2.13 | 1.56 |
| OOD    | frozen        | 0 | +0.10 | +0.09 | 1.04 | 1.03 |
| OOD    | ours          | 0 | +0.09 | +0.08 | 1.06 | 1.04 |
| OOD    | control       | 0 | +0.10 | +0.09 | 1.05 | 1.04 |
| OOD    | frozen        | 3 | +0.79 | +0.52 | 2.19 | 1.60 |
| OOD    | ours          | 3 | +0.79 | +0.52 | 2.22 | 1.61 |
| OOD    | control       | 3 | +0.80 | +0.52 | 2.21 | 1.61 |

Standard errors are 0.01-0.03 on the correlations. Findings.
1. With the tilt off, no model discriminates the peers at all: per token, attention is uniform over the
   six blocks (favourite 1.01-1.06, correct share 1.00-1.04, correlations at zero), and the four models
   are identical to within 0.03. Training under the tilt taught the model nothing about which peer to
   attend to. The hypothesis is refuted at the level of attention allocation.
2. With the tilt on, the favourite receives 2.1-2.2x the average peer's attention per token, attention
   correlates +0.8 with the record and +0.5-0.6 with actual correctness, and this is the same for all
   four models to within 0.02: the tilt is a mechanical, model-independent re-weighting, which is why
   it works as a test-time channel on every model.
3. The +0.9 extra lift that training under the tilt gives on high-spread events (21.7) is therefore
   not an attention-allocation effect at the prompt positions. It must live downstream of attention
   (how the attended content is resolved) or at the answer positions during decoding, which this
   measurement does not cover. That is the next thing to measure if the effect matters.
4. Depth: the preference for the favourite under the tilt is flat across layers (figure B); without
   the tilt every model attends slightly less to whichever peer the record ranks first in the first
   third of the stack, an artefact of block length, gone in the per-token view.

Figures in outputs/address/attention_mass/: figA (layers x reliability rank per model and gamma),
figB (share on the favourite by layer), figC (one event, peers ordered by the record, correctness
marked), figD (token-resolution map of the last prompt token for one event).

## 22. Next direction: train the judgement itself (decided 2026-09-11)

What the tilt experiments established (21.2-21.8): the memory is a test-time channel every model
reads identically; training under it makes the model tilt-dependent (better conversion of the same
record, worse unaided peer reading, net zero deployed); attention over the peers is uniform for every
model with the tilt off. Nothing we trained ever built epistemic ability, because nothing we rewarded
required it: the reward was final correctness, which is attainable while ignoring the peers, and with
the tilt on the record did the discrimination.

The destination (the user's framing): the model itself possessing the ability to judge, from
unlabeled evidence, which information is trustworthy and which to discard; small agents with limited
reasoning but strong epistemics; memory no longer an external harness. Two components with different
physics must be kept apart:
  per-event judgement   from the content of this event's answers alone; can be trained into weights;
  cross-event reputation  "this peer has been right on this kind of question before"; not present in
                          any single prompt; needs state across events (a matrix, or a fast weight
                          updated at test time - the record is the closed-form linear case of the
                          latter). Its value is the 3-5 points OOD the tilt adds on top of any judgement.

Recipes, cheapest first, each with its test:
  1. Anneal the tilt: gamma 3 -> 0 over the epoch (scheduled teacher forcing against tilt-dependency).
     One run on the existing pipeline. Test: accuracy at gamma 0 vs the control; attention at gamma 0.
  2. Reward the judgement: the model emits a per-peer trust verdict before answering; the verdict is
     rewarded against the verified peer labels we hold, jointly with the answer; no labels at test.
     Test: AUC of the model's own verdicts on OOD vs the record's 0.92; accuracy at gamma 0.
     Baseline to compute first: AUC of the frozen judge's stored label-free Yes/No log-odds (margins).
  3. Distil the record: auxiliary target = the record's p_i predicted from the prompt during training.
  4. Evidence seeking: tools on the peers' answers (run code, check a step), reward correct verdicts.
Decision: run 1 after the 8B control finishes; build 2 as the main bet.

Baseline measured 2026-09-11 (`scripts/judge_logodds_auc.py`): the frozen Qwen3-4B's own label-free Yes/No
log-odds per peer, no memory, separate right from wrong peers with AUC 0.629 in-dist (3,117 disagreeing
events) and 0.633 OOD (10,617); its favourite is right on 52.8% / 59.5% of disagreeing events; mean
Spearman with correctness +0.14 / +0.18. The record: AUC 0.92, favourite right 90.5% / 82.7%. So the
frozen model's content-only judgement is about a third of the way from chance to the record.

Method A implementation (2026-09-11): `feedback_state/verdict.py` (instruction, parser, strip, agreement
= 0.5(1 + Spearman) with tie-aware ranks), `training/sigma_rl/reward.py` (SigmaRewardManager adds
lambda * agreement on non-flat records, strips the line before grading; knobs memory.verdict_lambda /
verdict_flat), `build_rl_data.py --verdict` (appends the instruction to peers prompts; question-only
prompts untouched), `evaluate_memory_generator.py --verdict` (asks, strips, and reports verdict AUC vs
correctness, favourite hit rate, Spearman with the record), `launch_judge.sh` (A2 gamma 0 then A1 gamma 3,
tests incl. peers without the instruction, attention at gamma 0/3). Data: full_peers_verdict.parquet /
val_peers_verdict.parquet.

Method B stage 0 (label-free evidence lines computed by us and handed over in the prompt: sample I/O from the
statement run in the sandbox, arithmetic steps re-executed, answer span in the passage) was measured on
2026-09-11 and then removed from the plan the same evening (user): the checks were ours, not the model's,
so the arm would have tested whether the model can use execution results it never asked for, which is
not the ability the direction is about. Method B is stage 1 directly (the model writes the check, a
sandbox runs it; the two-pass rollout is not built yet). The stage-0 code (`feedback_state/evidence.py`,
`scripts/evidence_checks.py`, the `--evidence` options of build_rl_data / evaluate_memory_generator, the
B0 arm of launch_judge.sh) was removed; the computed lines stay in outputs/evidence/*.json. What the
measurement showed, the informativeness of label-free checks against the verified labels, is the case
for stage 1 and is kept here:

| stream  | check | peer answers checked | check passes -> right | check fails -> right | coverage |
|---------|-------|---------------------:|----------------------:|---------------------:|---------:|
| train   | code  |  7,416 | 72.7% | 3.0%  | 81.1% of all peer answers |
| train   | math  | 27,737 | 77.4% | 56.6% |  |
| train   | reading | 50,996 | 85.0% | 16.7% |  |
| in-dist | code  |  4,638 | 56.5% | 10.6% | 83.3% |
| in-dist | math  |  4,955 | 72.2% | 45.9% |  |
| in-dist | reading | 12,000 | 89.3% | 21.1% |  |
| OOD     | arithmetic only | 1,309 | 25.3% | 4.5% | 1.3% |

Reading: a failed sample test or an absent span is close to decisive (3-21% right), a pass is a strong but
not decisive sign (57-89%); arithmetic failures are the weakest (46-57% right, since an answer can be
right despite a mis-written step). In-distribution the evidence covers 83% of peer answers; on the OOD
stream almost nothing is checkable label-free (1.3%, BBH arithmetic only), so an evidence arm cannot help
OOD without new check types (option-consistency for multiple choice, passage checks for MultiRC).


### 22.1 A2 result (2026-09-11 23:11): the record distils into a verdict, not yet into accuracy

A2: one continuous epoch (276 steps of 64 events, group 8, KL 0.01 to the frozen model, no reset) from the frozen
Qwen3-4B; the reply opens with `Trust: a > b > ...`; reward = verifier + 0.5 * 0.5(1 + Spearman(rank, record p)) on
non-flat records; gamma 0 in training. Tests at step 276 on the whole streams (in-dist 4,319 / full OOD 17,403).

| condition (in-dist / OOD)            | A2          | control (no memory) | ours (tilt-trained) | frozen      |
|--------------------------------------|-------------|---------------------|---------------------|-------------|
| peers, gamma 0, with the Trust line  | 72.6 / 69.4 | -                   | -                   | -           |
| peers, gamma 0, no Trust instruction | 72.5 / 71.2 | 74.4 / 73.3         | 75.1 / 72.5         | 64.8 / 69.2 |
| peers + tilt gamma 3 (+ Trust line)  | 72.9 / 73.4 | 74.9 / 75.5         | 75.9 / 75.2         | 67.2 / 74.0 |
| swapped record (+ Trust line)        | 69.8 / 61.4 | 70.5 / 67.7         | 73.7 / 66.7         | 60.2 / 64.2 |
| question only                        | 71.0 / 69.9 | 71.4 / 72.7         | 73.4 / 70.7         | 60.5 / 67.8 |

The verdict, label-free, on the events where the peers disagree (AUC of the rank score against correctness / favourite right):

| judgement at gamma 0                   | in-dist (3,116)   | OOD (10,545)      |
|----------------------------------------|-------------------|-------------------|
| frozen judge's Yes/No log-odds (floor) | 0.629 / 52.8%     | 0.633 / 59.5%     |
| A2's Trust line                        | **0.845 / 92.0%** | **0.710 / 69.8%** |
| the record (the distillation target)   | 0.92 / 90.5%      | 0.92 / 82.7%      |
| A2 under the tilt, gamma 3             | 0.843 / 92.7%     | 0.728 / 76.5%     |
| A2 under the swapped record            | 0.658 / 68.8%     | 0.579 / 56.0%     |

Spearman of the Trust line with correctness 0.66 / 0.43, with the record 0.69 / 0.43 (gamma 0); parsed 100% / 99.2%.
Training: validation agreement with the record flattened from step 140 (final reading +0.75, math +0.66, code +0.22);
validation accuracy at step 276 reading 80.5, math 91.4, code 32.8.

Attention at gamma 0 (per-token density over the six peer blocks, query = last prompt token, all layers, 300 events;
scripts/attention_mass.py + attention_mass_summary.py): A2's favourite gets 1.12x in-dist / 1.08x OOD, corr(record)
+0.29 / +0.20, corr(correct) +0.22 / +0.17, against 1.01-1.05x, <= +0.10 and <= +0.09 for the frozen model, ours, the
control and the question-only arm (which sit within 0.03 of each other). Under gamma 3 every model is at 2.1-2.2x.

Reading. (1) Distillation trains a judgement: with no labels and no record at test time the model ranks the peers at
AUC 0.845 in-distribution, at the record's own favourite-right level (92.0 vs 90.5%), and 0.71 on the OOD stream, a
third of the way from the frozen judge (0.63) to the record (0.92). In-distribution the record's content is largely
peer-by-task reliability that the answers themselves reveal; on OOD the cross-event reputation the record carries is
not recoverable from one epoch of content, hence the gap. (2) It is visible inside the model: A2 is the first arm whose
attention at gamma 0 is not uniform over the peers; it leans toward the record's favourite and toward the correct
peers (1.12x, corr +0.29 against +0.05), an internal label-free tilt about a tenth the size of the engine's. (3) It
did not become accuracy: at gamma 0, 72.5 / 71.2 (no Trust instruction) against the control's 74.4 / 73.3, own
ability 71.0 / 69.9 against 71.4 / 72.7; the Trust line itself costs nothing in-distribution and 1.8 points on OOD
(the short-answer tasks, 45.7 vs 49.5). The tilt still adds +4.0 on OOD on top of the learned judgement (73.4), so the
record's information is not inside the model yet. Caveats: A2 ran one continuous epoch (276 steps) where the control
ran 40 + 277 with a reset, and the verdict term takes a third of the reward range (lambda 0.5), which dilutes the
answer signal. Next: A1 (gamma 3 in training, running from 23:11) tells whether the record as input makes the
distillation more complete; then the verdict rewarded against the labels instead of the record (the plan's main bet;
one flag: verdict_agreement(rank, labels)), and stage 1 of method B. Files: outputs/rl/judgeA2/hf/global_step_276/
eval_*, outputs/address/attention_mass/judgeA2_*.json.

## 23. Other central-model families, each with its own memory (2026-09-11)

Question (user): does the memory help central models other than Qwen3-4B? Models on the server:
Meta-Llama-3-8B (base), Meta-Llama-3.1-8B-Instruct, Ministral-8B-Instruct-2410, Qwen2.5-7B-Instruct, phi-4 (14B).
Comparison per model: peers + memory (the tilt, gamma 3) / peers, no memory / question only, on the in-distribution
stream (4,319) and the WHOLE OOD stream (17,403), thinking off, 768-token answers, greedy, vLLM; the frozen Qwen3-4B
row (outputs/gen/q3_4b/base6_nothink) is the reference. No swapped-record control for the families (user, 2026-09-11).

**Rule (user, 2026-09-11 22:xx): a family is tested with its OWN memory.** When the central model is Llama, the
record's address comes from Llama too (its own hidden states of the question and of each peer answer under the
plain-text Yes/No judge prompt), never from the Qwen3-4B judge. So per family the whole pipeline is rebuilt with that
model as the frozen judge: features (`scripts/encode_context_features.py`, caches outputs/context_features/<tag>_{big6,
indist6,6}_ph), PCA-256 addresses fit on train6, the Bayesian record run along each test stream read-before-write
(`scripts/build_generation_prompts.py --order shuffled0` = the probe files, under outputs/gen/<tag>/), its quality
(`scripts/record_quality.py` -> record_<stream>.json), then the three conditions. Driver:
`GPUS=... bash training/scripts/launchers/launch_family_memory.sh smoke|features|prompts|eval|all <tag>`; the
evaluation step is `launch_families.sh full <tag>` (ADDR=own by default). `ADDR=q3_4b` evaluates the same central
model on the Qwen3-4B-addressed prompt files instead (results with the suffix _q3addr): a comparison of the two
records, not the family's result. Tags in feedback_state/feature_streams.py MODELS.

Cost: every event needs one question pass and six judge prompts that each carry all six answers; measured on the smoke
run at about 0.5 s per event for an 8B model (0.7 s for phi-4), so about 5 GPU-hours per 8B family for the three
streams (39k events), about 8 for phi-4, plus about one GPU-hour of evaluation per family; shards on disk are skipped,
so runs resume. Caches are about 6 GB per 8B family (peer_hidden fp16, 3 x hidden size per candidate).

What is model-specific and what is not in the tilt itself:
- The tilt: b_i = gamma log(p_i / max p) added to the pre-softmax score of every query onto peer i's tokens, in every
  layer and head. softmax(s + b) = Norm(A * c) holds per head for ANY softmax attention (MHA or GQA, RoPE or not,
  any head count or size), so it applies to all five: Llama-3/3.1 (GQA 32/8, RoPE), Ministral (GQA 32/8, RoPE,
  sliding window 32k), Qwen2.5 (GQA 28/4), phi-4 (GQA 40/10). Not covered: linear-attention / SSM layers (no
  softmax, the tilt has no meaning there), ALiBi and iRoPE (the bias kernels assert).
- The engine patch (feedback_state/vllm_attn_bias.py) is model-agnostic: it works on KV slots inside vLLM's Triton
  kernels. The per-model work is locating the peer blocks' token positions under that model's tokenizer and chat
  template (feedback_state/attn_bias.py: character spans -> offsets -> per-token bias).
- Sliding window (Ministral, 32k): disabled in the engine (LLM(..., disable_sliding_window=True)), numerically a
  no-op because our contexts (<= 4k tokens) are shorter than the window; needed because the bias kernels are plain causal.
- Tokenisation: every condition now hands vLLM the same token ids, tokenised by us with add_special_tokens=False
  (the rendered template already carries BOS where the family uses one). Before this change the untilted conditions
  let the engine tokenise the string, which would have given Llama and Mistral a second BOS in those conditions only.
- A base model without a chat template (Meta-Llama-3-8B) gets a plain layout, BOS + system + user + "Answer:"
  (memory_generator.render_prompt); expect weak instruction following. The 3.1 Instruct is the meaningful Llama row.
- The judge prompt for the features is plain text (no chat template) and needs " Yes" / " No" to be single tokens,
  which holds for all five tokenizers (checked in the smoke run).

CPU check (scripts/family_prompt_check.py, first 100 in-dist events, gamma 3, on the Qwen3-4B-addressed prompts since
the messages are identical for every record): every tokenizer finds all six peer blocks; tilted-block coverage
99.0-99.1% for all five (Qwen3-4B: 99.1%); 96/100 prompts tilted; prompt tokens median 890-960, max 3.8k; BOS first
for Llama and Ministral, none for Qwen2.5 / phi-4 (their templates carry none). PROMPT_CHECK_OK.

Running it (hand-over, 2026-09-11 22:30): `bash run_families.sh` with every parameter in training/configs/families.yaml
(README_families.md explains the pipeline: address encoding, the record, the evaluation, the table); driver
scripts/run_families.py (features sharded over the GPUs -> prompts + record quality -> the three conditions -> table;
every task skipped when its output exists). The underlying launchers launch_family_memory.sh / launch_families.sh remain.
GPU smoke (2026-09-11 22:07-22:16, 48 events, whole pipeline, all five families, 0 failures; own record, three
conditions + swapped which was then dropped): Llama-3.1 72.9 / 70.8 / 58.3 (tilt / peers / solo), Ministral
66.7 / 64.6 / 50.0, phi-4 70.8 / 68.8 / 64.6, Qwen2.5 and Llama-3 base passed the same checks; 44-45/48 prompts
tilted, bias kernels taken, the tilt changed 18-44 of 48 generations. Encoder speed 0.5 s/event (8B), 0.7 s (phi-4).
