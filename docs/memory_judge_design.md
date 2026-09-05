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
| in-distribution (4,319) | 60.06 (math 89.5, reading 59.6, code 22.3) | **67.42** (math 90.8, reading 75.4, code 20.6) |
| OOD (17,403) | 67.57 (boolqa 83.5, mcqa 82.1, shortqa 42.0) | **68.48** (boolqa 77.0, mcqa 81.3, shortqa 50.8) |

The trained model beats the frozen one on both streams with math preserved; the in-distribution gain is reading comprehension
(the task where the peers are stronger than the model), the OOD gain is BIG-Bench Hard (+8.8) against a loss on the yes/no task
(−6.5). Attribution to the memory awaits the no-memory (`peers`), classical (`hint_label`) and plain-RLVR runs, chained next.
