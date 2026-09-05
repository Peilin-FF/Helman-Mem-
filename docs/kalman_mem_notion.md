# Kalman Mem — method in detail

*Σ-Mem v2, sigma-mem. Center models Qwen3-4B and Qwen3.5-4B, three peers (Gemma-3-4B, Phi-4-mini, Qwen2.5-Coder-7B). Written 2026-09-04. Code paths refer to the sigma-mem repository.*

---

## 1. Problem

A stream of events $t = 1, \dots, T$. Event $t$ brings a question $x_t$ and one answer $a_{p,t}$ from each of $P$ peers. The center model must act (select one answer, or write its own) **before** any label is seen; afterwards the correctness $y_{p,t} \in \{0,1\}$ of every peer is revealed and written into memory. The memory must therefore (i) learn online, (ii) be exact after a handful of events, (iii) not depend on the order of events, and (iv) never see dataset names or peer identities beyond an anonymous index.

Three evaluation streams (all cold start, decide-then-update): the training stream (GSM8K + SQuAD + APPS, 17,709 events), the in-distribution test stream (4,319), and the OOD stream (MCQA, boolean and short-answer benchmarks, 17,403). Fixed order and shuffled order are both reported.

## 2. Address: what the memory is indexed by

The frozen center model reads each candidate through the paper's Yes/No judge prompt ("Question … Response $p$ [candidate under review] … Should the candidate be selected?"). From that pass we keep, per candidate, the last-token hidden states of three layers ($L/3$, $2L/3$, $L$; $3H = 7{,}680$ dims), the mean over candidates as the question representation $h_t$, and the judge's own log-odds $z_{p,t}$.

Addresses are unsupervised projections fitted once on training features (no labels, no dataset names):

$$\psi_q(x) = \mathrm{PCA}_d\!\left(\frac{h - \mu}{\sigma}\right)/\kappa, \qquad \psi_c(x,p) = \mathrm{PCA}_d\!\left(\frac{h_p - \mu_c}{\sigma_c}\right)/\kappa_c$$

Design `qc` (the default) stacks a **peer-blocked** question part and a **shared** candidate part, so one linear model holds both "who is reliable on questions like this" and "does this answer look right":

$$\psi_{p} = [\,e_p \otimes \psi_q(x)\;;\;\psi_c(x,p)\;;\;1\,] \in \mathbb{R}^{P d + d + 1}$$

The block index $e_p$ is the peer's identity, never its position in the prompt. Design `q` uses only $[\psi_q;1]$ with one head per peer (used when candidate features are stale, e.g. under synthetic corruption).

## 3. State and update: the exact Bayesian competence posterior

Model per candidate row, with signed correctness $s = 2y-1$:

$$s = w^\top \psi + \varepsilon, \qquad w \sim \mathcal{N}(0, I/\lambda), \qquad \varepsilon \sim \mathcal{N}(0,1)$$

Sufficient statistics are sums, hence order-invariant and of fixed size:

$$\Lambda_t = \lambda I + \sum_{\tau \le t} \psi_\tau \psi_\tau^\top, \qquad b_t = \sum_{\tau \le t} s_\tau \psi_\tau, \qquad w \mid \mathcal{D}_t \sim \mathcal{N}(\Lambda_t^{-1} b_t,\ \Lambda_t^{-1})$$

Read-out for a query $\psi$ (posterior mean, posterior variance, probit predictive):

$$\mu(\psi) = \psi^\top \Lambda^{-1} b, \qquad v(\psi) = \psi^\top \Lambda^{-1} \psi, \qquad P(\text{correct}\mid\mathcal{D}) = \Phi\!\left(\frac{\mu}{\sqrt{1+v}}\right), \qquad \ell = \operatorname{logit} P$$

Write (Sherman–Morrison on $\mathbf{P} = \Lambda^{-1}$, $O(D^2)$ per event):

$$k = \frac{\mathbf{P}\psi}{1 + \psi^\top \mathbf{P}\psi}, \qquad \mathbf{P} \leftarrow \mathbf{P} - k\,\psi^\top \mathbf{P}, \qquad b \leftarrow b + s\,\psi \quad\Longleftrightarrow\quad m \leftarrow m + k\,(s - m^\top \psi)$$

The last form is **the delta rule with the Kalman gain**. DeltaNet, Gated DeltaNet and Kimi's KDA apply $S \leftarrow S(\alpha I - \beta k k^\top) + \beta v k^\top$: one gradient step of the same least-squares objective with a fixed or gated rate. The Kalman memory is its closed-form optimum with a direction-dependent step (large where evidence is scarce, small where abundant). A forgetting factor $\rho<1$ reproduces the gated decay: $\Lambda \leftarrow \rho\Lambda + (1-\rho)\lambda I + \psi\psi^\top$.

Evidence count along a query (0 at cold start): $n(\psi) = \dfrac{\psi^\top\psi/\lambda}{v(\psi)} - 1$.

Properties that predict the measurements:
- **Order invariance** ($\rho = 1$): the final state is the same for any permutation of the events.
- **Regret**: online ridge regression has cumulative regret $O(d \log T)$, so the average excess loss decays like $d\log T / T$ — the curve rises fast and then flattens once the linear class is exhausted.
- **Calibrated uncertainty**: the probit predictive shrinks cold reads to $1/2$; an unseen region cannot dominate a decision.
- **Spectral control**: each write is a symmetric rank-one update, so by Weyl every eigenvalue of $\Lambda$ moves by at most $\|\psi\|^2$.
- **Reversibility**: $(\Lambda, b)$ is a sufficient statistic; every write has a closed-form downdate; the read-out through $\Lambda^{-1}$ is the invertible change of basis that separates overlapping questions (raw squared cosine between random questions ≈ 0.5, whitened ≈ 0.002).

Code: `feedback_state/kalman_memory.py` (`KalmanMemory`, `DeltaMemory` ablation), `feedback_state/addresses.py`, `feedback_state/memory_runtime.py`; unit tests in `tests/unit/test_kalman_memory.py` (Sherman–Morrison vs direct posterior, order invariance, forgetting recursion).

## 4. Decision rules

- **Route**: select $\arg\max_p \ell_p$.
- **Vote** (Nitzan–Paroush): group candidates by canonical answer; pick the group with the largest $\sum_{p\in g} \ell_p$, then the most reliable member. Bayes-optimal under conditional independence with online-learned weights.
- **Self-vote**: the center model's own answer is a fourth source with its own head; the vote runs over $\{\text{own}, \text{peers}\}$. Its ceiling is the oracle over own + peers (79.3 in-distribution vs 75.2 for peers only).

## 5. Learned and reversible addresses (`train_address.py`)

The Kalman recursion is differentiable, so the address map can be trained on cold-start episodes of the training stream with the online selection loss (the area above the learning curve); gradients flow through the whole recursion. Three families (`AddressMap` in `addresses.py`):

| map ($d=64$) | property | in-dist fixed / shuffled | train fixed / shuffled | OOD fixed / shuffled |
|---|---|---|---|---|
| PCA | unsupervised | 68.05 / 67.86 | 75.04 / 75.24 | 61.86 / 61.88 |
| learned linear | PCA-initialised $W$ | 70.22 / 70.04 | 79.05 / 79.07 | 61.64 / 61.78 |
| learned orthonormal | $Q=\mathrm{qr}(W)$, norm-preserving, exactly decodable ($\hat x = Q z s$) | 69.25 / 68.84 | 77.36 / 77.53 | 61.68 / 61.87 |
| learned invertible flow | two affine-coupling layers on PCA coordinates, closed-form inverse | **70.22 / 70.25** | **80.51 / 80.50** | 61.56 / 61.97 |

Reading: the flow is the best address and loses nothing (bijection). OOD every address coincides: the transform is not the limit, the information in the frozen features is (hindsight 5-fold ridge on OOD: 62.4; capacity-unbounded kernel and nearest-neighbour memories add ≤ 0.15).

## 6. The judge that reads the memory (`memory_judge.py`, `train_memory_judge.py`)

Score of candidate $p$: $\text{score}_p = z_p(\theta;\ e_p \text{ in the residual stream}) + \kappa\,\ell_p$, with evidence vector $e_p = [\ell_p,\ \nu_p,\ \max_{p'\ne p}\ell_{p'},\ \operatorname{mean}_{p'\ne p}\ell_{p'}]$ injected into the upper decoder layers (paper's `ActivationSteerer`), the additive prior $\kappa\ell_p$ (Bayes under conditional independence), and optionally the memory's notes written into the prompt as text (`--memory_text on`). Trainable: LoRA adapters (rank 16), steerer, $\kappa$. Loss per event: $-\log \sum_{p\ \text{correct}} \operatorname{softmax}(\text{score})_p$. Episodes: cold-start memory, candidate slots permuted per event.

Reliability-shift episodes (hidden per-peer corruption rates during training) were tried and removed: on held-out streams the judge is indistinguishable with and without the memory (§7).

## 7. Results (Qwen3-4B unless stated; shuffled streams)

**Memory alone vs frozen judge**

| stream | frozen judge | Kalman `qc` memory | hindsight ridge (5-fold CV) | per-event oracle |
|---|---|---|---|---|
| OOD | 55.98 | 62.32 | 62.4 | 69.56 |
| in-distribution | 57.33 | 68.63 | 69.2 | 75.23 |
| training | 61.70 | 76.71 | 77.2 | 82.60 |

Qwen3.5-4B: OOD 59.54 → 62.40, in-distribution 59.99 → 68.70, training 61.7 → 77.0. Delta rule with a fixed step instead of the Kalman gain: −2.5 in-distribution, −0.3 OOD. Offline ridge fitted on the training stream, applied to OOD: 57.8 (trait overfitting in one number).

**Judge + memory (selection)**

| arm | in-dist | OOD | shifted in-dist (peer 1 wrong 90%) |
|---|---|---|---|
| frozen judge | 57.33 | 55.98 | 53.16 |
| frozen + memory prior ($\kappa=1$) | 65.83 | — | 59.46 |
| memory alone | 68.5 | 62.2 | 61.4 |
| trained without memory | 70.99 | 59.60 | 63.07 |
| trained with memory (steering + prior) | 70.94 | 61.75 | 63.42 |
| shift-trained, no memory | 71.38 | 59.04 | 63.09 |
| shift-trained, with memory | 70.97 | 60.79 | 63.23 |
| shift-trained, memory + invertible/learned address | 71.13 | 61.28 | 63.37 |
| shift-trained, memory as prompt text | 71.43 | (running) | **63.63** |
| trained with memory, adapting online at test time | 71.57 | **63.03** (62.5 → 63.6) | — |
| trained without memory, adapting online (control) | — | 63.08 (62.5 → 63.6) | — |

Reading: a trained verifier ties a trained verifier that also reads the memory, within ±0.5, on every stream where verification is possible. The memory is decisive where verification is unavailable: frozen judge (+8.5 / +6.2 / +6.3), OOD with frozen adapters (+2.2). Test-time adaptation makes the OOD curve rise with or without the explicit state: the adapters then are the memory. Qwen3.5-4B replicates every row (71.5 / 62.4 with memory; 71.8 / 60.1 without; 63.8 adapting).

**Generation (the center model answers)**

| Qwen3-4B, in-distribution | with peer responses | from the question only |
|---|---|---|
| frozen | 62.95 (raw responses) / 63.30 (with memory notes), first 2,000 events | 61.30 (first 2,000); 60.06 full stream (math 89.5, reading 59.6, code 22.3) |
| fine-tuned with memory notes | 63.02 (math 78.4, reading 74.0, code 20.9) | **46.82** (math 21.8, reading 78.8, code 16.0) |
| fine-tuned on peers only | 62.56 (math 74.9, reading 75.7, code 20.0) | 46.31 (math 21.5, reading 78.2, code 15.3) |
| full-parameter fine-tune on peer solutions (no adapters; control) | — | 53.37 (math 46.3, reading 77.5, code 14.4) |
| fine-tuned, self-distilled targets | 63.09 (math 84.8, reading 70.7, code 19.3) | **64.34** (math 87.8, reading 69.8, code 22.4) — peer knowledge internalized, no collapse |

Diagnosis (`scripts/analyze_generations.py`): fine-tuning on peer solutions halves output length (364 → 173 chars on math), copies a wrong peer five times as often (3.8% → 20.2%), never solves alone on code (0% when all peers fail vs 8.2% frozen), and — measured directly — loses its own math when no peer answers are offered (89.5 → 21.8). Supplying peer answers to the frozen model costs nothing (90.6 vs 89.5 on math).

Frozen Qwen3-4B answering directly on OOD (first 4,000 events): **67.07** (boolean 84.4, MCQA 81.4, short-answer 41.2) vs the peer-selection oracle 69.85 and the best selection 62.3.

**Self-vote** (`scripts/self_vote_sim.py`): memory-weighted vote over {own answer, peers}, no fine-tuning:

| stream | model alone | memory selection among peers | plain majority incl. own | **memory-weighted self-vote** | oracle own+peers |
|---|---|---|---|---|---|
| in-distribution | 60.06 | 67.59 | 62.91 | **70.48** (math 87.5, reading 83.2, code 22.5) | 79.28 |
| training | 67.86 | 76.11 | 70.70 | **80.28** (79.7 → 80.9 across the stream) | 87.15 |
| OOD (first 4,000, shuffled) | 67.07 | 61.18 | 67.07 | **68.17** (short-answer 41.2 → 44.8; MCQA / boolean kept at 81 / 84) | 79.17 (peers only 69.85) |
| in-distribution, Qwen3.5-4B | 61.77 | 67.35 | — | **69.16** (68.7 → 69.6) | 78.95 |

## 7b. Repairing the generator (in progress)

The collapse (46.8 from the question alone) has two structural fixes: **evidence-gated adapters** (LoRA active only when peer answers / memory evidence are in the prompt, so the no-peer behaviour is the base model exactly) and **on-policy targets** (rejection-sampling fine-tuning on the base model's own correct generations, with peers when that answer is correct, solo otherwise), i.e. policy improvement under the correctness reward with an implicit KL anchor instead of imitation of peers. Runs: `outputs/gen/q3_4b/rft_memory_gated`.

## 7c. Macro objective: memory-weighted distillation into the parameters (in progress)

The stream of peer solutions with cheap verdicts is a multi-teacher corpus without process annotation. The memory is the teacher-selection signal (per-peer reliability by question type, plus the student's own). Training is plain gradient descent on the full parameters with **question-only prompts**, targets chosen by the memory: the student's own peer-informed solution if correct, else its own solo solution if correct, else the most reliable verified-correct teacher's solution, else dropped. The student is evaluated alone; the trend across checkpoints is the trend with training steps. Runs: `outputs/gen/q3_4b/mwd_full`. *Note (2026-09-05): this uses the peers' labels before the student answers, i.e. it is the classical labels-before baseline of §7e, not the method.*

## 7d. Answering alone after training (Qwen3-4B, in-distribution, 4,319 events)

| training recipe | alone | math | reading | code |
|---|---|---|---|---|
| none (frozen) | 60.06 | 89.5 | 59.6 | 22.3 |
| LoRA imitation of peer solutions (with peers in the prompt) | 46.3–46.8 | 21.5–21.8 | 78.2–78.8 | 15.3–16.0 |
| full-parameter imitation of peer solutions (question-only prompts) | 53.37 | 46.3 | 77.5 | 14.4 |
| LoRA, self-distilled targets (own correct solution, else a correct peer's) | **64.34** | 87.8 | 69.8 | 22.4 |
| RLVR (GRPO, verifier reward), 1,500 prompts | 61.24 | 89.5 | 61.9 | 22.7 |
| RLVR + memory-guided hint distillation, 1,500 prompts | 61.03 | 89.8 | 61.3 | 22.7 |

Imitation of peers collapses the model's own solving whether or not adapters are used; targets drawn from the model's own correct solutions internalize peer knowledge (+4.3) while keeping math; verifier RL preserves math exactly and adds about a point per 1,500 prompts (checkpoint curve 60.4 → 61.5 → 61.9 on the first 1,000 events). Memory-weighted full-parameter distillation on likelihood-filtered targets, the own-words hinted data over the full stream, and a 4,000-prompt RLVR run are in progress.

## 7e. Labels only after the answer: the training protocol (training/)

At event t the central model gets the question and the peers' solutions, unlabeled. It decides what to trust from the memory (each peer's reliability on this question address, learned from the feedback of earlier events) and its own judgment, and answers; only then are its answer and the peers' solutions graded, and the memory updated. Its own answer is the deliverable, evaluated along the stream and afterwards alone (question only) on held-out streams. The classical regime — learning from solutions annotated as correct before learning — is the baseline.

Implementation: vendored verl 0.3.1 (from the ARPO repository) with plain GRPO (group-mean baseline, PPO clip, no critic, no KL, full parameters). Per event: n question-only samples plus one *guided* sample from the question with the peers' solutions and the memory's notes (`--guided memory`; `hint_memory` = only the most trusted peer's solution). All samples are graded after the fact. The guided answer is trained under the guided prompt and re-labelled under the question-only prompt (internalisation), in the same GRPO group (advantage +(1−μ)/σ with μ = 1/(n+1), e.g. +2.67 for n = 8). Baselines from the same code: labels before the answer (`--guided hint_label` / `hint_random` with `memory.guided_filter=verified`), labels after without memory (`--guided peers`), plain RLVR (`--guided none`), and SFT on the annotated solutions (`train_sft.sh`). Where no verifier is available after the answer, the memory's reliability-weighted peer vote is the reward. Verified end to end on 2 shared A100s with Qwen3-0.6B and Qwen3-4B; planned comparisons: memory vs peers vs hint_label vs none, and 30%-verified with vs without the pseudo-reward.

## 8. What the evidence supports

1. The Kalman memory is the correct linear memory for this problem: exact, order-invariant, calibrated, reaching the hindsight ceiling of its feature class on every stream, +6 to +11 over the frozen judge.
2. On these benchmarks a trained verifier subsumes peer-reliability information; the memory adds where verification fails (frozen judge, OOD, shifted streams with frozen adapters). This is a property of the task family (three checkable peers), not a defect of the memory.
3. Test-time adaptation of the adapters gives the rising OOD curve; the closed-form memory is the cheap, exact-after-few-events component, the adapters the slow one.
4. Imitating peers by fine-tuning destroys the center model's own solving; the configuration that beats both the frozen model and every no-memory model is the center model answering and the memory-weighted vote deciding (70.5 / 80.3), whose trainable part is the model's own solving.

## 9. Reproduce

Multi-GPU: `rproj run 'MODEL_TAG=q3_4b bash training/scripts/build_data.sh'` · `rproj submit 'GPUS=0,1,2,3 EXP=q3_4b_grpo_memory TRAIN=outputs/rl/data/q3_4b/train_memory.parquet VAL=outputs/rl/data/q3_4b/val_indist.parquet bash training/scripts/train_grpo.sh'` · `training/scripts/eval_hf.sh` (see `training/README.md`). Single-GPU: `scripts/memsim.py` (all memory arms, probes, sweeps, learned addresses, trained-judge scores) · `feedback_state/train_address.py --arch {linear,orthogonal,flow}` · `feedback_state/train_memory_judge.py` / `tests/experiments/common/evaluate_memory_judge.py` (`--cf_episodes`, `--cf_profile`, `--memory_text`, `--online_lr`) · `scripts/build_generation_prompts.py`, `feedback_state/train_memory_generator.py`, `tests/experiments/common/evaluate_memory_generator.py` · `scripts/self_vote_sim.py` · `scripts/analyze_generations.py` · full tables: `docs/memory_judge_design.md`; ledger page: https://claude.ai/code/artifact/3b635a27-7b9f-4f6a-989d-da5158960a69
