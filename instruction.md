# ============ CONSTANTS (fixed a priori — nothing is fitted) ============
P        = number_of_peers
r        = competence_direction_dim
gamma, eta       = ...            # same values as main experiments (M update)
gamma_G, eta_G   = ...            # same values as main experiments (G update)
LAM      = 0.1                    # ridge for solving with G; declared, not tuned
SEED     = 0                      # for all tie-breaking; repeat with 3 seeds

# ============ STATE ============
M = [zeros(r, r) for _ in range(P)]          # M_p^(0) = 0
G = eye(P)                                   # diag(G) = 1

def update_memory(phi, c):                   # c[p] in {-1,+1}: external labels
    for p in range(P):
        M[p] = gamma * M[p] + eta * c[p] * outer(phi, phi)      # Eq. 1
    q = c - mean(c)                                             # Eq. 2
    G[:] = gamma_G * G + eta_G * outer(q, q)                    # Eq. 3
    fill_diagonal(G, 1.0)

def route(phi, answers, t):
    # s[p] = decayed, similarity^2-weighted signed success rate of peer p
    #        along the current task direction (Rayleigh quotient of M_p at phi)
    s = [phi @ M[p] @ phi for p in range(P)]

    if max(s) - min(s) < 1e-9:               # cold start / exact tie
        p_hat = t % P                        # round-robin
    else:
        p_hat = argmax(s)
    return answers[p_hat]

# Algorithm 1 — Pre-hoc peer routing (M only)

# ---- baselines computed in the same pass ----
# (a) best fixed peer : accuracy of always answers[p], report best p (post hoc)
# (b) random routing  : rng(SEED).choice(P) per event
# (c) oracle-label dictionary (decayed counts keyed by TRUE dataset name):
D_num, D_den = defaultdict(float), defaultdict(float)

def dict_route(dataset_name, answers, t):
    rate = [D_num[(dataset_name, p)] / D_den[(dataset_name, p)]
            if D_den[(dataset_name, p)] > 0 else 0.0
            for p in range(P)]
    if max(rate) - min(rate) < 1e-9:
        return answers[t % P]
    return answers[argmax(rate)]

def dict_update(dataset_name, c):            # AFTER deciding, like M
    for p in range(P):
        D_num[(dataset_name, p)] = gamma * D_num[(dataset_name, p)] + (c[p] == +1)
        D_den[(dataset_name, p)] = gamma * D_den[(dataset_name, p)] + 1.0

# Algorithm 2 — Factorial reliability-weighted vote (2×2 over M, G)

def vote(answers, w):
    score = defaultdict(float)
    for p in range(P):
        score[canon(answers[p])] += w[p]     # canon(): normalized option label
    return argmax_key(score)                 # ties: lowest option index, fixed rule

def all_votes(phi, answers):
    s = array([phi @ M[p] @ phi for p in range(P)])
    return {
        "maj": vote(answers, ones(P) / P),                     # no memory (control)
        "G"  : vote(answers, solve(G + LAM*eye(P), ones(P))),  # G only (task-blind)
        "M"  : vote(answers, s),                               # M only (task-cond.)
        "MG" : vote(answers, solve(G + LAM*eye(P), s)),        # M+G (GLS)
    }
# appendix robustness: clipped variants max(w, 0); LAM in {0.05, 0.1, 0.5}

# Main loop (one pass, decide-then-update)

for t, ev in enumerate(stream):              # SAME stream & order as Sec. 4.3 runs
    phi = qwen_encoder(ev.task)               # frozen encoder, ||phi|| = 1
    # ---- decisions FIRST (labels untouched) ----
    out = {}
    out["route"]      = route(phi, ev.answers, t)                    # Alg. 1
    out["route_dict"] = dict_route(ev.dataset_name, ev.answers, t)   # baseline (c)
    out["route_rand"] = ev.answers[rng.choice(P)]                    # baseline (b)
    out.update(all_votes(phi, ev.answers))                           # Alg. 2
    log(t, ev.dataset_name, out,
        n_distinct=len(set(map(canon, ev.answers))),   # vote-active diagnostic
        flip=(out["maj"] != out["MG"]))                # for the flip table

    # ---- updates AFTER ----
    update_memory(phi, ev.labels)
    dict_update(ev.dataset_name, ev.labels)