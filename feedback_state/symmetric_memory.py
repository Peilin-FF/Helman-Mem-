"""Event-axis symmetric information-matrix trust memory for multi-agent selection.

Replaces the token-level Delta-Mem feedback channel (which self-overwrites: per-token
keep^~800 ~ 1e-7, so trust never survives to the next decision). Here the trust state
is updated ONCE PER TASK (event axis, response-length-independent) and is a real
SYMMETRIC matrix, so Weyl's inequality applies: a single noisy update moves the
spectrum by <= the update's norm, while only persistently aligned signals build a
dominant eigenvalue. See artifacts/symmetric_memory_design.html for the full rationale.

Two states:
  * per-peer  M_p  in R^{r x r}, symmetric, init 0:  M_p <- gamma*M_p + eta*s*phi phi^T
    where s in {+1,-1} = peer correct/incorrect, phi = L2-normalized soft-prototype address
    (a query soft-addressed over FIXED per-task CM centroids; see set_prototypes / phi_of).
    Read at decision time:  steer_vec = M_p @ phi*  -> a rank-dim direction injected into the
    frozen CM's activations (ActivationSteerer), re-ranking its " Peer j" log-probs. The trust
    is therefore conditioned on the task, not on this round's raw response.
  * CM-owned joint  G  in R^{P x P}, symmetric, init 0:  G <- gamma_G*G + eta*o o^T
    where o[p] in {+1,-1,0} = per-peer outcome on the task. Off-diagonal G[p,q] reads out
    redundancy (>0, agents right/wrong together) vs complementarity (<0).

gamma = exp(-softplus(theta)) in (0,1] (S4D-style: structurally stable for any theta;
||M|| <= eta/(1-gamma), so the state can neither blow up nor — while evidence keeps
arriving — collapse). Only this module's params train; the CM backbone stays frozen.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

# Task types used to build phi_t. Order fixes the embedding rows; "math/code/rag" cover
# the CF unified streams (feedback_state/tasks.py::task_type_of returns these lowercased).
DEFAULT_TASK_TYPES = ("math", "code", "rag")
RELIABILITY_ZSCORE_EPS = 1e-3


def zscore_peer_values(
    values: torch.Tensor, *, eps: float = RELIABILITY_ZSCORE_EPS
) -> torch.Tensor:
    """Standardize peer values with a finite near-tie sensitivity.

    ``sqrt(var + eps^2)`` behaves like an ordinary z-score once peers differ,
    while preventing a nearly constant reliability vector from amplifying
    floating-point noise or producing a gradient proportional to ``1e6``.
    """
    vals = torch.as_tensor(values)
    if vals.ndim != 1:
        vals = vals.reshape(-1)
    centered = vals - vals.mean()
    scale = (centered.square().mean() + float(eps) ** 2).sqrt()
    return centered / scale


class Whitener(nn.Module):
    """Running mean/var standardizer for raw CM hidden vectors (buffers, not params).

    Raw LM hidden states are strongly ANISOTROPIC: all problem embeddings collapse to
    cosine ~0.99 along one dominant direction, so a projection of the raw vector cannot
    address content (this is exactly why the earlier phi=desc/query failed). Subtracting
    the running mean and dividing by running std recovers the content structure (probed:
    within-task cosine 0.99 -> ~0.2). Stats update via EMA on each .observe(); .forward()
    standardizes. Stats are detached (a fixed-ish preprocessing frame), so gradient flows
    only through the downstream trainable projection, not into the whitening frame.
    """

    def __init__(self, dim: int, momentum: float = 0.99, dtype=torch.float32):
        super().__init__()
        self.momentum = float(momentum)
        self.register_buffer("mean", torch.zeros(dim, dtype=dtype))
        self.register_buffer("var", torch.ones(dim, dtype=dtype))
        self.register_buffer("count", torch.zeros(1, dtype=dtype))

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        x = x.detach().to(self.mean.dtype).reshape(-1)
        m = self.momentum if float(self.count) > 0 else 0.0
        self.mean.mul_(m).add_(x * (1 - m))
        self.var.mul_(m).add_((x - self.mean).pow(2) * (1 - m))
        self.count += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.dtype).detach()
        std = (self.var.to(x.dtype).detach() + 1e-5).sqrt()
        return (x - mean) / std



class SymmetricTrustMemory(nn.Module):
    """Per-peer symmetric trust matrices + an optional CM-owned joint agent matrix.

    All trainable knobs (phi embedding, decay theta, write strength eta) live here so the
    trainer can optimize ``self.parameters()`` together with the steerer's projection/gain.
    The matrices M_p / G are NON-trainable BUFFERS that evolve by the event-axis recurrence
    at train and eval time alike (they are state, not parameters).
    """

    def __init__(
        self,
        *,
        num_peers: int,
        rank: int = 16,
        task_types: Sequence[str] = DEFAULT_TASK_TYPES,
        phi_mode: str = "proto",         # only "proto" (soft task-centroid address); see set_prototypes()
        phi_in_dim: int | None = None,   # raw CM hidden dim the proto address whitens/compares (e.g. 1024)
        proto_tau: float = 0.1,          # softmax temperature over centroid cosines
        write_mode: str = "addr",        # "addr" (A: write phi phi^T) | "carry" (B: write u u^T from a value vec)
        value_in_dim: int | None = None, # raw dim of the carried value vector (carry mode)
        whiten: bool = False,            # standardize raw CM vectors before projection (fixes anisotropy)
        eta_init: float = 0.3,
        gamma_init: float = 0.9,
        decay_mode: str = "scalar",      # "scalar" (one gamma) | "diag" (per-direction gamma)
        use_joint: bool = True,
        spectral_clip: float | None = None,
        per_peer_decay: bool = False,
        device=None,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.num_peers = int(num_peers)
        self.rank = int(rank)
        self.task_types = tuple(t.lower() for t in task_types)
        self._task_index = {t: i for i, t in enumerate(self.task_types)}
        self.phi_mode = str(phi_mode)
        self.write_mode = str(write_mode)
        self.use_joint = bool(use_joint)
        self.spectral_clip = spectral_clip  # optional eigenvalue clip to [-rho, rho]
        self._mdtype = dtype

        # ---- trainable params (the only things the optimizer touches) ----
        # phi_t = L2-normalized context DIRECTION via the soft-prototype address:
        #   proto: phi = proto_proj(softmax(cos(whiten(h_q), task_centroids)/tau)). Centroids are
        #         FIXED, precomputed per-task CM means (NOT learned -> cannot collapse into a
        #         re-encoded task label like the dropped codebook). Seen tasks resolve to ~their
        #         own slot (recovers a reliable per-(peer,task) signal); an UNSEEN/related task
        #         lands as a soft MIXTURE over known centroids, so trust transfers in (the
        #         generalization win). The two rejected extremes -- task one-hot (collides but
        #         closed, no transfer) and raw continuous vec (open but never collides, ~NO-WRITE)
        #         -- were measured and dropped; proto is the validated middle regime.
        if self.phi_mode != "proto":
            raise ValueError(f"phi_mode must be 'proto', got {self.phi_mode}")
        assert phi_in_dim, "phi_mode='proto' needs phi_in_dim (raw CM hidden dim)"
        self.phi_in_dim = int(phi_in_dim)
        self._proto_tau = float(proto_tau)
        self.phi = None
        self.phi_proj = None
        self.phi_whitener = None
        # prototypes + whitening stats are FIXED buffers, filled by set_prototypes() before
        # training. n_proto defaults to #task_types but is overwritten on set_prototypes.
        n_proto = len(self.task_types)
        self.register_buffer("proto_centroids", torch.zeros(n_proto, self.phi_in_dim, dtype=dtype))
        self.register_buffer("proto_mean", torch.zeros(self.phi_in_dim, dtype=dtype))
        self.register_buffer("proto_std", torch.ones(self.phi_in_dim, dtype=dtype))
        self.register_buffer("proto_ready", torch.zeros(1, dtype=dtype))
        # the ONLY learnable part: maps the [n_proto] soft-address dist -> rank direction.
        self.proto_proj = nn.Linear(n_proto, self.rank, bias=False)
        nn.init.normal_(self.proto_proj.weight, std=1.0 / (n_proto ** 0.5))
        # Carry mode (B): the WRITTEN value direction u is a projection of a raw value vector
        # (e.g. CM_hidden(peer_answer) - CM_hidden(gold)), NOT phi. phi is still used for READS.
        if self.write_mode == "carry":
            assert value_in_dim, "write_mode='carry' needs value_in_dim (raw value vector dim)"
            self.value_in_dim = int(value_in_dim)
            self.diff_proj = nn.Linear(self.value_in_dim, self.rank, bias=False)
            nn.init.normal_(self.diff_proj.weight, std=1.0 / (self.value_in_dim ** 0.5))
            self.diff_whitener = Whitener(self.value_in_dim, dtype=dtype) if whiten else None
        else:
            self.diff_proj = None
            self.diff_whitener = None
        # write strength eta > 0 via softplus; bounds the per-event Weyl step ||E|| = eta.
        self._eta_raw = nn.Parameter(torch.tensor(float(_inv_softplus(eta_init))))
        # decay: gamma = exp(-softplus(theta)) in (0,1]; structurally stable for any theta.
        #   decay_mode="scalar": one theta -> one gamma for the whole matrix.
        #   decay_mode="diag":   a per-eigen-direction gamma vector in the FIXED basis (rank
        #     diagonal). Update keeps symmetry via the two-sided form
        #     M <- diag(g)^{1/2} M diag(g)^{1/2} + eta s phi phi^T, so directions can forget
        #     at different rates (long-memory dims vs fast-adapting dims). Stays bounded since
        #     every g_i in (0,1].
        self.decay_mode = str(decay_mode)
        # per_peer_decay: give EACH peer its own decay theta, so e.g. gemma's trust time-constant
        # is learned separately from phi's. Shape gains a leading num_peers axis. Default off keeps
        # the original single shared theta (and old checkpoints loadable).
        self.per_peer_decay = bool(per_peer_decay)
        np_ax = (self.num_peers,) if self.per_peer_decay else ()
        if self.decay_mode == "scalar":
            self._theta = nn.Parameter(torch.full(np_ax, float(_inv_softplus(-_log(gamma_init)))))
        elif self.decay_mode == "diag":
            self._theta = nn.Parameter(torch.full((*np_ax, self.rank), float(_inv_softplus(-_log(gamma_init)))))
        else:
            raise ValueError(f"decay_mode must be 'scalar' or 'diag', got {self.decay_mode}")
        self._theta_joint = nn.Parameter(torch.tensor(float(_inv_softplus(-_log(gamma_init)))))

        # ---- non-trainable state buffers (the matrices themselves) ----
        self.register_buffer("M", torch.zeros(self.num_peers, self.rank, self.rank, dtype=dtype))
        self.register_buffer("G", torch.zeros(self.num_peers, self.num_peers, dtype=dtype))
        if device is not None:
            self.to(device)

    # ---- parameter accessors (kept positive / in-range by construction) ----
    @property
    def eta(self) -> torch.Tensor:
        return nn.functional.softplus(self._eta_raw)

    def gamma(self) -> torch.Tensor:
        return torch.exp(-nn.functional.softplus(self._theta))

    def gamma_for(self, peer: int) -> torch.Tensor:
        """Decay for one peer: indexes the leading peer axis when per_peer_decay, else shared."""
        g = self.gamma()
        return g[peer] if self.per_peer_decay else g

    def gamma_joint(self) -> torch.Tensor:
        return torch.exp(-nn.functional.softplus(self._theta_joint))

    def gamma_scalar(self) -> float:
        """Mean gamma as a python float (works for scalar or diag decay; logging only)."""
        return float(self.gamma().mean())

    # ---- phi_t : L2-normalized context direction (soft-prototype address) ----
    def _device(self):
        return self.proto_proj.weight.device

    def set_prototypes(self, centroids: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install FIXED, precomputed task centroids and whitening stats for the proto address.
        centroids: [n_proto, phi_in_dim] (per-task CM mean, RAW unwhitened). mean/std: [phi_in_dim].
        Call once before train/eval. n_proto may differ from #task_types (e.g. train on a
        subset of tasks), so proto_proj is rebuilt to match. Centroids are not learned."""
        dev = self.proto_proj.weight.device
        n_proto = int(centroids.shape[0])
        c = centroids.to(dev, self.proto_centroids.dtype)
        # store centroids in WHITENED space so phi_of only whitens the query
        cw = (c - mean.to(dev)) / (std.to(dev) + 1e-6)
        self.proto_centroids = cw
        self.proto_mean = mean.to(dev, self.proto_mean.dtype)
        self.proto_std = std.to(dev, self.proto_std.dtype)
        if n_proto != self.proto_proj.in_features:
            self.proto_proj = nn.Linear(n_proto, self.rank, bias=False).to(dev)
            nn.init.normal_(self.proto_proj.weight, std=1.0 / (n_proto ** 0.5))
        self.proto_ready = torch.ones(1, device=dev, dtype=self.proto_ready.dtype)

    def phi_of(self, ctx) -> torch.Tensor:
        """ctx is a raw CM hidden vector (or (task, vector) tuple; the vector is used).

        Returns an L2-normalized rank-dim direction (||phi||=1 -> uniform Weyl step ||E||=eta):
        soft-address the query over FIXED task centroids, then project that distribution.
        """
        dev = self._device()
        assert float(self.proto_ready) > 0, "phi_mode='proto' needs set_prototypes() first"
        raw = ctx[1] if isinstance(ctx, tuple) else ctx
        raw = raw if torch.is_tensor(raw) else torch.as_tensor(raw, device=dev)
        raw = raw.to(dev, self.proto_proj.weight.dtype)
        q = (raw - self.proto_mean) / (self.proto_std + 1e-6)          # whiten query
        C = self.proto_centroids.to(self.proto_proj.weight.dtype)      # [n_proto, H], whitened
        sim = (C @ q) / (C.norm(dim=1) * q.norm() + 1e-9)              # [n_proto] cosine
        p = torch.softmax(sim / self._proto_tau, dim=0)               # soft address
        v = self.proto_proj(p)                                        # [rank]
        return v / (v.norm() + 1e-8)

    def value_of(self, raw_value) -> torch.Tensor:
        """Carry mode (B): L2-normalized rank-dim WRITE direction from a raw value vector
        (e.g. CM_hidden(peer_answer) - CM_hidden(gold)). Whitened then projected by diff_proj."""
        dev = self._device()
        raw = raw_value if torch.is_tensor(raw_value) else torch.as_tensor(raw_value, device=dev)
        raw = raw.to(dev, self.diff_proj.weight.dtype)
        if self.diff_whitener is not None:
            self.diff_whitener.observe(raw)
            raw = self.diff_whitener(raw)
        v = self.diff_proj(raw)
        return v / (v.norm() + 1e-8)

    # ---- state lifecycle ----
    def reset(self) -> None:
        with torch.no_grad():
            self.M.zero_()
            self.G.zero_()

    def snapshot(self) -> dict:
        return {"M": self.M.detach().clone(), "G": self.G.detach().clone()}

    def restore(self, snap: dict) -> None:
        with torch.no_grad():
            self.M.copy_(snap["M"])
            self.G.copy_(snap["G"])

    # ---- event-axis updates (called ONCE PER TASK, after the outcome is known) ----
    @torch.no_grad()
    def update(self, peer: int, sign: float, ctx, value_vec=None) -> None:
        """M_peer <- gamma*M_peer + eta*sign*(w w^T)  (symmetric rank-1, event-level).
        addr mode (A): w = phi(ctx)              -> trust addressed by content/task direction.
        carry mode (B): w = value_of(value_vec)  -> trust written along the ERROR direction
                        (peer_answer - gold), signed by correctness."""
        if not (0 <= peer < self.num_peers):
            return
        if self.write_mode == "carry":
            assert value_vec is not None, "carry mode update needs value_vec"
            w = self.value_of(value_vec).to(self._mdtype).detach()
        else:
            w = self.phi_of(ctx).to(self._mdtype).detach()
        g = self.gamma_for(peer).to(self._mdtype).detach()
        e = self.eta.to(self._mdtype).detach()
        outer = torch.outer(w, w)  # symmetric for any w -> M stays symmetric
        if self.decay_mode == "scalar":
            decayed = g * self.M[peer]
        else:  # diag: two-sided diag(g)^{1/2} M diag(g)^{1/2} preserves symmetry
            gh = g.sqrt()
            decayed = gh.unsqueeze(1) * self.M[peer] * gh.unsqueeze(0)
        self.M[peer] = decayed + e * float(sign) * outer
        if self.spectral_clip is not None:
            self.M[peer] = _clip_spectrum(self.M[peer], self.spectral_clip)

    @torch.no_grad()
    def update_joint(self, outcome_vec: Sequence[float]) -> None:
        """G <- gamma_G*G + eta*o o^T (symmetric joint state)."""
        if not self.use_joint:
            return
        o = torch.zeros(self.num_peers, dtype=self._mdtype, device=self.G.device)
        for p, v in enumerate(outcome_vec[: self.num_peers]):
            o[p] = float(v)
        g = self.gamma_joint().to(self._mdtype).detach()
        e = self.eta.to(self._mdtype).detach()
        self.G = g * self.G + e * torch.outer(o, o)

    @torch.no_grad()
    def decay_without_feedback(self) -> None:
        """Advance one event without adding a correctness-driven innovation.

        Selective-feedback experiments still advance the event clock on every
        example.  This applies exactly the decay term from ``update`` to every
        peer matrix and, when enabled, to the joint state ``G``.
        """
        for peer in range(self.num_peers):
            g = self.gamma_for(peer).to(self._mdtype).detach()
            if self.decay_mode == "scalar":
                self.M[peer].mul_(g)
            else:
                gh = g.sqrt()
                self.M[peer].copy_(
                    gh.unsqueeze(1) * self.M[peer] * gh.unsqueeze(0)
                )
        if self.use_joint:
            self.G.mul_(self.gamma_joint().to(self._mdtype).detach())

    def steer_vector(self, peer: int, ctx) -> torch.Tensor:
        """Rank-dim trust direction for steering: M_peer @ phi*  (differentiable thru phi*).

        Points along the peer's accumulated reliability response to the current task context;
        magnitude scales with how much aligned evidence M_peer holds in that direction.
        """
        dev = self._device()
        if not (0 <= peer < self.num_peers):
            return torch.zeros(self.rank, device=dev)
        phi = self.phi_of(ctx)
        M = self.M[peer].to(phi.dtype)
        return M @ phi

    def steer_vector_diff(self, peer: int, ctx, sign: float, value_vec=None) -> torch.Tensor:
        """Like steer_vector, but reads through ONE differentiable write applied to a detached
        copy of M_peer: M' = gamma*M_detached + eta*sign*(w w^T), then return M' @ phi.

        History stays a detached buffer (only this single step carries grad), so loss -> M' ->
        {theta (gamma), eta, phi/phi_proj}. This is what makes the decay/write-strength LEARNED
        instead of fixed -- gradient now flows to _theta/_eta_raw, which the plain (read-only)
        steer_vector path structurally cannot deliver. Train-only: sign is the ground-truth
        correctness, so eval must NOT call this (would leak the label).
        """
        dev = self._device()
        if not (0 <= peer < self.num_peers):
            return torch.zeros(self.rank, device=dev)
        phi = self.phi_of(ctx)
        if self.write_mode == "carry":
            assert value_vec is not None, "carry mode diff-write needs value_vec"
            w = self.value_of(value_vec).to(phi.dtype)
        else:
            w = phi
        g = self.gamma_for(peer).to(phi.dtype)             # differentiable (no detach)
        e = self.eta.to(phi.dtype)                          # differentiable
        M_hist = self.M[peer].to(phi.dtype).detach()        # history: detached buffer
        outer = torch.outer(w, w)
        if self.decay_mode == "scalar":
            M_new = g * M_hist + e * float(sign) * outer
        else:  # diag two-sided form, preserves symmetry
            gh = g.sqrt()
            M_new = gh.unsqueeze(1) * M_hist * gh.unsqueeze(0) + e * float(sign) * outer
        return M_new @ phi

    @torch.no_grad()
    def redundancy_penalty(self, peer: int, chosen: Sequence[int]) -> float:
        """Sum of G[peer, q] over already-chosen peers q (>0 = redundant with the team).

        For team assembly / anti-redundant tie-break: subtract this from a candidate's score
        so the CM does not stack near-duplicate agents. Read-only (no grad needed at decision).
        """
        if not self.use_joint or not chosen:
            return 0.0
        return float(sum(self.G[peer, q].item() for q in chosen if 0 <= q < self.num_peers))

    # ---- diagnostics ----
    @torch.no_grad()
    def symmetry_error(self) -> float:
        return float((self.M - self.M.transpose(-1, -2)).abs().max())

    @torch.no_grad()
    def spectral_norms(self) -> list[float]:
        return [float(torch.linalg.eigvalsh(self.M[p].float()).abs().max()) for p in range(self.num_peers)]


# --------------------------------------------------------------------------- helpers
def _log(x: float) -> float:
    import math
    return math.log(x)


def _inv_softplus(y: float) -> float:
    """Inverse of softplus so softplus(raw)=y exactly at init (y>0)."""
    import math
    return math.log(math.expm1(y)) if y > 0 else -10.0


def _clip_spectrum(M: torch.Tensor, rho: float) -> torch.Tensor:
    """Project a symmetric matrix's eigenvalues into [-rho, rho] (explicit stability cap)."""
    w, V = torch.linalg.eigh(M.float())
    w = w.clamp(-rho, rho)
    return (V @ torch.diag(w) @ V.transpose(-1, -2)).to(M.dtype)


@torch.no_grad()
def cm_context_vector(base_model, tokenizer, text: str, *, device, max_length: int = 2048,
                      layer_frac: float = 0.5) -> torch.Tensor:
    """Encode `text` with the frozen CM and mean-pool a mid-stack layer -> a raw CM hidden
    vector. Used by the proto address: each training task's pooled vectors form the FIXED
    centroids C, and a query's vector is soft-addressed against them in phi_of().
    Returns a [hidden] tensor on `device` (detached; proto_proj learns the down-projection).
    """
    enc = tokenizer(text if str(text).strip() else " ", return_tensors="pt",
                    truncation=True, max_length=max_length)
    # empty string tokenizes to a (1,0) float tensor -> guard so embedding never sees floats
    if enc["input_ids"].numel() == 0 or enc["input_ids"].dtype.is_floating_point:
        enc = tokenizer(" ", return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = base_model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
    hs = out.hidden_states
    layer = hs[max(1, int(len(hs) * layer_frac))][0]  # [T, H]
    return layer.mean(0).detach()


class ActivationSteerer(nn.Module):
    """Coupling (b): inject a trust-derived steering vector into the frozen CM's residual
    stream via forward hooks on the upper-half decoder layers.

    Self-contained (does NOT touch deltamem or joint_models). The trust read for the
    candidate being scored produces a rank-dim vector; a trainable projection maps it to
    hidden size and a learnable gain scales the residual add. Hooks are installed once and
    read ``self.steer_vec`` (set per candidate by the caller; None => no steering, so the
    backbone runs untouched). This is the higher-risk arm: injecting a slowly-varying
    cross-event signal into every token mirrors the content-steer path the ablations
    flagged — which is exactly what the experiment measures.
    """

    def __init__(self, base_model, *, rank: int, layer_frac: float = 0.5, dtype=torch.float32):
        super().__init__()
        self.hidden = int(base_model.config.hidden_size)
        self.rank = int(rank)
        self.proj = nn.Linear(self.rank, self.hidden, bias=False)
        nn.init.normal_(self.proj.weight, std=1e-3)  # near-zero start: step-0 ~= frozen base
        self.gain = nn.Parameter(torch.tensor(0.0))   # sigmoid-free scalar gain, learned
        self.steer_vec = None  # rank-dim tensor, set by caller per candidate; None = off
        self._handles = []
        layers = _decoder_layers(base_model)
        start = max(0, int(len(layers) * layer_frac))
        for layer in layers[start:]:
            self._handles.append(layer.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        if self.steer_vec is None:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        add = self.gain * self.proj(self.steer_vec.to(self.proj.weight.dtype))
        # Single candidate: [hidden]. Batched candidate scoring: [B, hidden].
        if add.ndim == 2:
            add = add.unsqueeze(1)
        elif add.ndim != 1:
            raise ValueError(f"steer_vec must be [rank] or [batch, rank], got {tuple(self.steer_vec.shape)}")
        # .to(hs.device) makes this safe under device_map sharding (the hooked layer may live
        # on a different GPU than the steerer's proj). No-op when everything is on one device.
        hs = hs + add.to(device=hs.device, dtype=hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def _decoder_layers(model):
    """Return the underlying decoder layer list, unwrapping PEFT/LoRA wrappers."""
    seen = set()
    stack = [model]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if hasattr(cur, "layers"):
            return cur.layers
        for attr in ("model", "base_model"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                stack.append(nxt)
    raise AttributeError("Could not locate decoder layers on central model")
