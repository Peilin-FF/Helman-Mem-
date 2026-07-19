"""Event-axis symmetric competence memory for multi-agent selection.

For each peer, the state is a symmetric matrix ``M_p`` updated once per event:
``M_p <- gamma*M_p + eta*s*phi phi^T``.  The task-conditioned readout
``M_p @ phi`` is projected into the frozen center model's residual stream.  The
second-order topology ``G`` used by Sigma-Mem is maintained by the evaluation
posterior, separately from this first-order memory module.
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


class SymmetricTrustMemory(nn.Module):
    """Peer-specific symmetric competence matrices and their learned dynamics."""

    def __init__(
        self,
        *,
        num_peers: int,
        rank: int = 16,
        task_types: Sequence[str] = DEFAULT_TASK_TYPES,
        phi_mode: str = "proto",         # only "proto" (soft task-centroid address); see set_prototypes()
        phi_in_dim: int | None = None,   # raw CM hidden dim the proto address whitens/compares (e.g. 1024)
        proto_tau: float = 0.1,          # softmax temperature over centroid cosines
        eta_init: float = 0.3,
        gamma_init: float = 0.9,
        device=None,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.num_peers = int(num_peers)
        self.rank = int(rank)
        self.task_types = tuple(t.lower() for t in task_types)
        self.phi_mode = str(phi_mode)
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
        # write strength eta > 0 via softplus; bounds the per-event Weyl step ||E|| = eta.
        self._eta_raw = nn.Parameter(torch.tensor(float(_inv_softplus(eta_init))))
        # decay: gamma = exp(-softplus(theta)) in (0,1]; structurally stable for any theta.
        self._theta = nn.Parameter(torch.tensor(float(_inv_softplus(-_log(gamma_init)))))

        # ---- non-trainable state buffer ----
        self.register_buffer("M", torch.zeros(self.num_peers, self.rank, self.rank, dtype=dtype))
        if device is not None:
            self.to(device)

    # ---- parameter accessors (kept positive / in-range by construction) ----
    @property
    def eta(self) -> torch.Tensor:
        return nn.functional.softplus(self._eta_raw)

    def gamma(self) -> torch.Tensor:
        return torch.exp(-nn.functional.softplus(self._theta))

    def gamma_scalar(self) -> float:
        """Shared scalar decay as a Python float for logging."""
        return float(self.gamma())

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

    # ---- state lifecycle ----
    def reset(self) -> None:
        with torch.no_grad():
            self.M.zero_()

    def snapshot(self) -> dict:
        return {"M": self.M.detach().clone()}

    def restore(self, snap: dict) -> None:
        with torch.no_grad():
            self.M.copy_(snap["M"])

    # ---- event-axis updates (called ONCE PER TASK, after the outcome is known) ----
    @torch.no_grad()
    def update(self, peer: int, sign: float, ctx) -> None:
        """Apply the symmetric event update ``M <- gamma M + eta sign phi phi^T``."""
        if not (0 <= peer < self.num_peers):
            return
        w = self.phi_of(ctx).to(self._mdtype).detach()
        g = self.gamma().to(self._mdtype).detach()
        e = self.eta.to(self._mdtype).detach()
        outer = torch.outer(w, w)  # symmetric for any w -> M stays symmetric
        decayed = g * self.M[peer]
        self.M[peer] = decayed + e * float(sign) * outer

    @torch.no_grad()
    def decay_without_feedback(self) -> None:
        """Advance one event without adding a correctness-driven innovation.

        Selective-feedback experiments still advance the event clock on every
        example.  This applies exactly the decay term from ``update`` to every
        peer matrix.
        """
        self.M.mul_(self.gamma().to(self._mdtype).detach())

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

    def training_steer_vector(self, peer: int, ctx, sign: float) -> torch.Tensor:
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
        g = self.gamma().to(phi.dtype)                     # differentiable (no detach)
        e = self.eta.to(phi.dtype)                          # differentiable
        M_hist = self.M[peer].to(phi.dtype).detach()        # history: detached buffer
        outer = torch.outer(phi, phi)
        M_new = g * M_hist + e * float(sign) * outer
        return M_new @ phi

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

    The trust read for the candidate being scored produces a rank-dim vector; a
    trainable projection maps it to
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
    """Return the underlying decoder layer list through common model containers."""
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
