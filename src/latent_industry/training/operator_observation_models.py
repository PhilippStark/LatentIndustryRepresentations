"""operator_observation_models.py — Phase 3 of the operator-preserving
regime. No neural network, no encoder: `Z` itself is the encoder (see the
implementation plan's strategic note, "no encoder in v1"). Each layer gets
its own `LayerObservationModel`, approximating the action of that layer's
diffusion operator from the shared coordinates `Z`.

The interface returns **logits** (masked scores, pre-softmax), not
log-probabilities — that keeps the softmax/log-softmax step in exactly one
place (operator_losses.py) instead of every observation model needing to
agree on a probability-space convention, and it's what lets mse_loss (which
needs probabilities, not log-probabilities) share the same forward-pass
contract as kl_loss/cross_entropy_loss without a special case.
"""

import torch
import torch.nn as nn


class LayerObservationModel(nn.Module):
    def forward(self, Z: torch.Tensor, domain_mask: torch.Tensor) -> torch.Tensor:
        """Returns masked logits, shape (n_domain, n_domain), diagonal = -inf.
        Softmax/log-softmax is NOT this class's job — see operator_losses.py."""
        raise NotImplementedError


class BilinearObservationModel(LayerObservationModel):
    """S_l = Z_l @ M_l @ Z_l.T, Z_l = Z[domain_mask]. `M_l` is a full
    (non-symmetric) d x d matrix — non-symmetric even for layers built from
    symmetric raw edges (labor/nace_text/io), because row-normalizing by
    different row sums already breaks symmetry in `P` itself; a symmetric
    `M_l` would be actively wrong there, not just a simplification (see the
    design doc §4)."""

    def __init__(self, dim: int):
        super().__init__()
        self.M = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.M)

    def forward(self, Z: torch.Tensor, domain_mask: torch.Tensor) -> torch.Tensor:
        Z_l = Z[domain_mask]
        S_l = (Z_l @ self.M @ Z_l.T).clone()
        S_l.fill_diagonal_(float("-inf"))
        return S_l


class MLPObservationModel(LayerObservationModel):
    """S_l(i,j) = MLP(concat(z_i, z_j)) — concat(2d) -> 64 -> 32 -> 1 (the
    implementation plan's Phase 8 architecture). Only reach for this after
    Phase 7 shows the bilinear form saturating for a given layer: unlike
    `Z_l @ M_l @ Z_l.T` (a single (n_l,d)@(d,d)@(d,n_l) matmul), this needs
    an explicit concat(z_i,z_j) for every one of the n_l^2 pairs before any
    of them can run through the hidden layers — materialised as an explicit
    (n_l^2, 2d) tensor every forward pass. For a domain the size of the
    full 4d-code set (order 1,600), that's on the order of 2.5M pairs,
    every epoch — a real memory/compute cost, not a style choice (see the
    plan's own accounting)."""

    def __init__(self, dim: int, hidden: tuple[int, int] = (64, 32)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(2 * dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, Z: torch.Tensor, domain_mask: torch.Tensor) -> torch.Tensor:
        Z_l = Z[domain_mask]
        n_l = Z_l.shape[0]
        z_i = Z_l.unsqueeze(1).expand(n_l, n_l, -1)
        z_j = Z_l.unsqueeze(0).expand(n_l, n_l, -1)
        pairs = torch.cat([z_i, z_j], dim=-1).reshape(n_l * n_l, -1)
        S_l = self.net(pairs).reshape(n_l, n_l).clone()
        S_l.fill_diagonal_(float("-inf"))
        return S_l


class CosineObservationModel(LayerObservationModel):
    """S(i,j) = cos(z_i, z_j) / tau — Phase 9 (shared-cosine, many-targets
    regime, docs/operator_preserving_implementation_plan.md). Deliberately
    has NO per-layer matrix: `M_l`/an MLP can rotate or reshape `Z_l`'s
    dot-product geometry to fit a target without `Z` itself changing (the
    exact failure mode Phase 7 diagnosed for `io`/`labor` and later
    `transition` — see project history). Every target sharing plain
    `cos(Z_l, Z_l)` and differing only by a learnable scalar temperature
    means any conflict between two targets (e.g. labor vs. io) can only be
    resolved by `Z` itself moving, not by a per-target transform absorbing
    it — "twenty targets, one geometry."

    `log_tau` (not `tau` directly) is the learnable parameter so temperature
    stays strictly positive: `tau = exp(log_tau)`. `log_tau` is additionally
    clamped to `[LOG_TAU_MIN, LOG_TAU_MAX]` before the exponential (v7 §1.1):
    an unregularised `log_tau` is an escape hatch — a target can minimise its
    loss by flattening its own prediction toward uniform (large `tau`)
    instead of moving `Z`, which is exactly what happened to `hierarchy_t1`
    (`tau` reached 10.3, pinned at `ln(domain)` for entire runs in v2-v6).
    The clamp is a differentiable `torch.clamp`: gradient flows normally
    inside the bound and vanishes outside it, so a `log_tau` that drifts past
    the bound simply stops moving (no explicit post-step clipping needed) —
    the `tau` actually used downstream never exceeds `exp(LOG_TAU_MAX)`."""

    LOG_TAU_MIN = -1.0
    LOG_TAU_MAX = 1.0

    def __init__(self, dim: int):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, Z: torch.Tensor, domain_mask: torch.Tensor) -> torch.Tensor:
        Z_l = torch.nn.functional.normalize(Z[domain_mask], p=2, dim=-1)
        tau = torch.exp(torch.clamp(self.log_tau, self.LOG_TAU_MIN, self.LOG_TAU_MAX))
        S_l = (Z_l @ Z_l.T / tau).clone()
        S_l.fill_diagonal_(float("-inf"))
        return S_l


OBSERVATION_MODEL_REGISTRY: dict[str, type[LayerObservationModel]] = {
    "bilinear": BilinearObservationModel,
    "mlp": MLPObservationModel,
    "cosine": CosineObservationModel,
}


def make_observation_model(kind: str, dim: int) -> LayerObservationModel:
    if kind not in OBSERVATION_MODEL_REGISTRY:
        raise ValueError(f"unknown observation model {kind!r} — expected one of {sorted(OBSERVATION_MODEL_REGISTRY)}")
    return OBSERVATION_MODEL_REGISTRY[kind](dim)
