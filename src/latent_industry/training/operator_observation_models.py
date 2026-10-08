"""
operator_observation_models.py — map the shared embedding Z to per-target
similarity logits. Z itself is the model (no encoder network).

The main run uses CosineObservationModel for every target. The bilinear and
MLP models are alternative, more flexible observation models that the
shared-cosine training does not use.

Models return masked logits (diagonal −inf); softmax / log-softmax is
applied in operator_losses.py.
"""

import torch
import torch.nn as nn


class LayerObservationModel(nn.Module):
    def forward(self, Z: torch.Tensor, domain_mask: torch.Tensor) -> torch.Tensor:
        """Returns masked logits, shape (n_domain, n_domain), diagonal = -inf.
        Softmax/log-softmax is NOT this class's job — see operator_losses.py."""
        raise NotImplementedError


class BilinearObservationModel(LayerObservationModel):
    """
    S_l = Z_l M_l Z_l^T with Z_l = Z[domain_mask] and a full (non-symmetric)
    d × d matrix M_l per target. Not used by the shared-cosine model.
    """

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
    """
    S_l(i, j) = MLP(concat(z_i, z_j)), 2d → 64 → 32 → 1, evaluated on all
    n_l² pairs. Not used by the shared-cosine model.
    """

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
    """
    S(i, j) = cos(z_i, z_j) / tau. There is no per-target matrix, so targets can
    only be fitted by moving Z itself: all targets share one geometry and differ
    only by a learnable scalar temperature.

    The learnable parameter is log_tau (tau = exp(log_tau) > 0), clamped to
    [LOG_TAU_MIN, LOG_TAU_MAX] before the exponential. Without the bound a
    target could lower its loss by flattening its prediction (large tau)
    instead of moving Z. Gradients vanish outside the bound.
    """

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
