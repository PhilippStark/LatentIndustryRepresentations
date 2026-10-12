"""
operator_observation_models.py — map the shared embedding Z to per-target
similarity logits. Z itself is the model (no encoder network).

The model returns masked logits (diagonal −inf); softmax / log-softmax is
applied in operator_losses.py.
"""

import torch
import torch.nn as nn


class CosineObservationModel(nn.Module):
    """
    S(i, j) = cos(z_i, z_j) / tau. There is no per-target matrix, so targets can
    only be fitted by moving Z itself: all targets share one geometry and differ
    only by a learnable scalar temperature.

    The learnable parameter is log_tau (tau = exp(log_tau) > 0), clamped to
    [LOG_TAU_MIN, LOG_TAU_MAX] before the exponential, i.e. tau ∈ [e^-1, e]
    ≈ [0.37, 2.72]. Without the bound a target could lower its loss by
    flattening its prediction (large tau) instead of moving Z. Gradients vanish
    outside the bound.
    """

    LOG_TAU_MIN = -1.0
    LOG_TAU_MAX = 1.0

    def __init__(self):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, Z: torch.Tensor, domain_mask: torch.Tensor) -> torch.Tensor:
        """Returns masked logits, shape (n_domain, n_domain), diagonal = -inf."""
        Z_l = torch.nn.functional.normalize(Z[domain_mask], p=2, dim=-1)
        tau = torch.exp(torch.clamp(self.log_tau, self.LOG_TAU_MIN, self.LOG_TAU_MAX))
        S_l = (Z_l @ Z_l.T / tau).clone()
        S_l.fill_diagonal_(float("-inf"))
        return S_l
