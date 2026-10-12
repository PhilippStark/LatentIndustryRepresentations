"""
operator_data.py — one container (OperatorData) with everything training
needs: targets, domains, active rows, loss types, per-target weights and
margins, and the target → family mapping. Training is full-batch (every
active row of every target in every epoch), so no Dataset/DataLoader.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from latent_industry.graph import artifacts
from latent_industry.training.operator_losses import layer_loss

# Likelihood per family: all are categorical (kl/ce share one optimum, see
# operator_losses.py). hierarchy_t1 is overridden to "ordinal"
# (DEFAULT_LOSS_TYPE_OVERRIDES).
DEFAULT_LOSS_TYPES = {
    "labor": "kl",
    "io": "kl",
    "hierarchy": "ce",
    "transition": "ce",
    "ownership": "kl",
}
DEFAULT_LOSS_TYPE_OVERRIDES = {"hierarchy_t1": "ordinal"}

# Family weights of the main run (shared_cosine_v16).
DEFAULT_LAMBDAS = {
    "labor": 0.30,
    "io": 0.25,
    "hierarchy": 0.10,
    "transition": 0.10,
    "ownership": 0.125,
}


@dataclass
class OperatorData:
    targets: dict[str, torch.Tensor]        # layer -> P_tilde, N x N
    domains: dict[str, torch.Tensor]         # layer -> N, bool
    active_masks: dict[str, torch.Tensor]    # layer -> N, bool, active <= domain
    loss_types: dict[str, str]               # layer -> "kl" | "ce" | "ordinal"
    weights: dict[str, float]                # layer -> lambda_l
    margins: dict[str, float] = field(default_factory=dict)  # layer -> saturation floor (default 0, no-op)
    families: dict[str, str] = field(default_factory=dict)   # target_id -> family, {} if unused

    @staticmethod
    def from_numpy(
        targets: dict[str, np.ndarray],
        domains: dict[str, np.ndarray],
        active_masks: dict[str, np.ndarray],
        loss_types: dict[str, str],
        weights: dict[str, float],
        margins: dict[str, float] | None = None,
        device: str = "cpu",
        families: dict[str, str] | None = None,
    ) -> "OperatorData":
        return OperatorData(
            targets={l: torch.tensor(t, dtype=torch.float32, device=device) for l, t in targets.items()},
            domains={l: torch.tensor(d, dtype=torch.bool, device=device) for l, d in domains.items()},
            active_masks={l: torch.tensor(a, dtype=torch.bool, device=device) for l, a in active_masks.items()},
            loss_types=dict(loss_types),
            weights=dict(weights),
            margins=dict(margins) if margins is not None else {},
            families=dict(families) if families is not None else {},
        )


def _null_loss(kind: str, target: np.ndarray, domain: np.ndarray, active: np.ndarray) -> float:
    """
    Loss of the trivial "predict uniform" baseline for one target, on the same
    domain-restricted, diagonal-excluded, row-renormalised target as
    operator_trainer.restrict_to_domain (re-derived here to avoid a circular
    import). Raw losses are divided by it so that domain size and target
    sharpness do not make a target look harder than it is.

    For kind == "ordinal" the target is a tier-rank matrix, not a distribution,
    so it is not renormalised; the baseline (all-equal logits) gives a
    non-zero null loss that depends only on the margin.
    """
    if kind == "ordinal":
        t = torch.tensor(target[domain][:, domain], dtype=torch.float32)
        a = torch.tensor(active[domain], dtype=torch.bool)
        if not a.any():
            return 1.0
        n_domain = t.shape[-1]
        zero_logits = torch.zeros(n_domain, n_domain)
        zero_logits.fill_diagonal_(float("-inf"))
        return max(layer_loss(kind, zero_logits, t, a).item(), 1e-12)

    t = torch.tensor(target[domain][:, domain], dtype=torch.float32)
    t.fill_diagonal_(0.0)
    row_sums = t.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    t = t / row_sums
    a = torch.tensor(active[domain], dtype=torch.bool)
    if not a.any():
        return 1.0
    n_domain = t.shape[-1]
    zero_logits = torch.zeros(n_domain, n_domain)
    zero_logits.fill_diagonal_(float("-inf"))
    return max(layer_loss(kind, zero_logits, t, a).item(), 1e-12)


def load_operator_data_expanded(
    name: str,
    loss_types_by_family: dict[str, str] | None = None,
    family_weights: dict[str, float] | None = None,
    saturation_frac: float = 0.2,
    device: str = "cpu",
    loss_type_overrides: dict[str, str] | None = None,
) -> OperatorData:
    """
    Load layer_diffusion_expanded_{name}.npz (targets keyed by
    target_id = f"{view}_t{t}") into an OperatorData for the shared-cosine model.

    Weights (two-stage normalisation): within each family, targets are weighted
    by the inverse of their null loss and rescaled so that the family's targets
    sum to its family weight,

        w_t = family_weights[family] * (1 / null_loss[t]) /
              Σ_{t' ∈ family(t)} (1 / null_loss[t'])

    so each family receives exactly its assigned weight regardless of how many
    scales it has or how different their null losses are.

    saturation_frac (main run: 0.2; 0 = off): margins[t] = saturation_frac *
    null_loss[t]; operator_trainer.train_joint subtracts this floor (clamped at
    0), so a target below the floor stops contributing gradient.

    loss_type_overrides: per-target loss kinds applied after the per-family
    loss types (default DEFAULT_LOSS_TYPE_OVERRIDES: hierarchy_t1 = "ordinal"
    while hierarchy_t2/t4 stay "ce").

    The returned OperatorData.families maps target_id → family, used to group
    per-epoch losses by family (train_shared_cosine_embedding.py --eval-every).
    """
    targets_np, domains_np, families = artifacts.load_layer_diffusion_targets_expanded(name)
    active_np = {t: (targets_np[t].sum(axis=1) > 0) for t in targets_np}

    loss_types_by_family = (dict(loss_types_by_family) if loss_types_by_family is not None
                             else dict(DEFAULT_LOSS_TYPES))
    family_weights = dict(family_weights) if family_weights is not None else dict(DEFAULT_LAMBDAS)
    loss_type_overrides = (dict(loss_type_overrides) if loss_type_overrides is not None
                           else dict(DEFAULT_LOSS_TYPE_OVERRIDES))

    loss_types = {t: loss_type_overrides.get(t, loss_types_by_family[families[t]]) for t in targets_np}

    null_losses: dict[str, float] = {}
    family_inv_null_sum: dict[str, float] = {}
    for target_id in targets_np:
        family = families[target_id]
        null = _null_loss(loss_types[target_id], targets_np[target_id], domains_np[target_id], active_np[target_id])
        null_losses[target_id] = null
        family_inv_null_sum[family] = family_inv_null_sum.get(family, 0.0) + 1.0 / null

    weights: dict[str, float] = {}
    margins: dict[str, float] = {}
    for target_id in targets_np:
        family = families[target_id]
        null = null_losses[target_id]
        weights[target_id] = family_weights[family] * (1.0 / null) / family_inv_null_sum[family]
        margins[target_id] = saturation_frac * null

    return OperatorData.from_numpy(targets_np, domains_np, active_np, loss_types, weights,
                                    margins=margins, device=device, families=families)
