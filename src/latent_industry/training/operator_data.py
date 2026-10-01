"""operator_data.py — Phase 2 of the operator-preserving regime (see
docs/operator_preserving_implementation_plan.md). One flat container for
everything training needs, so operator_trainer.py never touches `Layer` or
`NodeIndex` directly.

Not a `torch.utils.data.Dataset`: training here is full-batch — every epoch
uses every active row of every layer, there's no notion of an
individually-sampled "example" the way `Dataset.__getitem__` assumes, and
`N` is small enough (order 1,000-few-thousand nodes) that a `DataLoader`
would add batching machinery this problem doesn't need.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from latent_industry.graph import artifacts
from latent_industry.training.operator_losses import layer_loss

# Per-layer likelihood assignment (math doc §5, design doc §3): labor/
# nace_text/hierarchy/transition are categorical (kl/ce share one optimum,
# see operator_losses.py); io defaults to kl too, with an mse option
# available behind a config flag for experimentation.
DEFAULT_LOSS_TYPES = {
    "labor": "kl",
    "nace_text": "kl",
    "io": "kl",
    "hierarchy": "ce",
    "transition": "ce",
    "ownership": "kl",
}
DEFAULT_LAMBDAS = {lname: 1.0 / len(DEFAULT_LOSS_TYPES) for lname in DEFAULT_LOSS_TYPES}


@dataclass
class OperatorData:
    targets: dict[str, torch.Tensor]        # layer -> P_tilde, N x N
    domains: dict[str, torch.Tensor]         # layer -> N, bool
    active_masks: dict[str, torch.Tensor]    # layer -> N, bool, active <= domain
    loss_types: dict[str, str]               # layer -> "kl" | "ce" | "mse"
    weights: dict[str, float]                # layer -> lambda_l
    margins: dict[str, float] = field(default_factory=dict)  # layer -> saturation floor (default 0, no-op)
    families: dict[str, str] = field(default_factory=dict)   # target_id -> family (v7 §4.2), {} if unused

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
    """The loss `kind` reports for the trivial "predict uniform" baseline,
    restricted to this target's own domain/active rows exactly like
    operator_trainer.restrict_to_domain does (diagonal excluded, rows
    renormalised over the remaining support) — re-derived locally rather
    than imported from operator_trainer.py to avoid a circular import
    (operator_trainer.py already imports OperatorData from this module).
    Used to scale each expanded target's raw loss down to "fraction of
    trivial-baseline loss still unexplained" before weighting, since domain
    size and target sharpness both shift a categorical loss's raw scale
    (math doc §5 / implementation plan Phase 5) — without this, a target
    with a bigger domain or a near-one-hot target row would automatically
    look "harder" and end up dominating the joint gradient regardless of
    its assigned family weight.

    `kind == "ordinal"` (v7 §1.4) is special-cased: its `target` is a
    tier-rank matrix (values in {0..4}), not a probability distribution, so
    it must NOT go through the diagonal-zero/row-renormalise step every
    other kind gets here — that would corrupt the tier labels. The "trivial
    baseline" for an ordinal target is all-equal (zero) logits, which makes
    every present tier's mean similarity equal, so the null loss is a
    deterministic, non-zero function of `margin` — safe to divide by."""
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
    saturation_frac: float = 0.0,
    device: str = "cpu",
    loss_type_overrides: dict[str, str] | None = None,
) -> OperatorData:
    """Phase 9 loader (docs/operator_preserving_implementation_plan.md):
    reads layer_diffusion_expanded_{name}.npz, whose targets are keyed by
    `target_id = f"{layer}_t{t}"` rather than by layer — every (layer, t)
    scale kept as its own separate training target (see
    layer_diffusion.build_layer_diffusion_targets_expanded), scored by a
    shared-cosine observation model (operator_observation_models.CosineObservationModel)
    instead of a per-layer M_l.

    The returned `OperatorData.weights[target_id]` is a **two-stage**
    normalisation (v7 §1.3 — the single-stage version had a real bug, not
    just an approximation: `family_weights[family] /
    n_targets_in_family[family] / null_loss[target_id]` applies the
    `/n_targets_in_family` split BEFORE dividing by null loss, so the
    null-loss division — which varies enormously by domain/family scale —
    destroys the intended even split entirely. Measured in v5: io and labor
    ended up at 41.6%/41.0% of the joint loss against an intended 20% each,
    while hierarchy got 2.6%. The fix re-normalises a SECOND time, after the
    null-loss division, per family:

        w_t = family_weights[family] * (1 / null_loss[t]) /
              sum_{t' in family(t)} (1 / null_loss[t'])

    This still normalises each target's own raw loss by its own trivial-
    baseline scale (so a bigger domain or near-one-hot target doesn't
    automatically dominate), but now GUARANTEES `sum(w_t for t in family) ==
    family_weights[family]` exactly, regardless of how many scales that
    family has or how uneven their null losses are — the property the old
    formula never actually had.

    `saturation_frac` (default 0, no-op — user's item #1, 2026-07-27):
    `OperatorData.margins[target_id] = saturation_frac * null_loss[target_id]`,
    a per-target loss floor operator_trainer.train_joint subtracts before
    weighting (clamped at 0) — once a target's raw loss drops below
    `saturation_frac` of its own trivial-baseline loss, it's "satisfied"
    and contributes exactly zero further gradient, instead of continuing to
    compete (however faintly) with targets that are still unsatisfied. A
    reasonable starting point is small (e.g. 0.05-0.1): the empirical
    motivation (shared_cosine_v2) was `transition_SNI92_SNI02_t1` fitting to
    ~2% of its null loss while `transition_SNI02_SNI07_t1` stayed pinned
    near 100% of its own — a saturating floor lets the former stop pulling
    on shared SNI02 anchor nodes once it's clearly done, rather than
    continuing to sharpen indefinitely.

    `loss_type_overrides` (v7 §1.4, default None): per-TARGET loss-kind
    overrides applied after the family-based `loss_types_by_family` lookup —
    lets e.g. `hierarchy_t1` be `"ordinal"` while `hierarchy_t4` stays the
    family's normal `"ce"`, without needing a new top-level family. `_null_loss`
    special-cases `"ordinal"` targets (see below): their target matrix is a
    tier-rank matrix, not a probability distribution, so it must not go
    through the row-renormalisation every other kind gets.

    The returned `OperatorData.families` (v7 §4.2) is the same
    `target_id -> family` mapping this loader already reads from
    `artifacts.load_layer_diffusion_targets_expanded`, now threaded through
    instead of being discarded — lets a caller (e.g.
    train_shared_cosine_embedding.py's `--eval-every`) group per-epoch raw
    losses by family without needing to re-derive it from target_id naming
    conventions."""
    targets_np, domains_np, families = artifacts.load_layer_diffusion_targets_expanded(name)
    active_np = {t: (targets_np[t].sum(axis=1) > 0) for t in targets_np}

    family_names = sorted(set(families.values()))
    loss_types_by_family = (dict(loss_types_by_family) if loss_types_by_family is not None
                             else dict(DEFAULT_LOSS_TYPES))
    family_weights = (dict(family_weights) if family_weights is not None
                       else {f: 1.0 / len(family_names) for f in family_names})
    loss_type_overrides = dict(loss_type_overrides) if loss_type_overrides is not None else {}

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
