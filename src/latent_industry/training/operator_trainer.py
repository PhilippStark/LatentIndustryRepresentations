"""operator_trainer.py — Phase 5 of the operator-preserving regime. One
joint backward pass per epoch: every layer's loss is computed in the same
forward pass, summed as L = sum_l lambda_l * L_l, and a single
optimizer.step() updates Z and every layer's observation-model parameters
together (design doc §3) — not sequential per-layer backward/step calls.
"""

from typing import Callable

import torch

from latent_industry.training.operator_data import OperatorData
from latent_industry.training.operator_losses import layer_loss


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp(min=1e-12))).sum(dim=-1)


def hinge_loss(loss: torch.Tensor, margin: float) -> torch.Tensor:
    """Saturating loss floor (user's item #1, 2026-07-25): once `loss` drops
    below `margin` (typically a small fraction of that target's own null/
    trivial-baseline loss — see operator_data.load_operator_data_expanded's
    `saturation_frac`), this is exactly zero — both the reported loss AND
    its gradient — instead of continuing to compete, however faintly, with
    targets that are still unsatisfied. `margin=0` (every target's default,
    via OperatorData.margins' empty-dict default) is an exact no-op: `loss`
    is already non-negative (KL/CE/MSE all are), so `clamp(loss - 0, min=0)
    == loss`."""
    return torch.clamp(loss - margin, min=0.0)


def dynamic_reweight_ratio(running_loss_epoch1: float, running_loss_now: float, max_ratio: float) -> float:
    """user's item #2, 2026-07-25: how much to rescale a target's static
    weight so its CONTRIBUTION (weight * raw_loss) stays pinned near its
    epoch-1 value throughout training, instead of fading as the raw loss
    improves. `running_loss_now` -> 0 as a target saturates, so the ratio
    is clamped to `max_ratio` — unbounded growth here is exactly why this
    is meant to be paired with `hinge_loss`'s margin, which zeroes the
    contribution outright once the target is genuinely satisfied, rather
    than relying on this clamp alone."""
    return min(running_loss_epoch1 / max(running_loss_now, 1e-12), max_ratio)


def ema_update(previous: float, new_value: float, decay: float) -> float:
    """Exponential moving average — the `running_loss` estimate
    `dynamic_reweight_ratio` normalises against."""
    return decay * previous + (1 - decay) * new_value


def stall_capped_ratio(
    raw_ratio: float, running_loss_now: float, baseline_loss: float,
    stall_ratio: float, stall_min_improvement: float,
) -> tuple[float, bool]:
    """user's item #4, 2026-07-27: `dynamic_reweight_ratio` alone can't
    distinguish a target that's genuinely hard (loss slow but real to
    move) from one that's structurally unsatisfiable (loss pinned at its
    own null value regardless of weight) — both look identical to the
    reweighter, which just keeps boosting either. `transition_SNI02_SNI07_t1`
    got boosted past 10-15x in `shared_cosine_v4` BECAUSE its loss wouldn't
    move, not despite it — the reweighter's blindness to *why* a loss is
    high is exactly the failure mode this guards against.

    Once `raw_ratio` would cross `stall_ratio`, this requires the loss to
    have improved by at least `stall_min_improvement` (a fraction, e.g.
    0.1 = 10%) relative to `baseline_loss` (the running loss the last time
    the ratio was below `stall_ratio`, tracked by the caller) — if not,
    the ratio is capped AT `stall_ratio` (not allowed to climb further)
    and `True` is returned so the caller can flag it once. A target that
    keeps genuinely improving is never capped, however far its ratio
    climbs — only a target whose loss stops moving gets stopped.
    """
    if raw_ratio < stall_ratio:
        return raw_ratio, False
    improved_enough = running_loss_now <= baseline_loss * (1.0 - stall_min_improvement)
    if improved_enough:
        return raw_ratio, False
    return min(raw_ratio, stall_ratio), True


def restrict_to_domain(operator_data: OperatorData) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Domain-restrict each layer's target/active-mask ONCE (domain masks
    are static across epochs), and put the target into the exact shape/
    normalisation the loss functions expect: diagonal excluded (matching
    the observation model's -inf self-logit) and each row renormalised
    over j != i so it's a proper distribution over the same support the
    softmax produces (math doc §4 — "re-excluding the diagonal, consistent
    with how P̃_l was already zero on the diagonal-adjacent structure").
    Rows that end up all-zero (inactive, or one where the diagonal held all
    the mass) stay exactly zero — `active_mask` already excludes them from
    every loss's row-mean, so a degenerate zero row there is inert.

    Public (not trainer-private): operator_diagnostics.py's Phase 7 report
    needs the exact same domain-restricted/renormalised target shape the
    losses were actually trained against, not a second, potentially
    drifting reimplementation of this step.

    `"ordinal"`-kind targets (v7 §1.4) are special-cased: their target is a
    tier-rank matrix (values in {0..4}), not a probability distribution, so
    the diagonal-zero/row-renormalise step below must be skipped entirely —
    it would corrupt the tier labels (dividing rank integers by a row sum
    that means nothing for them).
    """
    targets, active = {}, {}
    for lname in operator_data.targets:
        dmask = operator_data.domains[lname]
        if operator_data.loss_types[lname] == "ordinal":
            targets[lname] = operator_data.targets[lname][dmask][:, dmask].clone()
            active[lname] = operator_data.active_masks[lname][dmask]
            continue
        t = operator_data.targets[lname][dmask][:, dmask].clone()
        t.fill_diagonal_(0.0)
        row_sums = t.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        targets[lname] = t / row_sums
        active[lname] = operator_data.active_masks[lname][dmask]
    return targets, active


def pretraining_diagnostics(
    Z: torch.Tensor, observation_models: dict[str, torch.nn.Module], operator_data: OperatorData,
    domain_targets: dict[str, torch.Tensor], domain_active: dict[str, torch.Tensor],
    hub_aggregation: torch.Tensor | None = None,
) -> None:
    """One forward (+ isolated backward) pass with the current Z/M_l, before
    the training loop starts. Logs domain size, target entropy, initial
    loss, and ||grad_Z L_l|| per layer — because domain sizes vary a lot
    across layers, uniform lambda_l does not automatically mean "equal
    influence per mechanism"; a bigger raw loss number isn't automatically
    "harder" (a categorical loss's scale grows with log(domain size)), and
    the loss value alone doesn't say whether a layer's gradient will be
    negligible or dominant once the joint loss starts training (see the
    implementation plan's Phase 5).

    `hub_aggregation` (v7 §1.2, default None): if given, every observation
    model sees `hub_aggregation @ Z` instead of raw `Z` — see train_joint's
    docstring. Entropy is skipped (printed as "n/a") for `"ordinal"`-kind
    targets (v7 §1.4): their target is a tier-rank matrix, not a probability
    distribution, so a Shannon entropy over it isn't meaningful.
    """
    print(f"\n{'layer':<12}{'domain':>8}{'entropy':>10}{'init_loss':>12}{'grad_norm':>12}")
    for lname in operator_data.targets:
        dmask = operator_data.domains[lname]
        target_l = domain_targets[lname]
        active_l = domain_active[lname]
        is_ordinal = operator_data.loss_types[lname] == "ordinal"

        if is_ordinal:
            ent_str = "n/a"
        else:
            p_active = target_l[active_l]
            ent = _entropy(p_active).mean().item() if active_l.any() else float("nan")
            ent_str = f"{ent:.4f}"

        Z_eff = hub_aggregation @ Z if hub_aggregation is not None else Z
        logits = observation_models[lname](Z_eff, dmask)
        loss = layer_loss(operator_data.loss_types[lname], logits, target_l, active_l)

        grad = torch.autograd.grad(loss, Z, retain_graph=False, allow_unused=True)[0]
        grad_norm = grad.norm().item() if grad is not None else 0.0

        print(f"{lname:<12}{int(dmask.sum()):>8}{ent_str:>10}{loss.item():>12.4f}{grad_norm:>12.4f}")


def train_joint(
    Z: torch.Tensor,
    observation_models: dict[str, torch.nn.Module],
    operator_data: OperatorData,
    lr: float,
    epochs: int,
    log_every: int = 20,
    dynamic_reweight: bool = False,
    reweight_decay: float = 0.9,
    reweight_max_ratio: float = 20.0,
    reweight_stall_ratio: float = 5.0,
    reweight_stall_min_improvement: float = 0.1,
    hub_aggregation: torch.Tensor | None = None,
    on_epoch: Callable[[int, torch.Tensor, dict[str, float]], None] | None = None,
) -> float:
    """Full-batch joint training loop. Returns the best (lowest) total loss
    observed.

    `dynamic_reweight` (default off, fully backward-compatible — user's
    item #2, 2026-07-27): with the default static `operator_data.weights`,
    a target's CONTRIBUTION to the joint loss (`weight * raw_loss`) shrinks
    automatically as its raw loss improves — which sounds right, but means
    a target that's already been solved keeps fading toward irrelevance
    while a target that's stuck near its initial loss keeps its full
    original share forever; nothing ever explicitly grows a stuck target's
    relative influence. `dynamic_reweight=True` re-derives the null-loss-
    style normalisation CONTINUOUSLY instead of only at epoch 1: it tracks
    an EMA of each target's own raw loss (`running_loss`) and rescales that
    target's weight by `running_loss(epoch=1) / running_loss(epoch)`
    (clamped to `reweight_max_ratio`, since this ratio -> infinity as a
    target's loss -> 0) — algebraically this holds `weight(epoch) *
    raw_loss(epoch) ≈ weight(1) * raw_loss(1)` for every epoch, i.e. every
    target's contribution stays pinned near its original fair share
    (`family_weight / n_targets_in_family`) throughout training instead of
    fading as it's solved. Combined with `operator_data.margins`'s
    saturating hinge below (user's item #1) — which is what actually stops
    an already-solved target's contribution, at exactly zero, once it
    crosses its own margin — the two together are what let capacity
    reallocate toward a stuck target instead of an already-solved sibling
    continuing to sharpen indefinitely.

    `reweight_stall_ratio`/`reweight_stall_min_improvement` (user's item
    #4, 2026-07-27): a guard on the guard. `dynamic_reweight` alone can't
    tell "hard" (loss moving slowly) from "structurally broken" (loss
    pinned regardless of weight) — both just look like "still high," so
    the ratio keeps climbing either way. Once a target's ratio would cross
    `reweight_stall_ratio` (default 5x), `stall_capped_ratio` requires its
    loss to have improved by `reweight_stall_min_improvement` (default 10%)
    since the last time it was below that threshold; if not, the ratio is
    capped there and a one-time WARNING is printed flagging the target as
    stalled rather than hard. This is what should have stopped
    `transition_SNI02_SNI07_t1` at ~5x instead of the ~10-15x
    `shared_cosine_v4` actually reached — its loss wasn't moving BECAUSE it
    was stuck, and the reweighter had no way to tell the difference.

    `hub_aggregation` (v7 §1.2, default None — fully backward-compatible for
    train_operator_embedding.py's bilinear/MLP regime): if given, an (N, N)
    matrix such that `hub_aggregation @ Z` gives every 4d row back unchanged
    and every hub row (3d/2d/section) the mean of its 4d descendants —
    see operator_embedding.build_hub_aggregation_matrix. Computed ONCE per
    epoch (not once per target — the matrix is identical across every
    layer) and used in place of raw `Z` for every observation model's
    forward pass, so hub rows stop being independently-trained free
    parameters and gradient reaches the 4d rows by construction.

    `on_epoch` (v7 §4.2, default None): called at the end of every epoch as
    `on_epoch(epoch, Z_eff, {target_id: raw_loss})` — lets a caller track
    per-family loss trajectories against an external reference metric
    without this function needing to know anything about what that metric
    is (see train_shared_cosine_embedding.py's `--eval-every`).
    """
    params = [Z] + [p for model in observation_models.values() for p in model.parameters()]
    optimizer = torch.optim.Adam(params, lr=lr)

    domain_targets, domain_active = restrict_to_domain(operator_data)

    pretraining_diagnostics(Z, observation_models, operator_data, domain_targets, domain_active,
                             hub_aggregation=hub_aggregation)

    print(f"\nTraining {epochs} epochs …")
    if dynamic_reweight:
        print(f"  dynamic_reweight=True (decay={reweight_decay}, max_ratio={reweight_max_ratio})")
    best_loss = float("inf")
    running_loss: dict[str, float] = {}
    running_loss_epoch1: dict[str, float] = {}
    stall_baseline: dict[str, float] = {}
    stalled: set[str] = set()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        total_loss = torch.zeros((), device=Z.device)
        per_layer_losses: dict[str, torch.Tensor] = {}
        per_layer_weights: dict[str, float] = {}
        per_layer_hinged: dict[str, float] = {}

        # Computed once per epoch, not once per target — hub_aggregation is
        # identical across every layer (v7 §1.2).
        Z_eff = hub_aggregation @ Z if hub_aggregation is not None else Z

        for lname in operator_data.targets:
            dmask = operator_data.domains[lname]
            logits = observation_models[lname](Z_eff, dmask)
            l = layer_loss(operator_data.loss_types[lname], logits, domain_targets[lname], domain_active[lname])
            per_layer_losses[lname] = l
            raw = l.item()

            weight = operator_data.weights[lname]
            if dynamic_reweight:
                if lname not in running_loss:
                    running_loss[lname] = raw
                    running_loss_epoch1[lname] = raw
                    stall_baseline[lname] = raw
                else:
                    running_loss[lname] = ema_update(running_loss[lname], raw, reweight_decay)
                raw_ratio = dynamic_reweight_ratio(running_loss_epoch1[lname], running_loss[lname], reweight_max_ratio)
                ratio, is_stalled = stall_capped_ratio(
                    raw_ratio, running_loss[lname], stall_baseline[lname],
                    reweight_stall_ratio, reweight_stall_min_improvement,
                )
                if not is_stalled:
                    stall_baseline[lname] = running_loss[lname]
                elif lname not in stalled:
                    print(f"  WARNING: {lname} stalled at ~{reweight_stall_ratio:.1f}x boost without "
                          f"{reweight_stall_min_improvement:.0%} loss improvement (loss={running_loss[lname]:.4f}, "
                          f"baseline={stall_baseline[lname]:.4f}) — capping further reweighting, "
                          f"flagging as structurally stalled, not just hard.")
                    stalled.add(lname)
                weight = weight * ratio
            per_layer_weights[lname] = weight

            margin = operator_data.margins.get(lname, 0.0)
            hinged = hinge_loss(l, margin)
            per_layer_hinged[lname] = hinged.item()
            total_loss = total_loss + weight * hinged

        total_loss.backward()
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(f"  epoch {epoch:4d}/{epochs}  total={total_loss.item():.5f}")
            for lname, l in per_layer_losses.items():
                lam = per_layer_weights[lname]
                hinged = per_layer_hinged[lname]
                print(f"    {lname:<12} raw={l.item():.5f}  lambda={lam:.3f}  "
                      f"hinged={hinged:.5f}  contribution={lam * hinged:.5f}")

        if on_epoch is not None:
            on_epoch(epoch, Z_eff.detach(), {lname: l.item() for lname, l in per_layer_losses.items()})

    return best_loss
