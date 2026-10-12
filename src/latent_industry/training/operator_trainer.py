"""
operator_trainer.py — joint full-batch training: in every epoch all targets'
losses are computed in one forward pass, combined as L = Σ_t w_t L_t, and a
single optimizer step updates Z and all observation-model parameters.
"""

from typing import Callable

import torch

from latent_industry.training.operator_data import OperatorData
from latent_industry.training.operator_losses import layer_loss


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp(min=1e-12))).sum(dim=-1)


def hinge_loss(loss: torch.Tensor, margin: float) -> torch.Tensor:
    """
    max(loss − margin, 0): once a target's loss is below its margin (a fraction
    of its null loss, see operator_data.load_operator_data_expanded), it
    contributes neither loss nor gradient. margin = 0 is a no-op.
    """
    return torch.clamp(loss - margin, min=0.0)


def dynamic_reweight_ratio(running_loss_epoch1: float, running_loss_now: float, max_ratio: float) -> float:
    """
    Factor by which a target's static weight is rescaled so that its
    contribution (weight × raw loss) stays near its epoch-1 value as the loss
    falls: running_loss_initial / running_loss_now, clamped to max_ratio.
    """
    return min(running_loss_epoch1 / max(running_loss_now, 1e-12), max_ratio)


def ema_update(previous: float, new_value: float, decay: float) -> float:
    """
    Exponential moving average, the running-loss estimate used by
    dynamic_reweight_ratio.
    """
    return decay * previous + (1 - decay) * new_value


def stall_capped_ratio(
    raw_ratio: float, running_loss_now: float, baseline_loss: float,
    stall_ratio: float, stall_min_improvement: float,
) -> tuple[float, bool]:
    """
    Cap for dynamic_reweight_ratio that separates hard targets from stalled
    ones. Once the raw ratio would exceed `stall_ratio`, the loss must have
    improved by at least `stall_min_improvement` (relative) since the ratio was
    last below `stall_ratio` (`baseline_loss`, tracked by the caller); otherwise
    the ratio is capped at `stall_ratio` and True is returned so the caller can
    flag the target as stalled.
    """
    if raw_ratio < stall_ratio:
        return raw_ratio, False
    improved_enough = running_loss_now <= baseline_loss * (1.0 - stall_min_improvement)
    if improved_enough:
        return raw_ratio, False
    return min(raw_ratio, stall_ratio), True


def restrict_to_domain(operator_data: OperatorData) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Restrict each target and active mask to the target's domain (once, as
    domains are static) and bring the target into the form the losses expect:
    diagonal excluded (matching the observation model's −inf self-logit) and
    each row renormalised over j ≠ i. All-zero rows stay zero and are excluded
    by the active mask.

    "ordinal" targets are tier-rank matrices and are not renormalised.
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
    """
    One forward (and isolated backward) pass before training: logs domain size,
    target entropy, initial loss and ||∇_Z L_t|| per target, to show how
    influential each target is at the start.

    With `hub_aggregation`, observation models see hub_aggregation @ Z.
    Entropy is not reported for "ordinal" targets.
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
    dynamic_reweight: bool = True,
    reweight_decay: float = 0.9,
    reweight_max_ratio: float = 20.0,
    reweight_stall_ratio: float = 2.0,
    reweight_stall_min_improvement: float = 0.1,
    hub_aggregation: torch.Tensor | None = None,
    on_epoch: Callable[[int, torch.Tensor, dict[str, float]], None] | None = None,
) -> None:
    """
    Full-batch joint training loop; updates Z and the observation models in place.

    Options
    -------
    dynamic_reweight
        Rescale each target's weight by loss(epoch 1) / running_loss(epoch)
        (running_loss an EMA, the ratio clamped to reweight_max_ratio), so that every
        target's contribution stays near its initial share instead of fading as
        it is fitted. Used together with the margins in operator_data.margins,
        which switch a target off once it falls below its floor.
    reweight_stall_ratio, reweight_stall_min_improvement
        Above reweight_stall_ratio, the ratio may only keep growing while the
        loss improves by at least reweight_stall_min_improvement; otherwise it is
        capped and the target is flagged as stalled (stall_capped_ratio).
    hub_aggregation
        (N, N) matrix with hub_aggregation @ Z = Z on 4d rows and the mean of the
        4d descendants on hub rows (operator_embedding.build_hub_aggregation_matrix).
        Applied once per epoch before the observation models, so hub rows are
        functions of their descendants rather than free parameters.
    on_epoch
        Callback on_epoch(epoch, Z_eff, {target_id: raw_loss}) at the end of each
        epoch (used by --eval-every).
    """
    params = [Z] + [p for model in observation_models.values() for p in model.parameters()]
    optimizer = torch.optim.Adam(params, lr=lr)

    domain_targets, domain_active = restrict_to_domain(operator_data)

    pretraining_diagnostics(Z, observation_models, operator_data, domain_targets, domain_active,
                             hub_aggregation=hub_aggregation)

    print(f"\nTraining {epochs} epochs …")
    if dynamic_reweight:
        print(f"  dynamic_reweight=True (decay={reweight_decay}, max_ratio={reweight_max_ratio})")
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
        # identical across every target.
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

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(f"  epoch {epoch:4d}/{epochs}  total={total_loss.item():.5f}")
            for lname, l in per_layer_losses.items():
                lam = per_layer_weights[lname]
                hinged = per_layer_hinged[lname]
                print(f"    {lname:<12} raw={l.item():.5f}  lambda={lam:.3f}  "
                      f"hinged={hinged:.5f}  contribution={lam * hinged:.5f}")

        if on_epoch is not None:
            on_epoch(epoch, Z_eff.detach(), {lname: l.item() for lname, l in per_layer_losses.items()})
