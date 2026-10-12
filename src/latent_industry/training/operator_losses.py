"""
operator_losses.py — losses between a target and the observation model's logits.

kl_loss and cross_entropy_loss share one computation (_categorical_nll) and
differ only in whether the target entropy H(p) is added back to the reported
value; their gradients are identical. Both are categorical-likelihood
observation models. ordinal_margin_loss enforces the hierarchy tier ordering.

Targets and active masks arrive domain-restricted, with the diagonal
excluded and rows renormalised over j ≠ i (operator_trainer.restrict_to_domain);
ordinal targets are tier-rank matrices and are not renormalised.
"""

import torch


def _categorical_nll(logits: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """
    mean over active rows i of −Σ_j target(i,j) · log_softmax(logits)(i,j).
    The logits' diagonal is −inf and the target's diagonal is 0; log-probabilities
    are set to 0 where the logit is −inf, so 0 · log 0 contributes 0 rather than NaN.
    """
    log_Q = torch.log_softmax(logits, dim=-1)
    log_Q = torch.where(torch.isneginf(logits), torch.zeros_like(log_Q), log_Q)
    ce_rows = -(target * log_Q).sum(dim=-1)
    return ce_rows[active_mask].mean()


def kl_loss(logits: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    ce = _categorical_nll(logits, target, active_mask)
    p = target[active_mask]
    entropy = -(p * torch.log(p.clamp(min=1e-12))).sum(dim=-1).mean()
    return ce - entropy


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    return _categorical_nll(logits, target, active_mask)


_ORDINAL_TIERS = (1, 2, 3, 4)


def ordinal_margin_loss(
    logits: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor, margin: float = 0.05,
) -> torch.Tensor:
    """
    Ordinal hierarchy constraint among 4-digit codes:
    cos(i, same 3-digit group) > cos(i, same division) > cos(i, same section)
    > cos(i, other section).

    `target` is a tier-rank matrix with values in {0, 1, 2, 3, 4} (0 = self /
    undefined; 1–4 = _ORDINAL_TIERS from nearest to farthest). `logits` are used
    directly as similarity scores (cos / tau), not log-softmaxed.

    Per active row: mean similarity within each tier present, then
    relu(margin + mean(tier_{k+1}) − mean(tier_k)) for every pair of adjacent
    tiers that are both present, averaged over pairs and then over active rows.
    Rows with fewer than two tiers contribute zero.
    """
    means, counts = [], []
    for t in _ORDINAL_TIERS:
        mask_t = target == t
        cnt = mask_t.sum(dim=-1).to(logits.dtype)
        s = torch.where(mask_t, logits, torch.zeros_like(logits)).sum(dim=-1)
        means.append(s / cnt.clamp(min=1))
        counts.append(cnt)
    means = torch.stack(means, dim=0)    # (len(_ORDINAL_TIERS), n)
    counts = torch.stack(counts, dim=0)  # (len(_ORDINAL_TIERS), n)

    row_loss_sum = torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
    row_pair_count = torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
    for k in range(len(_ORDINAL_TIERS) - 1):
        both_present = (counts[k] > 0) & (counts[k + 1] > 0)
        gap = torch.relu(margin + means[k + 1] - means[k])
        row_loss_sum = row_loss_sum + torch.where(both_present, gap, torch.zeros_like(gap))
        row_pair_count = row_pair_count + both_present.to(logits.dtype)

    row_loss = row_loss_sum / row_pair_count.clamp(min=1)
    return row_loss[active_mask].mean()


_DISPATCH = {"kl": kl_loss, "ce": cross_entropy_loss, "ordinal": ordinal_margin_loss}


def layer_loss(kind: str, logits: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    if kind not in _DISPATCH:
        raise ValueError(f"unknown loss kind {kind!r} — expected one of {sorted(_DISPATCH)}")
    return _DISPATCH[kind](logits, target, active_mask)
