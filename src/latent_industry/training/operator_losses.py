"""operator_losses.py — Phase 4 of the operator-preserving regime (math doc
§5). Two families, not three: `kl_loss`/`cross_entropy_loss` share one
internal computation (`_categorical_nll`) and differ only in whether the
target's own entropy H(p) is added back for the reported value — since
`H(p)` is a constant w.r.t. the parameters, ∇_θ KL(p‖q) = ∇_θ CE(p,q)
exactly, not just approximately. `mse_loss` is the one genuinely different
code path (needs Q = softmax(logits) explicitly).

Statistical framing (math doc §5): `kl`/`ce` are both the **categorical-
likelihood** observation model (labor, nace_text, io, hierarchy,
transition); `mse` is the **Gaussian-likelihood** alternative (io only,
behind a config flag) — a statement about what generative process each
economic mechanism is assumed to look like, not an ad hoc menu of loss
functions.

`target` and `active_mask` passed in here are already domain-restricted
(n_domain x n_domain / n_domain) with the diagonal excluded and the row
renormalised over j != i — see operator_trainer.py's
`restrict_to_domain`, which does that once per layer rather than every
epoch. `ordinal_margin_loss` (v7 §1.4) is the exception: its `target` is a
tier-rank matrix, not a probability distribution, so `restrict_to_domain`
skips the renormalisation step for it — see that function's docstring.
"""

import torch
import torch.nn.functional as F


def _categorical_nll(logits: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """mean_{i active} -sum_j target(i,j) * log_softmax(logits)(i,j).
    `logits`'s diagonal is -inf (self excluded, see BilinearObservationModel);
    `target`'s diagonal is exactly 0 for the same reason, so the two
    zero/-inf positions must line up — otherwise 0 * log(0) would evaluate
    to NaN under IEEE float rules even though the intended contribution is
    zero. `log_Q` is patched to 0 wherever the logit was masked to -inf so
    that 0 (target) * 0 (patched log-prob) stays exactly 0.
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


def mse_loss(logits: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    Q = torch.softmax(logits, dim=-1)
    return F.mse_loss(Q[active_mask], target[active_mask])


_ORDINAL_TIERS = (1, 2, 3, 4)


def ordinal_margin_loss(
    logits: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor, margin: float = 0.05,
) -> torch.Tensor:
    """v7 §1.4: `cos(i, same_3d) > cos(i, same_2d) > cos(i, same_section) >
    cos(i, other)` — an ordinal constraint on 4d-4d hierarchy relations,
    replacing `hierarchy_t1`'s old categorical (softmax-over-domain) target,
    which stops carrying information once §1.2 makes a hub row a function of
    its own children (a 4d code is then part of its own hub's definition by
    construction). This is what `A_treedist_rho` already measures, so target
    and metric finally align.

    `target` here is NOT a probability distribution — it's a tier-rank
    matrix, values in `{0, 1, 2, 3, 4}` (`0` = self/undefined, off-diagonal
    values `1..4` are `_ORDINAL_TIERS`, nearest to farthest). `logits` are
    used directly as raw similarity scores (`cos/tau`, already computed by
    `CosineObservationModel`) — NOT log-softmaxed, since this isn't a
    distribution-matching loss; the `-inf` diagonal `logits` never enters
    any tier's mean because the diagonal's `target` value (`0`) is never one
    of `_ORDINAL_TIERS`.

    Per active row: mean similarity within each tier present in that row,
    then a hinge on every pair of ADJACENT tiers both present —
    `relu(margin + mean(tier_{k+1}) - mean(tier_k))` for `k` in
    `(same_3d, same_2d), (same_2d, same_section), (same_section, other)` —
    averaged over the (up to 3) present adjacent pairs, then over active
    rows. A row with fewer than two tiers represented (no adjacent pair to
    compare) contributes exactly zero, not a crash."""
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


_DISPATCH = {"kl": kl_loss, "ce": cross_entropy_loss, "mse": mse_loss, "ordinal": ordinal_margin_loss}


def layer_loss(kind: str, logits: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    if kind not in _DISPATCH:
        raise ValueError(f"unknown loss kind {kind!r} — expected one of {sorted(_DISPATCH)}")
    return _DISPATCH[kind](logits, target, active_mask)
