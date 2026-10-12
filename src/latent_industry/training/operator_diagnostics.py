"""
operator_diagnostics.py — reference metrics tracked during training.

per_family_reference_metric gives one cheap reference metric per family,
computed from Z and the relatedness files (Spearman rho against labor / IO
relatedness, hierarchy tree-distance rho, or crosswalk MRR for transition);
used during training with --eval-every.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from latent_industry.common.paths import PROCESSED_DATA_DIR
from latent_industry.graph.layer_diffusion import build_hierarchy_ordinal_tiers
from latent_industry.graph.layers import CROSSWALK_CSVS, _crosswalk_pairs
from latent_industry.graph.nodes import NodeIndex


# ── Per-family loss <-> reference-metric correlation ──────────────────────
#
# train_shared_cosine_embedding.py --eval-every calls
# per_family_reference_metric periodically and correlates it with the
# family's training loss: the two should move in opposite directions; a
# non-negative correlation means the loss improves without the metric.
# The metrics read the relatedness sources directly (no text encoder), so
# they are cheap enough to compute during training.

def _labor_reference_metric(Z: torch.Tensor, ni: NodeIndex) -> float | None:
    """
    Spearman rho between cosine similarity and labor relatedness, per version
    (pairs evaluated in one batched tensor operation).
    """
    path = PROCESSED_DATA_DIR / "labor_relatedness.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    Z_norm = F.normalize(Z.detach(), p=2, dim=-1)
    rhos = []
    for ver in ("SNI07", "SNI02", "SNI92"):
        sub = df[df["coding"] == ver]
        s_ids, t_ids, phi = [], [], []
        for row in sub.itertuples():
            s = ni.get("4d", ver, str(row.sni_from))
            t = ni.get("4d", ver, str(row.sni_to))
            if s is None or t is None:
                continue
            s_ids.append(s); t_ids.append(t); phi.append(float(row.phi))
        if len(phi) < 10:
            continue
        s_idx = torch.tensor(s_ids, device=Z_norm.device)
        t_idx = torch.tensor(t_ids, device=Z_norm.device)
        sim = (Z_norm[s_idx] * Z_norm[t_idx]).sum(dim=-1).cpu().numpy()
        rho, _ = spearmanr(sim, phi)
        rhos.append(rho)
    return float(np.mean(rhos)) if rhos else None


def _io_reference_metric(Z: torch.Tensor, ni: NodeIndex) -> float | None:
    """
    Spearman rho between cosine similarity of the 2-digit hub embeddings and IO
    relatedness (one batched tensor operation).
    """
    path = PROCESSED_DATA_DIR / "io_relatedness_SNI07.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    Z_norm = F.normalize(Z.detach(), p=2, dim=-1)
    a_ids, b_ids, gt = [], [], []
    for row in df.itertuples():
        a, b = str(row.nace2d_from).zfill(2), str(row.nace2d_to).zfill(2)
        if a == b:
            continue
        ia, ib = ni.get("2d", "SNI07", a), ni.get("2d", "SNI07", b)
        if ia is None or ib is None:
            continue
        a_ids.append(ia); b_ids.append(ib); gt.append(float(row.sim))
    if len(gt) < 10:
        return None
    a_idx = torch.tensor(a_ids, device=Z_norm.device)
    b_idx = torch.tensor(b_ids, device=Z_norm.device)
    sim = (Z_norm[a_idx] * Z_norm[b_idx]).sum(dim=-1).cpu().numpy()
    rho, _ = spearmanr(sim, gt)
    return float(rho)


def _hierarchy_reference_metric(Z: torch.Tensor, ni: NodeIndex) -> float | None:
    """
    Spearman rho between cosine distance and hierarchy tree distance from
    code-prefix tiers.
    """
    tier, domain = build_hierarchy_ordinal_tiers(ni)
    idx = np.nonzero(domain)[0]
    if len(idx) < 2:
        return None
    Z_norm = F.normalize(Z.detach()[idx], p=2, dim=-1)
    sim = (Z_norm @ Z_norm.T).cpu().numpy()
    tier_sub = tier[np.ix_(idx, idx)]
    mask = tier_sub > 0
    if int(mask.sum()) < 10:
        return None
    rho, _ = spearmanr(1.0 - sim[mask], tier_sub[mask])
    return float(rho)


def _transition_reference_metric(Z: torch.Tensor, ni: NodeIndex) -> float | None:
    """
    Crosswalk MRR for SNI92 ↔ SNI02 and SNI02 ↔ SNI07, averaged (one batched
    similarity computation per pair).
    """
    Z_norm = F.normalize(Z.detach(), p=2, dim=-1)
    mrrs = []
    for csv_name, old_ver, new_ver, as_int in CROSSWALK_CSVS:
        old4, new4 = _crosswalk_pairs(csv_name, old_ver, new_ver, as_int)
        true_targets: dict[int, set[int]] = {}
        for o, n in zip(old4, new4):
            r, c = ni.get("4d", old_ver, o), ni.get("4d", new_ver, n)
            if r is not None and c is not None:
                true_targets.setdefault(r, set()).add(c)
        new_ids = [ni.get("4d", new_ver, c) for c in ni.sni_4d[new_ver]]
        new_ids = [i for i in new_ids if i is not None]
        if not true_targets or not new_ids:
            continue
        src_ids = list(true_targets.keys())
        V_new = Z_norm[new_ids]
        S = (Z_norm[src_ids] @ V_new.T).cpu().numpy()
        pair_mrr = []
        for row_i, src_id in enumerate(src_ids):
            tgt_ids = true_targets[src_id]
            ranked = [new_ids[j] for j in np.argsort(-S[row_i])]
            for rank, cand in enumerate(ranked, start=1):
                if cand in tgt_ids:
                    pair_mrr.append(1.0 / rank)
                    break
            else:
                pair_mrr.append(0.0)
        if pair_mrr:
            mrrs.append(float(np.mean(pair_mrr)))
    return float(np.mean(mrrs)) if mrrs else None


def per_family_reference_metric(Z: torch.Tensor, ni: NodeIndex, family: str) -> float | None:
    """One cheap, family-specific correlation/retrieval number computed
    directly from `Z` and already-on-disk relatedness data. Returns `None`
    if the family's static source file/domain isn't available or the
    family name isn't recognised — best-effort, never fatal (a missing
    metric should not crash training)."""
    if family == "labor":
        return _labor_reference_metric(Z, ni)
    if family == "io":
        return _io_reference_metric(Z, ni)
    if family == "hierarchy":
        return _hierarchy_reference_metric(Z, ni)
    if family == "transition":
        return _transition_reference_metric(Z, ni)
    return None
