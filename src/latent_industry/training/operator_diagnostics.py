"""
operator_diagnostics.py — training-state checkpoint and reconstruction diagnostics.

- save_training_state / load_training_state: Z plus every target's
  observation model (operator_state.pt in the run directory).
- layer_reconstruction_report: per-row Spearman rho, cosine similarity and
  top-k overlap between each target and the model's predicted distribution.
- per_family_reference_metric: one cheap reference metric per family,
  computed from Z and the relatedness files (Spearman rho against labor /
  IO relatedness, hierarchy tree-distance rho, or crosswalk MRR for
  transition); used during training with --eval-every.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from latent_industry.common.paths import METADATA_DIR, PROCESSED_DATA_DIR
from latent_industry.graph.layer_diffusion import build_hierarchy_ordinal_tiers
from latent_industry.graph.layers import CROSSWALK_CSVS, _crosswalk_pairs
from latent_industry.graph.nodes import NodeIndex
from latent_industry.training.operator_losses import layer_loss
from latent_industry.training.operator_observation_models import OBSERVATION_MODEL_REGISTRY, LayerObservationModel
from latent_industry.training.operator_trainer import restrict_to_domain

DEFAULT_TOP_KS = (5, 10, 20)

_KIND_BY_CLASS = {cls: kind for kind, cls in OBSERVATION_MODEL_REGISTRY.items()}


def save_training_state(Z: torch.Tensor, observation_models: dict[str, LayerObservationModel],
                         embedding_dim: int, out_path: Path) -> None:
    """
    Checkpoint of Z and every target's observation model (operator_state.pt).
    Not part of the exported embeddings.
    """
    state = {
        "embedding_dim": embedding_dim,
        "Z": Z.detach().cpu(),
        "model_kinds": {lname: _KIND_BY_CLASS[type(model)] for lname, model in observation_models.items()},
        "model_state_dicts": {
            lname: {k: v.detach().cpu() for k, v in model.state_dict().items()}
            for lname, model in observation_models.items()
        },
    }
    torch.save(state, out_path)


def load_training_state(path: Path) -> tuple[torch.Tensor, dict[str, LayerObservationModel]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — train_operator_embedding.py saves this alongside the exported "
            f"embeddings; re-run it if this is an older run predating operator_state.pt."
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    dim = state["embedding_dim"]
    Z = state["Z"]
    observation_models = {}
    for lname, kind in state["model_kinds"].items():
        model = OBSERVATION_MODEL_REGISTRY[kind](dim)
        model.load_state_dict(state["model_state_dicts"][lname])
        observation_models[lname] = model
    return Z, observation_models


def _row_reconstruction_metrics(P: np.ndarray, Q: np.ndarray, top_ks: tuple[int, ...]) -> dict[str, np.ndarray]:
    """P, Q: (n_active, n_domain) — true target rows and predicted (softmax)
    rows, same active subset, same domain-restricted/diagonal-excluded
    shape `restrict_to_domain` produces. Per active row: Spearman rho,
    cosine similarity, and top-k neighbour overlap (fraction of the true
    top-k destinations also in the predicted top-k)."""
    n = P.shape[0]
    row_spearman = np.full(n, np.nan)
    row_cosine = np.zeros(n)
    row_topk = {k: np.zeros(n) for k in top_ks}

    for i in range(n):
        p, q = P[i], Q[i]
        if np.ptp(p) > 1e-12 and np.ptp(q) > 1e-12:
            row_spearman[i] = spearmanr(p, q).statistic

        denom = np.linalg.norm(p) * np.linalg.norm(q)
        row_cosine[i] = float(p @ q / denom) if denom > 0 else 0.0

        for k in top_ks:
            k_eff = min(k, p.shape[0])
            true_top = set(np.argpartition(-p, k_eff - 1)[:k_eff].tolist())
            pred_top = set(np.argpartition(-q, k_eff - 1)[:k_eff].tolist())
            row_topk[k][i] = len(true_top & pred_top) / k_eff

    return {"row_spearman": row_spearman, "row_cosine": row_cosine, "row_topk_overlap": row_topk}


def layer_reconstruction_report(
    lname: str, Z: torch.Tensor, model: torch.nn.Module, domain_mask: torch.Tensor,
    target: torch.Tensor, active_mask: torch.Tensor, loss_kind: str,
    top_ks: tuple[int, ...] = DEFAULT_TOP_KS,
) -> dict:
    """`target`/`active_mask` must already be domain-restricted (see
    operator_trainer.restrict_to_domain) — same shape/normalisation the
    layer was actually trained against, not a second reimplementation."""
    with torch.no_grad():
        logits = model(Z, domain_mask)
        Q = torch.softmax(logits, dim=-1)
        scalar_loss = layer_loss(loss_kind, logits, target, active_mask).item()

        P_np = target[active_mask].cpu().numpy()
        Q_np = Q[active_mask].cpu().numpy()

    row_metrics = _row_reconstruction_metrics(P_np, Q_np, top_ks)

    return {
        "layer": lname,
        "loss_kind": loss_kind,
        "n_domain": int(domain_mask.sum()),
        "n_active": int(P_np.shape[0]),
        "scalar_loss": scalar_loss,
        **row_metrics,
    }


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
    code-prefix tiers; used for both the "hierarchy" and "nace_text" families.
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
    if family in ("hierarchy", "nace_text"):
        return _hierarchy_reference_metric(Z, ni)
    if family == "transition":
        return _transition_reference_metric(Z, ni)
    return None
