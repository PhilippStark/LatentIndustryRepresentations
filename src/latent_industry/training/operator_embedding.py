"""
operator_embedding.py — hub aggregation and export of the trained embedding Z.

- build_hub_aggregation_matrix: hub rows (3d/2d/section) are the mean of
  their 4-digit descendants, h_a = normalise(mean_{i ∈ D(a)} z_i), so hubs
  are functions of the 4-digit codes rather than free parameters.
- export_embeddings: L2-normalised 4-digit embeddings per version
  (embeddings_sni07.npz, embeddings_sni02.npz, embeddings_sni92.npz) and the
  SNI07 hub embeddings (embeddings_sni07_{3d,2d,section}.npz), each with
  keys `embeddings` and `codes`.
- export_hub_coherence: per-hub norm of the pre-normalisation mean vector.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from latent_industry.graph.nodes import NodeIndex

_TARGET_LABELS = [("4d", "SNI07", "sni07"), ("4d", "SNI02", "sni02"), ("4d", "SNI92", "sni92")]
_HUB_LABELS = [("3d", "SNI07", "sni07_3d"), ("2d", "SNI07", "sni07_2d"), ("section", "SNI07", "sni07_section")]
_HUB_NODE_TYPES = ("3d", "2d", "section")


def build_hub_aggregation_matrix(ni: NodeIndex) -> torch.Tensor:
    """
    M (N x N): M @ Z leaves every 4d row unchanged (identity) and turns every
    hub row (3d/2d/section) into the mean of its 4d descendants of the same
    version, u_a = mean_{i ∈ D(a)} z_i (normalised later by
    CosineObservationModel).

    Aggregation is flat over 4d descendants, not recursive over children, so a
    code in a small 3-digit group does not weigh more in its division's mean
    than a code in a large group. Descendants are looked up per hub version.
    """
    nodes = ni.nodes
    node_type = nodes["node_type"].values
    version = nodes["version"].values
    code = nodes["code"].values

    M = torch.zeros(ni.N, ni.N, dtype=torch.float32)
    for nid in range(ni.N):
        if node_type[nid] == "4d":
            M[nid, nid] = 1.0

    descendants: dict[int, list[int]] = {
        nid: [] for nid in range(ni.N) if node_type[nid] in _HUB_NODE_TYPES
    }
    for nid in range(ni.N):
        if node_type[nid] != "4d":
            continue
        ver, c = version[nid], code[nid]
        hub_3d = ni.get("3d", ver, c[:3])
        if hub_3d in descendants:
            descendants[hub_3d].append(nid)
        hub_2d = ni.get("2d", ver, c[:2])
        if hub_2d in descendants:
            descendants[hub_2d].append(nid)
        section_letter = ni.sections.get(ver, {}).get(c[:2])
        hub_sec = ni.get("section", ver, section_letter) if section_letter is not None else None
        if hub_sec in descendants:
            descendants[hub_sec].append(nid)

    for hub_id, desc_ids in descendants.items():
        if not desc_ids:
            continue  # no known descendants for this hub — row stays zero (see module docstring)
        w = 1.0 / len(desc_ids)
        for i in desc_ids:
            M[hub_id, i] = w
    return M


def export_hub_coherence(Z: torch.Tensor, ni: NodeIndex, hub_aggregation: torch.Tensor, out_path: Path) -> None:
    """
    Per hub, ||u_a||: the norm of the mean of its descendants' unit vectors,
    a measure of how internally coherent a section, division or group is in
    the learned space. Written to hub_coherence.csv.
    """
    nodes = ni.nodes
    node_type = nodes["node_type"].values
    with torch.no_grad():
        U = (hub_aggregation.to(Z.device) @ Z.detach()).cpu().numpy()
    norms = np.linalg.norm(U, axis=1)

    rows = [
        {"code": nodes.loc[nid, "code"], "node_type": nodes.loc[nid, "node_type"],
         "version": nodes.loc[nid, "version"], "norm": float(norms[nid])}
        for nid in range(ni.N) if node_type[nid] in _HUB_NODE_TYPES
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  hub coherence (‖u_a‖) -> {out_path}")


def export_embeddings(Z: torch.Tensor, ni: NodeIndex, out_dir: Path,
                       hub_aggregation: torch.Tensor | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if hub_aggregation is None:
        hub_aggregation = build_hub_aggregation_matrix(ni)
    Z_eff = hub_aggregation.to(Z.device) @ Z.detach()
    Z_norm = F.normalize(Z_eff, p=2, dim=-1).cpu().numpy()
    nodes = ni.nodes

    for node_type, version, label in _TARGET_LABELS + _HUB_LABELS:
        mask = (nodes["node_type"].values == node_type) & (nodes["version"].values == version)
        node_ids = nodes.index.values[mask]
        codes = nodes.loc[node_ids, "code"].values
        embs = Z_norm[node_ids].astype(np.float32)
        file_path = out_dir / f"embeddings_{label}.npz"
        np.savez_compressed(file_path, embeddings=embs, codes=np.array(codes))
        print(f"  {label}: {embs.shape} → {file_path.name}")
