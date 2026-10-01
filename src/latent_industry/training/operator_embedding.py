"""operator_embedding.py — Phase 5/6 glue. Retain `Z` only, discard every
`M_l`. Export L2-normalised rows in the exact `{sni07, sni02, sni92}` split
and `.npz` schema `sentence_embedding.embed_and_save` already produces
(math doc §6) — same file names too (`embeddings_sni07.npz` etc.), so this
is a byte-for-byte-schema drop-in for either regime. (`_shared_eval_context.py`'s
(the P02/P03 eval scripts' shared setup module, formerly part of
`evaluate_embeddings.py`) `load_or_encode` looks up "sni07_4d" internally
but falls back to this plain "sni07" name — see its own docstring for why;
that fallback is what lets both regimes' caches be found without this file
needing to know that module's internal label convention.)

Also exports the SNI07 hierarchy-hub levels (3d/2d/section), under the
exact labels P02_shared_geometry_reconstruction.py looks up for them
("sni07_3d"/"sni07_2d"/"sni07_section" — no legacy alias needed, the
sentence-transformer regime never caches these at all).

v7 §1.2: hub rows are no longer free parameters read straight out of `Z` —
`build_hub_aggregation_matrix` makes each hub row a FUNCTION of its 4d-SNI07
descendants (`h_a = normalize(mean_{i in D(a)} z_i)`), computed here (and,
during training, in operator_trainer.train_joint) rather than stored. This
is what gives hub nodes real gradient by construction (see the design doc
§4 and v7 §1.2) instead of relying on `hierarchy`/`nace_text` targets to
pull on them; without this, P02_shared_geometry_reconstruction.py would either fall back to
encoding hub text with a plain sentence-transformer, or (post-fix, pre-this
change) export a hub row that was still just an independently-trained free
parameter, decoupled from the 4d rows it's supposed to summarise.
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
    """M, N x N: `M @ Z` leaves every 4d row exactly as it was (M is the
    identity there) and turns every hub row (3d/2d/section) into the FLAT
    mean over its 4d descendants of the SAME version — `u_a = mean_{i in
    D(a)} z_i`, not yet normalised (CosineObservationModel's own
    `F.normalize` call handles that uniformly for leaf and hub rows alike,
    which is also where the epsilon-stabilised denominator the design note
    asks for comes from, for free).

    Flat aggregation over 4d descendants, NOT recursive over immediate
    children (v7 §1.2 design note): a code in a 2-member 3d group would
    otherwise count several times more toward its division's mean than one
    in a 10-member group.

    Version-generic, not hard-coded to SNI07: it looks up each hub's
    descendants via `ni.get(...)` for whatever version that hub row actually
    is. Only SNI07 hub nodes exist in `NodeIndex` today (extending
    nace_text/hierarchy to SNI92/SNI02 hub levels is v7 §2, explicitly
    deferred), so every non-SNI07 lookup here simply finds nothing and that
    hub type/version combination never arises as a row to fill — this
    function does not need to change when that coverage gap is eventually
    closed.
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
    """v7 §4.5: log ‖u_a‖ — the PRE-normalisation mean vector's norm — per
    hub. Normalised away by the aggregation above, but interpretable on its
    own: how internally coherent a division/group/section is in the learned
    space, independent of the loss."""
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
