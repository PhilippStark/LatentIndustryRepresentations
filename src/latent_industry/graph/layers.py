"""layers.py — one row-stochastic transition matrix per independent evidence
family. Each builder does exactly one thing: turn its own raw source into
`transitions.row_normalize`'s input and call it. No relation ever competes
against another relation here — that only happens later, in the supra-network
(see multilayer_diffusion.py). No top-k, no magnitude threshold beyond
clipping negative similarity to zero (a Markov chain can't carry negative
transition mass) — see repulsion.py for where the negative NACE evidence and
"officially excludes" pairs go instead.

Six layers, one per independent evidence family (not one per NACE
hierarchy level — those are correlated transformations of the same family,
so nace_text spans all four levels in a single block-diagonal matrix):

  labor       4d<->4d Neffke-Henning skill relatedness, within each SNI
              version separately (SNI07/02/92 never mix directly here).
  nace_text   cosine similarity between NACE description embeddings, at
              every level (4d/3d/2d/section) in one matrix. Levels never
              cross-connect, so this is block-diagonal by construction.
  io          flow-relatedness between 2d hub nodes (blended SCB/Eurostat/
              WIOD input-output tables).
  hierarchy   4d<->3d<->2d<->section parent/child, both directions combined.
  transition  SNI92<->02<->07 crosswalk, both directions combined. A
              zero-cost identity correspondence ("this old code IS that new
              code"), not a relatedness score.
  ownership   4d<->4d common-ownership (same OrgNr) co-occurrence, within
              each SNI version separately (2026-07-29).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from latent_industry.common.paths import METADATA_DIR, PROCESSED_DATA_DIR
from latent_industry.graph.nodes import NodeIndex
from latent_industry.graph.transitions import row_normalize


@dataclass
class Layer:
    name: str
    P: np.ndarray         # N x N row-stochastic transition matrix
    avail: np.ndarray     # N bool — which rows have outgoing mass in this layer *this run*
    domain: np.ndarray    # N bool — which nodes structurally belong to this layer, regardless
                           # of whether they currently have any edges (used by
                           # multilayer_diffusion.py to decide the supra-network's state space
                           # and how many layers a node's interlayer budget splits across —
                           # this is what replaces the old hand-maintained RELATION_NODE_MENU)
    edges: pd.DataFrame   # raw, pre-normalisation: src, dst, weight, type


@dataclass
class NaceEmbeddingContext:
    """Carries the loaded/centered NACE embeddings and the aligned 4d-level
    slice forward to repulsion.py, so it doesn't need to reload or re-align
    them itself."""
    raw: dict
    nids_4d07: list[int]
    codes_4d07: list[str]
    vecs_4d07: np.ndarray
    negative_edges: pd.DataFrame  # diagnostic-only max(-s,0) part, every level


def _mask_4d_all(ni: NodeIndex) -> np.ndarray:
    """Every 4d node, any version (SNI07/02/92) — the domain for labor and
    transition, the two layers that touch pre-2007 codes directly."""
    return (ni.nodes["node_type"].values == "4d")


def _mask_hierarchy_all_versions(ni: NodeIndex) -> np.ndarray:
    """Every 4d node (any version) plus every hub node (3d/2d/section, any
    version) — the domain for nace_text and hierarchy (v7 §2). A single
    combined mask across all three versions is safe even though hierarchy
    edges never cross versions: each version's 4d codes only ever connect
    to that SAME version's own hub nodes (build_hierarchy_layer loops per
    version), so nothing here implies any cross-version mixing — this mask
    only says which rows/columns the layer's P matrix has nonzero structure
    in, not who connects to whom."""
    nt = ni.nodes["node_type"].values
    return np.isin(nt, ["4d", "3d", "2d", "section"])


def _processed_path(path_or_name: str | Path) -> Path:
    path = Path(path_or_name)
    return path if path.is_absolute() else PROCESSED_DATA_DIR / path


def build_labor_layer(
    ni: NodeIndex, relatedness_file: str | Path = "labor_relatedness.parquet"
) -> Layer:
    """Within-version 4d<->4d phi edges (Neffke-Henning)."""
    path = _processed_path(relatedness_file)
    rows, cols, weights = [], [], []
    if path.exists():
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            ver = str(row["coding"])
            r = ni.get("4d", ver, str(row["sni_from"]))
            c = ni.get("4d", ver, str(row["sni_to"]))
            if r is not None and c is not None:
                rows.append(r); cols.append(c); weights.append(float(row["phi"]))
    P, avail = row_normalize(rows, cols, weights, ni.N, symmetric=True)
    edges = _symmetric_edge_table(rows, cols, weights, "labor")
    print(f"  labor      : {len(weights):6,} undirected entries")
    return Layer("labor", P, avail, _mask_4d_all(ni), edges)


def build_ownership_layer(
    ni: NodeIndex, relatedness_file: str | Path = "ownership_relatedness.parquet"
) -> Layer:
    """Within-version 4d<->4d phi edges from common-ownership co-occurrence
    (2026-07-29 — see build_ownership_relatedness.py): two SNI codes both
    worked under the same enterprise (OrgNr) in the same year are evidence
    of a related-diversification link independent of labor or IO flow. Same
    file convention as labor_relatedness.parquet (a `coding` column, one row
    per unordered pair per version) and the same never-split-by-version
    structure — build_expanded_layers (layer_diffusion.py) can split this
    into ownership_SNI92/02/07 the same way it splits `labor`, if a future
    diagnostic finds density/training-dynamics reasons to (v9 §... for
    labor's own version of that split); nothing here requires it yet."""
    path = _processed_path(relatedness_file)
    rows, cols, weights = [], [], []
    if path.exists():
        df = pd.read_parquet(path)
        for row in df.itertuples(index=False):
            ver = str(row.coding)
            r = ni.get("4d", ver, str(row.sni_from))
            c = ni.get("4d", ver, str(row.sni_to))
            if r is not None and c is not None:
                rows.append(r); cols.append(c); weights.append(float(row.phi))
    P, avail = row_normalize(rows, cols, weights, ni.N, symmetric=True)
    edges = _symmetric_edge_table(rows, cols, weights, "ownership")
    print(f"  ownership  : {len(weights):6,} undirected entries")
    return Layer("ownership", P, avail, _mask_4d_all(ni), edges)


def _crosswalk_pairs(csv_name: str, old_ver: str, new_ver: str, as_int: bool
                      ) -> tuple[pd.Series, pd.Series]:
    path = METADATA_DIR / csv_name
    if not path.exists():
        return pd.Series(dtype=object), pd.Series(dtype=object)
    df = pd.read_csv(path)
    old_col = next((c for c in df.columns if old_ver[:5] in c.upper()), df.columns[0])
    new_col = next((c for c in df.columns if new_ver[:5] in c.upper()), df.columns[1])

    def to4(v):
        try:
            s5 = str(int(v)).zfill(5) if as_int else str(v).zfill(5)
        except (ValueError, TypeError):
            return None
        return s5[:4]

    old4, new4 = df[old_col].map(to4), df[new_col].map(to4)
    valid = old4.notna() & new4.notna()
    return old4[valid], new4[valid]


CROSSWALK_CSVS = [
    ("mapping_02_07.csv", "SNI02", "SNI07", False),
    ("mapping_92_02.csv", "SNI92", "SNI02", True),
]


def build_transition_layer(ni: NodeIndex) -> Layer:
    """4d_SNI92<->4d_SNI02<->4d_SNI07 crosswalk, both directions combined
    into one directed layer. Weighted by fan-out on whichever side is the
    *source* for that direction: an old code that splits into 3 new codes
    sends 1/3 to each going up; a new code that 2 old codes merged into
    sends 1/2 to each going down."""
    rows, cols, weights = [], [], []
    for csv_name, old_ver, new_ver, as_int in CROSSWALK_CSVS:
        old4, new4 = _crosswalk_pairs(csv_name, old_ver, new_ver, as_int)
        fanout_old = old4.value_counts()
        fanout_new = new4.value_counts()
        for o, n in zip(old4, new4):
            r_old = ni.get("4d", old_ver, o)
            r_new = ni.get("4d", new_ver, n)
            if r_old is None or r_new is None:
                continue
            rows.append(r_old); cols.append(r_new); weights.append(1.0 / max(int(fanout_old.get(o, 1)), 1))
            rows.append(r_new); cols.append(r_old); weights.append(1.0 / max(int(fanout_new.get(n, 1)), 1))
    P, avail = row_normalize(rows, cols, weights, ni.N, symmetric=False)
    edges = pd.DataFrame({"src": rows, "dst": cols, "weight": weights, "type": "transition"})
    print(f"  transition : {len(weights):6,} directed entries (both directions)")
    return Layer("transition", P, avail, _mask_4d_all(ni), edges)


def build_node_concept_groups(ni: NodeIndex) -> dict[int, int]:
    """{node_id: group_id} — every 4d SNI92/02/07 node with a crosswalk
    correspondence mapped to a shared group id, so a concept's SNI92/02/07
    incarnations land in the same train/val/test split (see
    src/latent_industry/common/splits.py). Hub nodes and any 4d code missing
    a crosswalk row are left out — split_key_for's fallback treats them as
    their own singleton group."""
    from latent_industry.common.splits import union_find_groups
    pairs: list[tuple[int, int]] = []
    for csv_name, old_ver, new_ver, as_int in CROSSWALK_CSVS:
        old4, new4 = _crosswalk_pairs(csv_name, old_ver, new_ver, as_int)
        for o, n in zip(old4, new4):
            r = ni.get("4d", old_ver, o)
            c = ni.get("4d", new_ver, n)
            if r is not None and c is not None:
                pairs.append((r, c))
    return union_find_groups(pairs)


def build_hierarchy_layer(ni: NodeIndex) -> Layer:
    """4d<->3d<->2d<->section parent/child, both directions combined into
    one directed layer, built independently within EACH SNI version (v7 §2
    — previously SNI07 only; SNI92/SNI02 4d codes had no hierarchy edge of
    their own and reached this structure only indirectly through
    `transition`). Climbing (child->parent) has weight 1 per edge, each
    node has exactly one parent; descending (parent->child) mass splits
    across however many children a hub has — a coarser-level signal really
    is less specific per child than a direct edge would be. No edge ever
    crosses versions here — that is `transition`'s job."""
    rows, cols, weights = [], [], []
    for version in ("SNI07", "SNI02", "SNI92"):
        for code in ni.sni_4d[version]:
            r_4d = ni.get("4d", version, code)
            r_3d = ni.get("3d", version, code[:3])
            if r_4d is not None and r_3d is not None:
                rows += [r_4d, r_3d]; cols += [r_3d, r_4d]; weights += [1.0, 1.0]
        for code in ni.hub_3d[version]:
            r_3d = ni.get("3d", version, code)
            r_2d = ni.get("2d", version, code[:2])
            if r_3d is not None and r_2d is not None:
                rows += [r_3d, r_2d]; cols += [r_2d, r_3d]; weights += [1.0, 1.0]
        for code in ni.hub_2d[version]:
            r_2d = ni.get("2d", version, code)
            r_sec = ni.get("section", version, ni.sections[version].get(code, "?"))
            if r_2d is not None and r_sec is not None:
                rows += [r_2d, r_sec]; cols += [r_sec, r_2d]; weights += [1.0, 1.0]
    P, avail = row_normalize(rows, cols, weights, ni.N, symmetric=False)
    edges = pd.DataFrame({"src": rows, "dst": cols, "weight": weights, "type": "hierarchy"})
    print(f"  hierarchy  : {len(weights):6,} directed entries (both directions)")
    return Layer("hierarchy", P, avail, _mask_hierarchy_all_versions(ni), edges)


IO_RELATEDNESS_FILES_BY_VERSION: dict[str, str] = {
    "SNI07": "io_relatedness_SNI07.parquet",
    "SNI02": "io_relatedness_SNI02.parquet",
    # No SNI92 entry: no IO source reaches back that far (2026-07-29 rewrite
    # of build_input_output_relatedness.py — WIOD's earliest year, 1995, is
    # already SNI02-vintage by classification regardless of calendar year).
}

def build_io_layer(
    ni: NodeIndex,
    relatedness_files: dict[str, str | Path] | None = None,
) -> Layer:
    """Flow-relatedness edges between 2d hub nodes, built independently per
    SNI version (2026-07-29: `io` used to be SNI07-only — a single pooled
    `io_relatedness.parquet` blended sources spanning two classification
    vintages and a ~25-year window into one table. Now each version has its
    own relatedness file, built from only the sources that are native to
    that vintage (see build_input_output_relatedness.py), matching how
    `labor` and `hierarchy` already keep versions separate. No edge ever
    crosses versions here, same as those two."""
    relatedness_files = relatedness_files or IO_RELATEDNESS_FILES_BY_VERSION
    rows, cols, weights = [], [], []
    versions_with_data: list[str] = []
    for version, relatedness_file in relatedness_files.items():
        path = _processed_path(relatedness_file)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        sim_col = next((c for c in ["similarity", "io_similarity", "sim"] if c in df.columns), None)
        if sim_col is None:
            continue
        versions_with_data.append(version)
        for row in df.itertuples(index=False):
            div_from = str(getattr(row, "nace2d_from", "")).zfill(2)
            div_to = str(getattr(row, "nace2d_to", "")).zfill(2)
            sim = float(getattr(row, sim_col))
            if sim <= 0 or div_from == div_to:
                continue
            r = ni.get("2d", version, div_from)
            c = ni.get("2d", version, div_to)
            if r is not None and c is not None:
                rows.append(r); cols.append(c); weights.append(sim)
    P, avail = row_normalize(rows, cols, weights, ni.N, symmetric=True)
    edges = _symmetric_edge_table(rows, cols, weights, "io")
    print(f"  io         : {len(weights):6,} undirected entries  (versions: {versions_with_data or 'none'})")
    # Domain restricted to exactly the versions that have an IO source at all
    # (currently SNI07 + SNI02) — a plain `node_type == "2d"` mask would
    # incorrectly claim io "structurally belongs" to SNI92 divisions, which
    # have no IO source and never will under the current sources.
    nt, ver = ni.nodes["node_type"].values, ni.nodes["version"].values
    domain = (nt == "2d") & np.isin(ver, versions_with_data)
    return Layer("io", P, avail, domain, edges)


def _center_and_renormalize(vecs: np.ndarray, mean: np.ndarray) -> np.ndarray:
    centered = vecs.astype(np.float64) - mean
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (centered / norms).astype(np.float32)


def load_nace_embeddings(filename: str) -> dict | None:
    """Load a precomputed NACE embeddings file (see build_nace_embeddings.py),
    then mean-center each level before re-normalising — removes the shared
    anisotropic direction that otherwise spreads a node's mass almost
    uniformly across ~600 barely-distinguishable neighbours (see
    docs/conceptual_audit.md's NACE negative edges finding for the measured before/after spread)."""
    path = PROCESSED_DATA_DIR / filename
    if not path.exists():
        return None
    raw = np.load(path, allow_pickle=False)
    data = dict(raw)
    for level in ["class", "group", "division", "section"]:
        vecs_key, excl_key = f"{level}_vecs", f"{level}_excl_vecs"
        if vecs_key not in data or len(data[vecs_key]) == 0:
            continue
        mean = data[vecs_key].astype(np.float64).mean(axis=0)
        data[vecs_key] = _center_and_renormalize(data[vecs_key], mean)
        if excl_key in data and len(data[excl_key]) > 0:
            data[excl_key] = _center_and_renormalize(data[excl_key], mean)
    return data


_HUB_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _encode_version_hub_embeddings(ni: NodeIndex, version: str, model) -> dict:
    """v7 §2: SNI92/SNI02 have no NACE Rev.2 rich-text embeddings —
    build_nace_embeddings.py's cached nace_embeddings_{tag}.npz is SNI07
    only (that's the one version with full NACE Rev.2 documentation).
    But `nace_level_texts`'s own fallback for a code with no NACE Rev.2
    match is already just the plain SCB label (see that function's
    docstring) — the exact text NodeIndex already loaded into
    `sni_4d`/`hub_3d`/`hub_2d`/`hub_section` — so there is nothing richer
    to lose here for these two versions. Directly encode that plain text
    with the same lightweight, locally-cached sentence-transformer
    P02_shared_geometry_reconstruction.py (formerly evaluate_embeddings.py)
    already uses for hub-level fallback encoding,
    then mean-center/renormalise exactly like `load_nace_embeddings` does
    for the SNI07 OpenAI embeddings (same anisotropy-removal rationale,
    not SNI07-specific). One small one-time cost at graph-build time
    (~1,600 short texts total across both versions), not per training
    epoch."""
    level_lookups = {
        "class": ni.sni_4d[version],
        "group": ni.hub_3d[version],
        "division": ni.hub_2d[version],
        "section": ni.hub_section[version],
    }
    out: dict = {}
    for level, lookup in level_lookups.items():
        codes = sorted(lookup)
        texts = [lookup[c] for c in codes]
        if texts:
            vecs = model.encode(texts, batch_size=128, normalize_embeddings=True,
                                 show_progress_bar=False).astype(np.float32)
            mean = vecs.astype(np.float64).mean(axis=0)
            vecs = _center_and_renormalize(vecs, mean)
        else:
            vecs = np.zeros((0, 0), dtype=np.float32)
        out[f"{level}_codes"] = np.array(codes)
        out[f"{level}_vecs"] = vecs
    return out


def _align_codes(nace_emb: dict, level: str, node_type: str, version: str,
                  code_lookup: dict, ni: NodeIndex) -> tuple[list[int], list[str], np.ndarray]:
    level_codes = nace_emb.get(f"{level}_codes")
    if level_codes is None or len(level_codes) == 0:
        return [], [], np.zeros((0, 0), dtype=np.float32)
    vec_by_code = dict(zip(level_codes, nace_emb[f"{level}_vecs"]))
    nids, codes_used, vecs_list = [], [], []
    for code in sorted(code_lookup):
        nid = ni.get(node_type, version, code)
        vec = vec_by_code.get(code)
        if nid is None or vec is None:
            continue
        nids.append(nid); codes_used.append(code); vecs_list.append(vec)
    vecs = np.stack(vecs_list).astype(np.float32) if vecs_list else np.zeros((0, 0), dtype=np.float32)
    return nids, codes_used, vecs


def _level_pairs(nids: list[int], vecs: np.ndarray, sharpen: float
                  ) -> tuple[tuple[list, list, list], tuple[list, list, list]]:
    """All pairs at one NACE level, split by sign. Positive part (sharpened,
    then row-normalised elsewhere) feeds the layer; negative part is kept
    only as a diagnostic (see module docstring's "signed evidence" note —
    repulsion evidence comes from verified excludes pairs, not this)."""
    if len(nids) < 2:
        return ([], [], []), ([], [], [])
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, 0.0)
    iu, ju = np.triu_indices(len(nids), k=1)
    s = sims[iu, ju]

    pos = s > 0
    pos_w = s[pos] ** sharpen if sharpen != 1.0 else s[pos]
    pos_triple = ([nids[i] for i in iu[pos]], [nids[j] for j in ju[pos]], pos_w.tolist())

    neg = s < 0
    neg_triple = ([nids[i] for i in iu[neg]], [nids[j] for j in ju[neg]], (-s[neg]).tolist())
    return pos_triple, neg_triple


def build_nace_layer(ni: NodeIndex, nace_embedding_file: str, sharpen: float = 1.0
                      ) -> tuple[Layer, NaceEmbeddingContext]:
    """cosine-similarity edges between NACE description embeddings, at every
    level and every SNI version (v7 §2 — previously SNI07 only) in one
    matrix — levels AND versions never cross-connect, so this is
    block-diagonal by construction, not a per-level/per-version relation to
    be weighted separately (see module docstring). SNI92/SNI02 use a
    lightweight locally-encoded fallback instead of the cached SNI07-only
    NACE Rev.2 embeddings — see `_encode_version_hub_embeddings`.

    `sharpen` (default 1.0 = no-op): raw positive weight -> weight**sharpen
    before row-normalising, applied uniformly across all levels. Exists
    because NACE cosine similarity is naturally much flatter than flow data
    (labor/io) — short official descriptions share enough generic
    vocabulary that most pairs land around sim~0.2-0.3 regardless of real
    relatedness. This is local to this one layer because it's the one
    family that actually needs it; nothing else in the pipeline does.
    """
    domain = _mask_hierarchy_all_versions(ni)
    nace_emb = load_nace_embeddings(nace_embedding_file)
    if nace_emb is None:
        edges = pd.DataFrame(columns=["src", "dst", "weight", "type"])
        P, avail = row_normalize([], [], [], ni.N, symmetric=True)
        ctx = NaceEmbeddingContext(raw={}, nids_4d07=[], codes_4d07=[], vecs_4d07=np.zeros((0, 0)),
                                    negative_edges=edges)
        return Layer("nace_text", P, avail, domain, edges), ctx

    from sentence_transformers import SentenceTransformer
    _hub_text_model = SentenceTransformer(_HUB_TEXT_MODEL)
    nace_emb_by_version = {
        "SNI07": nace_emb,
        "SNI92": _encode_version_hub_embeddings(ni, "SNI92", _hub_text_model),
        "SNI02": _encode_version_hub_embeddings(ni, "SNI02", _hub_text_model),
    }

    levels = [
        ("class", "4d", "SNI07", ni.sni_4d["SNI07"], "nace_4d"),
        ("group", "3d", "SNI07", ni.sni07_3d, "nace_3d"),
        ("division", "2d", "SNI07", ni.sni07_2d, "nace_2d"),
        ("section", "section", "SNI07", ni.sni07_sec, "nace_sec"),
        ("class", "4d", "SNI92", ni.sni_4d["SNI92"], "nace_4d_92"),
        ("group", "3d", "SNI92", ni.hub_3d["SNI92"], "nace_3d_92"),
        ("division", "2d", "SNI92", ni.hub_2d["SNI92"], "nace_2d_92"),
        ("section", "section", "SNI92", ni.hub_section["SNI92"], "nace_sec_92"),
        ("class", "4d", "SNI02", ni.sni_4d["SNI02"], "nace_4d_02"),
        ("group", "3d", "SNI02", ni.hub_3d["SNI02"], "nace_3d_02"),
        ("division", "2d", "SNI02", ni.hub_2d["SNI02"], "nace_2d_02"),
        ("section", "section", "SNI02", ni.hub_section["SNI02"], "nace_sec_02"),
    ]

    all_rows, all_cols, all_weights = [], [], []
    level_edge_tables = []
    neg_records = []
    nids_4d07, codes_4d07, vecs_4d07 = [], [], np.zeros((0, 0), dtype=np.float32)

    for level, node_type, version, code_lookup, label in levels:
        nids, codes_used, vecs = _align_codes(nace_emb_by_version[version], level, node_type, version,
                                               code_lookup, ni)
        (pr, pc, pw), (nr, nc, nw) = _level_pairs(nids, vecs, sharpen)
        all_rows += pr; all_cols += pc; all_weights += pw
        # `type` keeps the fine-grained per-level label (nace_4d/3d/2d/sec) so
        # existing diagnostics that filter graph_edges_{name}.parquet by level
        # (e.g. P02_shared_geometry_reconstruction.py's H section) keep working — the P
        # matrix above is still the one combined nace_text layer; only this
        # provenance table distinguishes levels.
        level_edges = _symmetric_edge_table(pr, pc, pw, label)
        level_edges["layer"] = "nace_text"
        level_edge_tables.append(level_edges)
        for r, c, w in zip(nr, nc, nw):
            neg_records.append({"src": r, "dst": c, "weight": w, "type": f"{label}_neg"})
        print(f"  {label:<10} : {len(pw):6,} positive pairs  {len(nw):6,} negative pairs (diagnostic)"
              f"  ({len(nids)} nodes)")
        if level == "class" and version == "SNI07":
            nids_4d07, codes_4d07, vecs_4d07 = nids, codes_used, vecs

    P, avail = row_normalize(all_rows, all_cols, all_weights, ni.N, symmetric=True)
    edges = (pd.concat(level_edge_tables, ignore_index=True) if level_edge_tables
             else pd.DataFrame(columns=["src", "dst", "weight", "type", "layer"]))
    negative_edges = pd.DataFrame(neg_records, columns=["src", "dst", "weight", "type"])

    ctx = NaceEmbeddingContext(raw=nace_emb, nids_4d07=nids_4d07, codes_4d07=codes_4d07,
                                vecs_4d07=vecs_4d07, negative_edges=negative_edges)
    return Layer("nace_text", P, avail, domain, edges), ctx


def _symmetric_edge_table(rows: list, cols: list, weights: list, etype: str) -> pd.DataFrame:
    """Canonicalise a symmetric relation's (row, col) pairs to (low, high)
    once per pair for the saved edge table."""
    seen = set()
    records = []
    for r, c, w in zip(rows, cols, weights):
        key = (min(r, c), max(r, c))
        if key not in seen:
            seen.add(key)
            records.append({"src": key[0], "dst": key[1], "weight": float(w), "type": etype})
    return pd.DataFrame(records, columns=["src", "dst", "weight", "type"])


def build_all_layers(
    ni: NodeIndex,
    nace_embedding_file: str,
    nace_sharpen: float = 1.0,
    labor_relatedness_file: str | Path = "labor_relatedness.parquet",
    io_relatedness_files: dict[str, str | Path] | None = None,
    ownership_relatedness_file: str | Path = "ownership_relatedness.parquet",
) -> tuple[dict[str, Layer], NaceEmbeddingContext]:
    """Build every layer. Returns {layer_name: Layer} plus the NACE embedding
    context repulsion.py needs for the excludes-pairs lookup."""
    print("Building layers …")
    nace_layer, nace_ctx = build_nace_layer(ni, nace_embedding_file, sharpen=nace_sharpen)
    layers = {
        "labor": build_labor_layer(ni, labor_relatedness_file),
        "nace_text": nace_layer,
        "io": build_io_layer(ni, io_relatedness_files),
        "hierarchy": build_hierarchy_layer(ni),
        "transition": build_transition_layer(ni),
        "ownership": build_ownership_layer(ni, ownership_relatedness_file),
    }
    for lname, layer in layers.items():
        if "layer" not in layer.edges.columns:
            layer.edges["layer"] = lname
    return layers, nace_ctx
