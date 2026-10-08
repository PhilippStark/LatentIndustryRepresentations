"""
layers.py — one row-stochastic transition matrix per relational view.

Each builder turns its raw source into (rows, cols, weights) and calls
transitions.row_normalize. Views are not weighted against each other here,
and there is no top-k or magnitude threshold; negative NACE similarity is
clipped to zero (negative evidence is handled in repulsion.py).

Views
-----
  labor       4d ↔ 4d labor relatedness (Neffke–Henning), within each SNI version
  nace_text   cosine similarity of NACE description embeddings, at every level
              (4d/3d/2d/section) and version; block-diagonal by level and version
  io          input–output relatedness between 2d hub nodes, per version
  hierarchy   4d ↔ 3d ↔ 2d ↔ section parent/child edges, within each version
  transition  SNI92 ↔ SNI02 ↔ SNI07 correspondence (an identity mapping, not a
              relatedness score)
  ownership   4d ↔ 4d common-ownership relatedness, within each SNI version
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
                           # of whether they currently have any edges (the rows a training
                           # target is evaluated on)
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
    """
    Every 4d node and every hub node (3d/2d/section) of every version: the
    domain of nace_text and hierarchy. Edges never cross versions; the mask
    only marks which rows and columns carry structure.
    """
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
    """
    Within-version 4d ↔ 4d edges from common-ownership relatedness
    (ownership_relatedness.parquet; one row per unordered pair and version).
    """
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
    """
    4d ↔ 3d ↔ 2d ↔ section parent/child edges, both directions in one directed
    layer, built within each SNI version. Climbing (child → parent) has weight 1
    (each node has one parent); descending (parent → child) mass is split over
    the children. No edge crosses versions (that is `transition`).
    """
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
    # No SNI92 entry: no IO source covers SNI92 (WIOD's earliest year, 1995,
    # is classified in the SNI02 vintage).
}

def build_io_layer(
    ni: NodeIndex,
    relatedness_files: dict[str, str | Path] | None = None,
) -> Layer:
    """
    Input–output relatedness edges between 2d hub nodes, built per SNI version
    from that version's own relatedness file (io_relatedness_SNI07.parquet,
    io_relatedness_SNI02.parquet; there is no SNI92 source). No edge crosses
    versions.
    """
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
    """
    Load the precomputed NACE description embeddings (PROCESSED_DATA_DIR/
    `filename`) and mean-centre each level before re-normalising. Centring
    removes the shared anisotropic direction that otherwise makes most pairs
    look similar.
    """
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
    """
    Embeddings of SNI92/SNI02 nodes, which have no NACE Rev. 2 text: their
    plain SCB labels (as loaded by NodeIndex) are encoded with a local
    sentence-transformer (all-MiniLM-L6-v2) and mean-centred/re-normalised
    like load_nace_embeddings.
    """
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
    """
    All pairs at one NACE level, split by sign. The positive part feeds the
    layer; the negative part is kept as a diagnostic only.
    """
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
    """
    Cosine-similarity edges between NACE description embeddings at every level
    and every SNI version, in one block-diagonal matrix (levels and versions
    never connect). SNI07 uses the precomputed NACE Rev. 2 embeddings; SNI92 and
    SNI02 use _encode_version_hub_embeddings.

    `sharpen` (default 1.0 = off) raises positive weights to that power before
    row-normalising, to counter how flat description similarity is compared
    with flow-based relatedness.
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
