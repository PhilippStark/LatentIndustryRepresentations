"""
layers.py — one row-stochastic transition matrix per relational view.

Each builder turns its raw source into (rows, cols, weights) and calls
transitions.row_normalize. Views are not weighted against each other here,
and there is no top-k or magnitude threshold.

Views
-----
  labor       4d ↔ 4d labor relatedness (observed/expected worker flows),
              within each SNI version
  ownership   4d ↔ 4d common-ownership relatedness, within each SNI version
  io          input–output relatedness between 2d hub nodes, per version
  hierarchy   4d ↔ 3d ↔ 2d ↔ section parent/child edges, within each version
  transition_SNI92_SNI02, transition_SNI02_SNI07
              4d correspondence between consecutive SNI versions (an identity
              mapping, not a relatedness score), one layer per version pair
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
    domain: np.ndarray    # N bool — which nodes structurally belong to this layer, regardless
                           # of whether they currently have any edges (the rows a training
                           # target is evaluated on)


def _mask_4d_all(ni: NodeIndex) -> np.ndarray:
    """Every 4d node, any version (SNI07/02/92) — the domain for labor and
    ownership; build_labor_version_layers (layer_diffusion.py) narrows labor
    to one version per target."""
    return (ni.nodes["node_type"].values == "4d")


def _mask_4d_versions(ni: NodeIndex, versions: tuple[str, ...]) -> np.ndarray:
    """4d nodes in exactly the given version(s)."""
    nt, ver = ni.nodes["node_type"].values, ni.nodes["version"].values
    return (nt == "4d") & np.isin(ver, versions)


def _mask_hierarchy_all_versions(ni: NodeIndex) -> np.ndarray:
    """
    Every 4d node and every hub node (3d/2d/section) of every version: the
    domain of hierarchy. Edges never cross versions; the mask only marks which
    rows and columns carry structure.
    """
    nt = ni.nodes["node_type"].values
    return np.isin(nt, ["4d", "3d", "2d", "section"])


def _processed_path(path_or_name: str | Path) -> Path:
    path = Path(path_or_name)
    return path if path.is_absolute() else PROCESSED_DATA_DIR / path


def build_labor_layer(
    ni: NodeIndex, relatedness_file: str | Path = "labor_relatedness.parquet"
) -> Layer:
    """Within-version 4d<->4d labor relatedness edges (build_labor_relatedness.py)."""
    path = _processed_path(relatedness_file)
    df = pd.read_parquet(path)
    rows, cols, weights = [], [], []
    for _, row in df.iterrows():
        ver = str(row["coding"])
        r = ni.get("4d", ver, str(row["sni_from"]))
        c = ni.get("4d", ver, str(row["sni_to"]))
        if r is not None and c is not None:
            rows.append(r); cols.append(c); weights.append(float(row["phi"]))
    P = row_normalize(rows, cols, weights, ni.N, symmetric=True)
    print(f"  labor      : {len(weights):6,} undirected entries")
    return Layer("labor", P, _mask_4d_all(ni))


def build_ownership_layer(
    ni: NodeIndex, relatedness_file: str | Path = "ownership_relatedness.parquet"
) -> Layer:
    """
    Within-version 4d ↔ 4d edges from common-ownership relatedness
    (ownership_relatedness.parquet; one row per unordered pair and version).
    """
    path = _processed_path(relatedness_file)
    df = pd.read_parquet(path)
    rows, cols, weights = [], [], []
    for row in df.itertuples(index=False):
        ver = str(row.coding)
        r = ni.get("4d", ver, str(row.sni_from))
        c = ni.get("4d", ver, str(row.sni_to))
        if r is not None and c is not None:
            rows.append(r); cols.append(c); weights.append(float(row.phi))
    P = row_normalize(rows, cols, weights, ni.N, symmetric=True)
    print(f"  ownership  : {len(weights):6,} undirected entries")
    return Layer("ownership", P, _mask_4d_all(ni))


def _crosswalk_pairs(csv_name: str, old_ver: str, new_ver: str, as_int: bool
                      ) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(METADATA_DIR / csv_name)
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


def build_transition_pair_layer(ni: NodeIndex, csv_name: str, old_ver: str, new_ver: str,
                                 as_int: bool) -> Layer:
    """
    Transition layer for one correspondence pair (e.g. SNI92 ↔ SNI02), both
    directions, restricted to the two versions' 4-digit codes. Weighted by
    fan-out on whichever side is the source for that direction: an old code
    that splits into 3 new codes sends 1/3 to each going up; a new code that 2
    old codes merged into sends 1/2 to each going down.

    The two version pairs are kept as separate layers rather than one combined
    92 ↔ 02 ↔ 07 layer: that combined graph is a path without direct 92 ↔ 07
    edges, so even powers of its P could not carry any 92 ↔ 02 or 02 ↔ 07 mass,
    and the combined row-normalisation would dilute each direction by the
    other's fan-out.
    """
    old4, new4 = _crosswalk_pairs(csv_name, old_ver, new_ver, as_int)
    fanout_old = old4.value_counts()
    fanout_new = new4.value_counts()
    rows, cols, weights = [], [], []
    for o, n in zip(old4, new4):
        r_old = ni.get("4d", old_ver, o)
        r_new = ni.get("4d", new_ver, n)
        if r_old is None or r_new is None:
            continue
        rows.append(r_old); cols.append(r_new); weights.append(1.0 / max(int(fanout_old.get(o, 1)), 1))
        rows.append(r_new); cols.append(r_old); weights.append(1.0 / max(int(fanout_new.get(n, 1)), 1))
    P = row_normalize(rows, cols, weights, ni.N, symmetric=False)
    lname = f"transition_{old_ver}_{new_ver}"
    print(f"  {lname}: {len(weights):6,} directed entries (both directions)")
    return Layer(lname, P, _mask_4d_versions(ni, (old_ver, new_ver)))


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
    P = row_normalize(rows, cols, weights, ni.N, symmetric=False)
    print(f"  hierarchy  : {len(weights):6,} directed entries (both directions)")
    return Layer("hierarchy", P, _mask_hierarchy_all_versions(ni))


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
    for version, relatedness_file in relatedness_files.items():
        df = pd.read_parquet(_processed_path(relatedness_file))
        for row in df.itertuples(index=False):
            div_from = str(row.nace2d_from).zfill(2)
            div_to = str(row.nace2d_to).zfill(2)
            sim = float(row.sim)
            if sim <= 0 or div_from == div_to:
                continue
            r = ni.get("2d", version, div_from)
            c = ni.get("2d", version, div_to)
            if r is not None and c is not None:
                rows.append(r); cols.append(c); weights.append(sim)
    P = row_normalize(rows, cols, weights, ni.N, symmetric=True)
    print(f"  io         : {len(weights):6,} undirected entries  (versions: {list(relatedness_files)})")
    # Domain: the 2d divisions of the versions that have an IO source.
    nt, ver = ni.nodes["node_type"].values, ni.nodes["version"].values
    domain = (nt == "2d") & np.isin(ver, list(relatedness_files))
    return Layer("io", P, domain)


def build_all_layers(
    ni: NodeIndex,
    labor_relatedness_file: str | Path = "labor_relatedness.parquet",
    io_relatedness_files: dict[str, str | Path] | None = None,
    ownership_relatedness_file: str | Path = "ownership_relatedness.parquet",
) -> dict[str, Layer]:
    """Build every layer. Returns {layer_name: Layer}."""
    print("Building layers …")
    layers = {
        "labor": build_labor_layer(ni, labor_relatedness_file),
        "io": build_io_layer(ni, io_relatedness_files),
        "hierarchy": build_hierarchy_layer(ni),
        "ownership": build_ownership_layer(ni, ownership_relatedness_file),
    }
    for csv_name, old_ver, new_ver, as_int in CROSSWALK_CSVS:
        layer = build_transition_pair_layer(ni, csv_name, old_ver, new_ver, as_int)
        layers[layer.name] = layer
    return layers
