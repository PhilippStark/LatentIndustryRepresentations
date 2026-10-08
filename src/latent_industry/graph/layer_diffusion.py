"""
layer_diffusion.py — multi-scale diffusion targets per relational view.

Each view's transition matrix P_l is diffused on its own (P_l^t, no
cross-view mixing), and every (view, t) is kept as a separate training
target (build_layer_diffusion_targets_expanded). Labor and transition are
split by classification version; IO uses its native 2-digit layer.

lift_io_layer (IO lifted to 4-digit SNI07 codes via a membership matrix) is
not used for the training targets; diagnose_scale_redundancy.py uses it.
"""

import numpy as np
import pandas as pd

from latent_industry.graph.layers import CROSSWALK_CSVS, Layer, _crosswalk_pairs
from latent_industry.graph.transitions import row_normalize

def build_diffusion(layer: Layer, t: int) -> np.ndarray:
    """
    P_l^t as an ordinary matrix power. Powers of a row-stochastic matrix stay
    row-stochastic, and all-zero (inactive) rows stay zero.
    """
    return np.linalg.matrix_power(layer.P, t)


def build_membership_matrix(ni, fine: tuple[str, str] = ("4d", "SNI07"),
                             coarse: tuple[str, str] = ("2d", "SNI07"),
                             prefix_len: int = 2) -> np.ndarray:
    """
    H (N x N): H[i, k] = 1 iff node i is in the `fine` (node_type, version),
    node k is in the `coarse` (node_type, version) and i's first `prefix_len`
    code characters equal k's code (e.g. 4-digit → 2-digit with prefix_len=2).
    All other entries are 0, so H applies directly to a full N x N matrix.
    """
    nt = ni.nodes["node_type"].values
    ver = ni.nodes["version"].values
    code = ni.nodes["code"].values
    fine_mask = (nt == fine[0]) & (ver == fine[1])
    coarse_mask = (nt == coarse[0]) & (ver == coarse[1])
    coarse_code_to_id = {code[j]: j for j in np.nonzero(coarse_mask)[0]}

    H = np.zeros((ni.N, ni.N), dtype=np.float64)
    for i in np.nonzero(fine_mask)[0]:
        j = coarse_code_to_id.get(code[i][:prefix_len])
        if j is not None:
            H[i, j] = 1.0
    return H


def lift_to_finer_domain(W_coarse: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    H @ W_coarse @ H.T: lift a coarse-level raw (not yet row-normalised) weight
    matrix to the finer level. Lifting must happen before row-normalisation;
    lifting a normalised P would give coarse nodes with many children
    disproportionate mass.
    """
    return H @ W_coarse @ H.T


def _edges_to_matrix(edges: pd.DataFrame, n: int, symmetric: bool) -> np.ndarray:
    """Scatter a raw (src, dst, weight) edge table into a dense N x N
    matrix, pre-normalisation — mirrors transitions.row_normalize's own
    scatter step, minus the row-normalise."""
    W = np.zeros((n, n), dtype=np.float64)
    if len(edges):
        r = edges["src"].to_numpy(dtype=np.int64)
        c = edges["dst"].to_numpy(dtype=np.int64)
        w = edges["weight"].to_numpy(dtype=np.float64)
        if symmetric:
            r, c, w = np.concatenate([r, c]), np.concatenate([c, r]), np.concatenate([w, w])
        np.add.at(W, (r, c), w)
    return W


def _mask_4d_sni07(ni) -> np.ndarray:
    """The 4d-SNI07 subset of `_mask_hierarchy_all_versions`'s domain
    (layers.py) — the level `io` gets lifted to, matching
    `hierarchy`/`nace_text`."""
    nt, ver = ni.nodes["node_type"].values, ni.nodes["version"].values
    return (nt == "4d") & (ver == "SNI07")


def lift_io_layer(io_raw_edges: pd.DataFrame, ni) -> Layer:
    """
    Lift `io`'s raw 2-digit edge weights to the 4-digit SNI07 level via the
    membership matrix and return the row-normalised result as a `Layer` over the
    4-digit SNI07 domain. `io_raw_edges` must be the raw edge table
    (graph_edges_{name}.parquet filtered to type == "io"), not the normalised
    Layer.P.
    """
    H = build_membership_matrix(ni, fine=("4d", "SNI07"), coarse=("2d", "SNI07"), prefix_len=2)
    W_2d = _edges_to_matrix(io_raw_edges, ni.N, symmetric=True)
    W_4d = lift_to_finer_domain(W_2d, H)

    rows, cols = np.nonzero(W_4d)
    weights = W_4d[rows, cols]
    P_4d, avail_4d = row_normalize(rows.tolist(), cols.tolist(), weights.tolist(), ni.N, symmetric=False)

    domain_4d = _mask_4d_sni07(ni)
    edges = pd.DataFrame({"src": rows, "dst": cols, "weight": weights, "type": "io"})
    return Layer("io", P_4d, avail_4d, domain_4d, edges)


def _mask_4d_versions(ni, versions: tuple[str, ...]) -> np.ndarray:
    """4d nodes in exactly the given version(s) — the domain for a single
    crosswalk-pair layer (see build_transition_pair_layer), narrower than
    `layers._mask_4d_all`'s all-versions-combined domain."""
    nt, ver = ni.nodes["node_type"].values, ni.nodes["version"].values
    return (nt == "4d") & np.isin(ver, versions)


def build_transition_pair_layer(ni, csv_name: str, old_ver: str, new_ver: str, as_int: bool) -> Layer:
    """
    Transition layer for one correspondence pair (e.g. SNI92 ↔ SNI02), both
    directions, row-normalised on its own and restricted to the two versions'
    4-digit codes.

    The combined transition graph is a path 92 ↔ 02 ↔ 07 without direct 92 ↔ 07
    edges. A path graph is bipartite, so even powers of its P cannot carry any
    92 ↔ 02 or 02 ↔ 07 mass, and the combined row-normalisation would dilute each
    direction by the other's fan-out. Single-pair layers used only at t = 1
    avoid both problems.
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
    P, avail = row_normalize(rows, cols, weights, ni.N, symmetric=False)
    domain = _mask_4d_versions(ni, (old_ver, new_ver))
    lname = f"transition_{old_ver}_{new_ver}"
    edges = pd.DataFrame({"src": rows, "dst": cols, "weight": weights, "type": lname})
    return Layer(lname, P, avail, domain, edges)


TRANSITION_PAIR_SCALES: tuple[int, ...] = (1,)

# io is used as its native 2-digit layer, undiffused (t = 1).
IO_NATIVE_SCALES: tuple[int, ...] = (1,)


def build_hierarchy_ordinal_tiers(ni) -> tuple[np.ndarray, np.ndarray]:
    """
    (tier, domain) for the ordinal hierarchy target used as hierarchy_t1.

    Because hub embeddings are aggregates of their 4-digit descendants, the
    hierarchy is enforced as an ordinal constraint among 4-digit SNI07 codes:
    cos(i, same 3-digit group) > cos(i, same division) > cos(i, same section)
    > cos(i, other section).

    `tier[i, j]` ∈ {0, 1, 2, 3, 4}: 0 = self / outside `domain`; 1 = same
    3-digit group; 2 = same division, different group; 3 = same section,
    different division; 4 = different section. `domain` is the 4-digit SNI07
    codes.
    """
    domain = _mask_4d_sni07(ni)
    idx = np.nonzero(domain)[0]
    codes = ni.nodes.loc[idx, "code"].values
    prefix3 = np.array([c[:3] for c in codes])
    prefix2 = np.array([c[:2] for c in codes])
    sections = ni.sections["SNI07"]
    sec = np.array([sections.get(p2, "?") for p2 in prefix2])

    same_3d = prefix3[:, None] == prefix3[None, :]
    same_2d = prefix2[:, None] == prefix2[None, :]
    same_section = (sec[:, None] == sec[None, :]) & (sec[:, None] != "?")

    m = len(idx)
    tier_local = np.full((m, m), 4, dtype=np.int8)
    tier_local[same_section] = 3
    tier_local[same_2d] = 2
    tier_local[same_3d] = 1
    np.fill_diagonal(tier_local, 0)

    tier = np.zeros((ni.N, ni.N), dtype=np.int8)
    tier[np.ix_(idx, idx)] = tier_local
    return tier, domain


def build_transition_pair_layers(ni) -> dict[str, Layer]:
    """
    transition_SNI92_SNI02 (mapping_92_02.csv) and transition_SNI02_SNI07
    (mapping_02_07.csv), each built and row-normalised separately (see
    build_transition_pair_layer). Uses the same CROSSWALK_CSVS as
    layers.build_transition_layer.
    """
    return {
        f"transition_{old_ver}_{new_ver}": build_transition_pair_layer(ni, csv_name, old_ver, new_ver, as_int)
        for csv_name, old_ver, new_ver, as_int in CROSSWALK_CSVS
    }


LABOR_VERSIONS: tuple[str, ...] = ("SNI92", "SNI02", "SNI07")


def build_labor_version_layers(labor_layer: Layer, ni) -> dict[str, Layer]:
    """
    Split `labor` into one layer per SNI version (labor_SNI92 / labor_SNI02 /
    labor_SNI07), so each version is its own target and an easier-to-fit version
    cannot dominate the loss of the others.

    This is a domain restriction only: labor relatedness is computed within
    each version, so `labor`'s P is block-diagonal by version and each layer
    reuses the same P/avail/edges with the domain narrowed to that version's
    4-digit codes.
    """
    return {
        f"labor_{ver}": Layer(
            f"labor_{ver}", labor_layer.P, labor_layer.avail,
            _mask_4d_versions(ni, (ver,)) & labor_layer.domain, labor_layer.edges,
        )
        for ver in LABOR_VERSIONS
    }


# Default scales per view. diagnose_scale_redundancy.py shows that t = 1 is a
# distinct signal from t >= 2 for every view, while t = 2/4/8 are highly
# redundant with each other; (1, 4) keeps both kinds of structure.
EXPANDED_DEFAULT_SCALES: tuple[int, ...] = (1, 4)


def build_layer_diffusion_targets_expanded(
    layers: dict[str, Layer], ni,
    layer_scales: dict[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """
    One training target per (view, t): targets[f"{view}_t{t}"] = P_l^t
    (build_diffusion), with t from `layer_scales` (default
    EXPANDED_DEFAULT_SCALES).

    Returns (targets, domains, families), keyed by target_id:
    - targets[target_id]   P_l^t (N x N)
    - domains[target_id]   the view's domain mask
    - families[target_id]  the view (family) the target belongs to; loss
                           weights are set per family and split across its
                           targets (operator_data.py)

    Special cases:
    - hierarchy_t1 is the ordinal 4-digit tier matrix
      (build_hierarchy_ordinal_tiers) instead of P^1; family "hierarchy".
    - transition is split into transition_SNI92_SNI02 and
      transition_SNI02_SNI07 (build_transition_pair_layers), fixed at
      TRANSITION_PAIR_SCALES = (1,); family "transition".
    - labor is split into labor_SNI92 / labor_SNI02 / labor_SNI07
      (build_labor_version_layers), each with the labor scales; family "labor".
    - io uses its native 2-digit layer at IO_NATIVE_SCALES = (1,), not lifted
      to 4-digit codes, so the training signal acts on the 2-digit division
      embeddings that the IO evaluation (P02, section I) measures. 4-digit codes
      relate to IO structure through the hierarchy view.
    """
    layer_scales = dict(layer_scales) if layer_scales else {}

    effective_layers = dict(layers)
    effective_layers.pop("transition", None)
    transition_pairs = build_transition_pair_layers(ni)
    effective_layers.update(transition_pairs)
    families_by_layer = {pname: "transition" for pname in transition_pairs}
    for pname in transition_pairs:
        layer_scales.setdefault(pname, TRANSITION_PAIR_SCALES)

    labor_layer = effective_layers.pop("labor", None)
    if labor_layer is not None:
        labor_versions = build_labor_version_layers(labor_layer, ni)
        effective_layers.update(labor_versions)
        families_by_layer.update({pname: "labor" for pname in labor_versions})

    if "io" in effective_layers:
        layer_scales.setdefault("io", IO_NATIVE_SCALES)

    targets: dict[str, np.ndarray] = {}
    domains: dict[str, np.ndarray] = {}
    families: dict[str, str] = {}
    for lname, layer in effective_layers.items():
        l_scales = layer_scales.get(lname, EXPANDED_DEFAULT_SCALES)
        family = families_by_layer.get(lname, lname)
        for t in l_scales:
            target_id = f"{lname}_t{t}"
            # hierarchy_t1 is the ordinal 4d-4d tier target
            # (build_hierarchy_ordinal_tiers); other scales are P^t.
            if lname == "hierarchy" and t == 1:
                tier, ordinal_domain = build_hierarchy_ordinal_tiers(ni)
                targets[target_id] = tier
                domains[target_id] = ordinal_domain
            else:
                targets[target_id] = build_diffusion(layer, t)
                domains[target_id] = layer.domain
            families[target_id] = family
    return targets, domains, families
