"""
layer_diffusion.py — multi-scale diffusion targets per relational view.

Each view's transition matrix P_l is diffused on its own (P_l^t, no
cross-view mixing), and every (view, t) is kept as a separate training
target (build_layer_diffusion_targets_expanded). Labor is split by
classification version; IO uses its native 2-digit layer.
"""

import numpy as np

from latent_industry.graph.layers import Layer, _mask_4d_versions


def build_diffusion(layer: Layer, t: int) -> np.ndarray:
    """
    P_l^t as an ordinary matrix power. Powers of a row-stochastic matrix stay
    row-stochastic, and all-zero (inactive) rows stay zero.
    """
    return np.linalg.matrix_power(layer.P, t)


def _mask_4d_sni07(ni) -> np.ndarray:
    """The 4-digit SNI07 codes: the domain of the ordinal hierarchy target."""
    nt, ver = ni.nodes["node_type"].values, ni.nodes["version"].values
    return (nt == "4d") & (ver == "SNI07")


# Scales of the main run (shared_cosine_v16). transition and io are used at
# t = 1 only: transition is a direct correspondence, and io is kept as its
# native, undiffused 2-digit layer.
DEFAULT_LAYER_SCALES: dict[str, tuple[int, ...]] = {
    "labor": (1, 2, 4),
    "hierarchy": (1, 2, 4),
    "ownership": (1, 4),
    "io": (1,),
    "transition_SNI92_SNI02": (1,),
    "transition_SNI02_SNI07": (1,),
}


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


LABOR_VERSIONS: tuple[str, ...] = ("SNI92", "SNI02", "SNI07")


def build_labor_version_layers(labor_layer: Layer, ni) -> dict[str, Layer]:
    """
    Split `labor` into one layer per SNI version (labor_SNI92 / labor_SNI02 /
    labor_SNI07), so each version is its own target and an easier-to-fit version
    cannot dominate the loss of the others.

    This is a domain restriction only: labor relatedness is computed within
    each version, so `labor`'s P is block-diagonal by version and each layer
    reuses the same P with the domain narrowed to that version's 4-digit codes.
    """
    return {
        f"labor_{ver}": Layer(f"labor_{ver}", labor_layer.P, _mask_4d_versions(ni, (ver,)) & labor_layer.domain)
        for ver in LABOR_VERSIONS
    }


def build_layer_diffusion_targets_expanded(
    layers: dict[str, Layer], ni,
    layer_scales: dict[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """
    One training target per (view, t): targets[f"{view}_t{t}"] = P_l^t
    (build_diffusion), with t from `layer_scales` (default
    DEFAULT_LAYER_SCALES; labor's scales apply to each of its versions).

    Returns (targets, domains, families), keyed by target_id:
    - targets[target_id]   P_l^t (N x N)
    - domains[target_id]   the view's domain mask
    - families[target_id]  the view (family) the target belongs to; loss
                           weights are set per family and divided among its
                           targets (operator_data.py)

    Special cases:
    - hierarchy_t1 is the ordinal 4-digit tier matrix
      (build_hierarchy_ordinal_tiers) instead of P^1; family "hierarchy".
    - transition_SNI92_SNI02 and transition_SNI02_SNI07 form family "transition".
    - labor is split into labor_SNI92 / labor_SNI02 / labor_SNI07
      (build_labor_version_layers), each with the labor scales; family "labor".
    - io uses its native 2-digit layer, not lifted to 4-digit codes, so the
      training signal acts on the 2-digit division embeddings that the IO
      evaluation (P02, section I) measures. 4-digit codes relate to IO
      structure through the hierarchy view.
    """
    layer_scales = {**DEFAULT_LAYER_SCALES, **(layer_scales or {})}

    effective_layers = dict(layers)
    families_by_layer = {lname: "transition" for lname in layers if lname.startswith("transition_")}

    labor_layer = effective_layers.pop("labor")
    labor_versions = build_labor_version_layers(labor_layer, ni)
    effective_layers.update(labor_versions)
    for pname in labor_versions:
        families_by_layer[pname] = "labor"
        layer_scales[pname] = layer_scales["labor"]

    targets: dict[str, np.ndarray] = {}
    domains: dict[str, np.ndarray] = {}
    families: dict[str, str] = {}
    for lname, layer in effective_layers.items():
        family = families_by_layer.get(lname, lname)
        for t in layer_scales[lname]:
            target_id = f"{lname}_t{t}"
            if lname == "hierarchy" and t == 1:
                tier, ordinal_domain = build_hierarchy_ordinal_tiers(ni)
                targets[target_id] = tier
                domains[target_id] = ordinal_domain
            else:
                targets[target_id] = build_diffusion(layer, t)
                domains[target_id] = layer.domain
            families[target_id] = family
    return targets, domains, families
