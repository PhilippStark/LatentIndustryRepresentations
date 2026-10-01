"""layer_diffusion.py — Phase 1 of the operator-preserving regime (see
docs/operator_preserving_implementation_plan.md and
docs/operator_preserving_math.md §2.1/§3). Computes each layer's OWN
diffusion powers P_l^t independently — no supra-network, no coupling
parameter, no cross-layer mixing — and keeps every (layer, t) as its own
training target (`build_layer_diffusion_targets_expanded`). The earlier
regime that weight-averaged the scales into one P̃_l per layer is archived.

`lift_io_layer` (lift `io`'s 2d-only raw edges to 4d-SNI07 via a membership
matrix, math doc §2.1) is not used by the expanded targets, which train on
the native 2d `io` layer; it is kept for diagnose_scale_redundancy.py.
"""

import numpy as np
import pandas as pd

from latent_industry.graph.layers import CROSSWALK_CSVS, Layer, _crosswalk_pairs
from latent_industry.graph.transitions import row_normalize

def build_diffusion(layer: Layer, t: int) -> np.ndarray:
    """P_layer^t — a single ordinary matrix power. No renormalisation: matrix
    powers of a row-stochastic matrix are already row-stochastic, and any
    all-zero row (an inactive node) stays exactly zero at every power (see
    math doc §3)."""
    return np.linalg.matrix_power(layer.P, t)


def build_membership_matrix(ni, fine: tuple[str, str] = ("4d", "SNI07"),
                             coarse: tuple[str, str] = ("2d", "SNI07"),
                             prefix_len: int = 2) -> np.ndarray:
    """H, N x N: H[i,k] = 1 iff node i is in the `fine` (node_type, version)
    and node k is in the `coarse` (node_type, version) and i's first
    `prefix_len` code characters equal k's code — the same code-prefix
    relationship build_hierarchy_layer already uses for 4d->2d
    (prefix_len=2). All other entries are 0, including rows/columns for
    nodes outside `fine`/`coarse` respectively, so `H` can be used directly
    against a full N x N layer matrix without any separate submatrix
    bookkeeping.

    Parameterised by level pair (not hard-coded to 4d-SNI07<->2d-SNI07) so a
    future coarser-native layer (e.g. an innovation layer only observed at
    2d/3d) can reuse this unchanged — see the implementation plan's "adding
    a new layer" checklist.
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
    """H @ W_coarse @ H.T — lift a coarse-level raw (pre-normalisation)
    weight matrix to the finer level `H` maps into. Must run on raw weights,
    before row-normalisation (math doc §2.1): lifting an already-normalised
    coarse P would let coarse nodes with more fine children absorb
    disproportionate mass, since the same row value gets duplicated across
    every child without correcting for how many children each parent has.
    Row-normalising *after* lifting (see `lift_io_layer`) avoids that."""
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
    """`io`'s native domain is 2d-only — there is no 4d-level input-output
    flow data at all — so independent per-layer diffusion can never reach 4d
    rows from `io` alone (its domain never touches a 4d node, at any power).
    Lift `io`'s raw edge weights to the 4d-SNI07 level via the membership
    matrix *before* diffusing, and return the result as another `Layer` over
    the 4d-SNI07 domain (matching `hierarchy`/`nace_text`) — `io` stops
    being a 2d-only special case for the rest of the pipeline (math doc
    §2.1). `io_raw_edges` must be `io`'s pre-normalisation edge table (e.g.
    `graph_edges_{name}.parquet`, filtered to `type == "io"`), not the
    already row-normalised `Layer.P` cached in `layers_{name}.npz`.
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
    """One crosswalk pair only (e.g. SNI92<->SNI02), both directions,
    normalised independently of any other version pair — Phase 9's fix for
    a real, mechanistic problem with the combined `transition` layer
    (layers.build_transition_layer): its full graph is a strict path
    92<->02<->07 with NO direct 92<->07 edge in the source data at all.
    A path graph is bipartite by parity — even powers of its P (e.g. t=4,
    EXPANDED_DEFAULT_SCALES's second scale) can only ever connect a 92/07
    node back to {92, 07} and a 02 node back to other 02 nodes; they cannot
    carry ANY 92<->02 or 02<->07 mass, at any even t. Confirmed empirically
    (shared_cosine_v1 run, 2026-07-27): transition_t4 trained fine while
    transition_t1 (the only scale that COULD carry the real crosswalk)
    stayed flat all 500 epochs, and the resulting embedding's SNI02 codes
    got literally zero cross-version pull — visible directly in the
    t-SNE (92/07 merge into one cluster, 02 stays separate).

    Splitting into two single-hop-only layers, each restricted to just its
    own two versions' domain, removes the parity trap (t>1 is never used
    for these) and a secondary dilution problem: the combined layer
    row-normalises a 02 code's mass over its 92-parents AND 07-children
    together, so even at t=1 each direction was diluted by the other's
    fan-out. Independent normalisation here means each direction is its
    own clean distribution.
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

# Pinned to the raw (t=1, un-diffused) native io layer for the expanded
# regime — see build_layer_diffusion_targets_expanded's `io` note. Diffusing
# further (t>1) would just re-derive a version of the same lift-free-but-
# diffused signal the multi-scale default already offered before this fix;
# the point of this change is training directly on `io_relatedness.parquet`
# itself, at the same 2d-hub resolution P02_shared_geometry_reconstruction.py's
# (formerly evaluate_embeddings.py's) I_io_rho checks, not a smoothed/diffused
# derivative of it.
IO_NATIVE_SCALES: tuple[int, ...] = (1,)


def build_hierarchy_ordinal_tiers(ni) -> tuple[np.ndarray, np.ndarray]:
    """(tier, domain) for the v7 §1.4 ordinal hierarchy target, which
    replaces `hierarchy_t1`'s old categorical (softmax-over-domain) target:
    once §1.2 makes a hub row a function of its own 4d children, a 4d code
    is part of its own hub's definition by construction and `hierarchy_t1`
    stops carrying information. The replacement is an ordinal constraint on
    4d-4d relations only — `cos(i, same_3d) > cos(i, same_2d) >
    cos(i, same_section) > cos(i, other)` — which is exactly what
    P03_emergent_structure.py's (formerly evaluate_embeddings.py's)
    `A_treedist_rho` already measures.

    `domain` is 4d-SNI07 only — narrower than the categorical `hierarchy`
    layer's hub-inclusive domain (same precedent as `labor`'s per-version
    domain narrowing above in this file; other hierarchy scales, e.g. t=4,
    keep the original hub-inclusive categorical domain unchanged).

    `tier[i, j]` in `{0, 1, 2, 3, 4}`: `0` = self/undefined (also the value
    for any row/column outside `domain`); `1` = same 3d group; `2` = same 2d
    division, different 3d group; `3` = same section, different 2d
    division; `4` = different section. Same code-prefix logic
    `build_hierarchy_layer` already uses to build the actual parent/child
    edges, just compared pairwise here (m^2, m ~ 615) instead of walked
    level by level.
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
    """transition_SNI92_SNI02 (from mapping_92_02.csv) and
    transition_SNI02_SNI07 (from mapping_02_07.csv), each built and
    row-normalised independently — see build_transition_pair_layer's
    docstring for why the combined layer's single P can't provide a real
    92<->07 bridge at any even power. Same CROSSWALK_CSVS layers.py's
    build_transition_layer (the combined, non-split version used by every
    OTHER regime/pipeline) already iterates — reused here, not
    reimplemented, so both stay in sync if a crosswalk source ever changes."""
    return {
        f"transition_{old_ver}_{new_ver}": build_transition_pair_layer(ni, csv_name, old_ver, new_ver, as_int)
        for csv_name, old_ver, new_ver, as_int in CROSSWALK_CSVS
    }


LABOR_VERSIONS: tuple[str, ...] = ("SNI92", "SNI02", "SNI07")


def build_labor_version_layers(labor_layer: Layer, ni) -> dict[str, Layer]:
    """Splits `labor` into one layer per SNI version — Phase 9 fix for a
    WITHIN-target imbalance the family-level lambda can never reach: labor's
    single combined target pools all three versions' rows into one
    row-averaged categorical loss, so if one version's relatedness
    structure happens to be easier to fit than another's, joint training
    can improve the easy version's rows while leaving the harder version's
    rows essentially unimproved — exactly the same "one shared objective
    lets an easy piece dominate an unrelated hard piece" mechanism the
    transition split addressed, but here it's rows of ONE target, not
    separate targets, which no per-family lambda can rebalance (empirically
    observed: E_labor_rho_07 stuck around .25-.32 across every
    shared_cosine run while E_labor_rho_02/92 climbed to .6-.67).

    Splitting is a pure domain restriction, not a new computation:
    `build_labor_layer` already guarantees `labor`'s P has zero mass
    crossing version boundaries (Neffke-Henning relatedness is computed
    within each SNI version separately — see layers.py's own docstring),
    so each per-version layer reuses the identical, already block-diagonal
    P/avail/edges — only the domain mask narrows to that version's 4d
    codes."""
    return {
        f"labor_{ver}": Layer(
            f"labor_{ver}", labor_layer.P, labor_layer.avail,
            _mask_4d_versions(ni, (ver,)) & labor_layer.domain, labor_layer.edges,
        )
        for ver in LABOR_VERSIONS
    }


# Default per-layer scales for the *expanded* (many-targets, shared-cosine)
# regime — deliberately not the old averaged regime's (1,2,4,8,16). Picked
# from diagnose_scale_redundancy.py's own output (2026-07-27 run against
# multilayer_v1): for every one of the five layers, t=1 is close to
# uncorrelated-to-moderately-correlated with t>=2 (row rho from ~0.0
# (hierarchy/transition) to ~0.6-0.96 (labor/nace_text/io), cosine as low as
# 0.0-0.4) — a genuinely distinct signal, exactly the piece the original
# operator_v1 pipeline's weighted-average P̃_l blurred away. But t=2/4/8
# are themselves highly mutually redundant for every layer (rho/cosine
# mostly >0.9, several >0.99) — a third or fourth scale on top of one
# "diffused" scale adds little. Two scales per layer (t=1, t=4) capture
# most of the real, non-redundant structure the diagnostic found without
# padding the target count with near-duplicates that would just make the
# per-family weight-normalisation do more work for no informational gain.
EXPANDED_DEFAULT_SCALES: tuple[int, ...] = (1, 4)


def build_layer_diffusion_targets_expanded(
    layers: dict[str, Layer], ni,
    layer_scales: dict[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """Phase 9 variant of Phase 1: instead of collapsing each layer's scales
    into one weight-averaged P̃_l (the archived averaged regime), keep
    every (layer, t) power as its OWN separate target — `P_l^(t)` itself,
    via `build_diffusion`, not a blend. This is what lets the shared-cosine
    training regime see local (small t) and global (large t) reachability
    as distinct training signals instead of one pre-mixed compromise.

    `hierarchy_t1` (v7 §1.4) is the one exception to "every target is
    `P_l^(t)`": it's replaced with `build_hierarchy_ordinal_tiers`'s 4d-4d
    tier-rank matrix (domain narrowed to 4d-SNI07 only) — see that
    function's docstring for why. `families["hierarchy_t1"]` is still
    `"hierarchy"`; only its target/domain shape changes.

    No `io_raw_edges` parameter (unlike the archived averaged regime):
    this orchestrator does NOT lift `io` to 4d — see the `io` note below —
    so it never needs `io`'s raw pre-normalisation edges, only the already-
    cached native 2d-domain `Layer` already present in `layers`.

    Returns `(targets, domains, families)`, all keyed by `target_id =
    f"{layer}_t{t}"`:
    - `targets[target_id]` is `P_l^(t)` (N x N)
    - `domains[target_id]` is layer `l`'s effective domain (same for every
      `t` of a given layer)
    - `families[target_id] = l` — which layer this target belongs to, used
      downstream (operator_data.py) to group per-target loss weights by
      layer rather than by target, so a layer with more scales in
      `layer_scales` doesn't get proportionally more influence in the
      joint loss just for having more targets.

    `transition` is always split into `transition_SNI92_SNI02` and
    `transition_SNI02_SNI07` (build_transition_pair_layers), each pinned to
    `TRANSITION_PAIR_SCALES=(1,)` regardless of `layer_scales` — see
    build_transition_pair_layer's docstring for why the combined layer and
    any t>1 are both actively wrong for a crosswalk. Both stay grouped
    under the `families[target_id] = "transition"` family (not two new
    top-level families), so the existing per-family lambda budget just
    splits across these two targets instead of the old t=1/t=4 pair.

    `labor` is always split into `labor_SNI92`/`labor_SNI02`/`labor_SNI07`
    (build_labor_version_layers) — a pure domain restriction of the same
    already block-diagonal-by-version P, each still getting
    `layer_scales`/`EXPANDED_DEFAULT_SCALES` normally (no scale pin, unlike
    transition — labor's within-version relatedness graph is dense and
    genuinely benefits from multiple scales). All three stay grouped under
    `families[target_id] = "labor"`.

    `io` uses its NATIVE (2d-hub, un-lifted, un-diffused) layer, pinned to
    `IO_NATIVE_SCALES=(1,)` — deliberately NOT `lift_io_layer` (which the
    archived averaged regime used). Root-caused 2026-07-27:
    `P02_shared_geometry_reconstruction.py`'s (formerly `evaluate_embeddings.py`'s)
    `I_io_rho` correlates cosine similarity of
    the 2d-hub embeddings themselves against raw `io_relatedness.parquet`
    — but the lifted-to-4d `io_t1`/`io_t4` targets have domain=4d-SNI07
    only (`lift_io_layer`'s whole point), so they never put a single
    gradient on the 2d-hub rows `I_io_rho` actually measures; any
    correlation between io's training loss and `I_io_rho` was necessarily
    indirect at best. Confirmed empirically (`shared_cosine_v2`->`v4`):
    `io_t4`'s raw loss kept dropping under dynamic reweighting while
    `I_io_rho` got WORSE (0.598->0.239, ending up below even
    `multilayer_v1`'s 0.457) — loss improving while the actual target
    metric degrades is the signature of optimising a proxy that doesn't
    point at the thing being measured, not a weighting problem. Training
    directly on the native 2d layer puts real gradient on the exact rows
    `I_io_rho` checks; 4d nodes still reach IO structure indirectly via
    `hierarchy`'s 4d<->2d climb edges, the same "aggregate up through an
    adjacent family" pattern `transition` already relies on for SNI92/02
    4d codes reaching the SNI07 hierarchy graph.
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
            # v7 §1.4: hierarchy_t1 becomes the ordinal 4d-4d tier target
            # instead of the categorical hub-inclusive diffusion target —
            # see build_hierarchy_ordinal_tiers. Other hierarchy scales
            # (e.g. t=4 under EXPANDED_DEFAULT_SCALES) are untouched.
            if lname == "hierarchy" and t == 1:
                tier, ordinal_domain = build_hierarchy_ordinal_tiers(ni)
                targets[target_id] = tier
                domains[target_id] = ordinal_domain
            else:
                targets[target_id] = build_diffusion(layer, t)
                domains[target_id] = layer.domain
            families[target_id] = family
    return targets, domains, families
