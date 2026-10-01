"""splits.py — reproducible, leakage-safe train/val/test partitions for
evidence-family edge lists (labor, io, nace_4d/3d/2d/sec, ...).

Two leakage modes need blocking together, not just random-row splitting:

  1. Symmetric duplicates — every relation here is undirected, so a naive
     random split over individual (i, j) rows would let (i, j) train while
     (j, i), the same observation, lands in test.
  2. Cross-version near-duplicates — an SNI92 code, its SNI02 crosswalk
     successor, and its SNI07 successor are the same real industry. An SNI92
     edge (A, B) and the SNI07 edge (A'', B'') are near-duplicate
     observations of the same underlying relatedness, not independent
     evidence (see TODO.md §6.3).

Both are solved by splitting on a canonical PAIR-GROUP key instead of raw
edge rows: every node is mapped to a version-independent "concept group"
(its connected component under crosswalk correspondence — nodes with no
cross-version identity, e.g. hub nodes, just get their own singleton group).
An edge's split key is then tuple(sorted((group(i), group(j)))) — order-
independent (fixes symmetric duplicates) and vintage-independent (fixes
cross-version near-duplicates). Every edge sharing a split key is guaranteed
to land in the same partition.
"""

import hashlib
from typing import Hashable, Iterable

import pandas as pd


def union_find_groups(pairs: Iterable[tuple[Hashable, Hashable]]) -> dict[Hashable, Hashable]:
    """Union-find over "same concept" pairs. Returns {member: group_representative}
    for every member that appears in `pairs`. Members that never appear default
    to their own singleton group — see split_key_for's group.get(x, x) fallback,
    callers don't need to add an explicit entry for them."""
    parent: dict[Hashable, Hashable] = {}

    def find(x: Hashable) -> Hashable:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return {m: find(m) for m in parent}


def split_key_for(a: Hashable, b: Hashable, group: dict[Hashable, Hashable]) -> tuple:
    """Order-independent, group-aware split key for one edge."""
    ga, gb = group.get(a, a), group.get(b, b)
    return tuple(sorted((ga, gb), key=str))


def _key_fraction(key: Hashable, seed: int) -> float:
    """Deterministic pseudo-uniform value in [0, 1) for one key, salted by seed."""
    digest = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def assign_splits(keys: Iterable[Hashable], seed: int,
                   train_frac: float = 0.7, val_frac: float = 0.15) -> dict[Hashable, str]:
    """Deterministic salted-hash bucketing of each distinct key into
    'train'/'val'/'test' at approximately train_frac/val_frac/remainder
    proportions. Reproducible from `seed` alone — no persisted lookup table
    required — and stable under adding new keys later (existing keys don't
    get reshuffled just because the key set grew)."""
    assignment: dict[Hashable, str] = {}
    for key in set(keys):
        frac = _key_fraction(key, seed)
        if frac < train_frac:
            assignment[key] = "train"
        elif frac < train_frac + val_frac:
            assignment[key] = "val"
        else:
            assignment[key] = "test"
    return assignment


def split_edge_table(edges: pd.DataFrame, i_col: str, j_col: str,
                      group: dict[Hashable, Hashable] | None = None,
                      seed: int = 0, train_frac: float = 0.7, val_frac: float = 0.15
                      ) -> pd.DataFrame:
    """Returns a copy of `edges` with a 'split' column added
    ('train'/'val'/'test'), grouped/blocked per the module docstring.
    group=None is fine for single-vintage families (io, nace_4d/3d/2d/sec) —
    every node just gets an implicit singleton group. Pass a cross-version
    union-find mapping (see union_find_groups) for labor."""
    group = group or {}
    keys = [split_key_for(a, b, group) for a, b in zip(edges[i_col], edges[j_col])]
    assignment = assign_splits(keys, seed, train_frac, val_frac)
    out = edges.copy()
    out["split"] = [assignment[k] for k in keys]
    return out
