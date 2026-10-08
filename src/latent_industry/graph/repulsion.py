"""
repulsion.py — negative (repulsion) evidence between 4d SNI07 codes: pairs
whose official NACE class text explicitly excludes the other. Kept as a
separate matrix S_neg (saved in layers_{name}.npz), not as transition mass.

Negative NACE cosine similarity is not used as repulsion evidence: after the
per-level mean-centring in layers.load_nace_embeddings, a majority of pairs
have negative similarity, so the sign is not informative. It remains
available as a diagnostic (layers.NaceEmbeddingContext.negative_edges).
"""

import numpy as np

from latent_industry.graph.layers import NaceEmbeddingContext


def collect_class_excludes_pairs(nace_ctx: NaceEmbeddingContext) -> list[tuple[int, int]]:
    """4d_SNI07 node-id pairs whose official NACE class text explicitly
    excludes the other (closest excluded-text match)."""
    excl_codes = nace_ctx.raw.get("class_excl_codes")
    if excl_codes is None or len(excl_codes) == 0 or len(nace_ctx.nids_4d07) < 2:
        return []
    excl_vec_by_code = dict(zip(excl_codes, nace_ctx.raw["class_excl_vecs"]))
    code_pos = {code: i for i, code in enumerate(nace_ctx.codes_4d07)}
    pairs = []
    for code, excl_vec in excl_vec_by_code.items():
        li = code_pos.get(code)
        if li is None:
            continue
        sims = excl_vec.astype(np.float32) @ nace_ctx.vecs_4d07.T
        sims[li] = -1.0
        lj = int(np.argmax(sims))
        pairs.append((nace_ctx.nids_4d07[li], nace_ctx.nids_4d07[lj]))
    print(f"  excludes   : {len(pairs):6,} class-level repulsion pairs (post-hoc penalty, not a layer)")
    return pairs


def build_repulsion_matrix(nace_ctx: NaceEmbeddingContext, target_ids: list[int],
                            target_pos: dict[int, int]) -> np.ndarray:
    """
    Target × target repulsion matrix S_neg: weight 1.0 per excludes pair,
    deduplicated.
    """
    excludes_pairs = collect_class_excludes_pairs(nace_ctx)
    weight_by_pair: dict[tuple[int, int], float] = {}
    for i, j in excludes_pairs:
        weight_by_pair[(i, j)] = max(weight_by_pair.get((i, j), 0.0), 1.0)

    n = len(target_ids)
    S_neg = np.zeros((n, n), dtype=np.float64)
    for (i, j), w in weight_by_pair.items():
        pi, pj = target_pos.get(i), target_pos.get(j)
        if pi is not None and pj is not None:
            S_neg[pi, pj] = w
            S_neg[pj, pi] = w
    print(f"  repulsion  : {len(weight_by_pair):6,} 4d_SNI07 pairs (verified excludes only)")
    return S_neg
