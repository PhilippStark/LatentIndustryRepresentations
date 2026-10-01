"""repulsion.py — 4d_SNI07 repulsion evidence: verified "officially excludes"
class-text matches, applied as a direct post-hoc penalty on the final target
similarity (see multilayer_diffusion.finalize_target), not folded into any
layer's transition matrix, since a Markov chain can't carry negative
transition mass.

Sourced only from collect_class_excludes_pairs, not from negative NACE
cosine similarity — that similarity is negative largely because
layers.load_nace_embeddings mean-centers each level's vectors before
computing it (a per-level anisotropy-removal artifact, ~59% of all 4d pairs
measured negative post-centering, not a real ~1-2% exclusion rate — see
docs/conceptual_audit.md's NACE negative edges finding). The negative cosine score still exists as a
continuous diagnostic (layers.NaceEmbeddingContext.negative_edges) — it's
just not treated as repulsion here.
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
    """target x target repulsion weight matrix — flat weight 1.0 per
    verified excludes pair, deduplicated. This is what gets saved as S_neg
    and later scaled by --repulsion in finalize_target."""
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
