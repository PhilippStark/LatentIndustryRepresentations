"""
Embedding-based regional relatedness density, shared by the regional
scripts (scripts/regional/P04_*.py, P05, R01).

The density of industry j in municipality m in year t combines the
municipality's employment mix M_t[m, :] with a similarity kernel
K[i, j] = f(cos(z_i, z_j)):

    density[m, j] = (M_t @ K)[m, j] / normaliser[m, j]

Kernels
-------
  softmax          K[i,j] = softmax_i(τ·cos(z_i,z_j)), self excluded; high τ
                   concentrates weight on the closest industries
  inverse softmax  K[i,j] = softmax_i(−τ·cos(z_i,z_j)), self excluded; weight
                   on the most distant industries
  kNN              0/1 for the k nearest neighbours per column, column-normalised
  raw / relu       plain (or rectified) cosine, column-normalised

Employment-weighted densities (leave-one-out, no temperature)
-------------------------------------------------------------
  arithmetic:  D[m,j] = Σ_{i≠j} M_t[m,i]·cos(z_i,z_j) / Σ_{i≠j} M_t[m,i]
  geometric:   D[m,j] = exp( Σ_{i≠j} M_t[m,i]·log(s_ij) / Σ_{i≠j} M_t[m,i] ),
               s_ij = (cos(z_i,z_j) + 1) / 2, so one dissimilar industry with
               substantial employment lowers the density strongly

Passing np.log1p(M_t) as M_t gives the log-employment-weighted versions; the
arithmetic one, D^{A,log}, is the paper's main specification.
"""

from pathlib import Path

import numpy as np


def load_sni_embeddings(
    run_name: str,
    embed_base: Path,
    labels: tuple[str, ...] = ("sni07", "sni02", "sni92"),
    verbose: bool = True,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """
    Load {label: (vecs, codes)} from embed_base/run_name/embeddings_{label}.npz
    (e.g. sni07, sni02, sni92), as written by train_shared_cosine_embedding.py.
    """
    embed_dir = embed_base / run_name
    if verbose:
        print(f"\nLoading embeddings from {embed_dir} …")

    out: dict[str, tuple[np.ndarray, list[str]]] = {}
    for label in labels:
        path = embed_dir / f"embeddings_{label}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} — run train_shared_cosine_embedding.py first.")
        d = np.load(path, allow_pickle=True)
        vecs = d["embeddings"].astype(np.float32)
        codes = list(d["codes"].astype(str))
        if verbose:
            print(f"  {label}: {len(codes)} codes  dim={vecs.shape[1]}")
        out[label] = (vecs, codes)
    return out


def cosine_matrix(vecs: np.ndarray, zero_diagonal: bool = True) -> np.ndarray:
    """Full (n, n) cosine-similarity matrix. Assumes vecs are unit-normalised."""
    cos_sim = (vecs @ vecs.T).astype(np.float64)
    if zero_diagonal:
        np.fill_diagonal(cos_sim, 0.0)
    return cos_sim


def make_softmax_kernel(vecs: np.ndarray, tau: float) -> np.ndarray:
    """
    K[i,j] = softmax_i(τ·cos(x_i,x_j)), self excluded — column-stochastic.
    density_j = Σ_i K[i,j]·w_i / W is industry j's relatedness density: how
    close j sits to the existing employment mass, well-defined even if j
    itself has zero employment (K's diagonal is excluded by construction).
    """
    cos_sim = (vecs @ vecs.T).astype(np.float64)
    np.fill_diagonal(cos_sim, -np.inf)
    shifted = tau * cos_sim - (tau * cos_sim).max(axis=0, keepdims=True)
    E = np.exp(shifted)
    K = E / E.sum(axis=0, keepdims=True)
    return K.astype(np.float32)


def make_inv_softmax_kernel(vecs: np.ndarray, tau: float) -> np.ndarray:
    """
    Inverted softmax kernel: upweights DISSIMILAR (distant) industries.
    K_far[i,j] = softmax_i(−τ·cos(x_i,x_j)), self excluded.
    High τ → attention concentrated on the most distant industries in
    embedding space → density_far captures Jacobs-style unrelated variety.
    """
    cos_sim = (vecs @ vecs.T).astype(np.float64)
    raw_inv = -tau * cos_sim
    np.fill_diagonal(raw_inv, -np.inf)
    shifted = raw_inv - raw_inv.max(axis=0, keepdims=True)
    E = np.exp(shifted)
    K = E / E.sum(axis=0, keepdims=True)
    return K.astype(np.float32)


def make_kernel(vecs: np.ndarray, kernel: str, tau: float) -> np.ndarray:
    """
    Column-stochastic kernel K where K[i,j] = weight of industry i for
    computing the embedding neighbourhood density of industry j.

    raw      K = cos(x_i, x_j)
    relu     K = max(0, cos) column-normalised
    softmax  K = softmax_i(+τ · cos(x_i, x_j))  — see make_softmax_kernel()
    """
    if kernel == "softmax":
        return make_softmax_kernel(vecs, tau)

    cos_sim = (vecs @ vecs.T).astype(np.float64)
    np.fill_diagonal(cos_sim, 0.0)

    if kernel == "raw":
        K = cos_sim
    elif kernel == "relu":
        K = np.maximum(0.0, cos_sim)
        col_sums = K.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        K = K / col_sums
    else:
        raise ValueError(f"Unknown kernel {kernel!r} — expected raw/relu/softmax.")

    return K.astype(np.float32)


def make_knn_weights(cos_sim: np.ndarray, k: int) -> np.ndarray:
    """Binary (0/1): 1 for the k most similar neighbours per column, column-normalised."""
    n = cos_sim.shape[0]
    k = min(k, n - 1)
    W = np.zeros((n, n), dtype=np.float32)
    top_k = np.argpartition(-cos_sim, k, axis=0)[:k, :]
    cols = np.tile(np.arange(n), (k, 1))
    W[top_k, cols] = 1.0
    col_sums = W.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    return (W / col_sums).astype(np.float32)


def employment_weighted_density(M_t: np.ndarray, cos_sim: np.ndarray) -> np.ndarray:
    """
    Standard employment-weighted relatedness density, leave-one-out:
      D[m,j] = Σ_{i≠j} M_t[m,i]·cos_sim[i,j]  /  Σ_{i≠j} M_t[m,i]
    No temperature — plain weighted average of cosine similarity to
    industry j over the region's *other* employed industries.
    """
    cs = cos_sim.copy()
    np.fill_diagonal(cs, 0.0)
    W = M_t.sum(axis=1, keepdims=True)
    denom = W - M_t  # leave-one-out: total employment minus industry j itself
    denom_safe = np.where(denom <= 0, 1.0, denom)
    return (M_t @ cs) / denom_safe


def geometric_mean_employment_weighted_density(
    M_t: np.ndarray, cos_sim: np.ndarray, eps: float = 1e-12,
) -> np.ndarray:
    """
    Geometric-mean analogue of employment_weighted_density(), leave-one-out:
      D[m,j] = exp( Σ_{i≠j} M_t[m,i]·log(s[i,j]) / Σ_{i≠j} M_t[m,i] )
    where s = (cos_sim+1)/2 rescales cosine similarity from [-1,1] into
    (0,1] so the weighted geometric mean is well-defined (a weighted
    geometric mean needs strictly positive terms; cosine similarity itself
    can be negative). Unlike the arithmetic mean, the geometric mean is
    dragged down hard by any single dissimilar (low-s) neighbour that
    carries material employment weight, rather than averaging it away —
    a stricter reading of "close to the WHOLE local mix".
    """
    log_s = np.log(np.clip((cos_sim + 1.0) / 2.0, eps, 1.0))
    np.fill_diagonal(log_s, 0.0)  # excluded via the leave-one-out denominator below
    W = M_t.sum(axis=1, keepdims=True)
    denom = W - M_t  # leave-one-out: total weight minus industry j itself
    denom_safe = np.where(denom <= 0, 1.0, denom)
    return np.exp((M_t @ log_s) / denom_safe)


def empl_matrix(
    empl_df,
    all_municipalities: list[str],
    year: int,
    codes: list[str],
    year_col: str = "Year",
    industry_col: str = "sni_4d",
    municipality_col: str = "municipality",
    value_col: str = "workers",
) -> np.ndarray:
    """Return (n_muni, n_codes) employment matrix — zero for absent pairs."""
    sub = empl_df[(empl_df[year_col] == year) & (empl_df[industry_col].isin(codes))]
    piv = sub.pivot_table(
        index=municipality_col, columns=industry_col,
        values=value_col, aggfunc="sum", fill_value=0,
    )
    return (
        piv.reindex(index=all_municipalities, columns=codes, fill_value=0)
           .values.astype(np.float32)
    )
