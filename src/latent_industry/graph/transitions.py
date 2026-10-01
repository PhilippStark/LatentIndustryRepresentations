"""transitions.py — turn raw (rows, cols, weights) edge triples into a
row-stochastic transition matrix. The one normalization every layer needs
and nothing else: no cross-layer weighting, no sharpening beyond what an
individual layer builder explicitly opts into.
"""

import numpy as np


def row_normalize(rows: list, cols: list, weights: list, n: int,
                   symmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    """Dense row-normalised transition matrix for one relation, plus a bool
    mask of which rows have any outgoing mass at all ("avail")."""
    P = np.zeros((n, n), dtype=np.float64)
    if rows:
        r = np.asarray(rows, dtype=np.int64)
        c = np.asarray(cols, dtype=np.int64)
        w = np.asarray(weights, dtype=np.float64)
        if symmetric:
            r, c, w = np.concatenate([r, c]), np.concatenate([c, r]), np.concatenate([w, w])
        np.add.at(P, (r, c), w)
    row_sums = P.sum(axis=1)
    avail = row_sums > 0
    P[avail] /= row_sums[avail, None]
    return P, avail
