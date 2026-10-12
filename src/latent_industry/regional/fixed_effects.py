"""
Linear probability models of a regional outcome on relatedness density with
exactly absorbed fixed effects and two-way clustered standard errors.

Specifications (SPECS)
----------------------
  main      municipality×year + industry fixed effects (main specification)
  additive  municipality + year + industry fixed effects
  muni_ind  municipality×industry + year fixed effects

Fixed effects are absorbed by alternating projections iterated to
convergence; singleton observations are dropped first. Standard errors are
clustered two-way by municipality and industry (Cameron, Gelbach & Miller
2011): V = V_muni + V_industry − V_muni∩industry, each with a G/(G−1)
factor; a non-positive-definite estimate is repaired by clipping negative
eigenvalues. p-values and confidence intervals use a t distribution with
min(G_muni, G_industry) − 1 degrees of freedom.

Models are estimated separately for each classification period (SNI92,
SNI02, SNI07), because code strings overlap across versions but denote
different industries. density_effect gives one slope per period;
annual_density_effects gives year-specific slopes (density × year
indicators) within the same model and fixed effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SPECS: dict[str, dict] = {
    "main":     {"fe": [("municipality", "year"), ("industry",)],
                 "label": "Municipality×year + industry FE"},
    "additive": {"fe": [("municipality",), ("year",), ("industry",)],
                 "label": "Municipality + year + industry FE"},
    "muni_ind": {"fe": [("municipality", "industry"), ("year",)],
                 "label": "Municipality×industry + year FE"},
}
CLUSTERS = ("municipality", "industry")
TOL = 1e-10
MAX_ITER = 10_000


def standardise_density(panel: pd.DataFrame, col: str = "density", out: str = "density_std") -> float:
    """Divide density by its SD pooled over all periods (a fixed unit, so β is
    the effect of a one-SD increase). Adds `out` in place and returns the SD."""
    sd = float(panel[col].std())
    panel[out] = panel[col] / sd
    return sd


def _group_codes(df: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    return df.groupby(list(cols), sort=False, observed=True).ngroup().to_numpy()


def drop_singletons(df: pd.DataFrame, fe: list[tuple[str, ...]]) -> pd.DataFrame:
    """Iteratively drop observations that are alone in any fixed-effect group."""
    while True:
        keep = np.ones(len(df), dtype=bool)
        for cols in fe:
            sizes = df.groupby(list(cols), sort=False, observed=True)[cols[0]].transform("size")
            keep &= sizes.to_numpy() > 1
        if keep.all():
            return df
        df = df.loc[keep]


def absorb(M: np.ndarray, codes: list[np.ndarray], tol: float = TOL, max_iter: int = MAX_ITER) -> np.ndarray:
    """Residualise every column of M on all fixed effects (alternating projections)."""
    R = np.array(M, dtype=np.float64, copy=True)
    if R.ndim == 1:
        R = R[:, None]
    counts = [np.bincount(c) for c in codes]
    scale = np.maximum(R.std(axis=0), 1e-12)
    for _ in range(max_iter):
        max_change = 0.0
        for c, n in zip(codes, counts):
            for k in range(R.shape[1]):
                step = (np.bincount(c, weights=R[:, k]) / n)[c]
                R[:, k] -= step
                max_change = max(max_change, float(np.abs(step).max() / scale[k]))
        if max_change < tol:
            return R
    raise RuntimeError(f"fixed effects did not converge in {max_iter} iterations")


def _cluster_meat(scores: np.ndarray, cl: np.ndarray) -> tuple[np.ndarray, int]:
    g = int(cl.max()) + 1
    S = np.zeros((g, scores.shape[1]))
    np.add.at(S, cl, scores)
    return S.T @ S * g / (g - 1), g


def fit(df: pd.DataFrame, outcome: str, regressors: list[str], fe: list[tuple[str, ...]],
        clusters: tuple[str, str] = CLUSTERS) -> dict:
    """OLS of `outcome` on `regressors` with absorbed fixed effects `fe` and
    two-way clustered SE. Returns coefficients and inference per regressor."""
    df = drop_singletons(df, fe)
    codes = [_group_codes(df, cols) for cols in fe]
    Z = absorb(df[[outcome, *regressors]].to_numpy(np.float64), codes)
    y, X = Z[:, 0], Z[:, 1:]
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    scores = X * resid[:, None]

    c1 = pd.factorize(df[clusters[0]])[0]
    c2 = pd.factorize(df[clusters[1]])[0]
    c12 = pd.factorize(c1.astype(np.int64) * (c2.max() + 1) + c2)[0]
    meat1, g1 = _cluster_meat(scores, c1)
    meat2, g2 = _cluster_meat(scores, c2)
    meat12, _ = _cluster_meat(scores, c12)
    V = XtX_inv @ (meat1 + meat2 - meat12) @ XtX_inv
    w, U = np.linalg.eigh((V + V.T) / 2)
    V = (U * np.clip(w, 0, None)) @ U.T
    se = np.sqrt(np.diag(V))

    dof = min(g1, g2) - 1
    t = beta / np.where(se > 0, se, np.nan)
    crit = stats.t.ppf(0.975, dof)
    sst = float(y @ y)
    return {
        "coef": beta, "se": se, "t": t, "pval": 2 * stats.t.sf(np.abs(t), dof),
        "ci_low": beta - crit * se, "ci_high": beta + crit * se, "df": dof,
        "n": len(df), "g_muni": g1, "g_ind": g2,
        "r2_within": 1 - float(resid @ resid) / sst if sst > 0 else np.nan,
        "x_within_var_share": X.var(axis=0) / df[regressors].to_numpy().var(axis=0),
        "rate": float(df[outcome].mean()),
    }


def density_effect(df: pd.DataFrame, outcome: str, density_col: str = "density_std",
                   spec: str = "main") -> dict:
    """Slope of `outcome` on standardised density in one period (scalars)."""
    r = fit(df, outcome, [density_col], SPECS[spec]["fe"])
    out = {k: (float(v[0]) if isinstance(v, np.ndarray) else v) for k, v in r.items()}
    out["partial_r"] = float(np.sign(out["coef"]) * np.sqrt(max(out["r2_within"], 0.0)))
    out["pct_of_baseline"] = out["coef"] / out["rate"] * 100 if out["rate"] > 0 else float("nan")
    return out


def annual_density_effects(df: pd.DataFrame, outcome: str, density_col: str = "density_std",
                           spec: str = "main") -> pd.DataFrame:
    """Year-specific slopes within one period: `outcome` on density × 1[year = t]
    for every base year t, with the specification's fixed effects."""
    df = df.copy()
    years = sorted(df["year"].unique())
    cols = []
    for t in years:
        col = f"_d_{t}"
        df[col] = df[density_col] * (df["year"] == t)
        cols.append(col)
    r = fit(df, outcome, cols, SPECS[spec]["fe"])
    n_by_year = df.groupby("year").size()
    return pd.DataFrame({
        "year": years, "coef": r["coef"], "se": r["se"], "t": r["t"], "pval": r["pval"],
        "ci_low": r["ci_low"], "ci_high": r["ci_high"], "df": r["df"],
        "n": [int(n_by_year[t]) for t in years],
    })


def estimate_by_period(panel: pd.DataFrame, outcome: str, density_col: str = "density_std",
                       spec: str = "main", verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    """Period slopes and year-specific slopes for every classification period.

    Returns (results, year_df): results[period] from density_effect;
    year_df with one row per base year (year, period, coef, se, pval, ci_low, ci_high, n).
    """
    results, frames = {}, []
    for period in dict.fromkeys(panel["period"]):
        sub = panel[panel["period"] == period]
        res = density_effect(sub, outcome, density_col, spec)
        results[period] = res
        yr = annual_density_effects(sub, outcome, density_col, spec)
        yr.insert(1, "period", period)
        frames.append(yr)
        if verbose:
            print(f"  {period}: β={res['coef'] * 100:+.3f} pp  SE={res['se'] * 100:.3f}  "
                  f"95% CI [{res['ci_low'] * 100:+.3f}, {res['ci_high'] * 100:+.3f}]  "
                  f"p={res['pval']:.1e}  N={res['n']:,}  base rate={res['rate']:.4%}  "
                  f"({res['pct_of_baseline']:+.1f}% of base rate)")
    return results, pd.concat(frames, ignore_index=True)
