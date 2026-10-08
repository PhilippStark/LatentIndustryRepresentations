"""
P04_growth.py
=============
Relatedness density and local employment growth of industries (appendix).
Requires the restricted SCB individual-level register panel.

Outcome
-------
  growth_{j,m,t} = arcsinh(w_{j,m,t+1}) − arcsinh(w_{j,m,t})

for industry j, municipality m and year t (w = employment), within each
classification period (transition years 2001→2002 and 2006→2007 skipped).

Density measures
----------------
  density        base kernel density (--kernel, default softmax with --tau)
  density_near   softmax(+τ_near · cos) kernel: weight on the closest industries
  density_far    softmax(−τ_near · cos) kernel: weight on the most distant
                 industries (related vs. unrelated variety, Frenken et al. 2007)
  knn_density    mean similarity to the --knn-k nearest present industries
  density_ew     employment-weighted mean cosine similarity (leave-one-out)

Regressions use industry, municipality and year fixed effects (sequential
demeaning), SE clustered by municipality, per classification period, plus
municipality-specific slopes and maps.

Appendix result: growth_binary = 1{growth > 0} regressed on the main-text
density D^{A,log} (log-employment-weighted) with the main specification of
P04_entry.py (municipality×year + industry FE, two-way clustered SE).

Outputs: results/regional_growth/{run}{suffix}_* (figures, summaries,
coefficients, maps) and the panel under
<LATENT_DATA_ROOT>/processed/downstream_outcomes/growth/.

Usage (main run)
----------------
  python scripts/regional/P04_growth.py --run-name shared_cosine_v16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import statsmodels.api as sm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import EMBEDDING_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from latent_industry.common.municipalities import harmonization_table, load_municipality_crosswalk
from latent_industry.regional.fixed_effects import standardise_density
from latent_industry.regional.reporting import report_density_regressions
from latent_industry.regional.density import (
    load_sni_embeddings, make_kernel, make_inv_softmax_kernel, make_knn_weights,
    employment_weighted_density, geometric_mean_employment_weighted_density,
)
from latent_industry.regional.regional_figures import plot_density_effect_summary

DATA_ROOT    = Path("/lunarc/nobackup/projects/lu2025-17-39")
INDIVID_PATH = DATA_ROOT / "SCB" / "Preprocessed" / "individ_cleaned_1990-2019.parquet"
EMBED_BASE   = EMBEDDING_DIR
MUNI_XWALK   = harmonization_table(load_municipality_crosswalk())

# results_dir: small, paper-facing outputs only (figures + CSVs) — tracked in
# git. panel.parquet is a large intermediate cache (tens of MB) and goes to
# OUT_DIR on LUNARC storage instead, same convention as
# P04_startup_bridging.py — see that script's module docstring.
results_dir = project_root / "results" / "regional_growth"
results_dir.mkdir(parents=True, exist_ok=True)
OUT_DIR = PROCESSED_DATA_DIR / "downstream_outcomes" / "growth"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Defaults — edit here when running directly in VS Code ─────────────────────
DEFAULT_RUN_NAME  = "shared_cosine_v16"
DEFAULT_KERNEL    = "softmax"  # "raw" | "relu" | "softmax"
DEFAULT_TAU       = 1.0        # softmax temperature for the base density variable
DEFAULT_TAU_NEAR  = 5.0        # high τ → very peaked near cluster (related variety)
DEFAULT_KNN_K     = 50         # k for the kNN density robustness metric
DEFAULT_MIN_OBS   = 30         # min panel rows per municipality for muni-level regression

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--run-name",  default=DEFAULT_RUN_NAME,
                    help="Embedding run under EMBEDDING_DIR (train_shared_cosine_embedding.py --run-name).")
parser.add_argument("--out-suffix", default="",
                    help="Appended to OUT_NAME (output filename prefix) without touching "
                         "--run-name, so variant runs do not overwrite the main outputs.")
parser.add_argument("--kernel",    default=DEFAULT_KERNEL,
                    choices=["raw", "relu", "softmax"],
                    help="Similarity kernel for the BASE density variable.")
parser.add_argument("--tau",       type=float, default=DEFAULT_TAU,
                    help="Temperature for the base softmax kernel.")
parser.add_argument("--tau-near",  type=float, default=DEFAULT_TAU_NEAR,
                    help="Temperature for related/unrelated kernels. "
                         "High τ → very peaked near cluster (K_near). "
                         "Same τ used inverted for K_far (unrelated variety).")
parser.add_argument("--knn-k",    type=int, default=DEFAULT_KNN_K)
parser.add_argument("--min-obs",  type=int, default=DEFAULT_MIN_OBS,
                    help="Min observations for municipality-level regression.")
args, _ = parser.parse_known_args()

# OUT_NAME (not args.run_name) is used for every output path below, so
# --out-suffix can redirect a whole run's outputs to a separate namespace
# without touching --run-name (which selects the input embeddings).
OUT_NAME = args.run_name + args.out_suffix

print(f"Run: {args.run_name}   kernel: {args.kernel}"
      + (f"  τ={args.tau}" if args.kernel == "softmax" else "")
      + f"   τ_near={args.tau_near}   kNN-k: {args.knn_k}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load individual data and aggregate to (municipality, year, industry)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading {INDIVID_PATH} …")
raw = pl.read_parquet(
    INDIVID_PATH,
    columns=["Year", "Sni", "WorkplaceMunicipality"],
)

empl = (
    raw
    .filter(pl.col("Sni").is_not_null() & (pl.col("Sni") != 0))
    .filter(pl.col("WorkplaceMunicipality").is_not_null())
    .with_columns([
        pl.col("Sni").cast(pl.Utf8).str.zfill(5).str.slice(0, 4).alias("sni_4d"),
        pl.col("WorkplaceMunicipality").cast(pl.Utf8).str.zfill(4).alias("source_code"),
    ])
    .filter(pl.col("sni_4d") != "0000")
    # Inner join harmonizes historical municipality codes (e.g. the pre-1998
    # Skane/Vastra Gotaland county renumbering) to their canonical 290-code
    # successor and drops invalid/unmapped codes (0000, 9999) in one step —
    # see src/latent_industry/common/municipalities.py.
    .join(MUNI_XWALK, on=["Year", "source_code"], how="inner")
    .group_by(["municipality", "Year", "sni_4d"])
    .agg(pl.len().alias("workers"))
    .sort(["municipality", "Year", "sni_4d"])
    .to_pandas()
)

all_municipalities = sorted(empl["municipality"].unique())
print(f"  {len(empl):,} (municipality, year, industry) observations")
print(f"  Years: {empl['Year'].min()}–{empl['Year'].max()}")
print(f"  Municipalities: {len(all_municipalities)}")
print(f"  Industries in data: {empl['sni_4d'].nunique()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Load embeddings and compute kernel matrices
# ═══════════════════════════════════════════════════════════════════════════════

sni_embeddings = load_sni_embeddings(args.run_name, EMBED_BASE)
vecs07, codes07 = sni_embeddings["sni07"]
vecs02, codes02 = sni_embeddings["sni02"]
vecs92, codes92 = sni_embeddings["sni92"]

# Kernel builders (make_kernel, make_inv_softmax_kernel, make_knn_weights,
# employment_weighted_density) live in latent_industry.regional.density —
# shared with P04_entry.py / P04_exit.py / P04_startup_entry.py so the
# density definitions don't depend on this script continuing to exist.


print(f"Computing base kernel matrices  (kernel={args.kernel}, τ={args.tau}) …")
cos07 = (vecs07 @ vecs07.T).astype(np.float64)
cos02 = (vecs02 @ vecs02.T).astype(np.float64)
cos92 = (vecs92 @ vecs92.T).astype(np.float64)

K07 = make_kernel(vecs07, args.kernel, args.tau)
K02 = make_kernel(vecs02, args.kernel, args.tau)
K92 = make_kernel(vecs92, args.kernel, args.tau)

knn07 = make_knn_weights(cos07, args.knn_k)
knn02 = make_knn_weights(cos02, args.knn_k)
knn92 = make_knn_weights(cos92, args.knn_k)

print(f"Computing near/far kernels  (τ_near={args.tau_near}) …")
K07_near = make_kernel(vecs07, "softmax", args.tau_near)
K02_near = make_kernel(vecs02, "softmax", args.tau_near)
K92_near = make_kernel(vecs92, "softmax", args.tau_near)

K07_far  = make_inv_softmax_kernel(vecs07, args.tau_near)
K02_far  = make_inv_softmax_kernel(vecs02, args.tau_near)
K92_far  = make_inv_softmax_kernel(vecs92, args.tau_near)

# ── Effective neighbourhood size: N_eff(j) = 1 / Σ_i K[i,j]² ────────────────
print("\nEffective neighbourhood size  N_eff = 1/Σ K²  per industry:")
for label, K in [("base SNI07", K07), ("near SNI07", K07_near), ("far SNI07", K07_far)]:
    n_eff = 1.0 / (K.astype(np.float64) ** 2).sum(axis=0)
    print(f"  {label:12s}  min={n_eff.min():.1f}  median={np.median(n_eff):.1f}"
          f"  mean={n_eff.mean():.1f}  max={n_eff.max():.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Build panel
#    One row per (municipality, industry, year-pair) where at least one of the
#    two years has workers > 0.  arcsinh handles entry/exit/growth on one scale.
# ═══════════════════════════════════════════════════════════════════════════════

PERIODS = [
    ("SNI92", codes92, K92, knn92, K92_near, K92_far, cos92, range(1990, 2001)),
    ("SNI02", codes02, K02, knn02, K02_near, K02_far, cos02, range(2002, 2006)),
    ("SNI07", codes07, K07, knn07, K07_near, K07_far, cos07, range(2007, 2019)),
]


def empl_matrix(year: int, codes: list[str]) -> np.ndarray:
    """Return (n_muni, n_codes) employment matrix — zero for absent pairs."""
    sub = empl[(empl["Year"] == year) & (empl["sni_4d"].isin(codes))]
    piv = sub.pivot_table(
        index="municipality", columns="sni_4d",
        values="workers", aggfunc="sum", fill_value=0,
    )
    return (
        piv.reindex(index=all_municipalities, columns=codes, fill_value=0)
           .values.astype(np.float32)
    )


print("\nBuilding panel …")
panel_frames: list[pd.DataFrame] = []

for period_name, codes, K, knn_w, K_near, K_far, cos_sim, year_range in PERIODS:
    print(f"\n  {period_name}  ({len(codes)} codes):")

    for t in year_range:
        t1   = t + 1
        M_t  = empl_matrix(t,  codes)
        M_t1 = empl_matrix(t1, codes)

        W      = M_t.sum(axis=1, keepdims=True)
        W_safe = np.where(W == 0, 1.0, W)

        # Base density (softmax τ=1.0 by default) — primary single-variable result
        density     = (M_t @ K)      / W_safe
        knn_density = (M_t @ knn_w)  / W_safe

        # Related variety: near cluster (high τ softmax → concentrated on similar)
        density_near = (M_t @ K_near) / W_safe

        # Unrelated variety: distant industries (inverted softmax → concentrated on dissimilar)
        density_far  = (M_t @ K_far)  / W_safe

        # Standard employment-weighted relatedness density (no temperature,
        # leave-one-out denominator) — robustness check against the softmax
        # base density above.
        density_ew = employment_weighted_density(M_t, cos_sim)

        # Geometric-mean analogue of density_ew — same leave-one-out
        # employment weighting, but a weighted geometric (rather than
        # arithmetic) mean of rescaled cosine similarity; see
        # geometric_mean_employment_weighted_density() / module docstring.
        density_geo = geometric_mean_employment_weighted_density(M_t, cos_sim)

        # Log-employment-weighted analogue of density_ew — the baseline density
        # specification used by P04_entry.py / P04_exit.py / P04_startup_entry.py
        # (D^{A,log}). Not used by this script's own continuous multi-kernel
        # analysis above (which keeps density_ew raw-weighted, unchanged) —
        # only feeds the binarized-outcome appendix section near the end of
        # this script, so the two density_ew flavours don't get confused.
        density_ew_log = employment_weighted_density(np.log1p(M_t), cos_sim)

        growth = np.arcsinh(M_t1) - np.arcsinh(M_t)

        keep = (M_t + M_t1) > 0
        mi, ci = np.where(keep)

        panel_frames.append(pd.DataFrame({
            "period":        period_name,
            "year":          t,
            "municipality":  [all_municipalities[i] for i in mi],
            "industry":      [codes[j]              for j in ci],
            "w_t":           M_t         [mi, ci],
            "w_t1":          M_t1        [mi, ci],
            "density":       density     [mi, ci],
            "knn_density":   knn_density [mi, ci],
            "density_near":  density_near[mi, ci],
            "density_far":   density_far [mi, ci],
            "density_ew":    density_ew  [mi, ci],
            "density_geo":   density_geo [mi, ci],
            "density_ew_log": density_ew_log[mi, ci],
            "growth":        growth      [mi, ci],
        }))
        print(f"    {t}→{t+1}:  {len(panel_frames[-1]):,} rows")

panel = pd.concat(panel_frames, ignore_index=True)

n_entry = int(((panel["w_t"] == 0) & (panel["w_t1"] > 0)).sum())
n_exit  = int(((panel["w_t"] > 0) & (panel["w_t1"] == 0)).sum())
n_both  = int(((panel["w_t"] > 0) & (panel["w_t1"] > 0)).sum())
print(f"\nPanel: {len(panel):,} total rows")
print(f"  Entry (0→w):            {n_entry:,}")
print(f"  Exit  (w→0):            {n_exit:,}")
print(f"  Present in both years:  {n_both:,}")

# ── Standardise all density variables globally (÷ SD) before regression ───────
# β reads as: arcsinh-growth per 1-SD increase in the respective density measure.
_sd = {}
for var in ("density", "knn_density", "density_near", "density_far", "density_ew", "density_geo"):
    _sd[var] = float(panel[var].std())
    panel[var] = panel[var] / _sd[var]

print("\nDensity variables pre-standardised (÷ global SD):")
for var, sd in _sd.items():
    print(f"  {var:15s}  SD = {sd:.6f}")
print("  β now = arcsinh-growth per 1-SD increase in density")


def demean(df: pd.DataFrame, col: str, by: str | list[str]) -> pd.Series:
    return df[col] - df.groupby(by)[col].transform("mean")


def run_ols(y: pd.Series, x: pd.Series, cluster: pd.Series) -> dict:
    X = sm.add_constant(x.values)
    if cluster.nunique() > 1:
        res = sm.OLS(y.values, X).fit(
            cov_type="cluster", cov_kwds={"groups": cluster.values},
        )
    else:
        res = sm.OLS(y.values, X).fit(cov_type="HC3")
    ci = res.conf_int()
    r2 = float(res.rsquared)
    # Single-regressor OLS: R² is the squared (partial) correlation between x
    # and y net of the FE — its signed root gives a correlation-like number in
    # [-1, 1], comparable across periods/models unlike the raw coefficient,
    # whose scale depends on how much within-FE variance x has left. Same
    # convention as P04_entry.py / P04_exit.py.
    partial_r = float(np.sign(res.params[1]) * np.sqrt(r2))
    return {
        "coef":      float(res.params[1]),
        "se":        float(res.bse[1]),
        "t":         float(res.tvalues[1]),
        "pval":      float(res.pvalues[1]),
        "ci_low":    float(ci[1, 0]),
        "ci_high":   float(ci[1, 1]),
        "n":         int(res.nobs),
        "r2":        r2,
        "partial_r": partial_r,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. Per-classification-period regression — base density, industry +
#    municipality + year FE via triple demeaning (separate additive
#    municipality and year effects, not an interacted municipality×year
#    term), run separately per SNI period. This is the design entry/exit use
#    throughout (see P04_entry.py's docstring): SNI92/SNI02/SNI07 code
#    strings overlap heavily but refer to different classification vintages,
#    so pooling them under one industry FE (as section 4 below does, for the
#    near/far horse race and the kNN robustness check) would treat a 1995
#    code and a 2003 code that happen to share four digits as the same FE
#    level. This per-period version is the one plotted per-SNI in panel C
#    below, for consistency with entry/exit's own headline figure.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Per-period base-density regression  (industry + municipality + year FE, SE clustered by muni)")
print("=" * 60)

periods_order  = list(dict.fromkeys(panel["period"]))
growth_results: dict[str, dict] = {}

for period_name in periods_order:
    sub = panel[panel["period"] == period_name].copy()
    for var in ("growth", "density"):
        sub[f"{var}_r1"] = demean(sub, var,          "municipality")
        sub[f"{var}_r2"] = demean(sub, f"{var}_r1",  "year")
        sub[f"{var}_dm"] = demean(sub, f"{var}_r2",  "industry")
    res = run_ols(sub["growth_dm"], sub["density_dm"], sub["municipality"])
    # arcsinh(w_t1) - arcsinh(w_t) ~ ln(w_t1/w_t) away from zero (arcsinh(x) ~
    # ln(2x) for x >> 1), so exp(coef)-1 reads as an approximate proportional
    # growth effect per 1-SD density — entry/exit/growth's shared analog of
    # entry/exit's "% of baseline rate" label. It is only an approximation
    # (breaks down for cells near zero, i.e. entry/exit rows, which is most of
    # why growth's own baseline mean is itself near zero and unusable as a
    # denominator the way entry/exit's binary outcome rate is).
    res["pct_effect"] = (np.exp(res["coef"]) - 1) * 100
    growth_results[period_name] = res
    print(f"  {period_name}:  coef={res['coef']:+.6f}  SE={res['se']:.6f}  "
          f"t={res['t']:.3f}  p={res['pval']:.3e}  N={res['n']:,}  "
          f"partial r={res['partial_r']:+.4f}  ≈{res['pct_effect']:+.1f}% per 1-SD density")

# ── 3c. Same per-period regression, but for density_ew — the standard
#    employment-weighted relatedness density (no temperature, leave-one-out
#    denominator; see employment_weighted_density() / module docstring above).
#    Directly comparable to the softmax base-density result above since it's
#    the same FE spec on the same panel rows, just a different density variable.
print("\n" + "=" * 60)
print("Per-period density_ew regression  (employment-weighted, leave-one-out; industry + municipality + year FE)")
print("=" * 60)

growth_results_ew: dict[str, dict] = {}

for period_name in periods_order:
    sub = panel[panel["period"] == period_name].copy()
    for var in ("growth", "density_ew"):
        sub[f"{var}_r1"] = demean(sub, var,          "municipality")
        sub[f"{var}_r2"] = demean(sub, f"{var}_r1",  "year")
        sub[f"{var}_dm"] = demean(sub, f"{var}_r2",  "industry")
    res = run_ols(sub["growth_dm"], sub["density_ew_dm"], sub["municipality"])
    res["pct_effect"] = (np.exp(res["coef"]) - 1) * 100
    growth_results_ew[period_name] = res
    print(f"  {period_name}:  coef={res['coef']:+.6f}  SE={res['se']:.6f}  "
          f"t={res['t']:.3f}  p={res['pval']:.3e}  N={res['n']:,}  "
          f"partial r={res['partial_r']:+.4f}  ≈{res['pct_effect']:+.1f}% per 1-SD density_ew")

print("\n  Softmax density vs. employment-weighted density (density_ew), per period:")
print(f"  {'period':8s}  {'β softmax':>12s}  {'β density_ew':>14s}  {'r softmax':>10s}  {'r density_ew':>13s}")
for period_name in periods_order:
    a, b = growth_results[period_name], growth_results_ew[period_name]
    print(f"  {period_name:8s}  {a['coef']:+12.6f}  {b['coef']:+14.6f}  "
          f"{a['partial_r']:+10.4f}  {b['partial_r']:+13.4f}")

# ── 3d. Same per-period regression, but for density_geo — the geometric-mean
#    analogue of density_ew (see geometric_mean_employment_weighted_density()
#    / module docstring above). Same FE spec, same panel rows.
print("\n" + "=" * 60)
print("Per-period density_geo regression  (geometric-mean employment-weighted; industry + municipality + year FE)")
print("=" * 60)

growth_results_geo: dict[str, dict] = {}

for period_name in periods_order:
    sub = panel[panel["period"] == period_name].copy()
    for var in ("growth", "density_geo"):
        sub[f"{var}_r1"] = demean(sub, var,          "municipality")
        sub[f"{var}_r2"] = demean(sub, f"{var}_r1",  "year")
        sub[f"{var}_dm"] = demean(sub, f"{var}_r2",  "industry")
    res = run_ols(sub["growth_dm"], sub["density_geo_dm"], sub["municipality"])
    res["pct_effect"] = (np.exp(res["coef"]) - 1) * 100
    growth_results_geo[period_name] = res
    print(f"  {period_name}:  coef={res['coef']:+.6f}  SE={res['se']:.6f}  "
          f"t={res['t']:.3f}  p={res['pval']:.3e}  N={res['n']:,}  "
          f"partial r={res['partial_r']:+.4f}  ≈{res['pct_effect']:+.1f}% per 1-SD density_geo")

print("\n  Employment-weighted density_ew vs. geometric-mean density_geo, per period:")
print(f"  {'period':8s}  {'β density_ew':>14s}  {'β density_geo':>15s}  {'r density_ew':>13s}  {'r density_geo':>14s}")
for period_name in periods_order:
    a, b = growth_results_ew[period_name], growth_results_geo[period_name]
    print(f"  {period_name:8s}  {a['coef']:+14.6f}  {b['coef']:+15.6f}  "
          f"{a['partial_r']:+13.4f}  {b['partial_r']:+14.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Triple demeaning — absorbs municipality FE, then year FE, then industry
#    FE (separate additive municipality and year effects, not an interacted
#    municipality×year term), POOLED across all SNI periods. Fine for the
#    near/far horse race and the kNN robustness check below, since those
#    treat SNI codes as opaque FE levels and are not compared across
#    periods; NOT used for the headline density → growth effect, which is
#    per-period above.
# ═══════════════════════════════════════════════════════════════════════════════

for var in ("growth", "density", "knn_density", "density_near", "density_far", "density_ew", "density_geo"):
    panel[f"{var}_r1"] = demean(panel, var,          "municipality")
    panel[f"{var}_r2"] = demean(panel, f"{var}_r1",  "year")
    panel[f"{var}_dm"] = demean(panel, f"{var}_r2",  "industry")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Overall panel regression — base density
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Overall panel regression  (municipality + year + industry FE, SE clustered by muni)")
print("=" * 60)

res_main = run_ols(panel["growth_dm"], panel["density_dm"],     panel["municipality"])
res_knn  = run_ols(panel["growth_dm"], panel["knn_density_dm"], panel["municipality"])
res_ew   = run_ols(panel["growth_dm"], panel["density_ew_dm"],  panel["municipality"])
res_geo  = run_ols(panel["growth_dm"], panel["density_geo_dm"], panel["municipality"])

for label, res in [("soft density (base)", res_main), (f"kNN density (k={args.knn_k})", res_knn),
                    ("employment-weighted density (density_ew)", res_ew),
                    ("geometric-mean employment-weighted density (density_geo)", res_geo)]:
    print(f"\n  {label}")
    print(f"    coef={res['coef']:+.6f}  SE={res['se']:.6f}"
          f"  t={res['t']:.3f}  p={res['pval']:.3e}")
    print(f"    95% CI [{res['ci_low']:+.6f}, {res['ci_high']:+.6f}]")
    print(f"    N={res['n']:,}  R²={res['r2']:.5f}")
print("\n  β = arcsinh-growth per 1-SD increase in density (density pre-standardised)")


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. Related vs Unrelated variety — horse race
#
#   density_near  = softmax(+τ_near): K concentrated on CLOSEST industries
#                   → tests Marshall externalities / related variety
#   density_far   = softmax(−τ_near): K concentrated on MOST DISTANT industries
#                   → tests Jacobs externalities / unrelated variety
#
#   Horse race: both in same regression; each coefficient is conditional on
#   the other being in the model, so the estimates are orthogonalised.
#
#   Interpretation:
#     β_near > 0, β_far ≈ 0  → related variety drives growth
#     β_far  > 0, β_near ≈ 0 → unrelated variety / Jacobs externalities
#     both significant        → both parts of the similarity spectrum matter
#     both ≈ 0                → no density effect (or collinearity, check N_eff)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(f"Related vs Unrelated variety  (τ_near = {args.tau_near})")
print("=" * 60)

res_near = run_ols(panel["growth_dm"], panel["density_near_dm"], panel["municipality"])
res_far  = run_ols(panel["growth_dm"], panel["density_far_dm"],  panel["municipality"])

print(f"\n  Related variety alone   (softmax +τ={args.tau_near} → near cluster)")
print(f"    coef={res_near['coef']:+.6f}  SE={res_near['se']:.6f}"
      f"  t={res_near['t']:.3f}  p={res_near['pval']:.3e}")

print(f"\n  Unrelated variety alone (softmax −τ={args.tau_near} → distant industries)")
print(f"    coef={res_far['coef']:+.6f}  SE={res_far['se']:.6f}"
      f"  t={res_far['t']:.3f}  p={res_far['pval']:.3e}")

# Horse race: both density_near and density_far in the same model
X_hr = sm.add_constant(
    panel[["density_near_dm", "density_far_dm"]].values
)
if panel["municipality"].nunique() > 1:
    res_hr = sm.OLS(panel["growth_dm"].values, X_hr).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["municipality"].values},
    )
else:
    res_hr = sm.OLS(panel["growth_dm"].values, X_hr).fit(cov_type="HC3")

ci_hr = res_hr.conf_int()
print(f"\n  Horse race (both in same model)  N={int(res_hr.nobs):,}  R²={res_hr.rsquared:.5f}")
for k, label in [(1, f"related  (near, +τ={args.tau_near})"),
                 (2, f"unrelated (far,  −τ={args.tau_near})")]:
    print(f"    {label:40s}  coef={res_hr.params[k]:+.6f}  "
          f"SE={res_hr.bse[k]:.6f}  p={res_hr.pvalues[k]:.3e}  "
          f"CI [{ci_hr[k, 0]:+.6f}, {ci_hr[k, 1]:+.6f}]")

# Correlation between the two density measures (post-demeaning) for diagnostics
rho = float(panel[["density_near_dm", "density_far_dm"]].corr().iloc[0, 1])
print(f"\n  Correlation(density_near_dm, density_far_dm) = {rho:+.4f}")
print("  (high |ρ| suggests collinearity — interpret horse-race CIs cautiously)")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Year-by-year regression
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Year-by-year coefficients")
print("=" * 60)

year_records: list[dict] = []
for year, sub in panel.groupby("year"):
    sub = sub.copy()
    for var in ("growth", "density"):
        sub[f"{var}_r1"] = demean(sub, var,          ["municipality"])
        sub[f"{var}_dm"] = demean(sub, f"{var}_r1",  ["industry"])
    r         = run_ols(sub["growth_dm"], sub["density_dm"], sub["municipality"])
    r["year"] = int(year)
    year_records.append(r)
    print(f"  {year}→{year+1}:  coef={r['coef']:+.5f}  SE={r['se']:.5f}"
          f"  p={r['pval']:.3e}  N={r['n']:,}")

year_df = pd.DataFrame(year_records).sort_values("year")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Municipality-level heterogeneity
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(f"Municipality-level slopes  (min_obs={args.min_obs})")
print("=" * 60)

muni_records: list[dict] = []
for muni, sub in panel.groupby("municipality"):
    if len(sub) < args.min_obs:
        continue
    sub = sub.copy()
    for var in ("growth", "density"):
        sub[f"{var}_r1"] = demean(sub, var,          ["year"])
        sub[f"{var}_dm"] = demean(sub, f"{var}_r1",  ["industry"])
    r                 = run_ols(sub["growth_dm"], sub["density_dm"], sub["municipality"])
    r["municipality"] = muni
    r["n_obs"]        = len(sub)
    muni_records.append(r)

muni_df  = pd.DataFrame(muni_records)
pos_frac = float((muni_df["coef"] > 0).mean())
sig_pos  = int(((muni_df["coef"] > 0) & (muni_df["pval"] < 0.05)).sum())
sig_neg  = int(((muni_df["coef"] < 0) & (muni_df["pval"] < 0.05)).sum())
print(f"  Municipalities with enough obs: {len(muni_df)}")
print(f"  Positive slope:   {pos_frac:.1%}")
print(f"  Sig. positive:    {sig_pos}    Sig. negative: {sig_neg}")
print(f"  Median coef:      {muni_df['coef'].median():.5f}")
print(f"  IQR:              [{muni_df['coef'].quantile(0.25):.5f}, "
      f"{muni_df['coef'].quantile(0.75):.5f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Visualisations
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# A. Density VENTILE (q=20, finer than entry/exit's deciles — growth's
# near/far dip in earlier decile-level runs was worth resolving more finely)
# vs mean arcsinh growth, per classification period — same grouped-bar
# layout as P04_entry.py's panel A, since density's
# distribution differs by period (raw, un-demeaned density/growth here, by
# design — this panel shows the marginal pattern, not the FE-adjusted one).
N_BINS = 20
ax = axes[0]
width = 0.7 / len(periods_order)
for k, period_name in enumerate(periods_order):
    sub = panel[panel["period"] == period_name]
    bin_cat = pd.qcut(sub["density"], q=N_BINS, duplicates="drop")
    bin_id  = bin_cat.cat.codes + 1
    grp     = sub["growth"].groupby(bin_id)
    offset  = (k - (len(periods_order) - 1) / 2) * width
    ax.bar(grp.mean().index + offset, grp.mean().values, width=width, label=period_name)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(range(1, N_BINS + 1, 2))
ax.set_xlabel(f"Density ventile (q={N_BINS})  (1 = far from local mass  →  {N_BINS} = close, within period)")
ax.set_ylabel("Mean arcsinh growth")
ax.set_title("A. Growth rate by density")
ax.legend(fontsize=8, title="period")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# B. Year-by-year coefficient
ax = axes[1]
yr = year_df["year"] + 0.5
ax.fill_between(yr, year_df["ci_low"], year_df["ci_high"], alpha=0.25, color="steelblue")
ax.plot(yr, year_df["coef"], "o-", color="steelblue", linewidth=1.8, markersize=5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
for x_line in (2001.5, 2006.5):
    ax.axvline(x_line, color="grey", linewidth=1, linestyle=":")
ax.set_xlabel("Year")
ax.set_ylabel("β  (density → arcsinh growth)")
# muni + year + industry FE, clustered SE — methodological detail belongs in the
# paper's figure caption, not baked into the on-image title (matches
# P04_entry.py/P04_exit.py's panel-title convention).
ax.set_title("B. Effect over time")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# C. Per-period base-density effect — same layout as evaluate_industry_
# entry.py / P04_exit.py's panel C: one bar per SNI period,
# from the per-period FE regression in section 3b (NOT the pooled section-4/5
# regression, which mixes SNI vintages under one industry FE). Industry +
# municipality + year FE, clustered SE, within period.
ax = axes[2]
coefs = [growth_results[p]["coef"] for p in periods_order]
ses   = [growth_results[p]["se"]   for p in periods_order]
colors = ["steelblue" if c > 0 else "crimson" for c in coefs]
bars = ax.bar(periods_order, coefs, yerr=[1.96 * s for s in ses],
              color=colors, alpha=0.8, capsize=6, edgecolor="white", width=0.5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Classification period")
ax.set_ylabel("β  (arcsinh growth per 1-SD density)")
ax.set_title("C. Effect by period")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

ax.relim()
ax.autoscale_view()
y0, y1 = ax.get_ylim()
pad = 0.08 * (y1 - y0)
for bar, period_name in zip(bars, periods_order):
    res  = growth_results[period_name]
    coef = res["coef"]
    y_text = res["ci_high"] + pad if coef >= 0 else res["ci_low"] - pad
    ax.text(bar.get_x() + bar.get_width() / 2, y_text,
            f"r={res['partial_r']:+.3f}\n({res['pct_effect']:+.1f}% approx)",
            ha="center", va="bottom" if coef >= 0 else "top", fontsize=8)
ax.set_ylim(y0 - 2.4 * pad, y1 + 2.4 * pad)

kernel_label = args.kernel + (f" τ={args.tau}" if args.kernel == "softmax" else "")
print(f"  Embedding density → local industry growth  (run={args.run_name}, kernel={kernel_label})")
# No overall suptitle — run/kernel identification belongs in the paper's
# figure caption, not the image itself (matches P04_entry.py/P04_exit.py).
plt.tight_layout()
fig_path = results_dir / f"{OUT_NAME}_regional_density.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n  → {fig_path}")
plt.close()

# C2. Municipality-level slope distribution — still computed (section 7) and
# saved to muni_coefs.csv / used for the maps below, but moved to its own
# figure now that the main figure's panel C is the per-period bar chart.
fig, ax = plt.subplots(figsize=(6, 5))
ax.hist(muni_df["coef"], bins=40, color="steelblue", alpha=0.8, edgecolor="white")
ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.axvline(muni_df["coef"].median(), color="crimson", linewidth=1.5, linestyle="--",
           label=f"median = {muni_df['coef'].median():.4f}")
ax.set_xlabel("Municipality-level β  (density → growth)")
ax.set_ylabel("Count")
# run/n/%-positive (printed above) belong in the caption, not the on-image
# title (matches P04_entry.py/P04_exit.py's convention) — the median line is
# still labelled via the legend below.
ax.set_title("Distribution of local density slopes")
ax.legend(fontsize=8)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
muni_dist_path = results_dir / f"{OUT_NAME}_muni_slope_distribution.png"
fig.savefig(muni_dist_path, dpi=150, bbox_inches="tight")
print(f"  → {muni_dist_path}")
plt.close()

# D. Related vs unrelated variety comparison
fig, ax = plt.subplots(figsize=(7, 4))
labels  = [f"Related\n(near, +τ={args.tau_near})", f"Unrelated\n(far, −τ={args.tau_near})"]
coefs   = [res_near["coef"], res_far["coef"]]
ses     = [res_near["se"],   res_far["se"]]
colors  = ["steelblue" if c > 0 else "crimson" for c in coefs]
bars    = ax.bar(labels, coefs, yerr=[1.96 * s for s in ses],
                 color=colors, alpha=0.8, capsize=6, edgecolor="white", width=0.5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
for bar, coef, se in zip(bars, coefs, ses):
    ax.text(bar.get_x() + bar.get_width() / 2,
            coef + (1.96 * se + abs(coef) * 0.05) * np.sign(coef),
            f"β={coef:+.4f}\np={res_near['pval']:.2e}" if bar is bars[0]
            else f"β={coef:+.4f}\np={res_far['pval']:.2e}",
            ha="center", va="bottom" if coef >= 0 else "top", fontsize=8)
ax.set_ylabel("β  (arcsinh growth per 1-SD density)")
print(f"  Related vs Unrelated variety  |  run={args.run_name}  |  τ_near={args.tau_near}  "
      f"(municipality + year + industry FE, clustered SE, N={res_near['n']:,})")
# run/τ/FE spec belong in the caption, not the on-image title.
ax.set_title("Related vs Unrelated variety")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
rv_path = results_dir / f"{OUT_NAME}_related_vs_unrelated.png"
fig.savefig(rv_path, dpi=150, bbox_inches="tight")
print(f"  → {rv_path}")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Save outputs
# ═══════════════════════════════════════════════════════════════════════════════

panel_out = panel[["period", "year", "municipality", "industry",
                   "w_t", "w_t1", "density", "knn_density",
                   "density_near", "density_far", "density_ew", "density_geo", "growth"]]
panel_out.to_parquet(OUT_DIR / f"{OUT_NAME}_panel.parquet", index=False)
print(f"  → {OUT_DIR / f'{OUT_NAME}_panel.parquet'}  ({len(panel):,} rows)")

year_df.to_csv(results_dir / f"{OUT_NAME}_year_coefs.csv", index=False)
muni_df.to_csv(results_dir / f"{OUT_NAME}_muni_coefs.csv", index=False)

period_summary_df = pd.DataFrame([
    {"run": args.run_name, "period": p, "density_var": "softmax", **growth_results[p]}
    for p in periods_order
] + [
    {"run": args.run_name, "period": p, "density_var": "density_ew", **growth_results_ew[p]}
    for p in periods_order
] + [
    {"run": args.run_name, "period": p, "density_var": "density_geo", **growth_results_geo[p]}
    for p in periods_order
])
period_summary_df.to_csv(results_dir / f"{OUT_NAME}_period_summary.csv", index=False)

pd.DataFrame([{
    "run":              args.run_name,
    "kernel":           args.kernel,
    "tau":              args.tau if args.kernel == "softmax" else None,
    "tau_near":         args.tau_near,
    "knn_k":            args.knn_k,
    "n_obs":            len(panel),
    "n_entry":          n_entry,
    "n_exit":           n_exit,
    "n_municipalities": len(all_municipalities),
    "overall_coef":     res_main["coef"],
    "overall_se":       res_main["se"],
    "overall_pval":     res_main["pval"],
    "knn_coef":         res_knn["coef"],
    "knn_pval":         res_knn["pval"],
    "ew_coef":          res_ew["coef"],
    "ew_pval":          res_ew["pval"],
    "geo_coef":         res_geo["coef"],
    "geo_pval":         res_geo["pval"],
    "near_coef":        res_near["coef"],
    "near_pval":        res_near["pval"],
    "far_coef":         res_far["coef"],
    "far_pval":         res_far["pval"],
    "hr_near_coef":     float(res_hr.params[1]),
    "hr_near_pval":     float(res_hr.pvalues[1]),
    "hr_far_coef":      float(res_hr.params[2]),
    "hr_far_pval":      float(res_hr.pvalues[2]),
    "near_far_corr":    rho,
    "muni_pos_frac":    pos_frac,
    "muni_sig_pos":     sig_pos,
    "muni_sig_neg":     sig_neg,
}]).to_csv(results_dir / f"{OUT_NAME}_summary.csv", index=False)

print(f"  → {results_dir}/  (year_coefs.csv, muni_coefs.csv, summary.csv)")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Per-(municipality × year) slopes — hover info for the interactive map.
#     NOTE: ~30–100 obs per cell, true β ≈ 0.03 → SE >> β → very noisy.
#     Only shown on hover in the slider, NOT used for colour.
# ═══════════════════════════════════════════════════════════════════════════════

print("\nComputing municipality × year slopes …")
map_records: list[dict] = []

for year, yr_data in panel.groupby("year"):
    yr_data = yr_data.copy()
    yr_data["dens_dm"] = yr_data["density"] - yr_data.groupby("industry")["density"].transform("mean")
    yr_data["grow_dm"] = yr_data["growth"]  - yr_data.groupby("industry")["growth"].transform("mean")

    for muni, m_data in yr_data.groupby("municipality"):
        if len(m_data) < args.min_obs:
            continue
        x = m_data["dens_dm"].values
        y = m_data["grow_dm"].values
        var_x = np.var(x)
        if var_x == 0:
            continue
        coef = float(np.cov(x, y, ddof=1)[0, 1] / var_x)
        map_records.append({
            "municipality": muni,
            "year":         int(year),
            "coef":         coef,
            "n":            len(m_data),
        })

map_df = pd.DataFrame(map_records)
print(f"  {len(map_df):,} (municipality, year) cells  "
      f"({map_df['municipality'].nunique()} municipalities × {map_df['year'].nunique()} years)")
map_df.to_csv(results_dir / f"{OUT_NAME}_map_coefs.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Maps
#   11a. Static PNG: pooled municipality β  (stable, derived from section 7)
#   11b. Static PNG per year: mean arcsinh growth (directly observable)
#   11c. HTML slider: same data as 11b, one frame per year.
#        Colour = mean arcsinh growth (changes per frame).
#        Hover  = municipality name  +  per-(muni,year) β from section 10.
#        Title  = year-specific overall β from year_df (section 6).
# ═══════════════════════════════════════════════════════════════════════════════

import json
import geopandas as gpd

SHP_PATH = RAW_DATA_DIR / "shape_files" / "Municipalities_LargeScale.shp"
gdf_raw  = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
gdf_raw["municipality"] = gdf_raw["astkommun"].astype(str).str.zfill(4)
gdf_base = gdf_raw[["municipality", "KOMMUNNAMN", "geometry"]].copy()

maps_dir = results_dir / "maps"
maps_dir.mkdir(exist_ok=True)

# ── 11a. Static PNG: pooled municipality β ────────────────────────────────────
clim_pool = 2.0 * muni_df["coef"].std()
gdf_pool  = gdf_base.merge(muni_df[["municipality", "coef", "n_obs"]], on="municipality", how="left")

fig, ax = plt.subplots(1, 1, figsize=(6, 9))
gdf_base.plot(ax=ax, color="#e8e8e8", edgecolor="white", linewidth=0.2)
gdf_pool.dropna(subset=["coef"]).plot(
    ax=ax, column="coef",
    cmap="RdBu", vmin=-clim_pool, vmax=clim_pool,
    edgecolor="white", linewidth=0.2, legend=True,
    legend_kwds={"label": "β  (per 1-SD density, pooled 1990–2019)", "shrink": 0.6},
)
n_pos = (muni_df["coef"] > 0).sum()
ax.set_title(
    f"Embedding density → employment growth\n"
    f"Pooled 1990–2019  ({n_pos}/{len(muni_df)} municipalities positive)",
    fontsize=10, pad=6,
)
ax.axis("off")
plt.tight_layout()
pool_map_path = maps_dir / f"{OUT_NAME}_map_pooled.png"
fig.savefig(pool_map_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  → {pool_map_path}")

# ── 11b. Static PNG per year: mean employment growth per municipality ──────────
print(f"Saving per-year growth maps → {maps_dir}/")

muni_growth_yr = (
    panel.groupby(["municipality", "year"])["growth"]
    .mean().reset_index().rename(columns={"growth": "mean_growth"})
)
g_clim       = 2.0 * muni_growth_yr["mean_growth"].std()
years_sorted = sorted(muni_growth_yr["year"].unique())

for year in years_sorted:
    yr_g   = muni_growth_yr[muni_growth_yr["year"] == year]
    gdf_yr = gdf_base.merge(yr_g, on="municipality", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(6, 9))
    gdf_base.plot(ax=ax, color="#e8e8e8", edgecolor="white", linewidth=0.2)
    gdf_yr.dropna(subset=["mean_growth"]).plot(
        ax=ax, column="mean_growth",
        cmap="RdBu_r", vmin=-g_clim, vmax=g_clim,
        edgecolor="white", linewidth=0.2, legend=True,
        legend_kwds={"label": "Mean arcsinh growth", "shrink": 0.6},
    )
    ax.set_title(f"{year}→{year+1}  |  mean industry employment growth", fontsize=10, pad=6)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(maps_dir / f"{OUT_NAME}_growth_{year}.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  {year}→{year+1}", end="  ", flush=True)
print()

# ── 11c. HTML slider — per-year mean growth colour, name + β on hover ─────────
try:
    import plotly.graph_objects as go

    geojson    = json.loads(gdf_base.set_index("municipality").to_json())
    all_munis  = gdf_base["municipality"].tolist()
    muni_names = gdf_base["KOMMUNNAMN"].tolist()

    yr_beta = year_df.set_index("year")["coef"]

    if len(map_df) > 0:
        map_lookup = map_df.set_index(["municipality", "year"])["coef"]
    else:
        map_lookup = pd.Series(dtype=float)

    frames       = []
    slider_steps = []

    for i, year in enumerate(years_sorted):
        yr_coef   = float(yr_beta.get(year, float("nan")))
        yr_growth = (
            muni_growth_yr[muni_growth_yr["year"] == year]
            .set_index("municipality")["mean_growth"]
        )

        z_vals = [float(yr_growth.get(m, float("nan"))) for m in all_munis]
        b_vals = [float(map_lookup.get((m, year), float("nan"))) for m in all_munis]

        customdata = [[z, b] for z, b in zip(z_vals, b_vals)]

        yr_label = f"{yr_coef:+.4f}" if not np.isnan(yr_coef) else "—"
        trace = go.Choroplethmapbox(
            geojson=geojson,
            locations=all_munis,
            featureidkey="properties.municipality",
            z=z_vals,
            zmin=-g_clim, zmax=g_clim,
            colorscale="RdBu_r",
            colorbar=dict(title="Mean arcsinh<br>growth", thickness=14, len=0.6),
            text=muni_names,
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Mean arcsinh growth: %{z:+.4f}<br>"
                "β (density→growth): %{customdata[1]:+.4f}"
                "<extra></extra>"
            ),
            name=str(year),
            visible=(i == 0),
        )
        frames.append(trace)

        slider_steps.append(dict(
            method="update",
            args=[
                {"visible": [k == i for k in range(len(years_sorted))]},
                {"title": (
                    f"{year}→{year + 1}  |  Mean industry employment growth"
                    f"  |  overall β = {yr_label}<br>"
                    f"<sup>Colour = mean arcsinh growth  ·  "
                    f"hover for municipality name and density β  ·  "
                    f"run={args.run_name}</sup>"
                )},
            ],
            label=str(year),
        ))

    yr0       = years_sorted[0]
    yr0_coef  = float(yr_beta.get(yr0, float("nan")))
    yr0_label = f"{yr0_coef:+.4f}" if not np.isnan(yr0_coef) else "—"

    layout = go.Layout(
        title=(
            f"{yr0}→{yr0 + 1}  |  Mean industry employment growth"
            f"  |  overall β = {yr0_label}<br>"
            f"<sup>Colour = mean arcsinh growth  ·  "
            f"hover for municipality name and density β  ·  "
            f"run={args.run_name}</sup>"
        ),
        mapbox=dict(style="carto-positron", center={"lat": 62.5, "lon": 16.5}, zoom=3.8),
        height=750,
        margin={"r": 0, "t": 90, "l": 0, "b": 0},
        sliders=[dict(
            active=0,
            steps=slider_steps,
            currentvalue={"prefix": "Year: ", "visible": True, "xanchor": "center"},
            pad={"t": 50, "b": 10},
        )],
    )

    fig_html = go.Figure(data=frames, layout=layout)
    map_html = results_dir / f"{OUT_NAME}_map_slider.html"
    fig_html.write_html(str(map_html), config={"responsive": True})
    print(f"  → {map_html}")

except ImportError as e:
    print(f"  plotly not available — HTML slider skipped ({e})")
except Exception as e:
    import traceback
    print(f"  HTML slider failed: {e}")
    traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Appendix: binarised outcome growth_binary = 1{growth > 0}, regressed on the
#    main-text density D^{A,log} (density_ew_log) with municipality×year +
#    industry fixed effects and SE clustered by municipality and industry
#    (latent_industry.regional.reporting), per classification period.
# ═══════════════════════════════════════════════════════════════════════════════

panel["growth_binary"] = (panel["growth"] > 0).astype(float)
standardise_density(panel, "density_ew_log", "density_ew_log_std")
report_density_regressions(
    panel, "growth_binary", "Observed positive-growth rate",
    "Effect on positive-growth probability per 1-SD density (in %)", "darkorange",
    fig_path=results_dir / f"{OUT_NAME}_growth_binary.png",
    summary_path=results_dir / f"{OUT_NAME}_growth_binary_summary.csv",
    year_path=results_dir / f"{OUT_NAME}_growth_binary_year_coefs.csv",
    summary_extra={"run": args.run_name},
    density_col="density_ew_log_std", plot_density_col="density_ew_log",
)
# Small cache of this appendix panel, so it can be re-estimated without the microdata.
panel[["period", "year", "municipality", "industry", "density_ew_log", "growth_binary"]].to_parquet(
    OUT_DIR / f"{OUT_NAME}_growth_binary_panel.parquet", index=False)

print("\nDone.")
