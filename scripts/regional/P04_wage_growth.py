"""
P04_wage_growth.py
==================
Relatedness density and local wage growth of industries (appendix).
Requires the restricted SCB individual-level register panel.

Outcome
-------
For industry j, municipality m and year t, mean_wage is the average
employment income of the workers in the cell, and

  wage_growth_{j,m,t} = arcsinh(mean_wage_{j,m,t+1}) − arcsinh(mean_wage_{j,m,t})

Only cells with more than --min-workers workers in both t and t+1 are used
(intensive margin).

Design
------
Softmax density (--tau), industry, municipality and year fixed effects
(sequential demeaning), SE clustered by municipality, per classification
period. Appendix result: wage_growth_binary = 1{wage_growth > 0} regressed on
the main-text density D^{A,log} with the main specification of P04_entry.py
(municipality×year + industry FE, two-way clustered SE).

Outputs: results/regional_wage_growth/{run}{suffix}_* and the panel under
<LATENT_DATA_ROOT>/processed/downstream_outcomes/wage_growth/.

Usage (main run)
----------------
  python scripts/regional/P04_wage_growth.py --run-name shared_cosine_v16
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

from latent_industry.common.paths import EMBEDDING_DIR, PROCESSED_DATA_DIR
from latent_industry.common.municipalities import harmonization_table, load_municipality_crosswalk
from latent_industry.regional.density import employment_weighted_density, cosine_matrix
from latent_industry.regional.fixed_effects import standardise_density
from latent_industry.regional.reporting import report_density_regressions
from latent_industry.regional.regional_figures import plot_density_effect_summary

DATA_ROOT    = Path("/lunarc/nobackup/projects/lu2025-17-39")
INDIVID_PATH = DATA_ROOT / "SCB" / "Preprocessed" / "individ_cleaned_1990-2019.parquet"
EMBED_BASE   = EMBEDDING_DIR
MUNI_XWALK   = harmonization_table(load_municipality_crosswalk())

# results_dir: small, paper-facing outputs only (figures + CSVs) — tracked in
# git. panel.parquet is a large intermediate cache and goes to OUT_DIR on
# LUNARC storage instead, same convention as P04_startup_bridging.py.
results_dir = project_root / "results" / "regional_wage_growth"
results_dir.mkdir(parents=True, exist_ok=True)
OUT_DIR = PROCESSED_DATA_DIR / "downstream_outcomes" / "wage_growth"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Defaults — edit here when running directly in VS Code ─────────────────────
DEFAULT_RUN_NAME    = "shared_cosine_v16"
DEFAULT_TAU         = 1.0   # softmax temperature for the relatedness-density kernel
DEFAULT_MIN_WORKERS = 5     # both years need > this many workers for a wage cell to count

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                    help="Embedding run under EMBEDDING_DIR (train_shared_cosine_embedding.py --run-name).")
parser.add_argument("--out-suffix", default="",
                    help="Appended to OUT_NAME (output filename prefix) without touching "
                         "--run-name, so variant runs do not overwrite the main outputs.")
parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                    help="Softmax temperature for the relatedness-density kernel.")
parser.add_argument("--min-workers", type=int, default=DEFAULT_MIN_WORKERS,
                    help="Minimum workers required in BOTH t and t+1 for a cell's mean "
                         "wage to enter the sample (a wage needs a well-defined mean, "
                         "not an economic-significance floor).")
args, _ = parser.parse_known_args()

# OUT_NAME (not args.run_name) is used for every output path below, so
# --out-suffix can redirect a whole run's outputs to a separate namespace
# without touching --run-name (which selects the input embeddings).
OUT_NAME = args.run_name + args.out_suffix

embed_dir = EMBED_BASE / args.run_name
print(f"Run: {args.run_name}   tau={args.tau}   min_workers>{args.min_workers}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load individual data and aggregate to (municipality, year, industry):
#    worker count AND mean employment income
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading {INDIVID_PATH} …")
raw = pl.read_parquet(
    INDIVID_PATH,
    columns=["Year", "Sni", "WorkplaceMunicipality", "EmploymentIncome"],
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
    .join(MUNI_XWALK, on=["Year", "source_code"], how="inner")
    .group_by(["municipality", "Year", "sni_4d"])
    .agg([
        pl.len().alias("workers"),
        pl.col("EmploymentIncome").mean().alias("mean_income"),
    ])
    .sort(["municipality", "Year", "sni_4d"])
    .to_pandas()
)

all_municipalities = sorted(empl["municipality"].unique())
print(f"  {len(empl):,} (municipality, year, industry) observations")
print(f"  Years: {empl['Year'].min()}–{empl['Year'].max()}")
print(f"  Municipalities: {len(all_municipalities)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Load embeddings and build the relatedness-density kernel
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading embeddings from {embed_dir} …")

def load_emb(label: str) -> tuple[np.ndarray, list[str]]:
    path = embed_dir / f"embeddings_{label}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run train_shared_cosine_embedding.py first.")
    d     = np.load(path, allow_pickle=True)
    vecs  = d["embeddings"].astype(np.float32)
    codes = list(d["codes"].astype(str))
    print(f"  {label}: {len(codes)} codes  dim={vecs.shape[1]}")
    return vecs, codes

vecs07, codes07 = load_emb("sni07")
vecs02, codes02 = load_emb("sni02")
vecs92, codes92 = load_emb("sni92")


def make_softmax_kernel(vecs: np.ndarray, tau: float) -> np.ndarray:
    """K[i,j] = softmax_i(τ·cos(x_i,x_j)), self excluded — column-stochastic.
    density_j = Σ_i K[i,j]·w_i / W, same base kernel as P04_growth.py."""
    cos_sim = (vecs @ vecs.T).astype(np.float64)
    np.fill_diagonal(cos_sim, -np.inf)
    shifted = tau * cos_sim - (tau * cos_sim).max(axis=0, keepdims=True)
    E = np.exp(shifted)
    K = E / E.sum(axis=0, keepdims=True)
    return K.astype(np.float32)


def empl_matrix(year: int, codes: list[str]) -> np.ndarray:
    sub = empl[(empl["Year"] == year) & (empl["sni_4d"].isin(codes))]
    piv = sub.pivot_table(
        index="municipality", columns="sni_4d",
        values="workers", aggfunc="sum", fill_value=0,
    )
    return (
        piv.reindex(index=all_municipalities, columns=codes, fill_value=0)
           .values.astype(np.float64)
    )


def wage_matrix(year: int, codes: list[str]) -> np.ndarray:
    """Return (n_muni, n_codes) mean-income matrix — NaN where the cell is absent
    (mean of an empty group), distinct from empl_matrix's zero-fill."""
    sub = empl[(empl["Year"] == year) & (empl["sni_4d"].isin(codes))]
    piv = sub.pivot_table(
        index="municipality", columns="sni_4d",
        values="mean_income", aggfunc="mean",
    )
    return (
        piv.reindex(index=all_municipalities, columns=codes)
           .values.astype(np.float64)
    )


PERIODS = [
    ("SNI92", codes92, vecs92, range(1990, 2001)),
    ("SNI02", codes02, vecs02, range(2002, 2006)),
    ("SNI07", codes07, vecs07, range(2007, 2019)),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Build panel: (municipality, industry, year=t) cells with >min_workers
#    employees in BOTH t and t+1, density at t, wage growth to t+1
# ═══════════════════════════════════════════════════════════════════════════════

print("\nBuilding panel …")
panel_frames: list[pd.DataFrame] = []

for period_name, codes, vecs, year_range in PERIODS:
    K = make_softmax_kernel(vecs, args.tau)
    # cos_sim feeds density_ew_log below — the entry/exit/startup_entry
    # baseline density (log-employment-weighted, leave-one-out) — kept
    # separate from this script's own softmax-kernel "density" column, which
    # its continuous wage-growth analysis (sections 4-7 below) is unchanged
    # and still built on.
    cos_sim = cosine_matrix(vecs)
    print(f"\n  {period_name}  ({len(codes)} codes):")

    for t in year_range:
        t1      = t + 1
        M_t     = empl_matrix(t,  codes)
        M_t1    = empl_matrix(t1, codes)
        wage_t  = wage_matrix(t,  codes)
        wage_t1 = wage_matrix(t1, codes)

        W      = M_t.sum(axis=1, keepdims=True)
        W_safe = np.where(W == 0, 1.0, W)
        density = (M_t @ K) / W_safe
        density_ew_log = employment_weighted_density(np.log1p(M_t), cos_sim)

        keep = (M_t > args.min_workers) & (M_t1 > args.min_workers)
        mi, ci = np.where(keep)

        wage_growth = np.arcsinh(wage_t1) - np.arcsinh(wage_t)

        panel_frames.append(pd.DataFrame({
            "period":        period_name,
            "year":          t,
            "municipality":  [all_municipalities[i] for i in mi],
            "industry":      [codes[j]              for j in ci],
            "workers_t":     M_t        [mi, ci],
            "workers_t1":    M_t1       [mi, ci],
            "density":       density    [mi, ci],
            "density_ew_log": density_ew_log[mi, ci],
            "wage_growth":   wage_growth[mi, ci],
        }))
        print(f"    {t}→{t+1}:  {len(mi):,} cells (>{args.min_workers} workers both years)")

panel = pd.concat(panel_frames, ignore_index=True)
print(f"\nPanel: {len(panel):,} total rows")

_sd_density = float(panel["density"].std())
panel["density_std"] = panel["density"] / _sd_density
print(f"\nDensity pre-standardised (÷ SD = {_sd_density:.6f}) — β reads as "
      "arcsinh-wage-growth per 1-SD increase in density")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Per-classification-period demeaning and regression — industry +
#    municipality + year FE via triple demeaning (separate additive
#    municipality and year effects, not an interacted municipality×year
#    term), run separately per period (SNI code strings overlap across
#    vintages but never mean the same industry — see P04_entry.py's
#    docstring).
# ═══════════════════════════════════════════════════════════════════════════════

def demean(df: pd.DataFrame, col: str, by) -> pd.Series:
    return df[col] - df.groupby(by)[col].transform("mean")

def run_ols(y: pd.Series, x: pd.Series, cluster: pd.Series) -> dict:
    X = sm.add_constant(x.values)
    if cluster.nunique() > 1:
        res = sm.OLS(y.values, X).fit(cov_type="cluster", cov_kwds={"groups": cluster.values})
    else:
        res = sm.OLS(y.values, X).fit(cov_type="HC3")
    ci = res.conf_int()
    r2 = float(res.rsquared)
    partial_r = float(np.sign(res.params[1]) * np.sqrt(r2))
    return {
        "coef": float(res.params[1]), "se": float(res.bse[1]), "t": float(res.tvalues[1]),
        "pval": float(res.pvalues[1]), "ci_low": float(ci[1, 0]), "ci_high": float(ci[1, 1]),
        "n": int(res.nobs), "r2": r2, "partial_r": partial_r,
    }

print("\n" + "=" * 60)
print("Relatedness density → wage growth  (per classification period, industry + "
      "municipality + year FE, SE clustered by muni)")
print("=" * 60)

periods_order = list(dict.fromkeys(panel["period"]))
results: dict[str, dict] = {}
dm_frames: list[pd.DataFrame] = []

for period_name in periods_order:
    sub = panel[panel["period"] == period_name].copy()
    for var in ("wage_growth", "density_std"):
        sub[f"{var}_r1"] = demean(sub, var,          "municipality")
        sub[f"{var}_r2"] = demean(sub, f"{var}_r1",  "year")
        sub[f"{var}_dm"] = demean(sub, f"{var}_r2",  "industry")
    res = run_ols(sub["wage_growth_dm"], sub["density_std_dm"], sub["municipality"])
    # Same arcsinh-to-proportional-change approximation as evaluate_industry_
    # growth.py's pct_effect — mean_wage here is always positive (no entry/exit
    # margin, unlike headcount), so this approximation is on firmer footing.
    res["pct_effect"] = (np.exp(res["coef"]) - 1) * 100
    results[period_name] = res
    dm_frames.append(sub)
    print(f"\n  -- {period_name} --  N={len(sub):,}")
    print(f"    coef={res['coef']:+.6f}  SE={res['se']:.6f}  t={res['t']:.3f}  p={res['pval']:.3e}")
    print(f"    partial r={res['partial_r']:+.4f}  ≈{res['pct_effect']:+.1f}% per 1-SD density")

panel = pd.concat(dm_frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Year-by-year regression
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Year-by-year coefficients")
print("=" * 60)

year_records: list[dict] = []
for year, sub in panel.groupby("year"):
    sub = sub.copy()
    if sub["municipality"].nunique() < 2:
        continue
    for var in ("wage_growth", "density_std"):
        sub[f"{var}_dm"] = demean(sub, var, "municipality")
        sub[f"{var}_dm"] = demean(sub, f"{var}_dm", "industry")
    r = run_ols(sub["wage_growth_dm"], sub["density_std_dm"], sub["municipality"])
    r["year"] = int(year)
    year_records.append(r)
    print(f"  {year}→{year+1}:  coef={r['coef']:+.5f}  SE={r['se']:.5f}  p={r['pval']:.3e}  N={r['n']:,}")

year_df = pd.DataFrame(year_records).sort_values("year")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Visualisations
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# A. Density ventile vs mean wage growth, per period
N_BINS = 20
ax = axes[0]
width = 0.7 / len(periods_order)
for k, period_name in enumerate(periods_order):
    sub = panel[panel["period"] == period_name]
    bin_cat = pd.qcut(sub["density"], q=N_BINS, duplicates="drop")
    bin_id  = bin_cat.cat.codes + 1
    grp     = sub["wage_growth"].groupby(bin_id)
    offset  = (k - (len(periods_order) - 1) / 2) * width
    ax.bar(grp.mean().index + offset, grp.mean().values, width=width, label=period_name)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(range(1, N_BINS + 1, 2))
ax.set_xlabel(f"Density ventile (q={N_BINS})  (1 = far  →  {N_BINS} = close, within period)")
ax.set_ylabel("Mean arcsinh wage growth")
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
ax.set_ylabel("β  (density → arcsinh wage growth)")
# muni + industry FE, clustered SE — belongs in the paper's figure caption,
# not the on-image title (matches P04_entry.py/P04_exit.py's convention).
ax.set_title("B. Effect over time")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# C. Per-period coefficient comparison — same layout as entry/exit/growth.
# Industry + municipality + year FE, clustered SE, within period.
ax = axes[2]
coefs = [results[p]["coef"] for p in periods_order]
ses   = [results[p]["se"]   for p in periods_order]
colors = ["steelblue" if c > 0 else "crimson" for c in coefs]
bars = ax.bar(periods_order, coefs, yerr=[1.96 * s for s in ses],
              color=colors, alpha=0.8, capsize=6, edgecolor="white", width=0.5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Classification period")
ax.set_ylabel("β  (arcsinh wage growth per 1-SD density)")
ax.set_title("C. Effect by period")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

ax.relim()
ax.autoscale_view()
y0, y1 = ax.get_ylim()
pad = 0.08 * (y1 - y0)
for bar, period_name in zip(bars, periods_order):
    res  = results[period_name]
    coef = res["coef"]
    y_text = res["ci_high"] + pad if coef >= 0 else res["ci_low"] - pad
    ax.text(bar.get_x() + bar.get_width() / 2, y_text,
            f"r={res['partial_r']:+.3f}\n({res['pct_effect']:+.1f}% approx)",
            ha="center", va="bottom" if coef >= 0 else "top", fontsize=8)
ax.set_ylim(y0 - 2.4 * pad, y1 + 2.4 * pad)

print(f"  Embedding density → local industry wage growth  (run={args.run_name}, "
      f"τ={args.tau}, min_workers>{args.min_workers})")
# No overall suptitle — run/τ identification belongs in the paper's figure
# caption, not the image itself (matches P04_entry.py/P04_exit.py).
plt.tight_layout()
fig_path = results_dir / f"{OUT_NAME}_wage_growth.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n  → {fig_path}")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Save outputs
# ═══════════════════════════════════════════════════════════════════════════════

panel_out = panel[["period", "year", "municipality", "industry",
                   "workers_t", "workers_t1", "density", "wage_growth"]]
panel_out.to_parquet(OUT_DIR / f"{OUT_NAME}_panel.parquet", index=False)
print(f"  → {OUT_DIR / f'{OUT_NAME}_panel.parquet'}  ({len(panel):,} rows)")

year_df.to_csv(results_dir / f"{OUT_NAME}_year_coefs.csv", index=False)

period_summary_df = pd.DataFrame([
    {"run": args.run_name, "period": p, "min_workers": args.min_workers, **results[p]}
    for p in periods_order
])
period_summary_df.to_csv(results_dir / f"{OUT_NAME}_period_summary.csv", index=False)

print(f"  → {results_dir}/  ({OUT_NAME}_year_coefs.csv, _period_summary.csv)  "
      f"+ {OUT_DIR}/  ({OUT_NAME}_panel.parquet)")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Appendix: binarised outcome wage_growth_binary = 1{wage_growth > 0}, regressed on the
#    main-text density D^{A,log} (density_ew_log) with municipality×year +
#    industry fixed effects and SE clustered by municipality and industry
#    (latent_industry.regional.reporting), per classification period.
# ═══════════════════════════════════════════════════════════════════════════════

panel["wage_growth_binary"] = (panel["wage_growth"] > 0).astype(float)
standardise_density(panel, "density_ew_log", "density_ew_log_std")
report_density_regressions(
    panel, "wage_growth_binary", "Observed positive wage-growth rate",
    "Effect on positive wage-growth probability per 1-SD density (in %)", "purple",
    fig_path=results_dir / f"{OUT_NAME}_wage_growth_binary.png",
    summary_path=results_dir / f"{OUT_NAME}_wage_growth_binary_summary.csv",
    year_path=results_dir / f"{OUT_NAME}_wage_growth_binary_year_coefs.csv",
    summary_extra={"run": args.run_name},
    density_col="density_ew_log_std", plot_density_col="density_ew_log",
)
# Small cache of this appendix panel, so it can be re-estimated without the microdata.
panel[["period", "year", "municipality", "industry", "density_ew_log", "wage_growth_binary"]].to_parquet(
    OUT_DIR / f"{OUT_NAME}_wage_growth_binary_panel.parquet", index=False)

print("\nDone.")
