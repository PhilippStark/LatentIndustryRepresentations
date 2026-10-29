"""
P04_entry.py
============
Relatedness density and industry entry (Hidalgo et al. 2007; Neffke, Henning
& Boschma 2011): is a municipality more likely to gain an industry that sits
close to its existing employment structure in the learned space? Requires the
restricted SCB individual-level register panel (except with --from-cache).

Candidates and outcome
----------------------
Candidates are all (municipality, industry, year t) cells with zero
employment in t.

  entry_industry = 1 if the industry has positive employment in t+1.

Density
-------
For municipality m, industry j and year t, density is the employment-weighted
mean cosine similarity between j and the industries present in m, leaving j
itself out (latent_industry.regional.density):

  D_{j,m,t} = Σ_{i≠j} w_{i,m,t} cos(z_i, z_j) / Σ_{i≠j} w_{i,m,t}

with w = log(1 + employment) in the main specification D^{A,log}
(--density-var ew --log-employment-weights). Robustness variants:
--no-log-employment-weights (raw employment, D^A) and --density-var geo
(geometric-mean analogue, D^G / D^{G,log}); non-baseline variants get a
filename suffix (_ew_raw, _geo_log, _geo_raw).

Regression
----------
Linear probability model, outcome ~ density (standardised by its pooled SD),
with municipality×year and industry fixed effects absorbed exactly and SE
clustered by municipality and industry (latent_industry.regional.fixed_effects).
Estimated separately per classification period (SNI92, SNI02, SNI07): code
strings overlap across versions but denote different industries, so periods
are never pooled. Year-specific effects come from density × base-year
interactions in the same model. R02_fixed_effects_robustness.py compares
alternative fixed-effects specifications.

Options
-------
  --stratify-firm-size  splits entry by the size tercile of the new
                        establishment that triggers it (establishments with
                        more than --min-workers workers and at least
                        --min-worker-ratio of local employment); same
                        candidate set, separate figure and table
  --from-cache          re-estimates and replots from the cached panel (no microdata)

Outputs
-------
  results/regional_entry/{run}{suffix}_{entry.png, summary.csv, year_coefs.csv}
  <LATENT_DATA_ROOT>/processed/downstream_outcomes/entry/{run}{suffix}_panel.parquet

Usage (main run)
----------------
  python scripts/regional/P04_entry.py --run-name shared_cosine_v16
  python scripts/regional/P04_entry.py --run-name shared_cosine_v16 --from-cache
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import EMBEDDING_DIR, PROCESSED_DATA_DIR
from latent_industry.common.municipalities import harmonization_table, load_municipality_crosswalk
from latent_industry.regional.fixed_effects import density_effect, standardise_density
from latent_industry.regional.reporting import report_density_regressions
from latent_industry.regional.density import (
    load_sni_embeddings, make_softmax_kernel, cosine_matrix, employment_weighted_density,
    geometric_mean_employment_weighted_density, empl_matrix as _shared_empl_matrix,
)
from latent_industry.regional.regional_figures import plot_density_effect_by_tier

DATA_ROOT    = Path("/lunarc/nobackup/projects/lu2025-17-39")
INDIVID_PATH = DATA_ROOT / "SCB" / "Preprocessed" / "individ_cleaned_1990-2019.parquet"
EMBED_BASE   = EMBEDDING_DIR
MUNI_XWALK   = harmonization_table(load_municipality_crosswalk())

# results_dir: small, paper-facing outputs only (figures + CSVs) — tracked in
# git. panel.parquet is a large intermediate cache and goes to OUT_DIR on
# LUNARC storage instead, same convention as P04_startup_bridging.py.
results_dir = project_root / "results" / "regional_entry"
results_dir.mkdir(parents=True, exist_ok=True)
OUT_DIR = PROCESSED_DATA_DIR / "downstream_outcomes" / "entry"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Defaults — edit here when running directly in VS Code ─────────────────────
DEFAULT_RUN_NAME    = "shared_cosine_v16"
DEFAULT_TAU         = 1.0    # softmax temperature for the relatedness-density kernel
DEFAULT_MIN_WORKERS = 5      # "more than 5 workers" -> workers > 5
DEFAULT_MIN_WORKER_RATIO = 0.025  # additional floor: workers / municipality total
                                  # workforce that year; 0.0 disables it
SAMPLE_START_YEAR   = 1990   # establishments first seen this year are NOT counted as
                              # births — can't distinguish a true birth from a
                              # pre-existing firm first observed at the panel's edge

OUTCOME       = "entry_industry"
OUTCOME_LABEL = "Observed entry rate"
EFFECT_LABEL  = "Effect on entry probability per 1-SD density (in %)"
COLOR         = "crimson"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                    help="Embedding run under EMBEDDING_DIR (train_shared_cosine_embedding.py --run-name).")
parser.add_argument("--out-suffix", default="",
                    help="Appended to OUT_NAME (output filename prefix) without touching "
                         "--run-name, so variant runs do not overwrite the main outputs.")
parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                    help="Softmax temperature for the relatedness-density kernel.")
parser.add_argument("--density-var", default="ew", choices=["softmax", "ew", "geo"],
                    help="ew (default): plain employment-weighted mean cosine similarity, "
                         "leave-one-out, no temperature — see employment_weighted_density(). "
                         "Combined with --log-employment-weights (also on by default), this is "
                         "the paper's baseline density specification D^{A,log}. geo: "
                         "geometric-mean analogue of ew — see "
                         "geometric_mean_employment_weighted_density(). softmax: "
                         "temperature-weighted kernel, concentrates on the closest neighbours "
                         "(see --tau) — a non-baseline alternative kept for comparison. All "
                         "three live in latent_industry.regional.density. Outputs for any "
                         "combination other than the ew+log baseline are written with a "
                         "filename suffix so they never overwrite the baseline run's outputs.")
parser.add_argument("--log-employment-weights", default=True, action=argparse.BooleanOptionalAction,
                    help="Only affects --density-var ew/geo: weight by log(1+employment) "
                         "instead of raw employment, tempering the influence of a handful of "
                         "very large incumbent industries. On by default (paper baseline uses "
                         "log employment weights) — pass --no-log-employment-weights for the "
                         "raw-employment variant, which adds a filename suffix. No effect on "
                         "--density-var softmax (ignored there).")
parser.add_argument("--min-workers", type=int, default=DEFAULT_MIN_WORKERS,
                    help="--stratify-firm-size only: a new establishment qualifies as a "
                         "tagging event if workers > this value.")
parser.add_argument("--min-worker-ratio", type=float, default=DEFAULT_MIN_WORKER_RATIO,
                    help="--stratify-firm-size only: additional floor, new establishment's "
                         "workers must be >= this fraction of the municipality's total "
                         "workforce that year (e.g. 0.001 = 0.1%%). 0.0 disables this filter.")
parser.add_argument("--stratify-firm-size", default=False, action="store_true",
                    help="Tag each candidate cell with whether a qualifying new establishment "
                         "of a given size tercile (small/medium/large workers, terciles taken "
                         "over qualifying new establishments) triggered it, then regress "
                         "entry_industry_<tier> = entry_industry AND has_new_estab_<tier> "
                         "against density separately per tier — tests whether density predicts "
                         "industry entry differently depending on entrant size. "
                         "Builds an establishment-level panel (slower). Reported separately "
                         "from the main entry_industry figure, as "
                         "'{run}_entry_industry_firmsize.png' + '{run}_firmsize_summary.csv'. "
                         "Off by default.")
parser.add_argument("--from-cache", default=False, action="store_true",
                    help="Skip loading the SCB microdata and rerunning the regressions "
                         "entirely — replot the paper figure directly from this run's "
                         "already-written {run}_panel.parquet / _year_coefs.csv / "
                         "_summary.csv (all three are written by a normal full run). For "
                         "iterating on figure labels/styling without repeating the expensive "
                         "data load + regression pass.")
args, _ = parser.parse_known_args()

if args.log_employment_weights and args.density_var == "softmax":
    print("--log-employment-weights has no effect with --density-var softmax — ignoring.")

# Baseline is --density-var ew --log-employment-weights (paper's D^{A,log}) and
# gets no filename suffix; every other combination writes to a separate
# {run_name}_* namespace so it never overwrites the baseline run's already-
# written entry.png / panel.parquet / summary.csv for the same --run-name.
OUT_NAME = args.run_name
if args.density_var == "softmax":
    OUT_NAME += "_softmax"
elif args.density_var == "geo":
    OUT_NAME += "_geo" + ("_log" if args.log_employment_weights else "_raw")
elif args.density_var == "ew" and not args.log_employment_weights:
    OUT_NAME += "_ew_raw"
OUT_NAME += args.out_suffix

SUMMARY_EXTRA = {
    "run": args.run_name, "density_var": args.density_var,
    "log_employment_weights": args.log_employment_weights, "tau": args.tau,
    "min_workers": args.min_workers, "min_worker_ratio": args.min_worker_ratio,
}

print(f"Run: {args.run_name}   density_var={args.density_var}   tau={args.tau}   "
      f"min_workers>{args.min_workers}   min_worker_ratio>={args.min_worker_ratio:.4%}   "
      f"stratify_firm_size={args.stratify_firm_size}")


if args.from_cache:
    print(f"\n--from-cache: re-estimating from {OUT_DIR}/{OUT_NAME}_panel.parquet "
          f"(no SCB microdata needed).")
    _panel_path = OUT_DIR / f"{OUT_NAME}_panel.parquet"
    if not _panel_path.exists():
        raise SystemExit(f"--from-cache requires {_panel_path} — run this script without "
                          f"--from-cache once first.")
    _panel = pd.read_parquet(_panel_path)
    standardise_density(_panel)
    report_density_regressions(
        _panel, OUTCOME, OUTCOME_LABEL, EFFECT_LABEL, COLOR,
        fig_path=results_dir / f"{OUT_NAME}_entry.png",
        summary_path=results_dir / f"{OUT_NAME}_summary.csv",
        year_path=results_dir / f"{OUT_NAME}_year_coefs.csv",
        summary_extra=SUMMARY_EXTRA,
    )
    print("\nDone (--from-cache).")
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load individual data and aggregate to (municipality, year, industry)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading {INDIVID_PATH} …")
raw = pl.read_parquet(
    INDIVID_PATH,
    columns=["Year", "Sni", "WorkplaceMunicipality", "ArbstID"],
)

base = (
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
)

empl = (
    base
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
# 2. Establishment-level data → entrant size-tier tags (only if
#    --stratify-firm-size). No firm-level outcome is ever built here — this
#    section only produces has_new_estab_<tier> 0/1 TAGS, merged onto the
#    entry_industry candidate panel below to build entry_industry_<tier>.
# ═══════════════════════════════════════════════════════════════════════════════

tier_lookups: dict[str, pd.DataFrame] = {}
if args.stratify_firm_size:
    print("\nBuilding establishment-level panel for entrant size-tier tagging …")
    estab = (
        base
        .filter(pl.col("ArbstID").is_not_null() & (pl.col("ArbstID") != 0))
        .group_by(["ArbstID", "Year"])
        .agg([
            pl.len().alias("workers"),
            pl.col("sni_4d").first().alias("sni_4d"),
            pl.col("municipality").first().alias("municipality"),
        ])
        .to_pandas()
    )
    print(f"  {len(estab):,} (establishment, year) rows")

    first_year = estab.groupby("ArbstID")["Year"].min().rename("first_year")
    estab = estab.merge(first_year, on="ArbstID", how="left")

    muni_year_totals = (
        empl.groupby(["municipality", "Year"], as_index=False)["workers"]
        .sum().rename(columns={"workers": "muni_workers"})
    )
    estab = estab.merge(muni_year_totals, on=["municipality", "Year"], how="left")

    new_estab = estab[
        (estab["Year"] == estab["first_year"])
        & (estab["first_year"] > SAMPLE_START_YEAR)
        & (estab["workers"] > args.min_workers)
        & (estab["workers"] >= args.min_worker_ratio * estab["muni_workers"])
    ].copy()
    print(f"  {len(new_estab):,} qualifying new establishments "
          f"(first seen after {SAMPLE_START_YEAR}, >{args.min_workers} workers, "
          f"≥{args.min_worker_ratio:.4%} of municipality workforce)")

    # labels=False sidesteps qcut's "labels must match bin count" error when
    # duplicates="drop" collapses ties into fewer than 3 bins; map codes to
    # names afterwards instead of asking qcut to name fewer-than-3 bins itself.
    _tier_codes = pd.qcut(new_estab["workers"], q=3, duplicates="drop", labels=False).astype(int)
    _tier_names = ["small", "medium", "large"][: int(_tier_codes.max()) + 1]
    new_estab["size_tier"] = pd.Categorical.from_codes(_tier_codes.to_numpy(), categories=_tier_names)
    print(f"  entrant size terciles (workers): "
          + ", ".join(f"{t}: n={int((new_estab['size_tier'] == t).sum()):,}"
                       for t in _tier_names))
    for tier in _tier_names:
        col = f"has_new_estab_{tier}"
        tier_lookup = (
            new_estab[new_estab["size_tier"] == tier]
            .groupby(["sni_4d", "municipality", "Year"])
            .size().reset_index(name=f"n_new_estab_{tier}")
            .rename(columns={"sni_4d": "industry", "Year": "year_t1"})
        )
        tier_lookup[col] = True
        tier_lookups[tier] = tier_lookup


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Load embeddings and build the relatedness-density kernel
# ═══════════════════════════════════════════════════════════════════════════════

sni_embeddings = load_sni_embeddings(args.run_name, EMBED_BASE)
vecs07, codes07 = sni_embeddings["sni07"]
vecs02, codes02 = sni_embeddings["sni02"]
vecs92, codes92 = sni_embeddings["sni92"]

# make_softmax_kernel / employment_weighted_density live in
# latent_industry.regional.density — shared with P04_growth.py /
# P04_exit.py / P04_startup_entry.py.


def empl_matrix(year: int, codes: list[str]) -> np.ndarray:
    return _shared_empl_matrix(empl, all_municipalities, year, codes)


PERIODS = [
    ("SNI92", codes92, vecs92, range(1990, 2001)),
    ("SNI02", codes02, vecs02, range(2002, 2006)),
    ("SNI07", codes07, vecs07, range(2007, 2019)),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Build the candidate-entry set panel: industries absent at t, density at
#    t, entry outcomes observed at t+1
# ═══════════════════════════════════════════════════════════════════════════════

print("\nBuilding candidate-entry set (municipality-industry-year candidate "
      "observations where the industry is absent at year t) …")
panel_frames: list[pd.DataFrame] = []

for period_name, codes, vecs, year_range in PERIODS:
    if args.density_var in ("ew", "geo"):
        cos_sim = cosine_matrix(vecs)
    else:
        K = make_softmax_kernel(vecs, args.tau)
    print(f"\n  {period_name}  ({len(codes)} codes):")

    for t in year_range:
        t1   = t + 1
        M_t  = empl_matrix(t,  codes)
        M_t1 = empl_matrix(t1, codes)

        W      = M_t.sum(axis=1, keepdims=True)
        W_safe = np.where(W == 0, 1.0, W)
        if args.density_var in ("ew", "geo"):
            M_t_w = np.log1p(M_t) if args.log_employment_weights else M_t
            density = (
                employment_weighted_density(M_t_w, cos_sim) if args.density_var == "ew"
                else geometric_mean_employment_weighted_density(M_t_w, cos_sim)
            )  # defined even where M_t == 0
        else:
            density = (M_t @ K) / W_safe                         # defined even where M_t == 0

        absent = M_t == 0
        mi, ci = np.where(absent)

        # entry_industry: any positive employment at t1.
        entry_industry = M_t1 > 0

        panel_frames.append(pd.DataFrame({
            "period":         period_name,
            "year":           t,
            "municipality":   [all_municipalities[i] for i in mi],
            "industry":       [codes[j]              for j in ci],
            "density":        density       [mi, ci],
            "entry_industry": entry_industry[mi, ci],
        }))
        print(f"    {t}:  candidate rows={len(mi):,}  "
              f"entry_industry rate={entry_industry[mi, ci].mean():.4%}")

panel = pd.concat(panel_frames, ignore_index=True)
panel["entry_industry"] = panel["entry_industry"].astype(float)

firmsize_outcomes: list[str] = []
if tier_lookups:
    panel["year_t1"] = panel["year"] + 1
    for tier, tier_lookup in tier_lookups.items():
        tag_col = f"has_new_estab_{tier}"
        panel = panel.merge(
            tier_lookup[["industry", "municipality", "year_t1", tag_col]],
            on=["industry", "municipality", "year_t1"], how="left",
        )
        panel[tag_col] = (panel[tag_col] == True).astype(float)  # noqa: E712 — NaN -> False
        # Gated industry outcome: a "success" for this tier is entry_industry
        # AND a qualifying establishment of that tier's size triggered the
        # cell. All tiers share the same negative (entry_industry=0) sample.
        outcome_col = f"entry_industry_{tier}"
        panel[outcome_col] = panel["entry_industry"] * panel[tag_col]
        firmsize_outcomes.append(outcome_col)

print(f"\nCandidate-entry set panel: {len(panel):,} rows")
print(f"  entry_industry rate: {panel['entry_industry'].mean():.4%}")
print(f"    mean density | entry_industry=1: {panel.loc[panel['entry_industry']==1, 'density'].mean():.5f}")
print(f"    mean density | entry_industry=0: {panel.loc[panel['entry_industry']==0, 'density'].mean():.5f}")

_sd_density = standardise_density(panel)
print(f"\nDensity standardised by its pooled SD ({_sd_density:.6f}): β is the change in "
      f"P({OUTCOME}) per 1-SD increase in density")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Density regressions per classification period (and year-specific slopes):
#    municipality×year + industry fixed effects, SE clustered by municipality
#    and industry (latent_industry.regional.fixed_effects).
# ═══════════════════════════════════════════════════════════════════════════════

results, year_df = report_density_regressions(
    panel, OUTCOME, OUTCOME_LABEL, EFFECT_LABEL, COLOR,
    fig_path=results_dir / f"{OUT_NAME}_entry.png",
    summary_path=results_dir / f"{OUT_NAME}_summary.csv",
    year_path=results_dir / f"{OUT_NAME}_year_coefs.csv",
    summary_extra=SUMMARY_EXTRA,
)
periods_order = list(results)


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. Optional (--stratify-firm-size): entry_industry gated by entering
#    establishment size tercile, regressed against density separately — same
#    candidate panel, main specification, density_std as
#    above, just a different outcome column per tier. Reported and plotted
#    on its own (not folded into the main entry_industry figure) so it
#    doesn't complicate that figure's panels.
# ═══════════════════════════════════════════════════════════════════════════════

firmsize_results: dict[str, dict[str, dict]] = {}
if firmsize_outcomes:
    print("\n" + "=" * 60)
    print("Relatedness density → entry_industry, by entrant size tercile "
          "(" + ", ".join(firmsize_outcomes) + ")")
    print("=" * 60)

    for period_name in periods_order:
        sub = panel[panel["period"] == period_name]

        firmsize_results[period_name] = {}
        for outcome in firmsize_outcomes:
            res  = density_effect(sub, outcome)
            rate = res["rate"]
            firmsize_results[period_name][outcome] = res
            print(f"    {outcome:<24} coef={res['coef']:+.6f}  SE={res['se']:.6f}"
                  f"  p={res['pval']:.3e}  N={res['n']:,}  baseline rate={rate:.4%}"
                  f"  →  {res['pct_of_baseline']:+.1f}% of baseline")

    firmsize_rows = []
    for period_name in periods_order:
        for outcome in firmsize_outcomes:
            res = firmsize_results[period_name][outcome]
            firmsize_rows.append({
                "run": args.run_name, "period": period_name, "outcome": outcome,
                "tier": outcome.removeprefix("entry_industry_"),
                "coef": res["coef"], "se": res["se"], "pval": res["pval"],
                "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                "n_obs": res["n"], "rate": res["rate"],
                "pct_of_baseline": res["pct_of_baseline"],
            })
    firmsize_df = pd.DataFrame(firmsize_rows)
    firmsize_csv_path = results_dir / f"{OUT_NAME}_firmsize_summary.csv"
    firmsize_df.to_csv(firmsize_csv_path, index=False)
    print(f"\n  → {firmsize_csv_path}")

    tiers_present = [t.removeprefix("entry_industry_") for t in firmsize_outcomes]
    tier_results = {
        period_name: {
            tier: firmsize_results[period_name][f"entry_industry_{tier}"]
            for tier in tiers_present
        }
        for period_name in periods_order
    }
    firmsize_fig_path = plot_density_effect_by_tier(
        tier_results, periods_order, tiers_present,
        xlabel="Entering establishment's size tercile",
        effect_label="Effect on entry probability per 1-SD density (in %)",
        title="Density effect on entry, by entrant size",
        fig_path=results_dir / f"{OUT_NAME}_entry_industry_firmsize.png",
    )
    print(f"  → {firmsize_fig_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Save the panel
# ═══════════════════════════════════════════════════════════════════════════════

panel_out = panel[["period", "year", "municipality", "industry", "density", OUTCOME]]
panel_out.to_parquet(OUT_DIR / f"{OUT_NAME}_panel.parquet", index=False)
print(f"  → {OUT_DIR / f'{OUT_NAME}_panel.parquet'}  ({len(panel):,} rows)")


print("\nDone.")
