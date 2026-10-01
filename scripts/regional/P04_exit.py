"""
P04_exit.py
========================
Mirror of P04_entry.py: tests whether a region's chance of LOSING
an industry it already has depends on how close that industry sits to the
region's existing employment structure in embedding space (Hidalgo et al.
2007 "Product Space"; Neffke, Henning & Boschma 2011). Where entry asks
whether new activity appears close to or far from the local cluster, exit
asks the same question about activity that disappears — the expected sign
on relatedness density flips: an industry well embedded in the local
employment structure should be LESS likely to disappear, so the coefficient
on density is expected to be negative (the opposite of P04_entry.py).

Candidate-exit set
------------------
The set of potential exits: every (municipality, industry, year=t)
candidate observation where the industry has POSITIVE employment at t (the
complement of P04_entry.py's candidate-entry set, which requires
zero employment at t). The same relatedness-density kernel from
P04_growth.py applies unchanged — it already excludes the
industry's own self-similarity by construction, so density is well-defined
whether the industry is present or absent.

Exit outcome
------------
  exit_industry = the industry's regional employment, which was at a
                   non-negligible local size at t (within ± --sd-mult SDs,
                   arcsinh scale, of the other active industries in that
                   municipality at t), drops to ZERO by t+1.

This is the ONLY outcome regressed and plotted here — the outcome of interest
is industry-level exit, not firm-level exit.

--stratify-firm-size (optional): does NOT add a firm-level outcome. It builds
an establishment-level panel purely to TAG each candidate cell with whether a
qualifying closing establishment of a given size tercile (small/medium/large,
by the establishment's own worker count) triggered it, then regresses
  exit_industry_<tier> = exit_industry AND has_dead_estab_<tier>
against density, separately per tier. All three tiers share the exact same
negative (no-exit) sample, so each tier's regression reads as "does density
predict a significant exit that happens to be driven by a small/medium/large
closure", against the same base rate. Reported and plotted separately from
the main exit_industry figure.

Test
----
Linear probability model:  exit ~ density  (1-SD standardised), absorbing
industry + municipality + year fixed effects via triple demeaning (separate
additive municipality and year effects, not an interacted municipality×year
term) — same design as P04_entry.py. SE clustered by municipality.

SNI92/SNI02/SNI07 code strings overlap heavily across classification periods
(e.g. 488 of SNI92's 505 codes also appear in SNI02's own code list) but refer
to different classification vintages — a code never actually carries over
from one period to the next. Pooling them under one "industry" fixed effect
would treat a 1995 code and a 2003 code that happen to share the same digits
as the same FE level. The regression is therefore run separately within each
classification period (SNI92, SNI02, SNI07) — never pooled across periods,
same as P04_entry.py.

--stratify-firm-size's qualifying-establishment filter uses --min-workers as
an absolute headcount floor, which is not a meaningful bar in a big
municipality (e.g. Stockholm) where 5 workers is nothing. --min-worker-ratio
adds an optional second floor expressed as a fraction of the municipality's
total workforce in that year — e.g. 0.001 requires the closing establishment
to have been at least 0.1% of local employment. Defaults to 0.0 (disabled,
i.e. --min-workers alone applies).

Usage
-----
  python scripts/regional/P04_exit.py
  python scripts/regional/P04_exit.py --run-name graph_kl --min-workers 5 --sd-mult 2.0
  python scripts/regional/P04_exit.py --stratify-firm-size
  python scripts/regional/P04_exit.py --stratify-firm-size --min-worker-ratio 0.001
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
from latent_industry.regional.density import (
    load_sni_embeddings, make_softmax_kernel, cosine_matrix, employment_weighted_density,
    geometric_mean_employment_weighted_density, empl_matrix as _shared_empl_matrix,
)
from latent_industry.regional.regional_figures import plot_density_effect_summary, plot_density_effect_by_tier

DATA_ROOT    = Path("/lunarc/nobackup/projects/lu2025-17-39")
INDIVID_PATH = DATA_ROOT / "SCB" / "Preprocessed" / "individ_cleaned_1990-2019.parquet"
EMBED_BASE   = EMBEDDING_DIR
MUNI_XWALK   = harmonization_table(load_municipality_crosswalk())

# results_dir: small, paper-facing outputs only (figures + CSVs) — tracked in
# git. panel.parquet is a large intermediate cache and goes to OUT_DIR on
# LUNARC storage instead, same convention as P04_startup_bridging.py.
results_dir = project_root / "results" / "regional_exit"
results_dir.mkdir(parents=True, exist_ok=True)
OUT_DIR = PROCESSED_DATA_DIR / "downstream_outcomes" / "exit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Defaults — edit here when running directly in VS Code ─────────────────────
DEFAULT_RUN_NAME    = "shared_cosine_v16"
DEFAULT_TAU         = 1.0    # softmax temperature for the relatedness-density kernel
DEFAULT_MIN_WORKERS = 5      # "more than 5 workers" -> workers > 5
DEFAULT_MIN_WORKER_RATIO = 0.025  # additional floor: workers / municipality total
                                  # workforce that year; 0.0 disables it
DEFAULT_SD_MULT     = 2.0    # significance band (arcsinh scale) for exit_industry
SAMPLE_END_YEAR     = 2019   # establishments last seen this year are NOT counted as
                              # deaths — can't distinguish a true death from a firm
                              # that's still alive at the panel's edge

OUTCOME       = "exit_industry"
OUTCOME_LABEL = "Observed exit rate"
EFFECT_LABEL  = "Effect on exit probability per 1-SD density (in %)"
COLOR         = "steelblue"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                    help="Embedding run (from train_sentence_embedding.py --run-name).")
parser.add_argument("--out-suffix", default="",
                    help="Appended to OUT_NAME (output filename prefix) without touching "
                         "--run-name, so an investigative run (e.g. a fixed-effects spec "
                         "change) never overwrites this run's already-published outputs.")
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
                    help="--stratify-firm-size only: a closing establishment qualifies as a "
                         "tagging event if workers > this value.")
parser.add_argument("--min-worker-ratio", type=float, default=DEFAULT_MIN_WORKER_RATIO,
                    help="--stratify-firm-size only: additional floor, closing establishment's "
                         "workers must be >= this fraction of the municipality's total "
                         "workforce that year (e.g. 0.001 = 0.1%%). 0.0 disables this filter.")
parser.add_argument("--sd-mult", type=float, default=DEFAULT_SD_MULT,
                    help="SD multiplier (arcsinh scale) defining 'significant' industry presence.")
parser.add_argument("--stratify-firm-size", default=False, action="store_true",
                    help="Tag each candidate cell with whether a qualifying closing "
                         "establishment of a given size tercile (small/medium/large workers, "
                         "terciles taken over qualifying establishment deaths) triggered it, "
                         "then regress exit_industry_<tier> = exit_industry AND "
                         "has_dead_estab_<tier> against density separately per tier — tests "
                         "whether density predicts SIGNIFICANT industry exit differently "
                         "depending on closure size. Builds an establishment-level panel "
                         "(slower). Reported separately from the main exit_industry figure, "
                         "as '{run}_exit_industry_firmsize.png' + '{run}_firmsize_summary.csv'. "
                         "Off by default.")
parser.add_argument("--sd-mult-sweep", default="", type=str,
                    help="Appendix robustness check: comma-separated sd_mult values to "
                         "re-run the regression at (e.g. '1.0,1.5,2.0,2.5,3.0,4.0'), reusing "
                         "the density/employment matrices already computed for the main run "
                         "— only the outcome's significance band and the regression are "
                         "redone per value. Empty = skip (default). sd_mult is the main "
                         "robustness target because it defines the OUTCOME itself (which "
                         "drops-to-zero count as a 'significant' exit), not just a nuisance "
                         "control like tau.")
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
# written exit.png / panel.parquet / summary.csv for the same --run-name.
OUT_NAME = args.run_name
if args.density_var == "softmax":
    OUT_NAME += "_softmax"
elif args.density_var == "geo":
    OUT_NAME += "_geo" + ("_log" if args.log_employment_weights else "_raw")
elif args.density_var == "ew" and not args.log_employment_weights:
    OUT_NAME += "_ew_raw"
OUT_NAME += args.out_suffix

print(f"Run: {args.run_name}   density_var={args.density_var}   tau={args.tau}   "
      f"min_workers>{args.min_workers}   min_worker_ratio>={args.min_worker_ratio:.4%}   "
      f"sd_mult={args.sd_mult}   stratify_firm_size={args.stratify_firm_size}")

SD_MULT_SWEEP = [float(x) for x in args.sd_mult_sweep.split(",") if x.strip()] if args.sd_mult_sweep else []
if SD_MULT_SWEEP:
    print(f"  sd_mult robustness sweep requested: {SD_MULT_SWEEP}")


if args.from_cache:
    print(f"\n--from-cache: replotting from cached {OUT_DIR}/{OUT_NAME}_panel.parquet + "
          f"{results_dir}/{OUT_NAME}_{{year_coefs.csv,summary.csv}} — skipping SCB microdata "
          f"load and regressions entirely.")
    _panel_path   = OUT_DIR / f"{OUT_NAME}_panel.parquet"
    _year_path    = results_dir / f"{OUT_NAME}_year_coefs.csv"
    _summary_path = results_dir / f"{OUT_NAME}_summary.csv"
    for _p in (_panel_path, _year_path, _summary_path):
        if not _p.exists():
            raise SystemExit(f"--from-cache requires {_p} — run this script without "
                              f"--from-cache once first.")

    _panel   = pd.read_parquet(_panel_path)
    _year_df = pd.read_csv(_year_path)
    _summary = pd.read_csv(_summary_path)
    _periods_order = list(dict.fromkeys(_panel["period"]))

    _results: dict[str, dict] = {}
    for _, _row in _summary.iterrows():
        _coef, _se = float(_row[f"{OUTCOME}_coef"]), float(_row[f"{OUTCOME}_se"])
        _results[_row["period"]] = {
            "coef": _coef, "se": _se, "pval": float(_row[f"{OUTCOME}_pval"]),
            "partial_r": float(_row[f"{OUTCOME}_r"]),
            "pct_of_baseline": float(_row[f"{OUTCOME}_pct_of_baseline"]),
            "rate": float(_row[f"{OUTCOME}_rate"]),
            "ci_low": _coef - 1.96 * _se, "ci_high": _coef + 1.96 * _se,
        }

    _fig_path = plot_density_effect_summary(
        _panel, _year_df, _results, _periods_order, OUTCOME, OUTCOME_LABEL, EFFECT_LABEL, COLOR,
        results_dir / f"{OUT_NAME}_exit.png",
    )
    print(f"\n  → {_fig_path}")
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
# 2. Establishment-level data → closing-establishment size-tier tags (only if
#    --stratify-firm-size). No firm-level outcome is ever built here — this
#    section only produces has_dead_estab_<tier> 0/1 TAGS, merged onto the
#    exit_industry candidate panel below to build exit_industry_<tier>.
# ═══════════════════════════════════════════════════════════════════════════════

tier_lookups: dict[str, pd.DataFrame] = {}
if args.stratify_firm_size:
    print("\nBuilding establishment-level panel for closure size-tier tagging …")
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

    last_year = estab.groupby("ArbstID")["Year"].max().rename("last_year")
    estab = estab.merge(last_year, on="ArbstID", how="left")

    muni_year_totals = (
        empl.groupby(["municipality", "Year"], as_index=False)["workers"]
        .sum().rename(columns={"workers": "muni_workers"})
    )
    estab = estab.merge(muni_year_totals, on=["municipality", "Year"], how="left")

    dying_estab = estab[
        (estab["Year"] == estab["last_year"])
        & (estab["last_year"] < SAMPLE_END_YEAR)
        & (estab["workers"] > args.min_workers)
        & (estab["workers"] >= args.min_worker_ratio * estab["muni_workers"])
    ].copy()
    print(f"  {len(dying_estab):,} qualifying establishment deaths "
          f"(last seen before {SAMPLE_END_YEAR}, >{args.min_workers} workers, "
          f"≥{args.min_worker_ratio:.4%} of municipality workforce)")

    # labels=False sidesteps qcut's "labels must match bin count" error when
    # duplicates="drop" collapses ties into fewer than 3 bins; map codes to
    # names afterwards instead of asking qcut to name fewer-than-3 bins itself.
    _tier_codes = pd.qcut(dying_estab["workers"], q=3, duplicates="drop", labels=False).astype(int)
    _tier_names = ["small", "medium", "large"][: int(_tier_codes.max()) + 1]
    dying_estab["size_tier"] = pd.Categorical.from_codes(_tier_codes.to_numpy(), categories=_tier_names)
    print(f"  closure size terciles (workers): "
          + ", ".join(f"{t}: n={int((dying_estab['size_tier'] == t).sum()):,}"
                       for t in _tier_names))
    for tier in _tier_names:
        col = f"has_dead_estab_{tier}"
        tier_lookup = (
            dying_estab[dying_estab["size_tier"] == tier]
            .groupby(["sni_4d", "municipality", "Year"])
            .size().reset_index(name=f"n_dead_estab_{tier}")
            .rename(columns={"sni_4d": "industry", "Year": "year"})
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
# P04_entry.py / P04_startup_entry.py.


def empl_matrix(year: int, codes: list[str]) -> np.ndarray:
    return _shared_empl_matrix(empl, all_municipalities, year, codes)


PERIODS = [
    ("SNI92", codes92, vecs92, range(1990, 2001)),
    ("SNI02", codes02, vecs02, range(2002, 2006)),
    ("SNI07", codes07, vecs07, range(2007, 2019)),
]


def significance_band(arcsinh_vals: np.ndarray, active_mask: np.ndarray, sd_mult: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (municipality) ±sd_mult SD band, arcsinh scale, across that row's
    active industries — the local yardstick for 'non-negligible' presence."""
    n_rows = arcsinh_vals.shape[0]
    sig_lo = np.full(n_rows, np.nan)
    sig_hi = np.full(n_rows, np.nan)
    for row in range(n_rows):
        active = active_mask[row]
        if active.sum() < 2:
            continue
        vals = arcsinh_vals[row, active]
        mu, sd = float(vals.mean()), float(vals.std())
        if sd == 0:
            continue
        sig_lo[row] = mu - sd_mult * sd
        sig_hi[row] = mu + sd_mult * sd
    return sig_lo, sig_hi


def compute_exit_outcome(M_t: np.ndarray, M_t1: np.ndarray, arcsinh_t: np.ndarray, sd_mult: float) -> np.ndarray:
    """exit_industry: present at t, gone at t1, and already 'significant' at t
    (within ±sd_mult SDs, arcsinh scale, of the OTHER active industries in that
    municipality at t). sd_mult is the main robustness target (--sd-mult-sweep)
    since it defines the outcome itself, not just how the regressor is built."""
    active_t = M_t > 0
    sig_lo, sig_hi = significance_band(arcsinh_t, active_t, sd_mult)
    return (
        (M_t > 0) & (M_t1 == 0)
        & (arcsinh_t >= sig_lo[:, None]) & (arcsinh_t <= sig_hi[:, None])
    )


# Cache of per-(period, year) matrices for the sd_mult sweep — only populated if
# requested, since M_t/M_t1/density/arcsinh_t are dense (n_muni × n_codes) arrays
# and there's no reason to hold them in memory for the default run.
sweep_cache: list[dict] = []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Build the candidate-exit set panel: industries present at t, density at
#    t, exit outcomes observed at t+1
# ═══════════════════════════════════════════════════════════════════════════════

print("\nBuilding candidate-exit set (municipality-industry-year candidate "
      "observations where the industry is present at year t) …")
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
            )
        else:
            density = (M_t @ K) / W_safe

        present = M_t > 0
        mi, ci = np.where(present)

        # exit_industry: the industry must already be at a "significant"
        # local size at t (arcsinh scale, within ±sd_mult SDs of the OTHER
        # active industries in that municipality at t) before it can count
        # as a meaningful exit — mirrors entry_industry's significance band,
        # but evaluated on the departing side (t) instead of the arriving
        # side (t1).
        arcsinh_t = np.arcsinh(M_t)
        exit_industry = compute_exit_outcome(M_t, M_t1, arcsinh_t, args.sd_mult)

        if SD_MULT_SWEEP:
            sweep_cache.append({
                "period": period_name, "t": t, "codes": codes,
                "M_t": M_t, "M_t1": M_t1, "arcsinh_t": arcsinh_t,
                "density": density, "mi": mi, "ci": ci,
            })

        panel_frames.append(pd.DataFrame({
            "period":        period_name,
            "year":          t,
            "municipality":  [all_municipalities[i] for i in mi],
            "industry":      [codes[j]              for j in ci],
            "density":       density      [mi, ci],
            "exit_industry": exit_industry[mi, ci],
        }))
        print(f"    {t}:  candidate rows={len(mi):,}  "
              f"exit_industry rate={exit_industry[mi, ci].mean():.4%}")

panel = pd.concat(panel_frames, ignore_index=True)
panel["exit_industry"] = panel["exit_industry"].astype(float)

firmsize_outcomes: list[str] = []
if tier_lookups:
    for tier, tier_lookup in tier_lookups.items():
        tag_col = f"has_dead_estab_{tier}"
        panel = panel.merge(
            tier_lookup[["industry", "municipality", "year", tag_col]],
            on=["industry", "municipality", "year"], how="left",
        )
        panel[tag_col] = (panel[tag_col] == True).astype(float)  # noqa: E712 — NaN -> False
        # Gated industry outcome: a "success" for this tier is exit_industry
        # AND a qualifying establishment of that tier's size triggered the
        # cell. All tiers share the same negative (exit_industry=0) sample.
        outcome_col = f"exit_industry_{tier}"
        panel[outcome_col] = panel["exit_industry"] * panel[tag_col]
        firmsize_outcomes.append(outcome_col)

print(f"\nCandidate-exit set panel: {len(panel):,} rows")
print(f"  exit_industry rate: {panel['exit_industry'].mean():.4%}")
print(f"    mean density | exit_industry=1: {panel.loc[panel['exit_industry']==1, 'density'].mean():.5f}")
print(f"    mean density | exit_industry=0: {panel.loc[panel['exit_industry']==0, 'density'].mean():.5f}")

_sd_density = float(panel["density"].std())
panel["density_std"] = panel["density"] / _sd_density
print(f"\nDensity pre-standardised (÷ SD = {_sd_density:.6f}, pooled across all "
      "periods purely as a fixed unit) — β reads as Δ P(exit) per 1-SD "
      "increase in density")


# ═══════════════════════════════════════════════════════════════════════════════
# 5+6. Per-classification-period demeaning and regression — linear
#    probability model with industry + municipality + year FE via triple
#    demeaning (separate additive municipality and year effects, not an
#    interacted muni×year term). SNI92/SNI02/SNI07 code strings overlap
#    heavily (e.g. 488 of SNI92's 505 codes also appear in SNI02's own code
#    list) but refer to different classification vintages — pooling them
#    under one "industry" fixed effect would treat a 1995 code and a 2003
#    code that happen to share the same digits as the same FE level. Each
#    classification period is therefore demeaned and regressed
#    independently; results are never pooled across periods.
# ═══════════════════════════════════════════════════════════════════════════════

def demean(df: pd.DataFrame, col: str, by) -> pd.Series:
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
    # Single-regressor OLS: R² is the squared (partial) correlation between
    # x and y net of the FE — taking its root with the coefficient's sign
    # gives a correlation-like number in [-1, 1], unlike the raw coef, whose
    # scale depends on how much within-FE variance x has left (NOT directly
    # comparable across periods or embedding runs with different density
    # distributions).
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

print("\n" + "=" * 60)
print("Relatedness density → exit  (per classification period, industry + "
      "municipality + year FE, SE clustered by muni)")
print("=" * 60)

periods_order = list(dict.fromkeys(panel["period"]))   # preserve PERIODS order
dm_frames: list[pd.DataFrame] = []
results: dict[str, dict] = {}

for period_name in periods_order:
    sub = panel[panel["period"] == period_name].copy()
    for var in [OUTCOME, "density_std"]:
        sub[f"{var}_r1"] = demean(sub, var,         "municipality")
        sub[f"{var}_r2"] = demean(sub, f"{var}_r1",  "year")
        sub[f"{var}_dm"] = demean(sub, f"{var}_r2",  "industry")

    within_fe_sd = float(sub["density_std_dm"].std())
    print(f"\n  -- {period_name} --  (within-FE density_std SD={within_fe_sd:.6f}, "
          f"{within_fe_sd**2:.4%} of raw variance retained)")

    res  = run_ols(sub[f"{OUTCOME}_dm"], sub["density_std_dm"], sub["municipality"])
    rate = float(sub[OUTCOME].mean())
    # β in raw probability points is hard to judge against a base rate this
    # low — express it relative to the outcome's own baseline rate too, as a
    # second, scale-free read (doesn't change coef/SE/p, just adds context).
    res["pct_of_baseline"] = res["coef"] / rate * 100 if rate > 0 else float("nan")
    res["rate"] = rate
    results[period_name] = res
    print(f"    exit_industry (drops from significant local size to zero)")
    print(f"      coef={res['coef']:+.6f}  SE={res['se']:.6f}"
          f"  t={res['t']:.3f}  p={res['pval']:.3e}")
    print(f"      N={res['n']:,}  R²={res['r2']:.5f}  partial r={res['partial_r']:+.4f}")
    print(f"      baseline rate={rate:.4%}  →  β = {res['pct_of_baseline']:+.1f}% of baseline"
          f"  per 1-SD density")

    dm_frames.append(sub)

panel = pd.concat(dm_frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 6b. Optional (--stratify-firm-size): exit_industry gated by closing
#    establishment size tercile, regressed against density separately — same
#    candidate panel, industry + municipality + year FE, density_std as
#    above, just a different outcome column per tier. Reported and plotted
#    on its own (not folded into the main exit_industry figure) so it
#    doesn't complicate that figure's panels.
# ═══════════════════════════════════════════════════════════════════════════════

firmsize_results: dict[str, dict[str, dict]] = {}
if firmsize_outcomes:
    print("\n" + "=" * 60)
    print("Relatedness density → exit_industry, by exiting establishment size tercile "
          "(" + ", ".join(firmsize_outcomes) + ")")
    print("=" * 60)

    for period_name in periods_order:
        sub = panel[panel["period"] == period_name].copy()
        for var in firmsize_outcomes + ["density_std"]:
            sub[f"{var}_r1"] = demean(sub, var,         "municipality")
            sub[f"{var}_r2"] = demean(sub, f"{var}_r1",  "year")
            sub[f"{var}_dm"] = demean(sub, f"{var}_r2",  "industry")

        firmsize_results[period_name] = {}
        for outcome in firmsize_outcomes:
            res  = run_ols(sub[f"{outcome}_dm"], sub["density_std_dm"], sub["municipality"])
            rate = float(sub[outcome].mean())
            res["pct_of_baseline"] = res["coef"] / rate * 100 if rate > 0 else float("nan")
            res["rate"] = rate
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
                "tier": outcome.removeprefix("exit_industry_"),
                "coef": res["coef"], "se": res["se"], "pval": res["pval"],
                "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                "n_obs": res["n"], "rate": res["rate"],
                "pct_of_baseline": res["pct_of_baseline"],
            })
    firmsize_df = pd.DataFrame(firmsize_rows)
    firmsize_csv_path = results_dir / f"{OUT_NAME}_firmsize_summary.csv"
    firmsize_df.to_csv(firmsize_csv_path, index=False)
    print(f"\n  → {firmsize_csv_path}")

    tiers_present = [t.removeprefix("exit_industry_") for t in firmsize_outcomes]
    tier_results = {
        period_name: {
            tier: firmsize_results[period_name][f"exit_industry_{tier}"]
            for tier in tiers_present
        }
        for period_name in periods_order
    }
    firmsize_fig_path = plot_density_effect_by_tier(
        tier_results, periods_order, tiers_present,
        xlabel="Closing establishment's size tercile",
        effect_label="Effect on exit probability per 1-SD density (in %)",
        title="Density effect on exit, by exiting firm size",
        fig_path=results_dir / f"{OUT_NAME}_exit_industry_firmsize.png",
    )
    print(f"  → {firmsize_fig_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Year-by-year regression — already period-safe: a calendar year never
#    spans two classification periods, so the within-year industry FE never
#    mixes code systems even though it isn't filtered by period explicitly.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Year-by-year coefficients  (density → exit_industry)")
print("=" * 60)

year_records: list[dict] = []
for year, sub in panel.groupby("year"):
    sub = sub.copy()
    for var in [OUTCOME, "density_std"]:
        sub[f"{var}_dm"] = demean(sub, var, "municipality")
        sub[f"{var}_dm"] = demean(sub, f"{var}_dm", "industry")
    r = run_ols(sub[f"{OUTCOME}_dm"], sub["density_std_dm"], sub["municipality"])
    r["year"] = int(year)
    year_records.append(r)
    print(f"  {year}→{year+1}:  exit_industry β={r['coef']:+.5f} (p={r['pval']:.2e})   N={r['n']:,}")

year_df = pd.DataFrame(year_records).sort_values("year")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Visualisations
# ═══════════════════════════════════════════════════════════════════════════════

_fig_path = plot_density_effect_summary(
    panel, year_df, results, periods_order, OUTCOME, OUTCOME_LABEL, EFFECT_LABEL, COLOR,
    results_dir / f"{OUT_NAME}_exit.png",
)
print(f"\n  → {_fig_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Save outputs
# ═══════════════════════════════════════════════════════════════════════════════

panel_out = panel[["period", "year", "municipality", "industry", "density", OUTCOME]]
panel_out.to_parquet(OUT_DIR / f"{OUT_NAME}_panel.parquet", index=False)
print(f"  → {OUT_DIR / f'{OUT_NAME}_panel.parquet'}  ({len(panel):,} rows)")

year_df.to_csv(results_dir / f"{OUT_NAME}_year_coefs.csv", index=False)

summary_rows = []
for period_name in periods_order:
    in_period = panel["period"] == period_name
    row = {
        "run":               args.run_name,
        "density_var":       args.density_var,
        "log_employment_weights": args.log_employment_weights,
        "period":            period_name,
        "tau":               args.tau,
        "min_workers":       args.min_workers,
        "min_worker_ratio":  args.min_worker_ratio,
        "sd_mult":           args.sd_mult,
        "n_obs":             int(in_period.sum()),
        "n_municipalities":  len(all_municipalities),
        f"{OUTCOME}_rate":            panel.loc[in_period, OUTCOME].mean(),
        f"{OUTCOME}_coef":            results[period_name]["coef"],
        f"{OUTCOME}_se":              results[period_name]["se"],
        f"{OUTCOME}_pval":            results[period_name]["pval"],
        f"{OUTCOME}_r":               results[period_name]["partial_r"],
        f"{OUTCOME}_pct_of_baseline": results[period_name]["pct_of_baseline"],
    }
    summary_rows.append(row)
pd.DataFrame(summary_rows).to_csv(results_dir / f"{OUT_NAME}_summary.csv", index=False)

print(f"  → {results_dir}/  ({OUT_NAME}_year_coefs.csv, _summary.csv)  "
      f"+ {OUT_DIR}/  ({OUT_NAME}_panel.parquet)")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Appendix robustness check: sd_mult sweep (only if --sd-mult-sweep given)
#    sd_mult is the parameter most worth stress-testing here — unlike tau (which
#    only reweights the density regressor), sd_mult defines the OUTCOME itself
#    (which drops-to-zero count as a "significant" exit at all), so it directly
#    determines which rows enter the regression, not just how one predictor is
#    constructed. density, M_t and M_t1 don't depend on sd_mult, so the sweep
#    reuses the matrices cached in section 4 and only redoes the (cheap) outcome
#    recomputation + demeaning + OLS per value — no re-read of the individual
#    panel and no re-computation of embeddings/kernels.
# ═══════════════════════════════════════════════════════════════════════════════

if SD_MULT_SWEEP:
    print("\n" + "=" * 60)
    print(f"Appendix robustness check: sd_mult sweep {SD_MULT_SWEEP}")
    print("=" * 60)

    sweep_records: list[dict] = []
    for sd_mult_val in SD_MULT_SWEEP:
        frames = []
        for c in sweep_cache:
            mi, ci = c["mi"], c["ci"]
            mask = compute_exit_outcome(c["M_t"], c["M_t1"], c["arcsinh_t"], sd_mult_val)
            frames.append(pd.DataFrame({
                "period":        c["period"],
                "year":          c["t"],
                "municipality":  [all_municipalities[i] for i in mi],
                "industry":      [c["codes"][j]         for j in ci],
                "density_std":   c["density"][mi, ci] / _sd_density,
                "exit_industry": mask[mi, ci].astype(float),
            }))
        sweep_panel = pd.concat(frames, ignore_index=True)

        for period_name in periods_order:
            sub  = sweep_panel[sweep_panel["period"] == period_name].copy()
            rate = float(sub["exit_industry"].mean())
            sub["exit_r1"] = demean(sub, "exit_industry", "municipality")
            sub["exit_r2"] = demean(sub, "exit_r1",        "year")
            sub["exit_dm"] = demean(sub, "exit_r2",        "industry")
            sub["dens_r1"] = demean(sub, "density_std",    "municipality")
            sub["dens_r2"] = demean(sub, "dens_r1",         "year")
            sub["dens_dm"] = demean(sub, "dens_r2",         "industry")
            res = run_ols(sub["exit_dm"], sub["dens_dm"], sub["municipality"])
            pct_of_baseline = res["coef"] / rate * 100 if rate > 0 else float("nan")
            sweep_records.append({
                "run": args.run_name, "density_var": args.density_var,
                "period": period_name, "sd_mult": sd_mult_val,
                "n_obs": res["n"], "exit_industry_rate": rate,
                "coef": res["coef"], "se": res["se"], "pval": res["pval"],
                "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                "partial_r": res["partial_r"], "pct_of_baseline": pct_of_baseline,
            })
            print(f"  sd_mult={sd_mult_val:<4}  {period_name}:  rate={rate:.4%}  "
                  f"coef={res['coef']:+.6f}  pct_of_baseline={pct_of_baseline:+6.1f}%  "
                  f"p={res['pval']:.2e}  N={res['n']:,}")

    sweep_df = pd.DataFrame(sweep_records)
    sweep_csv_path = results_dir / f"{OUT_NAME}_sdmult_robustness.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)
    print(f"\n  → {sweep_csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for period_name in periods_order:
        sub = sweep_df[sweep_df["period"] == period_name].sort_values("sd_mult")
        ax.fill_between(sub["sd_mult"], sub["ci_low"], sub["ci_high"], alpha=0.15)
        ax.plot(sub["sd_mult"], sub["coef"], "o-", label=period_name)
    ax.axvline(args.sd_mult, color="black", linewidth=1, linestyle=":",
               label=f"baseline (sd_mult={args.sd_mult})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("sd_mult  (significance-band width, arcsinh SDs)")
    ax.set_ylabel("β  (P(exit) per 1-SD density)")
    ax.set_title("Coefficient vs sd_mult")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    ax = axes[1]
    for period_name in periods_order:
        sub = sweep_df[sweep_df["period"] == period_name].sort_values("sd_mult")
        ax.plot(sub["sd_mult"], sub["pct_of_baseline"], "o-", label=period_name)
    ax.axvline(args.sd_mult, color="black", linewidth=1, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("sd_mult  (significance-band width, arcsinh SDs)")
    ax.set_ylabel("β as % of that sd_mult's baseline exit rate")
    ax.set_title("Relative effect size vs sd_mult")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.suptitle(f"Appendix robustness: sd_mult sweep  (run={OUT_NAME})", fontsize=13, y=1.03)
    plt.tight_layout()
    sweep_fig_path = results_dir / f"{OUT_NAME}_sdmult_robustness.png"
    fig.savefig(sweep_fig_path, dpi=150, bbox_inches="tight")
    print(f"  → {sweep_fig_path}")
    plt.close()

print("\nDone.")
