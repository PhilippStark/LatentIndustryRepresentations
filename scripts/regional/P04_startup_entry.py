"""
P04_startup_entry.py
====================
Relatedness density and start-up entry: is a genuinely new firm more likely
to be founded in an industry that sits close to the municipality's existing
employment structure? Requires the restricted SCB register data (individual
panel and FAD firm demography), except with --from-cache.

Start-up definition (Grillitsch 2021), on the founding OrgNr in year t
-----------------------------------------------------------------------
  1. FAD firm-demography code Kvar in KVAR_STARTUP_CODES {1, 2, 3, 6, 7, 8}
     ("new" enterprise; mergers and continuing firms excluded; codebook in
     metadata/mapping_Kvar.json).
  2. The OrgNr is absent from the individual panel in t−1 (so 1990 cannot
     qualify).
  3. Exactly one establishment (CfarNr) in t.
  4. At most --max-founder-size employees in t (default 10).
  5. Optional: --max-growth-ratio excludes firms whose headcount in t+1
     exceeds this multiple of the founding size (0 = off, default).

Candidates and outcome
----------------------
Same candidates as P04_entry.py: (municipality, industry, year t) cells with
zero employment in t.

  startup_entry = 1 if a qualifying start-up is founded in the cell in t+1.

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
  --include-entry-industry  also report P04_entry.py's entry_industry outcome
                            for comparison
  --from-cache              re-estimate and replot from the cached panel (no microdata)

Outputs
-------
  results/regional_startup_entry/{run}{suffix}_{startup_entry.png, summary.csv, year_coefs.csv}
  <LATENT_DATA_ROOT>/processed/downstream_outcomes/startup_entry/{run}{suffix}_panel.parquet

Usage (main run)
----------------
  python scripts/regional/P04_startup_entry.py --run-name shared_cosine_v16
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import polars as pl

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import EMBEDDING_DIR, METADATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from latent_industry.common.municipalities import harmonization_table, load_municipality_crosswalk
from latent_industry.regional.density import (
    load_sni_embeddings, make_softmax_kernel, cosine_matrix, employment_weighted_density,
    geometric_mean_employment_weighted_density, empl_matrix as _shared_empl_matrix,
)
from latent_industry.regional.fixed_effects import standardise_density
from latent_industry.regional.reporting import report_density_regressions

DATA_ROOT    = Path("/lunarc/nobackup/projects/lu2025-17-39")
INDIVID_PATH = DATA_ROOT / "SCB" / "Preprocessed" / "individ_cleaned_1990-2019.parquet"
FAD_PATH     = RAW_DATA_DIR / "FAD_1990-2019.parquet"
EMBED_BASE   = EMBEDDING_DIR
MUNI_XWALK   = harmonization_table(load_municipality_crosswalk())

# results_dir: small, paper-facing outputs only (figures + CSVs) — tracked in
# git. panel.parquet is a large intermediate cache and goes to OUT_DIR on
# LUNARC storage instead, same convention as P04_startup_bridging.py. NOTE:
# P04_startup_bridging.py reads this run's panel.parquet back from OUT_DIR
# (as its own "startup_dir" input) — keep both in sync if this path changes.
results_dir = project_root / "results" / "regional_startup_entry"
results_dir.mkdir(parents=True, exist_ok=True)
OUT_DIR = PROCESSED_DATA_DIR / "downstream_outcomes" / "startup_entry"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with (METADATA_DIR / "mapping_Kvar.json").open(encoding="utf-8") as _f:
    KVAR_LABELS = json.load(_f)

# Genuine new-firm formation only — excludes merger-driven "new" entities
# (05/09) and surviving/continuing firms (11/12/13). See module docstring.
KVAR_STARTUP_CODES = {1, 2, 3, 6, 7, 8}

# ── Defaults — edit here when running directly in VS Code ─────────────────────
DEFAULT_RUN_NAME         = "shared_cosine_v16"
DEFAULT_TAU              = 1.0    # softmax temperature for the relatedness-density kernel
DEFAULT_MAX_FOUNDER_SIZE = 10     # Grillitsch (2021): founder size <= 10 employees
DEFAULT_MAX_GROWTH_RATIO = 0.0    # 0 disables the optional >30x jump-exclusion robustness check
DEFAULT_SD_MULT          = 2.0    # significance band (arcsinh scale); only used with --include-entry-industry
SAMPLE_START_YEAR        = 1990   # can't verify "absent in t-1" for this year — no 1989 data in the panel

OUTCOME       = "startup_entry"
OUTCOME_LABEL = "Observed start-up rate"
EFFECT_LABEL  = "Effect on start-up probability per 1-SD density (in %)"
COLOR         = "seagreen"

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
parser.add_argument("--max-founder-size", type=int, default=DEFAULT_MAX_FOUNDER_SIZE,
                    help="Founder-year headcount must be <= this (Grillitsch 2021's size threshold).")
parser.add_argument("--max-growth-ratio", type=float, default=DEFAULT_MAX_GROWTH_RATIO,
                    help="Optional robustness filter: exclude startups whose (t+1 headcount)/(founder "
                         "size) ratio exceeds this multiple. 0 disables it (default).")
parser.add_argument("--sd-mult", type=float, default=DEFAULT_SD_MULT,
                    help="SD multiplier (arcsinh scale) for entry_industry's significance band. "
                         "Only used together with --include-entry-industry.")
parser.add_argument("--include-entry-industry", default=False, action="store_true",
                    help="Also report P04_entry.py's crude entry_industry outcome side by side, as a "
                         "reference point only. Off by default.")
parser.add_argument("--from-cache", default=False, action="store_true",
                    help="Skip loading the SCB microdata and rerunning the regressions entirely — "
                         "replot the paper figure directly from this run's already-written "
                         "{run}_panel.parquet / _year_coefs.csv / _summary.csv (all three are written "
                         "by a normal full run).")
args, _ = parser.parse_known_args()

if args.log_employment_weights and args.density_var == "softmax":
    print("--log-employment-weights has no effect with --density-var softmax — ignoring.")

# Baseline is --density-var ew --log-employment-weights (paper's D^{A,log}) and
# gets no filename suffix; every other combination writes to a separate
# {run_name}_* namespace so it never overwrites the baseline run's already-
# written startup_entry.png / panel.parquet / summary.csv for the same --run-name.
OUT_NAME = args.run_name
if args.density_var == "softmax":
    OUT_NAME += "_softmax"
elif args.density_var == "geo":
    OUT_NAME += "_geo" + ("_log" if args.log_employment_weights else "_raw")
elif args.density_var == "ew" and not args.log_employment_weights:
    OUT_NAME += "_ew_raw"
OUT_NAME += args.out_suffix

print(f"Run: {args.run_name}   density_var={args.density_var}   tau={args.tau}   "
      f"max_founder_size<={args.max_founder_size}   "
      f"max_growth_ratio={args.max_growth_ratio}   include_entry_industry={args.include_entry_industry}")

OUTCOMES = ["startup_entry"] + (["entry_industry"] if args.include_entry_industry else [])
SUMMARY_EXTRA = {
    "run": args.run_name, "density_var": args.density_var,
    "log_employment_weights": args.log_employment_weights, "tau": args.tau,
    "max_founder_size": args.max_founder_size, "max_growth_ratio": args.max_growth_ratio,
}


def report(panel: pd.DataFrame) -> None:
    """Density regressions (main specification) for start-up entry and, with
    --include-entry-industry, for the entry_industry reference outcome."""
    report_density_regressions(
        panel, OUTCOME, OUTCOME_LABEL, EFFECT_LABEL, COLOR,
        fig_path=results_dir / f"{OUT_NAME}_startup_entry.png",
        summary_path=results_dir / f"{OUT_NAME}_summary.csv",
        year_path=results_dir / f"{OUT_NAME}_year_coefs.csv",
        summary_extra=SUMMARY_EXTRA,
    )
    if "entry_industry" in OUTCOMES and "entry_industry" in panel.columns:
        report_density_regressions(
            panel, "entry_industry", "Observed entry rate",
            "Effect on entry probability per 1-SD density (in %)", "steelblue",
            fig_path=results_dir / f"{OUT_NAME}_entry_industry_reference.png",
            summary_path=results_dir / f"{OUT_NAME}_entry_industry_reference_summary.csv",
            year_path=results_dir / f"{OUT_NAME}_entry_industry_reference_year_coefs.csv",
            summary_extra=SUMMARY_EXTRA,
        )


if args.from_cache:
    print(f"\n--from-cache: re-estimating from {OUT_DIR}/{OUT_NAME}_panel.parquet "
          f"(no SCB microdata needed).")
    _panel_path = OUT_DIR / f"{OUT_NAME}_panel.parquet"
    if not _panel_path.exists():
        raise SystemExit(f"--from-cache requires {_panel_path} — run this script without "
                          f"--from-cache once first.")
    _panel = pd.read_parquet(_panel_path)
    standardise_density(_panel)
    report(_panel)
    print("\nDone (--from-cache).")
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load individual data — one filtered/harmonized base frame reused for both
#    the (municipality, year, industry) employment panel (density input) and
#    the (OrgNr, year) aggregate (startup-eligibility input), so the 225M-row
#    parquet is scanned once, not twice.
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading {INDIVID_PATH} …")
base = (
    pl.scan_parquet(INDIVID_PATH)
    .select(["Year", "Sni", "WorkplaceMunicipality", "OrgNr", "CfarNr"])
    .filter(pl.col("Sni").is_not_null() & (pl.col("Sni") != 0))
    .filter(pl.col("WorkplaceMunicipality").is_not_null())
    .filter(pl.col("OrgNr").is_not_null() & (pl.col("OrgNr") != 0))
    .with_columns([
        pl.col("Sni").cast(pl.Utf8).str.zfill(5).str.slice(0, 4).alias("sni_4d"),
        pl.col("WorkplaceMunicipality").cast(pl.Utf8).str.zfill(4).alias("source_code"),
    ])
    .filter(pl.col("sni_4d") != "0000")
    .join(MUNI_XWALK.lazy(), on=["Year", "source_code"], how="inner")
    .collect()
)
print(f"  {len(base):,} filtered/harmonized individual-year rows")

empl = (
    base.group_by(["municipality", "Year", "sni_4d"])
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
# 2. Grillitsch (2021) startup identification
# ═══════════════════════════════════════════════════════════════════════════════

print("\nAggregating to (OrgNr, Year) for startup eligibility …")
org_year = base.group_by(["OrgNr", "Year"]).agg([
    pl.len().alias("founder_size"),
    pl.col("CfarNr").n_unique().alias("n_cfarnr"),
    pl.col("CfarNr").first().alias("cfarnr"),
    pl.col("sni_4d").first().alias("sni_4d"),
    pl.col("municipality").first().alias("municipality"),
])
print(f"  {len(org_year):,} (OrgNr, Year) rows")

# Condition 2: OrgNr absent in t-1 — a row (OrgNr, Y) in org_year shifted to
# (OrgNr, Y+1) marks "OrgNr existed the year before Y+1"; anti-joining
# org_year against that shifted table keeps exactly the rows where the OrgNr
# did NOT exist the previous year. Year 1990 can never pass this (no 1989
# data in the panel) and is dropped explicitly.
prev_existence = org_year.select(["OrgNr", "Year"]).with_columns((pl.col("Year") + 1).alias("Year"))
eligible = (
    org_year.join(prev_existence, on=["OrgNr", "Year"], how="anti")
    .filter(pl.col("Year") > SAMPLE_START_YEAR)
    # Condition 3 + 4: exactly one establishment, founder size within threshold.
    .filter((pl.col("n_cfarnr") == 1) & (pl.col("founder_size") <= args.max_founder_size))
)
print(f"  {len(eligible):,} OrgNr-year rows pass absent-in-t-1 + single-CfarNr + "
      f"founder-size<={args.max_founder_size} filters")

print(f"\nLoading {FAD_PATH} …")
print("  Kvar codes counted as genuine new-firm formation:")
for _c in sorted(KVAR_STARTUP_CODES):
    print(f"    Kvar={_c:02d}: {KVAR_LABELS.get(f'{_c:02d}', '?')}")

fad = (
    pl.scan_parquet(FAD_PATH)
    .select(["OrgNr", "CfarNr", "Year", "Kvar"])
    .filter(pl.col("Kvar").is_in(list(KVAR_STARTUP_CODES)))
    .with_columns([
        pl.col("OrgNr").cast(pl.UInt32),
        pl.col("CfarNr").cast(pl.UInt32).alias("cfarnr"),
        pl.col("Year").cast(pl.UInt16),
    ])
    .select(["OrgNr", "cfarnr", "Year", "Kvar"])
    .collect()
)
print(f"  {len(fad):,} FAD rows with a qualifying Kvar code")

# Condition 1: the eligible OrgNr-year must ALSO carry a qualifying Kvar
# transition for that exact (OrgNr, CfarNr, Year) — matched against
# individ's own CfarNr (condition 3 already pinned this to a single value),
# not blindly trusted from FAD.
startups = eligible.join(fad, on=["OrgNr", "cfarnr", "Year"], how="inner")
print(f"  {len(startups):,} OrgNr-year rows are Grillitsch-compliant startups "
      f"(pre growth-ratio filter)")

if args.max_growth_ratio > 0:
    next_size = (
        org_year.select(["OrgNr", "Year", "founder_size"])
        .rename({"founder_size": "next_size"})
        .with_columns((pl.col("Year") - 1).alias("Year"))  # align t+1's size onto birth year t
    )
    before = len(startups)
    startups = startups.join(next_size, on=["OrgNr", "Year"], how="left")
    ratio_ok = pl.col("next_size").is_null() | (
        pl.col("next_size") / pl.col("founder_size").clip(lower_bound=1) <= args.max_growth_ratio
    )
    startups = startups.filter(ratio_ok)
    print(f"  --max-growth-ratio {args.max_growth_ratio}: {before:,} -> {len(startups):,} rows "
          f"(firms disappearing at t+1 are not excluded — undefined ratio)")

startup_counts = (
    startups.group_by(["municipality", "sni_4d", "Year"])
    .agg(pl.len().alias("n_startups"))
    .rename({"sni_4d": "industry", "Year": "year_t1"})
    .with_columns(pl.lit(True).alias("startup_entry"))
    .to_pandas()
)
print(f"  {len(startup_counts):,} distinct (industry, municipality, year) cells "
      f"with at least one qualifying startup")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Load embeddings and build the relatedness-density kernel
#    (kernels shared with P04_entry.py / P04_exit.py / P04_growth.py via
#    latent_industry.regional.density)
# ═══════════════════════════════════════════════════════════════════════════════

sni_embeddings = load_sni_embeddings(args.run_name, EMBED_BASE)
vecs07, codes07 = sni_embeddings["sni07"]
vecs02, codes02 = sni_embeddings["sni02"]
vecs92, codes92 = sni_embeddings["sni92"]


def empl_matrix(year: int, codes: list[str]) -> np.ndarray:
    return _shared_empl_matrix(empl, all_municipalities, year, codes)


PERIODS = [
    ("SNI92", codes92, vecs92, range(1990, 2001)),
    ("SNI02", codes02, vecs02, range(2002, 2006)),
    ("SNI07", codes07, vecs07, range(2007, 2019)),
]


def significance_band(arcsinh_vals: np.ndarray, active_mask: np.ndarray, sd_mult: float) -> tuple[np.ndarray, np.ndarray]:
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


def compute_entry_industry_outcome(M_t1: np.ndarray, arcsinh_t1: np.ndarray, sd_mult: float) -> np.ndarray:
    active1 = M_t1 > 0
    sig_lo, sig_hi = significance_band(arcsinh_t1, active1, sd_mult)
    return (
        (M_t1 > 0)
        & (arcsinh_t1 >= sig_lo[:, None]) & (arcsinh_t1 <= sig_hi[:, None])
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Build the candidate-entry set panel: industries absent at t, density at
#    t, startup_entry (and optionally entry_industry) outcomes observed at t+1
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

        absent = M_t == 0
        mi, ci = np.where(absent)

        row = {
            "period":       period_name,
            "year":         t,
            "municipality": [all_municipalities[i] for i in mi],
            "industry":     [codes[j]              for j in ci],
            "density":      density[mi, ci],
        }
        if args.include_entry_industry:
            M_t1 = empl_matrix(t1, codes)
            arcsinh_t1 = np.arcsinh(M_t1)
            entry_industry = compute_entry_industry_outcome(M_t1, arcsinh_t1, args.sd_mult)
            row["entry_industry"] = entry_industry[mi, ci]

        panel_frames.append(pd.DataFrame(row))
        print(f"    {t}:  candidate rows={len(mi):,}")

panel = pd.concat(panel_frames, ignore_index=True)

panel["year_t1"] = panel["year"] + 1
panel = panel.merge(
    startup_counts[["industry", "municipality", "year_t1", "startup_entry"]],
    on=["industry", "municipality", "year_t1"], how="left",
)
panel["startup_entry"] = (panel["startup_entry"] == True).astype(float)  # noqa: E712 — NaN -> False
if args.include_entry_industry:
    panel["entry_industry"] = panel["entry_industry"].astype(float)

print(f"\nCandidate-entry set panel: {len(panel):,} rows")
for outcome in OUTCOMES:
    print(f"  {outcome:<14} rate: {panel[outcome].mean():.4%}")
    print(f"    mean density | {outcome}=1: {panel.loc[panel[outcome]==1, 'density'].mean():.5f}")
    print(f"    mean density | {outcome}=0: {panel.loc[panel[outcome]==0, 'density'].mean():.5f}")

_sd_density = standardise_density(panel)
print(f"\nDensity standardised by its pooled SD ({_sd_density:.6f}): β is the change in "
      f"P(start-up entry) per 1-SD increase in density")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Density regressions per classification period (and year-specific slopes):
#    municipality×year + industry fixed effects, SE clustered by municipality
#    and industry (latent_industry.regional.fixed_effects).
# ═══════════════════════════════════════════════════════════════════════════════

report(panel)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Save the panel
# ═══════════════════════════════════════════════════════════════════════════════

panel_out = panel[["period", "year", "municipality", "industry", "density"] + OUTCOMES]
panel_out.to_parquet(OUT_DIR / f"{OUT_NAME}_panel.parquet", index=False)
print(f"  → {OUT_DIR / f'{OUT_NAME}_panel.parquet'}  ({len(panel):,} rows)")

print("\nDone.")
