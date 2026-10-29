"""
build_labor_relatedness.py
==========================
Labor (skill) relatedness between 4-digit industries from worker flows,
following Neffke & Henning (2013). Requires the restricted SCB individual-level
register panel.

Worker flows
------------
A flow i→j is a worker whose 4-digit industry is i in one year and j ≠ i in
the next (stayers are excluded; an employer change within the same industry
is not a flow). Flows are counted only within a classification
period (SNI92: 1990–2001, SNI02: 2002–2006, SNI07: 2007–2019); year pairs
that straddle a reclassification are skipped, because a code change there is
a coding artifact rather than a move between industries.

Observed-over-expected relatedness
----------------------------------
Per period, a Poisson regression over the full grid of industry pairs
(unobserved pairs zero-filled) predicts the flow expected from industry
characteristics alone:

  log E[f(i→j)] = b0 + b1 log(size_i) + b2 log(size_j) + b3 growth_i
                     + b4 growth_j + b5 log(income_i) + b6 log(income_j)

with size (headcount), growth (year-on-year headcount change) and income
(mean employment income) computed from the register panel. Relatedness is the
smoothed observed/expected ratio, symmetrised:

  R(i,j) = (f(i→j) + k) / (E[f(i→j)] + k),   k = SHRINKAGE_K = 1
  phi(i,j) = (R(i,j) + R(j,i)) / 2

Poisson pseudo-maximum likelihood gives a consistent conditional mean under
over-dispersion, which is all that is used here. The additive smoothing k
pulls pairs with very few observed movers and a near-zero expected flow back
towards 1.

Minimum evidence
----------------
A pair is kept only if at least --min-labor-count workers moved between the
two industries (both directions, pooled over the period); default 10, as in
the main run.

Spatial hold-out
----------------
--exclude-municipalities FILE drops every person-year whose (harmonised)
workplace municipality is listed in FILE before any flow or covariate is
computed (scripts/data/_municipality_exclusion.py), so no flow that starts
or ends there enters the relatedness. Off by default.

Output
------
  <LATENT_DATA_ROOT>/processed/labor_relatedness.parquet
    sni_from, sni_to, phi, coding (SNI92 / SNI02 / SNI07)

Usage
-----
  python scripts/data/build_labor_relatedness.py
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import METADATA_DIR, PROCESSED_DATA_DIR

DATA_ROOT    = Path("/lunarc/nobackup/projects/lu2025-17-39")
INDIVID_PATH = DATA_ROOT / "SCB" / "Preprocessed" / "individ_cleaned_1990-2019.parquet"
metadata_dir = METADATA_DIR
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--max-year",
    type=int,
    default=2019,
    help="Last register year allowed in transitions and expected-flow covariates.",
)
parser.add_argument(
    "--output-file",
    default="labor_relatedness.parquet",
    help="Absolute path or filename under PROCESSED_DATA_DIR.",
)
parser.add_argument(
    "--min-labor-count",
    type=int,
    default=10,
    help="Minimum total observed movers (both directions, pooled over the whole period) "
         "for a pair to be kept at all. "
         "Raising this keeps only higher-evidence pairs (fewer, more confident edges).",
)
parser.add_argument(
    "--exclude-municipalities",
    default=None,
    help="CSV with a `municipality` column (harmonised codes): person-years working "
         "there are dropped before any flow is counted (spatial hold-out). Off by default.",
)
args, _ = parser.parse_known_args()
if args.max_year < 1991:
    raise ValueError("--max-year must permit at least one adjacent-year transition")
output_path = Path(args.output_file)
if not output_path.is_absolute():
    output_path = PROCESSED_DATA_DIR / output_path

# Minimum total observed movers (both directions, pooled over the whole
# period) for a pair to be kept at all — see module docstring.
MIN_LABOR_COUNT = args.min_labor_count

# Additive smoothing on the observed/expected ratio itself: R = (count+k)/
# (expected+k) — see module docstring.
SHRINKAGE_K = 1.0

# Year coverage of each classification regime — used both for the flow-matrix
# year pairs below and for which years' headcount/income feed that regime's
# size/growth/income stats. Growth is never computed across a boundary (a
# digit string can mean a different industry under a different SNI version).
def capped_years(start: int, end: int) -> range:
    capped_end = min(end, args.max_year)
    return range(start, capped_end + 1) if capped_end >= start else range(0)


REGIME_YEARS = {
    "SNI92": capped_years(1990, 2001),
    "SNI02": capped_years(2002, 2006),
    "SNI07": capped_years(2007, 2019),
}
UNIVERSE_FILE_TAG = {"SNI92": "1992", "SNI02": "2002", "SNI07": "2007"}

# ── Load and prepare raw individual data ─────────────────────────────────────────
# Expected columns: PersonNr, Year, Sni, EmploymentIncome.
# Sni holds the period-appropriate code for every year; SNI02 is an auxiliary
# overlap column and is ignored here.

print(f"Loading individual data from {INDIVID_PATH} …")
raw = pl.read_parquet(INDIVID_PATH)
raw = raw.filter(pl.col("Year") <= args.max_year)

exclusion_stats = None
if args.exclude_municipalities:
    from _municipality_exclusion import drop_excluded_municipalities, load_excluded_municipalities

    raw, exclusion_stats = drop_excluded_municipalities(
        raw, load_excluded_municipalities(args.exclude_municipalities)
    )
    exclusion_stats["file"] = str(Path(args.exclude_municipalities).resolve())
    print(f"  --exclude-municipalities: {exclusion_stats['n_excluded_municipalities']} municipalities, "
          f"{exclusion_stats['rows_dropped']:,} of {exclusion_stats['rows_before']:,} person-years dropped "
          f"({exclusion_stats['rows_unmapped_kept']:,} unmapped kept)")

if "Sni" not in raw.columns:
    raise ValueError("Expected column 'Sni' in individual data.")

print(f"  Years: {raw['Year'].min()}–{raw['Year'].max()}  |  rows: {len(raw):,}")


# ── Core flow builder ────────────────────────────────────────────────────────────

def build_flow_matrix(
    df: pl.DataFrame,
    year_from: int,
    year_to: int,
    src_col: str,
    tgt_col: str,
    atom_len: int,
) -> pd.DataFrame:
    """
    Count worker moves from src_col industry (year_from) to tgt_col industry (year_to).

    Returns a DataFrame with columns [sni_from, sni_to, count].
    sni_from and sni_to are zero-padded strings truncated to atom_len digits.
    """
    left  = df.filter(pl.col("Year") == year_from).select(["PersonNr", pl.col(src_col).alias("_src")])
    right = df.filter(pl.col("Year") == year_to  ).select(["PersonNr", pl.col(tgt_col).alias("_tgt")])

    joined = (
        left.join(right, on="PersonNr", how="inner")
        .filter(
            pl.col("_src").is_not_null() & pl.col("_tgt").is_not_null()
            & (pl.col("_src") != 0) & (pl.col("_tgt") != 0)
        )
        .with_columns([
            pl.col("_src").cast(pl.Utf8).str.zfill(5).str.slice(0, atom_len).alias("sni_from"),
            pl.col("_tgt").cast(pl.Utf8).str.zfill(5).str.slice(0, atom_len).alias("sni_to"),
        ])
        .filter(pl.col("sni_from") != pl.col("sni_to"))  # exclude stayers
        .group_by(["sni_from", "sni_to"])
        .agg(pl.len().alias("count"))
    )
    return joined.to_pandas()


# ── Industry size / growth / income panel ────────────────────────────────────────

print("\nBuilding per-code-year industry panel (headcount, mean income) …")
code_year_panel = (
    raw.filter(pl.col("Sni").is_not_null() & (pl.col("Sni") != 0))
    .with_columns(pl.col("Sni").cast(pl.Utf8).str.zfill(5).str.slice(0, 4).alias("code"))
    .group_by(["Year", "code"])
    .agg([
        pl.col("PersonNr").n_unique().alias("headcount"),
        pl.col("EmploymentIncome").mean().alias("mean_income"),
    ])
    .to_pandas()
)


def regime_stats(panel: pd.DataFrame, years: range) -> pd.DataFrame:
    """Per-code mean size/growth/income over a single classification regime's years."""
    sub = panel[panel["Year"].isin(years)].sort_values(["code", "Year"]).copy()
    growth = sub.groupby("code")["headcount"].pct_change()
    sub["growth"] = growth.replace([np.inf, -np.inf], np.nan).clip(-0.9, 3.0)
    return sub.groupby("code").agg(
        size=("headcount", "mean"), growth=("growth", "mean"), income=("mean_income", "mean")
    )


print("Computing regime-level size/growth/income stats …")
STATS = {ver: regime_stats(code_year_panel, years) for ver, years in REGIME_YEARS.items()}
for ver, stats in STATS.items():
    print(f"  {ver}: {len(stats):,} codes with panel data")


def load_universe(version: str) -> set[str]:
    path = metadata_dir / f"mapping_Sni_4digits_{UNIVERSE_FILE_TAG[version]}.json"
    return set(json.loads(path.read_text()).keys())


UNIVERSE = {ver: load_universe(ver) for ver in REGIME_YEARS}


# ── Expected-flow regression ─────────────────────────────────────────────────────

def fit_expected_flow(observed: pd.DataFrame, stats: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    """
    Poisson regression of flow counts on origin/destination size, growth, and
    income, fit over the full pair grid (observed flows zero-filled) for one
    coding period. See module docstring for the model and why plain Poisson.
    """
    pairs = list(itertools.permutations(sorted(universe), 2))
    grid = pd.DataFrame(pairs, columns=["sni_from", "sni_to"])

    grid = grid.merge(stats.add_suffix("_i"), left_on="sni_from", right_index=True, how="left")
    grid = grid.merge(stats.add_suffix("_j"), left_on="sni_to", right_index=True, how="left")
    grid = grid.merge(observed, on=["sni_from", "sni_to"], how="left")
    grid["count"] = grid["count"].fillna(0.0)

    feature_src_cols = ["size_i", "size_j", "growth_i", "growth_j", "income_i", "income_j"]
    fit_df = grid[grid[feature_src_cols].notna().all(axis=1)].copy()

    X = pd.DataFrame({
        "log_size_i":   np.log1p(fit_df["size_i"]),
        "log_size_j":   np.log1p(fit_df["size_j"]),
        "growth_i":     fit_df["growth_i"],
        "growth_j":     fit_df["growth_j"],
        "log_income_i": np.log1p(fit_df["income_i"]),
        "log_income_j": np.log1p(fit_df["income_j"]),
    })
    X = sm.add_constant(X)
    model = sm.GLM(fit_df["count"], X, family=sm.families.Poisson()).fit()
    fit_df["expected"] = model.predict(X)
    return fit_df[["sni_from", "sni_to", "count", "expected"]]


def symmetrize_ratio(fit_df: pd.DataFrame) -> pd.DataFrame:
    """
    R(i,j) = (count + k) / (expected + k); phi = (R(i,j) + R(j,i)) / 2.

    phi is symmetric, so one row per unordered pair is kept (sni_from < sni_to);
    the graph builder mirrors it into both directions. Pairs are dropped unless
    the total observed movers in both directions (count + count_rev) reach
    MIN_LABOR_COUNT.
    """
    f = fit_df.copy()
    f["R"] = (f["count"] + SHRINKAGE_K) / (f["expected"] + SHRINKAGE_K)
    rev = f[["sni_from", "sni_to", "R", "count"]].rename(
        columns={"sni_from": "sni_to", "sni_to": "sni_from", "R": "R_rev", "count": "count_rev"}
    )
    f = f.merge(rev, on=["sni_from", "sni_to"], how="left")
    f["R_rev"] = f["R_rev"].fillna(0.0)
    f["count_rev"] = f["count_rev"].fillna(0.0)
    f["phi"] = 0.5 * (f["R"] + f["R_rev"])
    total_count = f["count"] + f["count_rev"]
    out = f.loc[total_count >= MIN_LABOR_COUNT, ["sni_from", "sni_to", "phi"]]
    lo = np.minimum(out["sni_from"], out["sni_to"])
    hi = np.maximum(out["sni_from"], out["sni_to"])
    return out.assign(sni_from=lo, sni_to=hi).drop_duplicates(["sni_from", "sni_to"])


# ── Collect all flows ────────────────────────────────────────────────────────────

all_flows: list[pd.DataFrame] = []

print("\nBuilding within-coding flow matrices …")

# Adjacent transitions within each coding regime, capped by --max-year.
for coding, years in REGIME_YEARS.items():
    year_list = list(years)
    for y in year_list[:-1]:
        df_y = build_flow_matrix(raw, y, y + 1, "Sni", "Sni", atom_len=4)
        df_y["coding"] = coding
        all_flows.append(df_y)
        print(f"  {coding} {y}→{y+1}: {len(df_y):,} atom pairs")

print("\nFitting expected-flow regressions and computing observed/expected relatedness …")
phi_frames: list[pd.DataFrame] = []

for coding, group_df in pd.concat(all_flows).groupby("coding"):
    observed = group_df.groupby(["sni_from", "sni_to"])["count"].sum().reset_index()

    fit_df = fit_expected_flow(observed, STATS[coding], UNIVERSE[coding])
    phi = symmetrize_ratio(fit_df)
    phi["coding"] = coding
    phi_frames.append(phi)
    n_any_flow = int((fit_df["count"] > 0).sum())
    print(f"  {coding}: {len(phi):,} pairs kept (>= {MIN_LABOR_COUNT} total movers) out of "
          f"{n_any_flow:,} pairs with any observed flow ({len(fit_df):,} grid pairs fit)  "
          f"(phi range [{phi['phi'].min():.4f}, {phi['phi'].max():.4f}])")

result = pd.concat(phi_frames, ignore_index=True)

result.to_parquet(output_path, index=False)
metadata_path = output_path.with_suffix(".metadata.json")
metadata = {
    "max_year": args.max_year,
    "regime_years": {
        coding: [min(years), max(years)] if len(years) else None
        for coding, years in REGIME_YEARS.items()
    },
    "min_labor_count": MIN_LABOR_COUNT,
    "shrinkage_k": SHRINKAGE_K,
    "rows": len(result),
    "output": str(output_path),
}
if exclusion_stats is not None:
    metadata["exclude_municipalities"] = exclusion_stats
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"\nSaved {len(result):,} pairs → {output_path}")
print(f"Metadata → {metadata_path}")
print("Done.")
