"""
build_labor_relatedness.py — Neffke-Henning skill relatedness from worker transitions.

The raw data has a single column `Sni` that holds the period-appropriate code:
  year ≤ 2001  → SNI92
  2002 ≤ year ≤ 2006  → SNI02
  year ≥ 2007  → SNI07

Only within-coding transitions are used — workers who change employer within
the same classification period:
                   SNI92 → SNI92  (year pairs 1990-91 through 2000-01)
                   SNI02 → SNI02  (year pairs 2002-03 through 2005-06)
                   SNI07 → SNI07  (year pairs 2007-08 through 2018-19)

Transitions in a year straddling a reclassification (SNI92→SNI02 at 2001-02,
SNI02→SNI07 at 2006-07) are not calculated at all: build_diffusion_targets.py's
build_labor_relation() only ever consumed within-coding rows (a reclassification
year's move is a coding artifact, not a real change of industry, and folding
it into "labor" would mix that noise into what should be a clean signal), so
there is nothing downstream to feed.

All flows are aggregated to 4-digit atom level across all three versions.

Observed-over-expected relatedness (Neffke & Henning 2013, SMJ):

  For each coding period, fit a Poisson regression of worker-flow counts on
  origin/destination industry size, growth, and income, over the FULL pair
  grid for that period's code universe (observed flows zero-filled in, not
  just the pairs that happened to have a move):

    log E[f(i->j)] = b0 + b1*log(size_i) + b2*log(size_j)
                        + b3*growth_i    + b4*growth_j
                        + b5*log(income_i) + b6*log(income_j)

  size/growth/income are computed straight from the individual-level panel
  (headcount, year-over-year headcount change, mean EmploymentIncome per
  4-digit code), not from the flow table's own row/column sums — a code with
  huge employment but low turnover should count as "large", even though it
  would look small if size were inferred from flow degree alone.

    R(i,j)   = (f(i->j) + k) / (E[f(i->j)] + k)      (>1: more flow than
                                           industry characteristics alone
                                           predict, i.e. more related than
                                           expected; k = SHRINKAGE_K)
    R_sym    = 0.5 * (R(i,j) + R(j,i))

  This is the actual Neffke & Henning measure (ratio of observed to a
  model's predicted flow), with one deliberate simplification: plain Poisson
  instead of their zero-inflated negative binomial. Poisson's log-link mean
  is a consistent estimator of the conditional mean even when the true
  distribution is over-dispersed (the standard "Poisson PML" argument from
  the gravity-model literature), and a full ZINB fit buys inferential
  correctness (proper SEs) we don't need — we only ever use the point
  prediction E[f(i->j)], never a p-value, and fitting one across 5 coding
  periods x up to ~380k grid rows each is unnecessary weight for that.

  R is a ratio of two noisy quantities, so a naive f/E has a heavy right
  tail for pairs where the model predicts almost no flow but a handful of
  moves happened anyway (small-sample noise, not relatedness) — confirmed
  with measured evidence, not assumed: evaluate_route_weights.py at
  diffusion_t=1 with every RELATION_BETA=0 still showed labor's weighted-mean
  diffusion lift at ~73x the bare-hierarchy baseline, versus 2-5x for every
  other relation, even after the MIN_LABOR_COUNT floor above. The additive
  smoothing R = (count+k)/(expected+k) (SHRINKAGE_K, default 1.0) is the fix:
  it leaves well-evidenced pairs (count and expected both reasonably large)
  essentially untouched, while pulling a count=1-3 pair against a
  near-zero predicted rate back toward 1 instead of letting it blow up.

  An earlier non-parametric alternative — phi(i,j) = (f(i->j) + f(j->i)) /
  (out(i) + in(j) + out(j) + in(i)), still in utils.symmetric_flow_phi and
  used for IO relatedness in build_input_output_relatedness.py — has no notion of "expected"
  flow at all: it normalizes by the flow network's own row/column sums, so
  an industry that is large but has little employee turnover looks "small"
  by that measure, and the share-based formula has no way to use
  growth or income. It is bounded [0, 1] by construction, which the
  observed/expected ratio above is not.

transition_up/down are excluded from the balance objective: their baseline is
structurally 0 (4d/SNI92 and 4d/SNI02 nodes have no hierarchy_up of their own,
so the bare-hierarchy baseline leaves them with zero outgoing mass and
lift_ratio is an uninformative inf regardless of beta) — they are an identity
correspondence, not a relatedness density to balance against the others.

Pairs are kept once total observed movers (both directions, pooled over the
whole period) reach MIN_LABOR_COUNT — not a top-k or magnitude-based prune (no
relative-rank or relatedness-score cutoff: the typed-relation diffusion in
build_diffusion_targets.py gates entire relation types per node, not individual edges, so
thinning by rank would only throw away real signal without changing how labor
competes against other relation types). This is purely a minimum-evidence
floor: 1-2 movers over a ~12-year pooled window is anecdotal, not evidence of
relatedness, and was the dominant source of both the near-complete-graph
density at the 4d level (most "barely above expected" pairs only had a
couple of movers) and the heaviest outlier ratios (a tiny predicted rate plus
one lucky mover gives an arbitrarily large R).

Output
------
  <LUNARC_DATA_ROOT>/processed/labor_relatedness.parquet
    columns: sni_from [str], sni_to [str], phi [float], coding [str]
    coding ∈ {SNI07, SNI02, SNI92}
    phi is the observed/expected ratio R_sym described above (kept under the
    old column name so build_diffusion_targets.py needs no changes).
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
    default=3,
    help="Minimum total observed movers (both directions, pooled over the whole period) "
         "for a pair to be kept at all — see module docstring's MIN_LABOR_COUNT discussion. "
         "Raising this keeps only higher-evidence pairs (fewer, more confident edges).",
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
    R(i,j) = (count+k)/(expected+k); phi = 0.5*(R(i,j) + R(j,i)) — symmetric
    by construction, so only one canonical row per unordered pair is kept
    (sni_from < sni_to). Storing both directions would be pure duplication,
    and build_diffusion_targets.py's row_normalize() already mirrors every "labor"-type
    row back into both directions when it builds the transition matrix (it
    has to: most other consumers of a symmetric relation, e.g. nace_4d's
    triu_indices, only ever hand it one direction per pair too). Keeping both
    here would double-count every pair that happens to have flow recorded in
    both directions relative to the pairs that only have one direction
    recorded — a storage artifact, not a real difference in relatedness.

    The SHRINKAGE_K additive smoothing tempers pairs where expected is tiny
    and count is small (see module docstring) — negligible once both are
    reasonably large.

    Pairs are then dropped unless total observed movers (count + count_rev)
    reach MIN_LABOR_COUNT — see module docstring. The threshold uses the raw
    counts, not the smoothed ratio, since it's evidence-based, not magnitude-based.
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
metadata_path.write_text(
    json.dumps(
        {
            "max_year": args.max_year,
            "regime_years": {
                coding: [min(years), max(years)] if len(years) else None
                for coding, years in REGIME_YEARS.items()
            },
            "min_labor_count": MIN_LABOR_COUNT,
            "shrinkage_k": SHRINKAGE_K,
            "rows": len(result),
            "output": str(output_path),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"\nSaved {len(result):,} pairs → {output_path}")
print(f"Metadata → {metadata_path}")
print("Done.")
