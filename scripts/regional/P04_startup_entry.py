"""
P04_startup_entry.py
========================
Implements Grillitsch (2021, Strategic Entrepreneurship Journal,
https://sms.onlinelibrary.wiley.com/doi/full/10.1002/sej.1375)'s definition of
a genuine new-firm "startup" — as opposed to P04_entry.py's crude entry_firm
proxy (any new establishment with > N workers, regardless of whether it is a
spin-off, a merger remnant, or a reclassification) — and tests the same
relatedness-density hypothesis (Hidalgo et al. 2007; Neffke, Henning &
Boschma 2011) against it: does a genuinely new firm's chance of forming in a
given (municipality, industry) cell favour industries that already sit close
to the region's existing employment structure?

Startup definition (all conditions on the founding OrgNr/year)
----------------------------------------------------------------
  1. Kvar in KVAR_STARTUP_CODES — a genuine "new" firm-demography event in
     FAD_1990-2019.parquet, not a merger-driven "new" entity (Kvar 05/09,
     excluded) and not a surviving/continuing firm (Kvar 11/12/13, excluded).
     See metadata/mapping_Kvar.json (SCB's FAD Kvar codebook): 01/02/03 are
     "new" under the original methodology, 06/07/08 are their 2004
     methodology-adjustment counterparts. Kvar=03 ("new, no predecessor
     found, zero employees the previous year") never actually fires in this
     1990-2019 extract — the column's realised value set is
     {1,2,5,6,7,8,9,11,12,13}, no 3 — but is kept in the filter for
     conceptual completeness (harmless if it never matches; confirmed with
     the user rather than silently dropped).
  2. OrgNr absent in year t-1 in individ_cleaned_1990-2019.parquet. This is
     specifically "absent the immediately preceding year", not "never seen
     before anywhere in the panel" (a different, stricter condition) — a
     firm that had a genuine gap year would still qualify. Year 1990
     candidates can never satisfy this (there is no 1989 row in the panel)
     and are excluded (SAMPLE_START_YEAR).
  3. Exactly one CfarNr for that OrgNr in year t (single-establishment firm)
     — computed directly from individ, not trusted from FAD's own CfarNr
     field (FAD only supplies the Kvar-coded transition; individ is the
     source of truth for the microdata-based checks per the definition).
  4. Founder size <= --max-founder-size employees (default 10) — headcount
     at that OrgNr in year t, from individ.
  5. Optional (--max-growth-ratio, default 0 = disabled): exclude firms
     whose (t+1 headcount) / (founder size) ratio exceeds the given
     multiple — Grillitsch (2021)'s robustness check against implausible
     ramps that more likely reflect mis-coded M&A activity than organic
     growth. A firm that disappears at t+1 (no headcount) is NOT excluded
     by this filter — an undefined ratio is not evidence of an implausible
     jump.

Regional density test
----------------------
Exactly P04_entry.py's design, reused rather than reinvented: relatedness-
density kernel (softmax over cosine similarity, employment-share-weighted,
self excluded), computed per classification period at year t; outcome
("startup_entry") is whether a qualifying startup is born in that
(industry, municipality) cell at t+1; linear probability model with
industry + municipality + year FE via triple demeaning (separate additive
municipality and year effects, not an interacted municipality×year term),
SE clustered by municipality, run separately per SNI classification period
(SNI92/SNI02/SNI07 — never pooled, for the same reason P04_entry.py never
pools them: overlapping code strings across vintages are not the same
industry).

--include-entry-industry additionally reports P04_entry.py's own crude
"industry emerges to non-negligible local size" outcome side by side, purely
as a reference point for how much the compliant startup definition changes
the answer relative to that cruder proxy — not because that outcome is
itself of interest here. Off by default.

Usage
-----
  python scripts/regional/P04_startup_entry.py
  python scripts/regional/P04_startup_entry.py --run-name graph_kl --max-founder-size 10
  python scripts/regional/P04_startup_entry.py --max-growth-ratio 30
  python scripts/regional/P04_startup_entry.py --include-entry-industry
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
import statsmodels.api as sm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import EMBEDDING_DIR, METADATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from latent_industry.common.municipalities import harmonization_table, load_municipality_crosswalk
from latent_industry.regional.density import (
    load_sni_embeddings, make_softmax_kernel, cosine_matrix, employment_weighted_density,
    geometric_mean_employment_weighted_density, empl_matrix as _shared_empl_matrix,
)
from latent_industry.regional.regional_figures import plot_density_effect_summary

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
    _year_df = _year_df[_year_df["outcome"] == OUTCOME] if "outcome" in _year_df.columns else _year_df
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
        results_dir / f"{OUT_NAME}_startup_entry.png",
    )
    print(f"\n  → {_fig_path}")
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

_sd_density = float(panel["density"].std())
panel["density_std"] = panel["density"] / _sd_density
print(f"\nDensity pre-standardised (÷ SD = {_sd_density:.6f}, pooled across all "
      "periods purely as a fixed unit) — β reads as Δ P(entry) per 1-SD "
      "increase in density")


# ═══════════════════════════════════════════════════════════════════════════════
# 5+6. Per-classification-period demeaning and regression — linear
#    probability model with industry + municipality + year FE via triple
#    demeaning (separate additive municipality and year effects, not an
#    interacted muni×year term). Never pooled across SNI92/SNI02/SNI07 (see
#    module docstring).
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

LABELS = {
    "startup_entry":   "startup_entry  (start-up growth: new, single-establishment, <=10-employee firm)",
    "entry_industry":  "entry_industry (jumps to significant local size — reference only)",
}

print("\n" + "=" * 60)
print("Relatedness density → startup entry  (per classification period, industry + "
      "municipality + year FE, SE clustered by muni)")
print("=" * 60)

periods_order = list(dict.fromkeys(panel["period"]))
dm_frames: list[pd.DataFrame] = []
results: dict[str, dict[str, dict]] = {}

for period_name in periods_order:
    sub = panel[panel["period"] == period_name].copy()
    for var in OUTCOMES + ["density_std"]:
        sub[f"{var}_r1"] = demean(sub, var,         "municipality")
        sub[f"{var}_r2"] = demean(sub, f"{var}_r1",  "year")
        sub[f"{var}_dm"] = demean(sub, f"{var}_r2",  "industry")

    within_fe_sd = float(sub["density_std_dm"].std())
    print(f"\n  -- {period_name} --  (within-FE density_std SD={within_fe_sd:.6f}, "
          f"{within_fe_sd**2:.4%} of raw variance retained)")

    results[period_name] = {}
    for outcome in OUTCOMES:
        res  = run_ols(sub[f"{outcome}_dm"], sub["density_std_dm"], sub["municipality"])
        rate = float(sub[outcome].mean())
        res["pct_of_baseline"] = res["coef"] / rate * 100 if rate > 0 else float("nan")
        res["rate"] = rate
        results[period_name][outcome] = res
        print(f"    {LABELS[outcome]}")
        print(f"      coef={res['coef']:+.6f}  SE={res['se']:.6f}"
              f"  t={res['t']:.3f}  p={res['pval']:.3e}")
        print(f"      N={res['n']:,}  R²={res['r2']:.5f}  partial r={res['partial_r']:+.4f}")
        print(f"      baseline rate={rate:.4%}  →  β = {res['pct_of_baseline']:+.1f}% of baseline"
              f"  per 1-SD density")

    dm_frames.append(sub)

panel = pd.concat(dm_frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Year-by-year regression
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Year-by-year coefficients  (density → " + ", ".join(OUTCOMES) + ")")
print("=" * 60)

year_records: list[dict] = []
for year, sub in panel.groupby("year"):
    sub = sub.copy()
    period_name = sub["period"].iloc[0]
    for var in OUTCOMES + ["density_std"]:
        sub[f"{var}_dm"] = demean(sub, var, "municipality")
        sub[f"{var}_dm"] = demean(sub, f"{var}_dm", "industry")
    year_res = {}
    for outcome in OUTCOMES:
        r = run_ols(sub[f"{outcome}_dm"], sub["density_std_dm"], sub["municipality"])
        r["year"], r["outcome"], r["period"] = int(year), outcome, period_name
        year_records.append(r)
        year_res[outcome] = r
    msg = f"  {year}→{year+1}:  " + "   ".join(
        f"{outcome} β={year_res[outcome]['coef']:+.5f} (p={year_res[outcome]['pval']:.2e})"
        for outcome in OUTCOMES
    )
    print(msg + f"   N={year_res[OUTCOMES[0]]['n']:,}")

year_df = pd.DataFrame(year_records).sort_values(["outcome", "year"])


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Visualisation — the main paper figure covers startup_entry only; if
#    --include-entry-industry is set, that reference outcome's regression
#    still runs and is saved to the CSVs above, it's just not folded into
#    this figure (the shared plotting function is single-outcome, matching
#    P04_entry.py/P04_exit.py).
# ═══════════════════════════════════════════════════════════════════════════════

_startup_results = {p: results[p][OUTCOME] for p in periods_order}
_startup_year_df = year_df[year_df["outcome"] == OUTCOME]
_fig_path = plot_density_effect_summary(
    panel, _startup_year_df, _startup_results, periods_order, OUTCOME, OUTCOME_LABEL, EFFECT_LABEL, COLOR,
    results_dir / f"{OUT_NAME}_startup_entry.png",
)
print(f"\n  → {_fig_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Save outputs
# ═══════════════════════════════════════════════════════════════════════════════

panel_out = panel[["period", "year", "municipality", "industry", "density"] + OUTCOMES]
panel_out.to_parquet(OUT_DIR / f"{OUT_NAME}_panel.parquet", index=False)
print(f"  → {OUT_DIR / f'{OUT_NAME}_panel.parquet'}  ({len(panel):,} rows)")

year_df.to_csv(results_dir / f"{OUT_NAME}_year_coefs.csv", index=False)

summary_rows = []
for period_name in periods_order:
    in_period = panel["period"] == period_name
    row = {
        "run":                args.run_name,
        "density_var":        args.density_var,
        "log_employment_weights": args.log_employment_weights,
        "period":             period_name,
        "tau":                args.tau,
        "max_founder_size":   args.max_founder_size,
        "max_growth_ratio":   args.max_growth_ratio,
        "n_obs":              int(in_period.sum()),
        "n_municipalities":   len(all_municipalities),
    }
    for outcome in OUTCOMES:
        row[f"{outcome}_rate"]            = panel.loc[in_period, outcome].mean()
        row[f"{outcome}_coef"]            = results[period_name][outcome]["coef"]
        row[f"{outcome}_se"]              = results[period_name][outcome]["se"]
        row[f"{outcome}_pval"]            = results[period_name][outcome]["pval"]
        row[f"{outcome}_r"]               = results[period_name][outcome]["partial_r"]
        row[f"{outcome}_pct_of_baseline"] = results[period_name][outcome]["pct_of_baseline"]
    summary_rows.append(row)
pd.DataFrame(summary_rows).to_csv(results_dir / f"{OUT_NAME}_summary.csv", index=False)

print(f"  → {results_dir}/  ({OUT_NAME}_year_coefs.csv, _summary.csv)  "
      f"+ {OUT_DIR}/  ({OUT_NAME}_panel.parquet)")

print("\nDone.")
