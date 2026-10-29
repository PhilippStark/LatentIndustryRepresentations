"""
H0_employment_cache.py
======================
Builds the employment cache every robustness check uses and verifies that it
reproduces the published v16 panels. Requires the restricted SCB
individual-level register panel (only for building the cache).

Employment cache
----------------
Workers per (municipality, Year, sni_4d), aggregated exactly as in step 1 of
scripts/regional/P04_entry.py: 4-digit Sni (0/missing dropped), harmonised
WorkplaceMunicipality (inner join with the year-sensitive crosswalk), one
row per person-year. Restricted data: it stays under
<HOLDOUT_ROOT>/common/data/ and is never copied into the repository.

Checks (the script exits with an error if one fails)
----------------------------------------------------
  1. density recomputed from the cache with the v16 embedding equals the
     panels' `density` (max |Δ| < 1e-6), for entry, exit and startup_entry;
  2. the h = 1 outcome rules of _holdout_common reproduce `entry_industry`
     and `exit_industry` exactly;
  3. the main-specification estimate on the published panel reproduces
     results/regional_<outcome>/shared_cosine_v16_summary.csv (β and SE).

Outputs
-------
  <HOLDOUT_ROOT>/common/data/employment_muni_year_sni4d.parquet (+ .json)
  results/holdout_checks/shared_cosine_v16_H0_checks.json

Usage
-----
  python scripts/robustness/H0_employment_cache.py            # builds the cache once, then checks
  python scripts/robustness/H0_employment_cache.py --rebuild
"""

import argparse
import hashlib
import sys

import numpy as np
import pandas as pd
import polars as pl

from _holdout_common import (
    EMPLOYMENT_CACHE, MAIN_RUN, OUTCOMES, REPO_ROOT, Panel, estimate, load_main, result_path, write_json,
)
from latent_industry.common.municipalities import harmonization_table, load_municipality_crosswalk

INDIVID_PATH = "/lunarc/nobackup/projects/lu2025-17-39/SCB/Preprocessed/individ_cleaned_1990-2019.parquet"

parser = argparse.ArgumentParser()
parser.add_argument("--rebuild", action="store_true", help="Rebuild the cache even if it exists.")
args = parser.parse_args()


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ── 1. Employment cache ───────────────────────────────────────────────────────

if args.rebuild or not EMPLOYMENT_CACHE.exists():
    print(f"Loading {INDIVID_PATH} …")
    raw = pl.read_parquet(INDIVID_PATH, columns=["Year", "Sni", "WorkplaceMunicipality"])
    empl = (
        raw
        .filter(pl.col("Sni").is_not_null() & (pl.col("Sni") != 0))
        .filter(pl.col("WorkplaceMunicipality").is_not_null())
        .with_columns([
            pl.col("Sni").cast(pl.Utf8).str.zfill(5).str.slice(0, 4).alias("sni_4d"),
            pl.col("WorkplaceMunicipality").cast(pl.Utf8).str.zfill(4).alias("source_code"),
        ])
        .filter(pl.col("sni_4d") != "0000")
        .join(harmonization_table(load_municipality_crosswalk()), on=["Year", "source_code"], how="inner")
        .group_by(["municipality", "Year", "sni_4d"])
        .agg(pl.len().alias("workers"))
        .sort(["municipality", "Year", "sni_4d"])
        .to_pandas()
    )
    EMPLOYMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    empl.to_parquet(EMPLOYMENT_CACHE, index=False)
    write_json(EMPLOYMENT_CACHE.with_suffix(".json"), {
        "source": INDIVID_PATH,
        "aggregation": "P04_entry.py step 1: workers per (municipality, Year, sni_4d)",
        "rows": len(empl),
        "years": [int(empl["Year"].min()), int(empl["Year"].max())],
        "municipalities": int(empl["municipality"].nunique()),
        "sha256": sha256(EMPLOYMENT_CACHE),
    })
    print(f"  → {EMPLOYMENT_CACHE}  ({len(empl):,} rows)")
    del raw, empl
else:
    print(f"Employment cache exists: {EMPLOYMENT_CACHE}")

# ── 2. Checks against the published v16 panels ────────────────────────────────

emp, v16 = load_main()
checks: dict = {"municipalities": len(emp.municipalities)}
failures: list[str] = []
if len(emp.municipalities) != 290:
    failures.append(f"cache has {len(emp.municipalities)} municipalities, expected 290")

for outcome, spec in OUTCOMES.items():
    print(f"\n{outcome}:")
    panel = Panel(outcome, emp)
    col = spec["col"]

    dens = panel.density(emp, v16)
    max_diff = float(np.nanmax(np.abs(dens - panel.df["density"].to_numpy(dtype=float))))
    n_missing = int(np.isnan(dens).sum())
    print(f"  density: max |Δ| = {max_diff:.2e}, missing = {n_missing}")
    if n_missing or max_diff >= 1e-6:
        failures.append(f"{outcome}: density not reproduced (max |Δ| {max_diff:.2e}, missing {n_missing})")

    n_mismatch = None
    if outcome in ("entry", "exit"):
        n_mismatch = 0
        for (period, year), rows in panel.groups.items():
            y = panel.outcome_at_horizon(emp, period, year, 1)
            n_mismatch += int((y != panel.df[col].to_numpy()[rows]).sum())
        print(f"  h = 1 outcome: {n_mismatch} mismatching rows")
        if n_mismatch:
            failures.append(f"{outcome}: h = 1 outcome rule differs from the panel in {n_mismatch} rows")

    effects, _ = estimate(panel.df, col, panel.df["density"].to_numpy(), annual=False)
    published = pd.read_csv(REPO_ROOT / "results" / f"regional_{outcome}" / f"{MAIN_RUN}_summary.csv")
    merged = effects.merge(published[["period", f"{col}_coef", f"{col}_se"]], on="period")
    coef_diff = float(np.max(np.abs(merged["coef"] - merged[f"{col}_coef"]) / np.abs(merged[f"{col}_coef"])))
    se_diff = float(np.max(np.abs(merged["se"] - merged[f"{col}_se"]) / merged[f"{col}_se"]))
    print(f"  regression vs published summary: max rel. Δβ = {coef_diff:.2e}, max rel. ΔSE = {se_diff:.2e}")
    if len(merged) != 3 or coef_diff > 1e-6 or se_diff > 1e-6:
        failures.append(f"{outcome}: published estimates not reproduced (rel. Δβ {coef_diff:.2e}, ΔSE {se_diff:.2e})")

    checks[outcome] = {"density_max_abs_diff": max_diff, "density_missing": n_missing,
                       "h1_outcome_mismatches": n_mismatch,
                       "coef_max_rel_diff": coef_diff, "se_max_rel_diff": se_diff}

checks["failures"] = failures
write_json(result_path("checks", "H0_checks.json"), checks)
if failures:
    print("\nCHECKS FAILED:\n  " + "\n  ".join(failures))
    sys.exit(1)
print("\nAll checks passed.")
