"""
build_ownership_relatedness.py
==============================
Ownership relatedness between 4-digit industries: how often two industries
are active under the same enterprise (OrgNr) in the same year (related
diversification, Teece 1980). Requires the restricted SCB individual-level
register panel.

Co-occurrence
-------------
For every (year, OrgNr), the set of distinct industry codes of its employees
is taken; each pair of codes in the set co-occurs once, irrespective of
headcount. Enterprises with more than MAX_CODES_PER_ORG (30) codes are
skipped. Counts are pooled over the years of each classification period
(SNI92: 1990–2001, SNI02: 2002–2006, SNI07: 2007–2019), never across periods,
and pairs with fewer than MIN_OWNERSHIP_COUNT (3) co-occurrences are dropped.
Relatedness is the symmetric flow-strength measure
latent_industry.common.flow_metrics.symmetric_flow_phi.

Spatial hold-out
----------------
--exclude-municipalities FILE drops every person-year whose (harmonised)
workplace municipality is listed in FILE before the co-occurrence step
(scripts/data/_municipality_exclusion.py), so employees there do not enter
any OrgNr's code set. Off by default.

Outputs (<LATENT_DATA_ROOT>/processed/)
---------------------------------------
  ownership_relatedness.parquet    sni_from, sni_to, phi, coding (one row per
                                   unordered pair and period, sni_from < sni_to)

Usage
-----
  python scripts/data/build_ownership_relatedness.py
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import METADATA_DIR, PROCESSED_DATA_DIR
from latent_industry.common.flow_metrics import symmetric_flow_phi

DATA_ROOT    = Path("/lunarc/nobackup/projects/lu2025-17-39")
INDIVID_PATH = DATA_ROOT / "SCB" / "Preprocessed" / "individ_cleaned_1990-2019.parquet"
metadata_dir = METADATA_DIR
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--max-year", type=int, default=2019,
                     help="Last register year allowed.")
parser.add_argument("--output-file", default="ownership_relatedness.parquet",
                     help="Absolute path or filename under PROCESSED_DATA_DIR.")
parser.add_argument("--exclude-municipalities", default=None,
                     help="CSV with a `municipality` column (harmonised codes): person-years "
                          "working there are dropped before co-occurrence is counted (spatial "
                          "hold-out). Off by default.")
args, _ = parser.parse_known_args()
if args.max_year < 1991:
    raise ValueError("--max-year must permit at least one within-regime year")
output_path = Path(args.output_file)
if not output_path.is_absolute():
    output_path = PROCESSED_DATA_DIR / output_path

# Minimum pooled (within-version) co-occurrence count for a pair to be kept —
# same role as MIN_LABOR_COUNT in build_labor_relatedness.py: an evidence
# floor, not a magnitude/rank cutoff.
MIN_OWNERSHIP_COUNT = 3

# Max distinct codes per (Year, OrgNr) group before it's dropped from the
# pairwise expansion. A handful of OrgNr values are misc./umbrella codes
# covering hundreds of establishments — without a cap, one such OrgNr would
# emit tens of thousands of pairs (n*(n-1)/2) that says nothing about real
# industry coherence, just that the identifier is a catch-all.
MAX_CODES_PER_ORG = 30


def capped_years(start: int, end: int) -> range:
    capped_end = min(end, args.max_year)
    return range(start, capped_end + 1) if capped_end >= start else range(0)


REGIME_YEARS = {
    "SNI92": capped_years(1990, 2001),
    "SNI02": capped_years(2002, 2006),
    "SNI07": capped_years(2007, 2019),
}
YEAR_TO_REGIME = {y: ver for ver, years in REGIME_YEARS.items() for y in years}

print(f"Loading individual data from {INDIVID_PATH} …")
exclusion_stats = None
if args.exclude_municipalities:
    from _municipality_exclusion import drop_excluded_municipalities, load_excluded_municipalities

    raw = pl.read_parquet(INDIVID_PATH, columns=["Year", "OrgNr", "Sni", "WorkplaceMunicipality"])
    raw, exclusion_stats = drop_excluded_municipalities(
        raw.filter(pl.col("Year") <= args.max_year),
        load_excluded_municipalities(args.exclude_municipalities),
    )
    raw = raw.drop("WorkplaceMunicipality")
    exclusion_stats["file"] = str(Path(args.exclude_municipalities).resolve())
    print(f"  --exclude-municipalities: {exclusion_stats['n_excluded_municipalities']} municipalities, "
          f"{exclusion_stats['rows_dropped']:,} of {exclusion_stats['rows_before']:,} person-years dropped "
          f"({exclusion_stats['rows_unmapped_kept']:,} unmapped kept)")
else:
    raw = pl.read_parquet(INDIVID_PATH, columns=["Year", "OrgNr", "Sni"])
raw = raw.filter(
    (pl.col("Year") <= args.max_year)
    & pl.col("Sni").is_not_null() & (pl.col("Sni") != 0)
    & pl.col("OrgNr").is_not_null() & (pl.col("OrgNr") != 0)
).with_columns(
    pl.col("Sni").cast(pl.Utf8).str.zfill(5).str.slice(0, 4).alias("code")
)
print(f"  Years: {raw['Year'].min()}–{raw['Year'].max()}  |  rows: {len(raw):,}")


def year_cooccurrence(df_year: pl.DataFrame) -> pd.DataFrame:
    """Distinct codes per OrgNr this year -> unordered co-occurrence counts,
    one row per (sni_from < sni_to) pair, value = number of OrgNr this year
    where both codes were present. Groups above MAX_CODES_PER_ORG codes are
    dropped before pairing (see module-level comment)."""
    org_codes = (
        df_year.group_by("OrgNr")
        .agg(pl.col("code").unique().alias("codes"))
        .to_pandas()
    )
    counts: dict[tuple[str, str], int] = {}
    for codes in org_codes["codes"]:
        if len(codes) < 2 or len(codes) > MAX_CODES_PER_ORG:
            continue
        for a, b in itertools.combinations(sorted(set(codes)), 2):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    if not counts:
        return pd.DataFrame(columns=["sni_from", "sni_to", "count"])
    rows = [{"sni_from": a, "sni_to": b, "count": c} for (a, b), c in counts.items()]
    return pd.DataFrame(rows)


print("\nComputing per-year OrgNr co-occurrence …")
year_frames = []
for year in sorted(YEAR_TO_REGIME):
    df_year = raw.filter(pl.col("Year") == year)
    cooc = year_cooccurrence(df_year)
    if cooc.empty:
        continue
    cooc.insert(0, "year", year)
    cooc.insert(0, "coding", YEAR_TO_REGIME[year])
    year_frames.append(cooc)
    print(f"  {YEAR_TO_REGIME[year]} {year}: {cooc['sni_from'].nunique() + cooc['sni_to'].nunique()} "
          f"codes involved, {len(cooc):,} pairs")

flows_by_year = pd.concat(year_frames, ignore_index=True) if year_frames else pd.DataFrame(
    columns=["coding", "year", "sni_from", "sni_to", "count"]
)

print("\nPooling within each classification period and computing phi …")
phi_frames = []
for coding, group_df in flows_by_year.groupby("coding"):
    pooled = group_df.groupby(["sni_from", "sni_to"], as_index=False)["count"].sum()

    # symmetric_flow_phi expects directed rows; co-occurrence is symmetric by
    # construction (a==b co-occurrence is undefined/excluded already, since
    # itertools.combinations never pairs a code with itself), so both
    # directions carry the identical count — this is what makes each code's
    # out/in degree correctly reflect its TOTAL co-occurrence mass across
    # every partner, not just the partners where it happened to sort first.
    doubled = pd.concat([
        pooled.rename(columns={"sni_from": "src", "sni_to": "tgt"}),
        pooled.rename(columns={"sni_from": "tgt", "sni_to": "src"}),
    ], ignore_index=True)
    phi = symmetric_flow_phi(doubled, "src", "tgt", "count")
    phi = phi.rename(columns={"src": "sni_from", "tgt": "sni_to"})
    phi = phi.merge(pooled, on=["sni_from", "sni_to"], how="left")

    kept = phi[phi["count"] >= MIN_OWNERSHIP_COUNT][["sni_from", "sni_to", "phi"]].copy()
    kept["coding"] = coding
    phi_frames.append(kept)
    print(f"  {coding}: {len(kept):,} pairs kept (>= {MIN_OWNERSHIP_COUNT} pooled co-occurrences) "
          f"out of {len(pooled):,} observed pairs  "
          f"(phi range [{kept['phi'].min() if len(kept) else float('nan'):.4f}, "
          f"{kept['phi'].max() if len(kept) else float('nan'):.4f}])")

result = pd.concat(phi_frames, ignore_index=True) if phi_frames else pd.DataFrame(
    columns=["sni_from", "sni_to", "phi", "coding"]
)
result.to_parquet(output_path, index=False)

metadata_path = output_path.with_suffix(".metadata.json")
metadata = {
    "max_year": args.max_year,
    "regime_years": {
        coding: [min(years), max(years)] if len(years) else None
        for coding, years in REGIME_YEARS.items()
    },
    "min_ownership_count": MIN_OWNERSHIP_COUNT,
    "max_codes_per_org": MAX_CODES_PER_ORG,
    "rows": len(result),
    "output": str(output_path),
}
if exclusion_stats is not None:
    metadata["exclude_municipalities"] = exclusion_stats
metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"\nSaved {len(result):,} pairs → {output_path}")
print(f"Metadata → {metadata_path}")
print("Done.")
