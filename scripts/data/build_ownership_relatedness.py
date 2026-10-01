"""
build_ownership_relatedness.py — relatedness between SNI industries revealed
by common ownership: how often two industries appear together under the
same enterprise (OrgNr) in the same year.

Rationale
---------
Every other relatedness family here is either a worker-transition flow
(labor) or a national accounts flow (io) — neither captures the diversification
channel a firm's own portfolio reveals: a conglomerate active in both
industry i and industry j is direct revealed evidence those two activities
sit together under one ownership/management structure, independent of
whether workers or intermediate goods ever flow between them (Teece 1980;
this is the same logic behind Neffke & Henning's later firm-coherence work
and the general "related diversification" literature).

The raw individual-level panel (same source as build_labor_relatedness.py)
has an `OrgNr` column — Sweden's enterprise identifier. It is NOT stable
across years (an OrgNr can be reused/reassigned over time), but it IS exact
within a single year, which is all this needs: for each (Year, OrgNr), take
the SET of distinct SNI codes worked under that enterprise that year. Two
codes "co-occur" once per (Year, OrgNr) where both are present, regardless
of how many employees are under each — headcount-weighting would let one
huge multi-division employer dominate the whole measure; a plain co-occurrence
count, corrected by each code's own base rate via `symmetric_flow_phi` (the
same formula build_input_output_relatedness.py and this project's IO/labor
layers already use), is the simpler and more defensible choice here.

Version handling — same rule as labor (see build_labor_relatedness.py):
`Sni` already holds the period-appropriate code per row, and OrgNr-year
co-occurrence is only ever counted WITHIN one year, so it is automatically
within one classification vintage. Pooling happens across years WITHIN a
vintage (SNI92: 1990-2001, SNI02: 2002-2006, SNI07: 2007-2019), never across
a reclassification boundary — an OrgNr's code identity across a
reclassification year is a coding artifact, not a real ownership fact.

All flows are also kept year-tagged (not pooled away first) for future
temporal-embedding work, same as build_input_output_relatedness.py's
io_flows_by_year.parquet (2026-07-29).

Output
------
  <LUNARC_DATA_ROOT>/processed/ownership_relatedness.parquet
    columns: sni_from [str], sni_to [str], phi [float], coding [str]
    coding ∈ {SNI07, SNI02, SNI92}. One row per unordered pair per period
    (sni_from < sni_to), same convention as labor_relatedness.parquet.

  <LUNARC_DATA_ROOT>/processed/ownership_flows_by_year.parquet
    columns: coding, year, sni_from, sni_to, count — NOT yet pooled across years.

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
parser.add_argument("--flows-by-year-file", default="ownership_flows_by_year.parquet",
                     help="Absolute path or filename under PROCESSED_DATA_DIR.")
args, _ = parser.parse_known_args()
if args.max_year < 1991:
    raise ValueError("--max-year must permit at least one within-regime year")
output_path = Path(args.output_file)
if not output_path.is_absolute():
    output_path = PROCESSED_DATA_DIR / output_path
flows_by_year_path = Path(args.flows_by_year_file)
if not flows_by_year_path.is_absolute():
    flows_by_year_path = PROCESSED_DATA_DIR / flows_by_year_path

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
flows_by_year.to_parquet(flows_by_year_path, index=False)
print(f"\nSaved year-tagged co-occurrence ({len(flows_by_year):,} rows) → {flows_by_year_path}")

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
metadata_path.write_text(
    json.dumps(
        {
            "max_year": args.max_year,
            "regime_years": {
                coding: [min(years), max(years)] if len(years) else None
                for coding, years in REGIME_YEARS.items()
            },
            "min_ownership_count": MIN_OWNERSHIP_COUNT,
            "max_codes_per_org": MAX_CODES_PER_ORG,
            "rows": len(result),
            "output": str(output_path),
            "flows_by_year": str(flows_by_year_path),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"\nSaved {len(result):,} pairs → {output_path}")
print(f"Metadata → {metadata_path}")
print("Done.")
