"""
_municipality_exclusion.py
==========================
Optional municipality exclusion for the register-based relatedness builders
(build_labor_relatedness.py, build_ownership_relatedness.py:
--exclude-municipalities FILE). Used by the spatial hold-out
(scripts/robustness/): the relational views are rebuilt without any
person-year whose workplace lies in a held-out municipality.

FILE is a CSV with a `municipality` column of harmonised 4-character codes
(latent_industry.common.municipalities). WorkplaceMunicipality is harmonised
exactly as in scripts/regional/P04_entry.py (zfill(4), then the
year-sensitive crosswalk). Rows whose code cannot be mapped (0000/9999,
missing) are kept: they cannot be assigned to any municipality and are the
same in every fold.
"""

from pathlib import Path

import pandas as pd
import polars as pl

from latent_industry.common.municipalities import harmonization_table, load_municipality_crosswalk


def load_excluded_municipalities(path: str | Path) -> list[str]:
    """Harmonised municipality codes listed in the `municipality` column of `path`."""
    codes = pd.read_csv(path, dtype={"municipality": str})["municipality"].str.zfill(4)
    return sorted(set(codes))


def drop_excluded_municipalities(raw: pl.DataFrame, excluded: list[str]) -> tuple[pl.DataFrame, dict]:
    """Drop every row of `raw` (needs Year, WorkplaceMunicipality) whose
    harmonised workplace municipality is in `excluded`. Returns the kept rows
    (original columns only) and counts for the builder's metadata."""
    xwalk = harmonization_table(load_municipality_crosswalk()).rename(
        {"source_code": "_source_code", "municipality": "_municipality"}
    )
    if xwalk.select(["Year", "_source_code"]).is_duplicated().any():
        raise ValueError("municipality crosswalk is not unique on (Year, source_code)")
    unknown = sorted(set(excluded) - set(xwalk["_municipality"].unique().to_list()))
    if unknown:
        raise ValueError(f"--exclude-municipalities lists codes outside the crosswalk: {unknown}")

    keyed = (
        raw.with_columns(pl.col("WorkplaceMunicipality").cast(pl.Utf8).str.zfill(4).alias("_source_code"))
        .join(xwalk, on=["Year", "_source_code"], how="left", maintain_order="left")
    )
    if keyed.height != raw.height:
        raise RuntimeError("municipality join changed the number of rows")
    in_excluded = pl.col("_municipality").is_in(excluded).fill_null(False)
    n_dropped = keyed.select(in_excluded.sum()).item()
    n_unmapped = keyed.select(pl.col("_municipality").is_null().sum()).item()
    kept = keyed.filter(~in_excluded).drop(["_source_code", "_municipality"])
    stats = {
        "n_excluded_municipalities": len(excluded),
        "excluded_municipalities": excluded,
        "rows_before": raw.height,
        "rows_dropped": int(n_dropped),
        "rows_unmapped_kept": int(n_unmapped),
    }
    return kept, stats
