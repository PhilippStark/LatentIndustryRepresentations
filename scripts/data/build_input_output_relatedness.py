"""
build_input_output_relatedness.py
=================================
Input–output (IO) relatedness between 2-digit industry divisions, built
separately for each classification vintage from public IO tables.

Sources (cached under <LATENT_DATA_ROOT>/raw/io/)
-------------------------------------------------
  SCB symmetric IO table, 2008–2016 (domestic intermediate use, product ×
      product, NACE Rev. 2 groupings). Vintage: SNI07.
  Eurostat naio_10_cp1700, Sweden, domestic flows, 2010–2019. Vintage: SNI07.
  WIOD 2013 release, Sweden, 1995/2000/2005 (NACE Rev. 1.1 industries).
      Vintage: SNI02.

Each source's flows are expanded from its own aggregated rows to 2-digit
divisions and kept in its native vintage; no crosswalk is applied across
classification versions. There is no IO source for SNI92.

Relatedness
-----------
Per source, flows are summed over its years and converted to a symmetric
flow-strength measure (latent_industry.common.flow_metrics.symmetric_flow_phi):

  phi(i,j) = (f(i→j) + f(j→i)) / (out(i) + in(j) + out(j) + in(i))

For SNI07 the SCB and Eurostat measures are averaged over the sources
available for a pair.

Outputs (<LATENT_DATA_ROOT>/processed/)
---------------------------------------
  io_relatedness_SNI07.parquet   nace2d_from, nace2d_to, sim, phi_scb, phi_eurostat, n_sources
  io_relatedness_SNI02.parquet   nace2d_from, nace2d_to, sim, phi_wiod, n_sources
  io_flows_by_year.parquet       source, version, year, nace2d_from, nace2d_to, value

In the main run (shared_cosine_v16) these tables feed the `io` view of the
`layers_v1` graph (scripts/graph/build_layer_transition_matrices.py).

Usage
-----
  python scripts/data/build_input_output_relatedness.py
  python scripts/data/build_input_output_relatedness.py --show-sample
"""

import argparse
import gzip
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.sni_codes import SNI_2digits_to_sections
from latent_industry.common.flow_metrics import symmetric_flow_phi
from latent_industry.common.paths import METADATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

metadata_dir = METADATA_DIR
io_dir = RAW_DATA_DIR / "io"
io_dir.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--show-sample", action="store_true")
parser.add_argument(
    "--max-year",
    type=int,
    default=2019,
    help="Last benchmark/year retained from every time-varying IO source.",
)
parser.add_argument(
    "--output-dir",
    default=None,
    help="Directory for the three output files (defaults to PROCESSED_DATA_DIR).",
)
args, _ = parser.parse_known_args()
if args.max_year < 2008:
    raise ValueError("--max-year below 2008 is not supported by the current SCB/WIOD blend")
OUT_DIR = Path(args.output_dir) if args.output_dir else PROCESSED_DATA_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
FLOWS_BY_YEAR_PATH = OUT_DIR / "io_flows_by_year.parquet"
OUT_PATH_SNI07 = OUT_DIR / "io_relatedness_SNI07.parquet"
OUT_PATH_SNI02 = OUT_DIR / "io_relatedness_SNI02.parquet"

# ── Source configuration ─────────────────────────────────────────────────────────

SCB_URL = "https://www.scb.se/contentassets/3ec4ba4f12e045ba858304efeb576f15/nrio_siot_181108.xlsx"
SCB_PATH = io_dir / "nrio_siot.xlsx"
SCB_YEARS = [str(y) for y in range(2008, min(2016, args.max_year) + 1)]

EUROSTAT_BULK_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
    "naio_10_cp1700?format=TSV&compressed=true"
)
EUROSTAT_PATH = io_dir / "eurostat_naio_10_cp1700_SE.csv"
EUROSTAT_DIMS = ["freq", "unit", "stk_flow", "prd_use", "prd_ava", "geo"]
EUROSTAT_YEARS = set(range(2010, min(2019, args.max_year) + 1))

WIOD_ZIP_URL = "https://dataverse.nl/api/access/datafile/199123"
WIOD_TMP_ZIP = io_dir / "_wiod2013_tmp.zip"
WIOD_YEAR_FILES = {
    year: filename
    for year, filename in {
        1995: "WIOT95_ROW_Apr12.xlsx",
        2000: "WIOT00_ROW_Apr12.xlsx",
        2005: "WIOT05_ROW_Apr12.xlsx",
    }.items()
    if year <= args.max_year
}
WIOD_SHEET_NAMES = {year: f"WIOT_{year}" for year in WIOD_YEAR_FILES}
WIOD_CODES_PATH = io_dir / "wiod_industry_codes.csv"

# Official Eurostat classification metadata behind the prd_use/prd_ava codes — the
# ground truth used to validate expand_code()'s output.
# Cached here rather than re-fetched each run, and kept as a citable artifact.
EUROSTAT_CODELIST_URL = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/codelist/ESTAT/PRD_AVA?format=TSV"
EUROSTAT_CODELIST_PATH = io_dir / "eurostat_prd_codelist.csv"

CODE_REFERENCE_PATH = io_dir / "code_reference.csv"


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url} -> {dest.name}")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def load_eurostat_codelist() -> pd.DataFrame:
    """
    Official Eurostat code → description list for the prd_use/prd_ava dimensions,
    used to validate expand_code()'s output.
    """
    if not EUROSTAT_CODELIST_PATH.exists():
        resp = requests.get(EUROSTAT_CODELIST_URL, timeout=60)
        resp.raise_for_status()
        rows = [line.split("\t", 1) for line in resp.text.splitlines() if "\t" in line]
        pd.DataFrame(rows, columns=["code", "description"]).to_csv(EUROSTAT_CODELIST_PATH, index=False)
    return pd.read_csv(EUROSTAT_CODELIST_PATH)


# ── Generic A64/A35-style code -> 2-digit division expansion ────────────────────
# Every source's industry/product codes reduce to the same shape: an optional
# "CPA_" prefix, a leading section letter, and zero, one, or two digit groups
# (e.g. "A01", "C10T12", "C10-12", "G45-47", "L68A", "B", "AtB", "T-U",
# "15t16"). All observed multi-digit codes are short consecutive ranges.

def _invert_section_map(version: str) -> dict[str, list[str]]:
    """{section_letter: [2-digit division codes]} for a given SNI version."""
    fwd = SNI_2digits_to_sections(version)  # {2-digit code: letter}
    inv: dict[str, list[str]] = {}
    for code, letter in fwd.items():
        if letter == "Unknown":
            continue
        inv.setdefault(letter, []).append(code)
    return inv


def expand_code(code, section_map_inv: dict[str, list[str]]) -> list[str]:
    """Expand an aggregated industry/product code to the 2-digit divisions it covers."""
    code = str(code).strip()
    if code.startswith("CPA_"):
        code = code[4:]
    digits = re.findall(r"\d+", code)
    if digits:
        if len(digits) == 1:
            return [digits[0].zfill(2)]
        lo, hi = int(digits[0]), int(digits[-1])
        return [str(d).zfill(2) for d in range(lo, hi + 1)]
    letters = re.findall(r"[A-Z]", code)
    if len(letters) == 1:
        return list(section_map_inv.get(letters[0], []))
    if len(letters) >= 2:
        lo, hi = sorted((letters[0], letters[-1]))
        return sorted({d for L, divs in section_map_inv.items() if lo <= L <= hi for d in divs})
    return []


def _allocate_shares(divisions: list[str]) -> list[tuple[str, float]]:
    """
    Turn a list of covered divisions into (division, share) pairs summing to 1.

    Shares are equal across the covered divisions (no employment or output
    split is available at this level). An aggregate cell's value is therefore
    split across its covered division pairs rather than copied to each of them,
    so total flow is conserved.
    """
    n = len(divisions)
    if n == 0:
        return []
    return [(d, 1.0 / n) for d in divisions]


def broadcast_to_divisions(flows: pd.DataFrame, section_map_inv: dict[str, list[str]],
                            extra_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Expand (code_from, code_to, value, *extra_cols) rows to (div_from,
    div_to, value, *extra_cols) rows, ALLOCATING each aggregate value across
    its covered divisions via _allocate_shares (equal-share) so that summing
    the expanded value over every (div_from, div_to) pair reproduces the
    original (code_from, code_to) value exactly. `extra_cols` (e.g. "year")
    are carried through unchanged — this is what lets the same function
    serve both the year-tagged tidy table and the pooled-across-years
    per-source phi input."""
    expanded = []
    share_cache: dict[str, list[tuple[str, float]]] = {}
    cols = ["code_from", "code_to", "value", *extra_cols]
    for row in flows[cols].itertuples(index=False):
        code_from, code_to, value, *extras = row
        for c in (code_from, code_to):
            if c not in share_cache:
                share_cache[c] = _allocate_shares(expand_code(c, section_map_inv))
        for d_from, w_from in share_cache[code_from]:
            for d_to, w_to in share_cache[code_to]:
                expanded.append((d_from, d_to, value * w_from * w_to, *extras))
    return pd.DataFrame(expanded, columns=["div_from", "div_to", "value", *extra_cols])


def _check_conservation(label: str, raw_total: float, expanded_total: float) -> None:
    """Print the conservation ratio (expanded / raw flow total); ~1.0 is expected."""
    ratio = expanded_total / raw_total if raw_total else float("nan")
    print(f"  IO conservation check ({label}): raw={raw_total:.2f}  "
          f"expanded={expanded_total:.2f}  ratio={ratio:.4f}")


def source_phi_by_year(flows: pd.DataFrame, section_map_inv: dict[str, list[str]],
                        label: str, source_name: str, version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expand a source's per-year flows to 2-digit divisions and return:
      (a) the year-tagged division-level table (source, version, year,
          nace2d_from, nace2d_to, value) for io_flows_by_year.parquet;
      (b) the source's symmetric_flow_phi, computed on the flows pooled over
          its years (all within one classification vintage).
    """
    expanded = broadcast_to_divisions(flows, section_map_inv, extra_cols=("year",))
    pooled = expanded.groupby(["div_from", "div_to"], as_index=False)["value"].sum()
    _check_conservation(label, flows["value"].sum(), expanded["value"].sum())
    phi = symmetric_flow_phi(pooled, "div_from", "div_to", "value")

    tidy = expanded.groupby(["div_from", "div_to", "year"], as_index=False)["value"].sum()
    tidy.insert(0, "source", source_name)
    tidy.insert(1, "version", version)
    tidy = tidy.rename(columns={"div_from": "nace2d_from", "div_to": "nace2d_to"})
    tidy["nace2d_from"] = tidy["nace2d_from"].str.zfill(2)
    tidy["nace2d_to"] = tidy["nace2d_to"].str.zfill(2)
    return tidy, phi


# ── Source 1: SCB domestic SIOT (native SNI07) ────────────────────────────────────

def load_scb_flows() -> pd.DataFrame:
    if not SCB_PATH.exists():
        print("SCB SIOT:")
        _download(SCB_URL, SCB_PATH)
    frames = []
    for year in SCB_YEARS:
        df = pd.read_excel(SCB_PATH, sheet_name=year, header=None)
        codes = df.iloc[3:62, 1].tolist()
        block = df.iloc[3:62, 2:61].copy()
        block.index = codes
        block.columns = codes
        long = block.stack().reset_index()
        long.columns = ["code_from", "code_to", "value"]
        long["year"] = int(year)
        frames.append(long)
    flows = pd.concat(frames, ignore_index=True)
    print(f"  SCB        : {len(SCB_YEARS)} years ({SCB_YEARS[0]}-{SCB_YEARS[-1]}), "
          f"{flows['code_from'].nunique()} codes, {len(flows):,} flow rows")
    return flows


# ── Source 2: Eurostat naio_10_cp1700, SE/DOM only (native SNI07) ────────────────

def load_eurostat_flows() -> pd.DataFrame:
    if not EUROSTAT_YEARS:
        print("  Eurostat   : unavailable before 2010; omitted for this cutoff")
        return pd.DataFrame(columns=["code_from", "code_to", "value", "year"])
    if not EUROSTAT_PATH.exists():
        print("Eurostat naio_10_cp1700:")
        print("  downloading + filtering bulk file to geo=SE, stk_flow=DOM (one-time) ...")
        resp = requests.get(EUROSTAT_BULK_URL, timeout=300)
        resp.raise_for_status()
        raw = gzip.decompress(resp.content).decode("utf-8")
        lines = raw.splitlines()
        year_cols = [c.strip() for c in lines[0].split("\t")[1:]]
        kept = []
        for line in lines[1:]:
            first, *vals = line.split("\t")
            dims = first.split(",")
            if dims[-1] != "SE" or dims[2] != "DOM":
                continue
            # Eurostat reports every value in both MIO_EUR and MIO_NAC (national
            # currency, SEK) for non-euro countries — keep exactly one or every
            # flow gets double-counted (once per currency) before summing.
            if dims[1] != "MIO_NAC":
                continue
            if not dims[3].startswith("CPA_") or not dims[4].startswith("CPA_"):
                continue
            kept.append(dims + vals)
        wide = pd.DataFrame(kept, columns=EUROSTAT_DIMS + year_cols)
        wide.to_csv(EUROSTAT_PATH, index=False)
    else:
        wide = pd.read_csv(EUROSTAT_PATH, dtype=str)

    year_cols = [c for c in wide.columns if c not in EUROSTAT_DIMS]
    long = wide.melt(id_vars=EUROSTAT_DIMS, value_vars=year_cols, var_name="year", value_name="raw_value")
    long["year"] = long["year"].astype(int)
    long = long[long["year"].isin(EUROSTAT_YEARS)]
    long["value"] = pd.to_numeric(long["raw_value"].str.extract(r"([\d.]+)")[0], errors="coerce")
    long = long.dropna(subset=["value"])
    flows = long.rename(columns={"prd_ava": "code_from", "prd_use": "code_to"})[
        ["code_from", "code_to", "value", "year"]
    ]
    print(f"  Eurostat   : years {min(EUROSTAT_YEARS)}-{max(EUROSTAT_YEARS)}, "
          f"{flows['code_from'].nunique()} codes, {len(flows):,} flow rows")
    return flows


# ── Source 3: WIOD 2013 release, pre-2008 years (native SNI02) ───────────────────

def load_wiod_flows() -> pd.DataFrame:
    print("WIOD 2013 release:")
    frames = []
    code_descriptions: dict[str, str] = {}
    for year, fname in WIOD_YEAR_FILES.items():
        cache_path = io_dir / f"wiod2013_SE_{year}.csv"
        if not cache_path.exists() or not WIOD_CODES_PATH.exists():
            if not WIOD_TMP_ZIP.exists():
                print("  downloading WIOD 2013 release zip (~500MB, one-time) ...")
                _download(WIOD_ZIP_URL, WIOD_TMP_ZIP)
            with zipfile.ZipFile(WIOD_TMP_ZIP) as zf:
                with zf.open(fname) as f:
                    raw = pd.read_excel(f, sheet_name=WIOD_SHEET_NAMES[year], header=None)
            country_col = raw.iloc[:, 2]
            country_row = raw.iloc[4]
            swe_rows = country_col[country_col == "SWE"].index.tolist()
            swe_cols = country_row[country_row == "SWE"].index.tolist()[: len(swe_rows)]
            codes = raw.iloc[swe_rows, 0].astype(str).tolist()
            descriptions = raw.iloc[swe_rows, 1].astype(str).tolist()
            code_descriptions.update(dict(zip(codes, descriptions)))
            if not cache_path.exists():
                block = raw.iloc[swe_rows, swe_cols].copy()
                block.index = codes
                block.columns = codes
                long = block.stack().reset_index()
                long.columns = ["code_from", "code_to", "value"]
                long.to_csv(cache_path, index=False)
        year_df = pd.read_csv(cache_path, dtype={"code_from": str, "code_to": str})
        year_df["year"] = int(year)
        frames.append(year_df)
    if WIOD_TMP_ZIP.exists():
        WIOD_TMP_ZIP.unlink()
    if code_descriptions and not WIOD_CODES_PATH.exists():
        pd.DataFrame(sorted(code_descriptions.items()), columns=["code", "description"]).to_csv(
            WIOD_CODES_PATH, index=False
        )
    flows = pd.concat(frames, ignore_index=True)
    print(f"  WIOD       : years {sorted(WIOD_YEAR_FILES)}, "
          f"{flows['code_from'].nunique()} codes, {len(flows):,} flow rows")
    return flows


def build_code_reference(
    scb_codes: set[str],
    eurostat_codes: set[str],
    wiod_codes: set[str],
    sni07_section_inv: dict[str, list[str]],
    sni02_section_inv: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Audit table: every raw code used by each source, its description, and the
    divisions expand_code() resolves it to. Written to raw/io/code_reference.csv.
    """
    eurostat_desc = load_eurostat_codelist().set_index("code")["description"].to_dict()
    wiod_desc = (
        pd.read_csv(WIOD_CODES_PATH).set_index("code")["description"].to_dict()
        if WIOD_CODES_PATH.exists()
        else {}
    )
    sni07_desc = json.loads((metadata_dir / "mapping_Sni_2digits_2007.json").read_text())
    sni02_desc = json.loads((metadata_dir / "mapping_Sni_2digits_2002.json").read_text())

    def _scb_description(code: str) -> str:
        for candidate in (f"CPA_{code}", f"CPA_{code.replace('T', '-')}", f"CPA_{code.replace('-', '_')}"):
            if candidate in eurostat_desc:
                return eurostat_desc[candidate]
        # No single official Eurostat aggregate covers this exact grouping (SCB groups
        # some divisions, e.g. G45-47, that Eurostat reports separately) — synthesize
        # from the individual divisions' own official descriptions instead.
        divisions = expand_code(code, sni07_section_inv)
        atoms = [eurostat_desc.get(f"CPA_{code[0]}{d}", "") for d in divisions]
        if all(atoms):
            return "SCB-specific grouping of: " + "; ".join(atoms)
        return ""

    rows = []
    for source, codes, desc_lookup, section_inv, division_desc in [
        ("scb", scb_codes, _scb_description, sni07_section_inv, sni07_desc),
        ("eurostat", eurostat_codes, lambda c: eurostat_desc.get(c, ""), sni07_section_inv, sni07_desc),
        ("wiod", wiod_codes, lambda c: wiod_desc.get(c, ""), sni02_section_inv, sni02_desc),
    ]:
        for code in sorted(codes):
            divisions = expand_code(code, section_inv)
            description = desc_lookup(code) if callable(desc_lookup) else desc_lookup
            rows.append({
                "source": source,
                "code": code,
                "description": description,
                "divisions": ",".join(divisions),
                "division_descriptions": " | ".join(division_desc.get(d, "?") for d in divisions),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

sni07_section_inv = _invert_section_map("SNI07")
sni02_section_inv = _invert_section_map("SNI02")

print("Loading IO sources …")
scb_flows = load_scb_flows()
eurostat_flows = load_eurostat_flows()
wiod_flows = load_wiod_flows()

print("\nBuilding code reference (audit trail for the code -> division mapping) …")
code_ref = build_code_reference(
    scb_codes=set(scb_flows["code_from"]) | set(scb_flows["code_to"]),
    eurostat_codes=set(eurostat_flows["code_from"]) | set(eurostat_flows["code_to"]),
    wiod_codes=set(wiod_flows["code_from"]) | set(wiod_flows["code_to"]),
    sni07_section_inv=sni07_section_inv,
    sni02_section_inv=sni02_section_inv,
)
code_ref.to_csv(CODE_REFERENCE_PATH, index=False)
print(f"  {len(code_ref)} codes -> {CODE_REFERENCE_PATH}")

print("\nComputing per-source, year-tagged division flows + pooled phi …")
tidy_scb, phi_scb = source_phi_by_year(scb_flows, sni07_section_inv, "SCB", "scb", "SNI07")
phi_scb = phi_scb.rename(columns={"phi": "phi_scb"})
if not eurostat_flows.empty:
    tidy_eurostat, phi_eurostat = source_phi_by_year(
        eurostat_flows, sni07_section_inv, "Eurostat", "eurostat", "SNI07"
    )
    phi_eurostat = phi_eurostat.rename(columns={"phi": "phi_eurostat"})
else:
    print("  Eurostat   : unavailable at this cutoff")
    tidy_eurostat = pd.DataFrame(columns=["source", "version", "year", "nace2d_from", "nace2d_to", "value"])
    phi_eurostat = pd.DataFrame(columns=["div_from", "div_to", "phi_eurostat"])

# WIOD stays in its native SNI02 divisions (no crosswalk to SNI07). Its codes
# are NACE Rev. 1.1 for every benchmark year (1995/2000/2005), hence SNI02.
tidy_wiod, phi_wiod = source_phi_by_year(wiod_flows, sni02_section_inv, "WIOD", "wiod", "SNI02")
phi_wiod = phi_wiod.rename(columns={"phi": "phi_wiod"})

for name, phi_df, col in [("SCB", phi_scb, "phi_scb"), ("Eurostat", phi_eurostat, "phi_eurostat"),
                           ("WIOD", phi_wiod, "phi_wiod")]:
    if phi_df.empty:
        print(f"  {name:<10}: omitted at this cutoff")
    else:
        print(f"  {name:<10}: {len(phi_df):,} pairs  (phi range [{phi_df[col].min():.4f}, "
              f"{phi_df[col].max():.4f}])")

# ── Year-tagged tidy table (all sources, native divisions, not yet pooled) ────────

flows_by_year = pd.concat([tidy_scb, tidy_eurostat, tidy_wiod], ignore_index=True)
flows_by_year.to_parquet(FLOWS_BY_YEAR_PATH, index=False)
print(f"\nSaved year-tagged division flows ({len(flows_by_year):,} rows) → {FLOWS_BY_YEAR_PATH}")

# ── Per-version blend ─────────────────────────────────────────────────────────────

print("\nBlending SNI07 sources (SCB + Eurostat) …")
blended_07 = phi_scb.merge(phi_eurostat, on=["div_from", "div_to"], how="outer")
phi_cols_07 = ["phi_scb", "phi_eurostat"]
blended_07["sim"] = blended_07[phi_cols_07].mean(axis=1, skipna=True)
blended_07["n_sources"] = blended_07[phi_cols_07].notna().sum(axis=1)
blended_07 = blended_07.rename(columns={"div_from": "nace2d_from", "div_to": "nace2d_to"})
blended_07["nace2d_from"] = blended_07["nace2d_from"].str.zfill(2)
blended_07["nace2d_to"] = blended_07["nace2d_to"].str.zfill(2)
blended_07 = blended_07[blended_07["nace2d_from"] != blended_07["nace2d_to"]].reset_index(drop=True)
result_07 = blended_07[["nace2d_from", "nace2d_to", "sim", *phi_cols_07, "n_sources"]]

print("SNI02 (WIOD only, no blending needed) …")
result_02 = phi_wiod.rename(columns={"div_from": "nace2d_from", "div_to": "nace2d_to"}).copy()
result_02["nace2d_from"] = result_02["nace2d_from"].str.zfill(2)
result_02["nace2d_to"] = result_02["nace2d_to"].str.zfill(2)
result_02 = result_02[result_02["nace2d_from"] != result_02["nace2d_to"]].reset_index(drop=True)
result_02["sim"] = result_02["phi_wiod"]
result_02["n_sources"] = 1
result_02 = result_02[["nace2d_from", "nace2d_to", "sim", "phi_wiod", "n_sources"]]

# No top-k pruning — every division pair with at least one source's flow data
# is kept. The typed-relation diffusion gates whole relation types per node
# (io vs. nace_2d vs. hierarchy, ...), not individual edges, so thinning
# this graph would only discard real signal.
for label, result, out_path in [("SNI07", result_07, OUT_PATH_SNI07), ("SNI02", result_02, OUT_PATH_SNI02)]:
    n_divisions = pd.unique(result[["nace2d_from", "nace2d_to"]].values.ravel()).size
    print(f"\n{label}: {len(result):,} pairs across {n_divisions} divisions")
    print("  pairs by n_sources:")
    print(result["n_sources"].value_counts().sort_index().to_string())
    result.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}")

metadata_path = OUT_DIR / "io_relatedness.metadata.json"
metadata_path.write_text(
    json.dumps(
        {
            "max_year": args.max_year,
            "sni07": {
                "output": str(OUT_PATH_SNI07),
                "sources": ["scb", "eurostat"],
                "scb_years": [int(y) for y in SCB_YEARS],
                "eurostat_years": sorted(EUROSTAT_YEARS),
                "rows": len(result_07),
            },
            "sni02": {
                "output": str(OUT_PATH_SNI02),
                "sources": ["wiod"],
                "wiod_years": sorted(WIOD_YEAR_FILES),
                "rows": len(result_02),
            },
            "flows_by_year": str(FLOWS_BY_YEAR_PATH),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"\nMetadata → {metadata_path}")

if args.show_sample:
    pd.set_option("display.max_colwidth", 80)
    print("\nSample SNI07 (top 20 by sim):")
    print(result_07.sort_values("sim", ascending=False).head(20).to_string())
    print("\nSample SNI02 (top 20 by sim):")
    print(result_02.sort_values("sim", ascending=False).head(20).to_string())
