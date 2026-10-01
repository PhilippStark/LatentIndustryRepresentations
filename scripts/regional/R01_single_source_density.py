"""
R01_single_source_density.py
=============================
Robustness check: does the entry/exit/startup_entry density result depend
on the FUSED, trained embedding — or would any single raw evidence source,
used alone as the similarity kernel, produce a similar (or different)
result?

For each of 5 single-source kernels — labor, io, ownership, hierarchy, and
the untrained 64d text-prior-init+PCA space (the same warm start
shared_cosine_v16's own Z began training from, see
results/shared_cosine_v16_SPECS.md) — this rebuilds the SAME candidate-set +
industry+municipality+year-FE regression (separate additive municipality
and year effects, not an interacted municipality×year term)
P04_entry.py/P04_exit.py/P04_startup_entry.py use, substituting that
source's own raw relatedness values for the trained cosine-similarity
kernel. Only the similarity matrix changes — the
regression TARGET is identical across every evaluation here, exactly as it
is across those three original scripts: the candidate-panel construction
(industries absent at t), demeaning, and OLS specification are copied
verbatim (not imported — kept fully self-contained here so this robustness
check can never silently drift if the main scripts change).

startup_entry uses the EXACT SAME absent-at-t candidate panel as entry —
P04_startup_entry.py's own candidate/density construction is P04_entry.py's,
reused rather than reinvented (see that script's own docstring: "Regional
density test ... Exactly P04_entry.py's design, reused rather than
reinvented"). The only real difference is its outcome: instead of a
function of (M_t, M_t1), it's a merge against `startup_counts` — a lookup
table built ONCE (shared across every source, since it doesn't depend on
density at all) from the Grillitsch (2021) OrgNr+FAD genuine-new-firm
identification. That one-time lookup is the only extra machinery
startup_entry needs; the regression itself is unchanged.

Per-source construction, and where each one is domain-limited (flagged
directly, not silently patched over):

  labor / ownership   Direct (sni_from, sni_to, phi) lookup from
                       labor_relatedness.parquet / ownership_relatedness.parquet,
                       symmetrised, 0 for any pair not present (same convention
                       the actual layer-building code uses). All three periods
                       (SNI92/02/07) — both sources are natively per-version.

  io                   io_relatedness_{SNI07,SNI02}.parquet is 2-DIGIT only
                       (no 4d input-output data exists at all — same reason
                       the real io training layer needs lifting). Lifted to
                       4d here by giving every (i, j) pair with i, j in
                       DIFFERENT 2-digit divisions their divisions' io sim;
                       same-division pairs get 0 (io has no self-loop, same
                       convention build_io_layer/lift_io_layer already use).
                       NO SNI92 — no io source reaches back that far at all
                       (same gap the real training pipeline has); skipped for
                       that period, not fabricated.

  hierarchy            No raw pairwise relatedness table exists for
                       hierarchy — it's a parent/child tree, not a flat
                       relation. Built here directly from each version's own
                       code strings (same code-prefix logic
                       build_hierarchy_layer/build_hierarchy_ordinal_tiers
                       already use, generalised to all three versions instead
                       of just SNI07's precomputed helper): sim=3 same
                       3-digit group, 2 same 2-digit division, 1 same
                       section, 0 different section. All three periods.

  baseline (init)      The exact text-prior-init + PCA recipe
                       train_shared_cosine_embedding.py uses (encode every
                       node's own NACE description with
                       sentence-transformers/all-MiniLM-L6-v2, PCA to 64d,
                       re-normalise) — regenerated here since the initial Z
                       is never itself persisted as its own artifact (only
                       the post-training Z is saved). Same --seed 42 as v16's
                       actual training run, for a reproducible PCA fit. All
                       three periods (encodes every node regardless of
                       version).

Scope, flagged directly rather than silently included/omitted:
  - entry, exit, AND startup_entry are all covered — the same three the
    paper's core regional analysis reports. growth/wage_growth are
    appendix-only material and are not part of this check.
  - shared_cosine_v16's own numbers are NOT recomputed here — they're read
    straight from its already-written results/regional_{entry,exit,
    startup_entry}/shared_cosine_v16_summary.csv, for a like-for-like
    comparison against what's already the paper's own reported result.

Output: NOT written to the git-tracked project results/ tree — this is a
one-off robustness check, not paper material — goes to
<LUNARC_DATA_ROOT>/robustness/single_source_density/ instead (comparison
CSV + boxplot PNG).

Usage
-----
  python scripts/regional/R01_single_source_density.py
  python scripts/regional/R01_single_source_density.py --run-name shared_cosine_v16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm

matplotlib.use("Agg")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import EMBEDDING_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, METADATA_DIR, LUNARC_DATA_ROOT
from latent_industry.common.municipalities import harmonization_table, load_municipality_crosswalk
from latent_industry.common.sni_codes import SNI_2digits_to_sections
from latent_industry.regional.density import load_sni_embeddings, employment_weighted_density, empl_matrix as _shared_empl_matrix

DATA_ROOT    = Path("/lunarc/nobackup/projects/lu2025-17-39")
INDIVID_PATH = DATA_ROOT / "SCB" / "Preprocessed" / "individ_cleaned_1990-2019.parquet"
FAD_PATH     = RAW_DATA_DIR / "FAD_1990-2019.parquet"
MUNI_XWALK   = harmonization_table(load_municipality_crosswalk())

OUT_DIR = LUNARC_DATA_ROOT / "robustness" / "single_source_density"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_RUN_NAME = "shared_cosine_v16"
DEFAULT_SD_MULT  = 2.0
SAMPLE_START_YEAR = 1990

# Grillitsch (2021) startup identification — identical defaults/logic to
# P04_startup_entry.py (see that script's module docstring for the full
# definition and rationale for each condition).
DEFAULT_MAX_FOUNDER_SIZE = 10
DEFAULT_MAX_GROWTH_RATIO = 0.0
with (METADATA_DIR / "mapping_Kvar.json").open(encoding="utf-8") as _f:
    KVAR_LABELS = json.load(_f)
KVAR_STARTUP_CODES = {1, 2, 3, 6, 7, 8}

PERIOD_YEARS = {
    "SNI92": range(1990, 2001),
    "SNI02": range(2002, 2006),
    "SNI07": range(2007, 2019),
}

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                     help="Main fused-embedding run to compare against (its own results are read "
                          "from results/regional_{entry,exit,startup_entry}/{run}{out_suffix}_summary.csv, "
                          "not recomputed).")
parser.add_argument("--out-suffix", default="",
                     help="Selects which {run}{out_suffix}_summary.csv to compare against (see "
                          "P04_entry.py's own --out-suffix), and is appended to this script's own "
                          "output filenames so an investigative run doesn't overwrite "
                          "single_source_density_comparison.{csv,png}.")
parser.add_argument("--sd-mult", type=float, default=DEFAULT_SD_MULT)
parser.add_argument("--max-founder-size", type=int, default=DEFAULT_MAX_FOUNDER_SIZE)
parser.add_argument("--max-growth-ratio", type=float, default=DEFAULT_MAX_GROWTH_RATIO)
parser.add_argument("--seed", type=int, default=42,
                     help="Must match the main run's --seed for the baseline PCA fit to be the "
                          "same one that run's own Z actually started from.")
args = parser.parse_args()

print(f"Run to compare against: {args.run_name}   sd_mult={args.sd_mult}   "
      f"max_founder_size<={args.max_founder_size}   max_growth_ratio={args.max_growth_ratio}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load individual data ONCE — one filtered/harmonized base frame reused
#    for the (municipality, year, industry) employment panel (density input,
#    shared by all three outcomes) AND the (OrgNr, year) startup-eligibility
#    aggregate — same single-scan strategy P04_startup_entry.py itself uses,
#    and needed here for the SAME reason: including OrgNr/CfarNr in the base
#    filter (P04_entry.py/P04_exit.py's own `base` doesn't) so entry/exit/
#    startup_entry all come from one identical employment panel within this
#    script — internal consistency across the 5 sources is what matters here,
#    not bit-identical reproduction of P04_entry.py's own standalone numbers
#    (shared_cosine_v16's own official numbers are read from its existing
#    summary.csv separately, untouched by this choice).
# ═══════════════════════════════════════════════════════════════════════════

print(f"\nLoading {INDIVID_PATH} …")
raw = pl.read_parquet(INDIVID_PATH, columns=["Year", "Sni", "WorkplaceMunicipality", "OrgNr", "CfarNr"])

base = (
    raw
    .filter(pl.col("Sni").is_not_null() & (pl.col("Sni") != 0))
    .filter(pl.col("WorkplaceMunicipality").is_not_null())
    .filter(pl.col("OrgNr").is_not_null() & (pl.col("OrgNr") != 0))
    .with_columns([
        pl.col("Sni").cast(pl.Utf8).str.zfill(5).str.slice(0, 4).alias("sni_4d"),
        pl.col("WorkplaceMunicipality").cast(pl.Utf8).str.zfill(4).alias("source_code"),
    ])
    .filter(pl.col("sni_4d") != "0000")
    .join(MUNI_XWALK, on=["Year", "source_code"], how="inner")
)

empl = (
    base.group_by(["municipality", "Year", "sni_4d"])
    .agg(pl.len().alias("workers"))
    .sort(["municipality", "Year", "sni_4d"])
    .to_pandas()
)
all_municipalities = sorted(empl["municipality"].unique())
print(f"  {len(empl):,} (municipality, year, industry) observations, "
      f"{len(all_municipalities)} municipalities")


def empl_matrix(year: int, codes: list[str]) -> np.ndarray:
    return _shared_empl_matrix(empl, all_municipalities, year, codes)


# ═══════════════════════════════════════════════════════════════════════════
# 1b. Grillitsch (2021) startup identification — computed ONCE, shared by
#    every source below (it doesn't depend on density/embeddings at all).
#    Copied verbatim from P04_startup_entry.py's own steps 1-2 — see that
#    script's module docstring for the full definition of each condition.
# ═══════════════════════════════════════════════════════════════════════════

print("\nAggregating to (OrgNr, Year) for startup eligibility …")
org_year = base.group_by(["OrgNr", "Year"]).agg([
    pl.len().alias("founder_size"),
    pl.col("CfarNr").n_unique().alias("n_cfarnr"),
    pl.col("CfarNr").first().alias("cfarnr"),
    pl.col("sni_4d").first().alias("sni_4d"),
    pl.col("municipality").first().alias("municipality"),
])
print(f"  {len(org_year):,} (OrgNr, Year) rows")

prev_existence = org_year.select(["OrgNr", "Year"]).with_columns((pl.col("Year") + 1).alias("Year"))
eligible = (
    org_year.join(prev_existence, on=["OrgNr", "Year"], how="anti")
    .filter(pl.col("Year") > SAMPLE_START_YEAR)
    .filter((pl.col("n_cfarnr") == 1) & (pl.col("founder_size") <= args.max_founder_size))
)
print(f"  {len(eligible):,} OrgNr-year rows pass absent-in-t-1 + single-CfarNr + "
      f"founder-size<={args.max_founder_size} filters")

print(f"\nLoading {FAD_PATH} …")
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
_kvar_desc = ", ".join(f"{c}={KVAR_LABELS.get(f'{c:02d}', '?')}" for c in sorted(KVAR_STARTUP_CODES))
print(f"  {len(fad):,} FAD rows with a qualifying Kvar code ({_kvar_desc})")

startups = eligible.join(fad, on=["OrgNr", "cfarnr", "Year"], how="inner")
print(f"  {len(startups):,} OrgNr-year rows are Grillitsch-compliant startups (pre growth-ratio filter)")

if args.max_growth_ratio > 0:
    next_size = (
        org_year.select(["OrgNr", "Year", "founder_size"])
        .rename({"founder_size": "next_size"})
        .with_columns((pl.col("Year") - 1).alias("Year"))
    )
    before = len(startups)
    startups = startups.join(next_size, on=["OrgNr", "Year"], how="left")
    ratio_ok = pl.col("next_size").is_null() | (
        pl.col("next_size") / pl.col("founder_size").clip(lower_bound=1) <= args.max_growth_ratio
    )
    startups = startups.filter(ratio_ok)
    print(f"  --max-growth-ratio {args.max_growth_ratio}: {before:,} -> {len(startups):,} rows")

startup_counts = (
    startups.group_by(["municipality", "sni_4d", "Year"])
    .agg(pl.len().alias("n_startups"))
    .rename({"sni_4d": "industry", "Year": "year_t1"})
    .with_columns(pl.lit(True).alias("startup_entry"))
    .to_pandas()
)
print(f"  {len(startup_counts):,} distinct (industry, municipality, year) cells "
      f"with at least one qualifying startup")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Code universe per period — SAME codes shared_cosine_v16 itself was
#    evaluated against (its own embeddings_sni*.npz), so every source here
#    is compared over an identical candidate universe, not a source-specific
#    one. Vectors themselves are discarded; only the code lists are used.
# ═══════════════════════════════════════════════════════════════════════════

_sni_emb = load_sni_embeddings(args.run_name, EMBEDDING_DIR)
CODES = {"SNI92": _sni_emb["sni92"][1], "SNI02": _sni_emb["sni02"][1], "SNI07": _sni_emb["sni07"][1]}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Per-source cos_sim builders — each returns {period: (n, n) ndarray},
#    aligned to CODES[period]'s own order. Missing pairs -> 0 (no evidence,
#    same convention the real layer-building code uses for both labor and
#    ownership already).
# ═══════════════════════════════════════════════════════════════════════════

def _pairwise_lookup_source(parquet_path: Path, from_col: str, to_col: str, val_col: str,
                             coding_col: str | None, versions: list[str]) -> dict[str, np.ndarray]:
    """Shared builder for labor/ownership (coding_col="coding", per-version rows)."""
    if not parquet_path.exists():
        print(f"  SKIPPED — {parquet_path} not found.")
        return {}
    df = pd.read_parquet(parquet_path)
    out: dict[str, np.ndarray] = {}
    for ver in versions:
        sub = df[df[coding_col] == ver] if coding_col else df
        codes = CODES[ver]
        pos = {c: i for i, c in enumerate(codes)}
        n = len(codes)
        M = np.zeros((n, n), dtype=np.float64)
        hits = 0
        for row in sub.itertuples(index=False):
            a, b = str(getattr(row, from_col)), str(getattr(row, to_col))
            ia, ib = pos.get(a), pos.get(b)
            if ia is None or ib is None:
                continue
            v = float(getattr(row, val_col))
            M[ia, ib] = v
            M[ib, ia] = v
            hits += 1
        np.fill_diagonal(M, 0.0)
        out[ver] = M
        print(f"    {ver}: {hits:,} pairs matched onto {n} codes")
    return out


def build_labor_source() -> dict[str, np.ndarray]:
    print("\nBuilding labor single-source kernel …")
    return _pairwise_lookup_source(
        PROCESSED_DATA_DIR / "labor_relatedness.parquet",
        "sni_from", "sni_to", "phi", "coding", ["SNI92", "SNI02", "SNI07"],
    )


def build_ownership_source() -> dict[str, np.ndarray]:
    print("\nBuilding ownership single-source kernel …")
    return _pairwise_lookup_source(
        PROCESSED_DATA_DIR / "ownership_relatedness.parquet",
        "sni_from", "sni_to", "phi", "coding", ["SNI92", "SNI02", "SNI07"],
    )


def build_io_source() -> dict[str, np.ndarray]:
    """2d-level io_relatedness_{ver}.parquet lifted to 4d: pair (i, j) with i, j
    in different 2-digit divisions gets that division-pair's sim; same-division
    pairs (and any division io_relatedness has no row for) get 0. No SNI92 —
    no io source covers that period at all."""
    print("\nBuilding io single-source kernel (lifted 2d->4d; no SNI92 source) …")
    out: dict[str, np.ndarray] = {}
    for ver in ["SNI02", "SNI07"]:
        path = PROCESSED_DATA_DIR / f"io_relatedness_{ver}.parquet"
        if not path.exists():
            print(f"  SKIPPED {ver} — {path} not found.")
            continue
        df = pd.read_parquet(path)
        sim2d: dict[tuple[str, str], float] = {}
        for row in df.itertuples(index=False):
            a, b = str(row.nace2d_from).zfill(2), str(row.nace2d_to).zfill(2)
            sim2d[(a, b)] = float(row.sim)
            sim2d[(b, a)] = float(row.sim)
        codes = CODES[ver]
        n = len(codes)
        div = [c[:2] for c in codes]
        M = np.zeros((n, n), dtype=np.float64)
        hits = 0
        for i in range(n):
            for j in range(i + 1, n):
                if div[i] == div[j]:
                    continue
                v = sim2d.get((div[i], div[j]))
                if v is None:
                    continue
                M[i, j] = v
                M[j, i] = v
                hits += 1
        out[ver] = M
        print(f"    {ver}: {hits:,} cross-division 4d pairs assigned a lifted io sim "
              f"(same-division pairs left at 0)")
    return out


def build_hierarchy_source() -> dict[str, np.ndarray]:
    """No raw hierarchy relatedness table exists — built directly from each
    version's own code-prefix structure (same logic build_hierarchy_layer /
    build_hierarchy_ordinal_tiers already use, generalised to all 3 versions
    instead of just the precomputed SNI07-only helper): sim=3 same 3-digit
    group, 2 same 2-digit division (different 3-digit group), 1 same section
    (different division), 0 different section."""
    print("\nBuilding hierarchy single-source kernel (code-prefix tiers) …")
    out: dict[str, np.ndarray] = {}
    for ver in ["SNI92", "SNI02", "SNI07"]:
        codes = CODES[ver]
        sec_map = SNI_2digits_to_sections(ver)
        d3 = [c[:3] for c in codes]
        d2 = [c[:2] for c in codes]
        sec = [sec_map.get(c[:2], "?") for c in codes]
        n = len(codes)
        d3a, d2a = np.array(d3), np.array(d2)
        seca = np.array(sec)
        same3 = d3a[:, None] == d3a[None, :]
        same2 = d2a[:, None] == d2a[None, :]
        samesec = seca[:, None] == seca[None, :]
        M = np.where(same3, 3.0, np.where(same2, 2.0, np.where(samesec, 1.0, 0.0)))
        np.fill_diagonal(M, 0.0)
        out[ver] = M
        print(f"    {ver}: {n} codes, tier distribution (off-diagonal pairs) "
              f"3={int((M == 3).sum() / 2):,} 2={int((M == 2).sum() / 2):,} "
              f"1={int((M == 1).sum() / 2):,} 0={int(((M == 0).sum() - n) / 2):,}")
    return out


def build_baseline_source(seed: int) -> dict[str, np.ndarray]:
    """The untrained text-prior-init + PCA space shared_cosine_v16's own Z
    started from (see train_shared_cosine_embedding.py's --text-prior-init
    --text-prior-pca path) — regenerated here since the initial Z isn't
    itself saved as a persistent artifact (only post-training Z is)."""
    print("\nBuilding baseline (untrained text-prior-init+PCA) kernel …")
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from latent_industry.graph.nodes import NodeIndex

    ni = NodeIndex()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = ni.nodes["description"].tolist()
    vecs = model.encode(texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    pca = PCA(n_components=64, random_state=seed)
    vecs = pca.fit_transform(vecs)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    print(f"    384d -> 64d ({pca.explained_variance_ratio_.sum():.1%} of variance retained)")

    out: dict[str, np.ndarray] = {}
    for ver in ["SNI92", "SNI02", "SNI07"]:
        codes = CODES[ver]
        idx = [ni.get("4d", ver, c) for c in codes]
        if any(i is None for i in idx):
            print(f"  SKIPPED {ver} — some codes missing from NodeIndex (unexpected).")
            continue
        v = vecs[idx]
        cos = (v @ v.T).astype(np.float64)
        np.fill_diagonal(cos, 0.0)
        out[ver] = cos
    return out


SOURCES = {
    "labor":     build_labor_source,
    "io":        build_io_source,
    "ownership": build_ownership_source,
    "hierarchy": build_hierarchy_source,
    "baseline":  lambda: build_baseline_source(args.seed),
}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Candidate-set + FE regression — copied from P04_entry.py/P04_exit.py
#    (not imported, deliberately self-contained here), parameterised by an
#    injected cos_sim per period instead of a trained embedding's own.
# ═══════════════════════════════════════════════════════════════════════════

def significance_band(arcsinh_vals: np.ndarray, active_mask: np.ndarray, sd_mult: float):
    n_rows = arcsinh_vals.shape[0]
    sig_lo, sig_hi = np.full(n_rows, np.nan), np.full(n_rows, np.nan)
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


def entry_outcome(M_t, M_t1, sd_mult):
    arcsinh_t1 = np.arcsinh(M_t1)
    active1 = M_t1 > 0
    sig_lo, sig_hi = significance_band(arcsinh_t1, active1, sd_mult)
    return (M_t1 > 0) & (arcsinh_t1 >= sig_lo[:, None]) & (arcsinh_t1 <= sig_hi[:, None])


def exit_outcome(M_t, M_t1, sd_mult):
    arcsinh_t = np.arcsinh(M_t)
    active0 = M_t > 0
    sig_lo, sig_hi = significance_band(arcsinh_t, active0, sd_mult)
    within_band = (arcsinh_t >= sig_lo[:, None]) & (arcsinh_t <= sig_hi[:, None])
    return (M_t > 0) & (M_t1 == 0) & within_band


OUTCOME_SPECS = {
    "entry": {"candidate_mask": lambda M_t: M_t == 0, "kind": "fn", "outcome_fn": entry_outcome,
              "col": "entry_industry"},
    "exit":  {"candidate_mask": lambda M_t: M_t > 0,  "kind": "fn", "outcome_fn": exit_outcome,
              "col": "exit_industry"},
    # Same absent-at-t candidate set as entry — P04_startup_entry.py's own
    # design, reused rather than reinvented (see its module docstring).
    # Outcome comes from a merge against the startup_counts lookup built
    # once in section 1b, not a function of (M_t, M_t1).
    "startup_entry": {"candidate_mask": lambda M_t: M_t == 0, "kind": "merge", "col": "startup_entry"},
}


def demean(df, col, by):
    return df[col] - df.groupby(by)[col].transform("mean")


def run_ols(y, x, cluster):
    X = sm.add_constant(x.values)
    if cluster.nunique() > 1:
        res = sm.OLS(y.values, X).fit(cov_type="cluster", cov_kwds={"groups": cluster.values})
    else:
        res = sm.OLS(y.values, X).fit(cov_type="HC3")
    r2 = float(res.rsquared)
    partial_r = float(np.sign(res.params[1]) * np.sqrt(r2))
    return {"coef": float(res.params[1]), "se": float(res.bse[1]), "pval": float(res.pvalues[1]),
            "n": int(res.nobs), "partial_r": partial_r}


def run_outcome_for_source(outcome_name: str, source_name: str, cos_sim_by_period: dict[str, np.ndarray],
                            sd_mult: float) -> list[dict]:
    spec = OUTCOME_SPECS[outcome_name]
    rows = []
    for period_name, cos_sim in cos_sim_by_period.items():
        codes = CODES[period_name]
        frames = []
        for t in PERIOD_YEARS[period_name]:
            t1 = t + 1
            M_t = empl_matrix(t, codes)
            M_t_log = np.log1p(M_t)
            density = employment_weighted_density(M_t_log, cos_sim)
            mask = spec["candidate_mask"](M_t)
            mi, ci = np.where(mask)
            frame = pd.DataFrame({
                "municipality": [all_municipalities[i] for i in mi],
                "industry":     [codes[j] for j in ci],
                "year":         t,
                "density":      density[mi, ci],
            })
            if spec["kind"] == "fn":
                M_t1 = empl_matrix(t1, codes)
                outcome = spec["outcome_fn"](M_t, M_t1, sd_mult)
                frame[spec["col"]] = outcome[mi, ci].astype(float)
            else:  # "merge" — startup_entry
                frame["year_t1"] = t1
            frames.append(frame)
        panel = pd.concat(frames, ignore_index=True)
        if spec["kind"] == "merge":
            panel = panel.merge(
                startup_counts[["industry", "municipality", "year_t1", spec["col"]]],
                on=["industry", "municipality", "year_t1"], how="left",
            )
            panel[spec["col"]] = (panel[spec["col"]] == True).astype(float)  # noqa: E712 — NaN -> False
        sd_density = float(panel["density"].std())
        if sd_density == 0 or np.isnan(sd_density):
            print(f"    {source_name} / {outcome_name} / {period_name}: SKIPPED — zero-variance "
                  f"density (source has no coverage over this period's active industries).")
            continue
        panel["density_std"] = panel["density"] / sd_density
        for var in [spec["col"], "density_std"]:
            panel[f"{var}_r1"] = demean(panel, var,         "municipality")
            panel[f"{var}_r2"] = demean(panel, f"{var}_r1", "year")
            panel[f"{var}_dm"] = demean(panel, f"{var}_r2", "industry")
        res = run_ols(panel[f"{spec['col']}_dm"], panel["density_std_dm"], panel["municipality"])
        rate = float(panel[spec["col"]].mean())
        res["pct_of_baseline"] = res["coef"] / rate * 100 if rate > 0 else float("nan")
        res["rate"] = rate
        res.update({"source": source_name, "outcome": outcome_name, "period": period_name})
        rows.append(res)
        print(f"    {source_name:<10} {outcome_name:<6} {period_name}: "
              f"coef={res['coef']:+.5f}  partial_r={res['partial_r']:+.4f}  "
              f"pct_of_baseline={res['pct_of_baseline']:+6.1f}%  N={res['n']:,}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 5. Run every source x outcome
# ═══════════════════════════════════════════════════════════════════════════

all_rows: list[dict] = []
for source_name, builder in SOURCES.items():
    cos_sim_by_period = builder()
    if not cos_sim_by_period:
        continue
    for outcome_name in OUTCOME_SPECS:
        all_rows.extend(run_outcome_for_source(outcome_name, source_name, cos_sim_by_period, args.sd_mult))

comparison_df = pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Fold in shared_cosine_v16's own already-computed numbers (read, not
#    recomputed) for a like-for-like comparison row set.
# ═══════════════════════════════════════════════════════════════════════════

print(f"\nReading {args.run_name}{args.out_suffix}'s own already-computed entry/exit/startup_entry "
      "results for comparison …")
for outcome_name, subdir in [("entry", "regional_entry"), ("exit", "regional_exit"),
                              ("startup_entry", "regional_startup_entry")]:
    summary_path = project_root / "results" / subdir / f"{args.run_name}{args.out_suffix}_summary.csv"
    if not summary_path.exists():
        print(f"  SKIPPED — {summary_path} not found.")
        continue
    summary = pd.read_csv(summary_path)
    col = OUTCOME_SPECS[outcome_name]["col"]
    for _, row in summary.iterrows():
        all_rows.append({
            "source": "main_run", "outcome": outcome_name, "period": row["period"],
            "coef": row[f"{col}_coef"], "se": row[f"{col}_se"], "pval": row[f"{col}_pval"],
            "n": row["n_obs"], "partial_r": row[f"{col}_r"],
            "pct_of_baseline": row[f"{col}_pct_of_baseline"], "rate": row[f"{col}_rate"],
        })

comparison_df = pd.DataFrame(all_rows)
csv_path = OUT_DIR / f"single_source_density_comparison{args.out_suffix}.csv"
comparison_df.to_csv(csv_path, index=False)
print(f"\n  → {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Boxplot — one box per source (data points = that source's available
#    periods), entry/exit/startup_entry side by side. partial_r is the
#    metric: unlike the raw coefficient (whose scale depends on how much
#    within-FE variance the density regressor has left, not comparable
#    across sources with very different density distributions) it's a
#    bounded, correlation-like number explicitly meant to be comparable
#    across runs/periods (see P04_entry.py's own run_ols docstring).
#
#    Styling matches latent_industry.regional.regional_figures.
#    plot_density_effect_summary (the entry.png/exit.png/startup_entry.png
#    figures this compares against): no figure-level suptitle (panel titles
#    carry the framing), axhline(0, black, "--") as the zero reference, same
#    grid style, and — the actual fix here — each panel keeps its OWN
#    y-range via relim()/autoscale_view() instead of a shared one. exit's
#    values are all negative; forcing entry's positive range onto it (the
#    previous sharey=True) compressed its boxes into a sliver of the shared
#    axis instead of using its own full height.
# ═══════════════════════════════════════════════════════════════════════════

SOURCE_ORDER = ["baseline", "hierarchy", "io", "ownership", "labor", "main_run"]
# Display labels only — SOURCE_ORDER/comparison_df's "source" values (used for
# filtering) are unchanged; "main_run" is the fused, trained embedding this
# whole check is stress-testing, i.e. the paper's actual opportunity density.
SOURCE_LABELS = {"main_run": "Opportunity\nDensity"}
OUTCOME_TITLES = {"entry": "Entry", "exit": "Exit", "startup_entry": "Start-up"}
OUTCOME_YLABELS = {"entry": "Effect on entry (partial r)", "exit": "Effect on exit (partial r)",
                    "startup_entry": "Effect on start-up (partial r)"}
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for ax, outcome_name in zip(axes, OUTCOME_SPECS):
    sub = comparison_df[comparison_df["outcome"] == outcome_name]
    box_data, labels = [], []
    for source_name in SOURCE_ORDER:
        vals = sub.loc[sub["source"] == source_name, "partial_r"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        box_data.append(vals)
        # Point count per box = number of classification periods (SNI92/02/07)
        # this source has a regression for — 3 for every source except io
        # (2: no SNI92 io source exists at all, see build_io_source's own
        # docstring) — noted in the figure's caption/description, not on the
        # plot itself.
        labels.append(SOURCE_LABELS.get(source_name, source_name))
    ax.boxplot(box_data, tick_labels=labels, showmeans=True)
    for i, vals in enumerate(box_data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.03, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="black", s=14, zorder=3, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(OUTCOME_TITLES[outcome_name])
    ax.set_ylabel(OUTCOME_YLABELS[outcome_name])
    ax.tick_params(axis="x", labelsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.relim()
    ax.autoscale_view()
    y0, y1 = ax.get_ylim()
    pad = 0.08 * (y1 - y0)
    ax.set_ylim(y0 - pad, y1 + pad)
plt.tight_layout()
fig_path = OUT_DIR / f"single_source_density_comparison{args.out_suffix}.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  → {fig_path}")

print("\nDone.")
