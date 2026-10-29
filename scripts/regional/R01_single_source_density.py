"""
R01_single_source_density.py
============================
Does opportunity density (the learned embedding) carry information beyond
the individual relational views? Estimated for industry entry, industry exit,
startup formation and, as in the SI, positive employment growth and positive
wage growth (samples and outcomes as in P04_entry.py, P04_exit.py,
P04_startup_entry.py, P04_growth.py and P04_wage_growth.py).

All densities are built identically on the same panel: for municipality m,
industry j and year t, the log-employment-weighted, leave-one-out mean of a
national industry-pair similarity (latent_industry.regional.density), with
one similarity matrix per classification version. Only the similarity
differs:

  opportunity    cosine similarity of the main run's embeddings
  labor          labor relatedness as used to train the embedding
                 (layers_{graph}.npz: observed/expected worker flows, row-normalised)
  ownership      ownership relatedness as used in training (flow share, row-normalised)
  io             2-digit IO relatedness assigned to all 4-digit pairs in
                 different divisions, 0 within a division; no SNI92 source
  hierarchy      3 = same 3-digit group, 2 = same division, 1 = same section, 0 otherwise
  baseline       the untrained 64-d text-prior initialisation of the main run

Estimates, with the fixed-effects specification --spec (default: main,
municipality×year + industry FE; SE clustered by municipality and industry):

  single source  outcome on one density (partial r is compared across sources,
                 as the densities have different scales)
  multiple regression  outcome on the opportunity density and one single-view
                 density in the same model (each standardised within the period)

The opportunity density rebuilt here is checked against the main run's own
results (results/regional_*/{run}_summary.csv); small differences come from
this script's employment panel, which also requires a firm identifier.

Outputs (<LATENT_DATA_ROOT>/robustness/single_source_density/, suffix _<spec>
for non-main specifications)
  single_source_density_comparison.csv         single-density estimates
  multiple_regression.csv                      multiple regressions, every view
  (paper figure and tables: scripts/paper/plot_multiple_regression.py)

Requires the restricted SCB register data.

Usage (main run)
----------------
  python scripts/regional/R01_single_source_density.py --run-name shared_cosine_v16
  python scripts/regional/R01_single_source_density.py --run-name shared_cosine_v16 --spec additive
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import EMBEDDING_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, METADATA_DIR, LUNARC_DATA_ROOT
from latent_industry.common.municipalities import harmonization_table, load_municipality_crosswalk
from latent_industry.common.sni_codes import SNI_2digits_to_sections
from latent_industry.graph.nodes import NodeIndex
from latent_industry.regional.fixed_effects import SPECS, density_effect, fit
from latent_industry.regional.density import (
    load_sni_embeddings, employment_weighted_density, cosine_matrix, empl_matrix as _shared_empl_matrix,
)

DATA_ROOT    = Path("/lunarc/nobackup/projects/lu2025-17-39")
INDIVID_PATH = DATA_ROOT / "SCB" / "Preprocessed" / "individ_cleaned_1990-2019.parquet"
FAD_PATH     = RAW_DATA_DIR / "FAD_1990-2019.parquet"
MUNI_XWALK   = harmonization_table(load_municipality_crosswalk())

OUT_DIR = LUNARC_DATA_ROOT / "robustness" / "single_source_density"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_RUN_NAME = "shared_cosine_v16"
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
                     help="Main run whose embeddings give the opportunity density.")
parser.add_argument("--out-suffix", default="",
                     help="Selects which {run}{out_suffix}_summary.csv to compare against (see "
                          "P04_entry.py's own --out-suffix), and is appended to this script's own "
                          "output filenames so an investigative run doesn't overwrite "
                          "single_source_density_comparison.{csv,png}.")
parser.add_argument("--spec", default="main", choices=["main", "additive", "muni_ind"],
                     help="Fixed-effects specification (latent_industry.regional.fixed_effects.SPECS); "
                          "non-main specifications add a filename suffix.")
parser.add_argument("--graph-name", default="layers_v1",
                     help="Graph whose training-input matrices (layers_{name}.npz) give the "
                          "labor and ownership sources.")
parser.add_argument("--max-founder-size", type=int, default=DEFAULT_MAX_FOUNDER_SIZE)
parser.add_argument("--max-growth-ratio", type=float, default=DEFAULT_MAX_GROWTH_RATIO)
parser.add_argument("--seed", type=int, default=42,
                     help="Must match the main run's --seed for the baseline PCA fit to be the "
                          "same one that run's own Z actually started from.")
args = parser.parse_args()

print(f"Run to compare against: {args.run_name}   "
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
raw = pl.read_parquet(INDIVID_PATH, columns=["Year", "Sni", "WorkplaceMunicipality", "OrgNr", "CfarNr",
                                             "EmploymentIncome"])

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
    .agg([pl.len().alias("workers"), pl.col("EmploymentIncome").mean().alias("mean_income")])
    .sort(["municipality", "Year", "sni_4d"])
    .to_pandas()
)
all_municipalities = sorted(empl["municipality"].unique())
print(f"  {len(empl):,} (municipality, year, industry) observations, "
      f"{len(all_municipalities)} municipalities")


def empl_matrix(year: int, codes: list[str]) -> np.ndarray:
    return _shared_empl_matrix(empl, all_municipalities, year, codes)


def wage_matrix(year: int, codes: list[str]) -> np.ndarray:
    """Mean employment income per (municipality, industry); NaN where the cell is absent."""
    sub = empl[(empl["Year"] == year) & (empl["sni_4d"].isin(codes))]
    piv = sub.pivot_table(index="municipality", columns="sni_4d", values="mean_income", aggfunc="mean")
    return piv.reindex(index=all_municipalities, columns=codes).to_numpy(np.float64)


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

def build_training_input_source(view: str) -> dict[str, np.ndarray]:
    """The view's transition matrix exactly as the embedding was trained on it
    (layers_{graph}.npz, built by build_layer_transition_matrices.py): raw
    relatedness values, row-normalised per industry. Entry [i, j] is the share
    of industry i's relatedness that goes to j, so the density of j sums these
    shares over the local industries i."""
    print(f"\nBuilding {view} training-input kernel (layers_{args.graph_name}.npz) …")
    layers = np.load(PROCESSED_DATA_DIR / f"layers_{args.graph_name}.npz", allow_pickle=True)
    P = layers[f"{view}__P"]
    ni = NodeIndex()  # node order of layers_{graph}.npz
    out: dict[str, np.ndarray] = {}
    for ver in ["SNI92", "SNI02", "SNI07"]:
        idx = [ni.get("4d", ver, c) for c in CODES[ver]]
        if any(i is None for i in idx):
            print(f"  SKIPPED {ver} — codes missing from the graph's node index")
            continue
        K = P[np.ix_(idx, idx)].astype(np.float64)
        np.fill_diagonal(K, 0.0)
        out[ver] = K
        print(f"    {ver}: {int((K > 0).sum()):,} directed pairs on {len(idx)} codes")
    return out


def build_opportunity_source() -> dict[str, np.ndarray]:
    """Cosine similarity of the main run's embeddings — the opportunity
    density itself, rebuilt on this script's panel so that the multiple regressions
    compare both densities on identical observations."""
    print("\nBuilding opportunity-density kernel (main-run embeddings) …")
    return {ver: cosine_matrix(_sni_emb[label][0].astype(np.float64))
            for ver, label in [("SNI92", "sni92"), ("SNI02", "sni02"), ("SNI07", "sni07")]}


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
    "opportunity": build_opportunity_source,
    "labor":       lambda: build_training_input_source("labor"),
    "ownership":   lambda: build_training_input_source("ownership"),
    "io":          build_io_source,
    "hierarchy":   build_hierarchy_source,
    "baseline":    lambda: build_baseline_source(args.seed),
}
VIEWS = [name for name in SOURCES if name != "opportunity"]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Candidate sets and outcomes as in P04_entry.py / P04_exit.py /
#    P04_startup_entry.py, with a source's similarity matrix in place of the
#    embedding's; regression with the main specification
#    (latent_industry.regional.fixed_effects).
# ═══════════════════════════════════════════════════════════════════════════

WAGE_MIN_WORKERS = 5   # as P04_wage_growth.py: more than 5 workers in both years


def growth_outcome(c: dict) -> np.ndarray:
    """Positive employment growth: asinh(E_t+1) − asinh(E_t) > 0."""
    return np.arcsinh(c["M_t1"]) - np.arcsinh(c["M_t"]) > 0


def wage_growth_outcome(c: dict) -> np.ndarray:
    """Positive wage growth: asinh(mean income t+1) − asinh(mean income t) > 0."""
    diff = np.arcsinh(wage_matrix(c["t"] + 1, c["codes"])) - np.arcsinh(wage_matrix(c["t"], c["codes"]))
    return np.nan_to_num(diff, nan=0.0) > 0


# Candidate sets and outcomes as in P04_entry.py, P04_exit.py,
# P04_startup_entry.py and the binarised appendix outcomes of P04_growth.py /
# P04_wage_growth.py. candidate(M_t, M_t1) -> bool matrix; outcome(context) ->
# bool matrix, or "merge" for start-ups (lookup in startup_counts).
OUTCOME_SPECS = {
    "entry": {"candidate": lambda M_t, M_t1: M_t == 0, "col": "entry_industry",
              "outcome": lambda c: c["M_t1"] > 0},
    "exit": {"candidate": lambda M_t, M_t1: M_t > 0, "col": "exit_industry",
             "outcome": lambda c: c["M_t1"] == 0},
    "startup_entry": {"candidate": lambda M_t, M_t1: M_t == 0, "col": "startup_entry", "outcome": "merge"},
    "growth": {"candidate": lambda M_t, M_t1: (M_t + M_t1) > 0, "col": "growth_binary",
               "outcome": growth_outcome},
    "wage_growth": {"candidate": lambda M_t, M_t1: (M_t > WAGE_MIN_WORKERS) & (M_t1 > WAGE_MIN_WORKERS),
                    "col": "wage_growth_binary", "outcome": wage_growth_outcome},
}


def build_panel(outcome_name: str, period_name: str, kernels: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """Candidate panel of one outcome and period with one density column per
    source (density_<source>), all from the same log-employment weights."""
    spec = OUTCOME_SPECS[outcome_name]
    codes = CODES[period_name]
    frames = []
    for t in PERIOD_YEARS[period_name]:
        M_t, M_t1 = empl_matrix(t, codes), empl_matrix(t + 1, codes)
        M_t_log = np.log1p(M_t)
        mi, ci = np.where(spec["candidate"](M_t, M_t1))
        frame = pd.DataFrame({
            "municipality": [all_municipalities[i] for i in mi],
            "industry":     [codes[j] for j in ci],
            "year":         t,
        })
        for source, by_period in kernels.items():
            if period_name in by_period:
                frame[f"density_{source}"] = employment_weighted_density(M_t_log, by_period[period_name])[mi, ci]
        if spec["outcome"] == "merge":   # start-up entry: lookup in startup_counts
            frame["year_t1"] = t + 1
        else:
            outcome = spec["outcome"]({"M_t": M_t, "M_t1": M_t1, "t": t, "codes": codes})
            frame[spec["col"]] = outcome[mi, ci].astype(float)
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    if spec["outcome"] == "merge":
        panel = panel.merge(startup_counts[["industry", "municipality", "year_t1", spec["col"]]],
                            on=["industry", "municipality", "year_t1"], how="left")
        panel[spec["col"]] = (panel[spec["col"]] == True).astype(float)  # noqa: E712 — NaN -> False
    # Standardise every density within the period (effects per 1 SD).
    for source in kernels:
        col = f"density_{source}"
        if col in panel:
            sd = float(panel[col].std())
            panel[f"{col}_std"] = panel[col] / sd if sd > 0 else np.nan
    return panel


# ═══════════════════════════════════════════════════════════════════════════
# 5. Single-density estimates and multiple regressions (opportunity density
#    and one single-view density in the same model), every outcome and period, with the
#    chosen fixed-effects specification and two-way clustered SE.
# ═══════════════════════════════════════════════════════════════════════════

kernels = {name: builder() for name, builder in SOURCES.items()}
kernels = {name: k for name, k in kernels.items() if k}
fe = SPECS[args.spec]["fe"]
single_rows: list[dict] = []
mr_rows: list[dict] = []
KEEP = ("coef", "se", "pval", "ci_low", "ci_high", "n", "partial_r", "rate", "pct_of_baseline")

for outcome_name, ospec in OUTCOME_SPECS.items():
    col = ospec["col"]
    for period_name in PERIOD_YEARS:
        print(f"\n{outcome_name} / {period_name}: building panel …")
        panel = build_panel(outcome_name, period_name, kernels)
        for source in kernels:
            x = f"density_{source}_std"
            if x not in panel or panel[x].isna().all():
                continue
            r = density_effect(panel, col, density_col=x, spec=args.spec)
            single_rows.append({"source": source, "outcome": outcome_name, "period": period_name,
                                **{k: r[k] for k in KEEP}})
            print(f"  single  {source:<13} β={r['coef'] * 100:+.3f} pp  partial r={r['partial_r']:+.4f}  "
                  f"N={r['n']:,}")
        for view in VIEWS:
            x = f"density_{view}_std"
            if x not in panel or panel[x].isna().all():
                continue
            r = fit(panel, col, ["density_opportunity_std", x], fe)
            row = {"view": view, "outcome": outcome_name, "period": period_name, "n": r["n"], "df": r["df"]}
            for k, name in enumerate(["opportunity", "view"]):
                for stat in ("coef", "se", "pval", "ci_low", "ci_high"):
                    row[f"{name}_{stat}"] = float(r[stat][k])
            mr_rows.append(row)
            print(f"  multiple regression with {view:<10} opportunity β={row['opportunity_coef'] * 100:+.3f} pp "
                  f"[{row['opportunity_ci_low'] * 100:+.3f}, {row['opportunity_ci_high'] * 100:+.3f}]   "
                  f"{view} β={row['view_coef'] * 100:+.3f} pp "
                  f"[{row['view_ci_low'] * 100:+.3f}, {row['view_ci_high'] * 100:+.3f}]")
        del panel

single_df = pd.DataFrame(single_rows)
mr_df = pd.DataFrame(mr_rows)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Check: the opportunity density rebuilt here reproduces the main run
#    (P04 scripts, main specification). Partial r is compared because P04
#    standardises density over all periods and this script within periods.
# ═══════════════════════════════════════════════════════════════════════════

if args.spec == "main":
    print("\nCheck against the main run (results/regional_*/{run}_summary.csv):")
    for outcome_name, subdir, stem in [
            ("entry", "regional_entry", ""), ("exit", "regional_exit", ""),
            ("startup_entry", "regional_startup_entry", ""),
            ("growth", "regional_growth", "_growth_binary"),
            ("wage_growth", "regional_wage_growth", "_wage_growth_binary")]:
        path = project_root / "results" / subdir / f"{args.run_name}{args.out_suffix}{stem}_summary.csv"
        if not path.exists():
            print(f"  {path} not found — skipped")
            continue
        col = OUTCOME_SPECS[outcome_name]["col"]
        ref = pd.read_csv(path).set_index("period")
        mine = single_df[(single_df["source"] == "opportunity") & (single_df["outcome"] == outcome_name)]
        for _, r in mine.iterrows():
            print(f"  {outcome_name:<14}{r['period']}: partial r {r['partial_r']:+.4f} here vs "
                  f"{ref.loc[r['period'], f'{col}_r']:+.4f} main run;  N {int(r['n']):,} vs "
                  f"{int(ref.loc[r['period'], 'n_obs']):,}")

OUT_TAG = args.out_suffix + ("" if args.spec == "main" else f"_{args.spec}")
single_path = OUT_DIR / f"single_source_density_comparison{OUT_TAG}.csv"
mr_path = OUT_DIR / f"multiple_regression{OUT_TAG}.csv"
single_df.to_csv(single_path, index=False)
mr_df.to_csv(mr_path, index=False)
print(f"\n  → {single_path}\n  → {mr_path}")


print("\nFigure and tables: scripts/paper/plot_multiple_regression.py")
print("\nDone.")
