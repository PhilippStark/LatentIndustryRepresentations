"""
_holdout_common.py
==================
Shared code of the hold-out and seed-stability checks of the main run
(scripts/robustness/H*.py).

Only the embedding changes between variants. Candidates and outcomes are
those of the published v16 panels
(<MAIN_DATA_ROOT>/processed/downstream_outcomes/<outcome>/shared_cosine_v16_panel.parquet);
density is recomputed for every panel row from the local employment of
that municipality and year (the employment cache of H0_employment_cache.py)
and the variant's embedding, in the main specification D^{A,log}
(latent_industry.regional.density, log(1 + employment) weights). The
regression is the main specification: municipality×year + industry FE,
SE clustered by municipality and industry (latent_industry.regional.fixed_effects).
When one regression stacks rows whose density comes from different
embeddings (spatial cross-fit, stacked temporal windows), the industry FE are
interacted with the embedding (`fe_group` of `estimate`): each embedding sets
its own density level for every industry, and a shared industry FE would
leave these level differences in the regressor as noise (attenuation). The
pooled slope is then a precision-weighted combination of the slopes of the
single embeddings.

Timing: a panel row of year t has density from M_t and an outcome realised
at t+1 (P04_entry.py, P04_exit.py, P04_startup_entry.py).

Locations
---------
  MAIN_DATA_ROOT   data root of the main run (LATENT_MAIN_DATA_ROOT overrides)
  HOLDOUT_ROOT     working area of the checks: common/, spatial_holdout/,
                   temporal_holdout/, seed_stability/ (HOLDOUT_ROOT overrides)
  RESULTS_ROOT     <project root>/results/ (aggregate output only):
                   holdout_checks/, regional_spatial_holdout/,
                   regional_temporal_holdout/, seed_stability/ (files
                   prefixed shared_cosine_v16_) and the SI tables in
                   paper/appendix/ (RESULT_DIRS, result_path)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import PROJECT_ROOT as REPO_ROOT
from latent_industry.regional.density import (
    cosine_matrix, empl_matrix, employment_weighted_density, load_sni_embeddings,
)
from latent_industry.regional.fixed_effects import (
    density_effect, estimate_by_period, fit, standardise_density,
)

MAIN_DATA_ROOT = Path(os.environ.get(
    "LATENT_MAIN_DATA_ROOT", "/lunarc/nobackup/projects/lu2025-17-39/DataLatentIndustryRepresentations",
))
HOLDOUT_ROOT = Path(os.environ.get("HOLDOUT_ROOT", MAIN_DATA_ROOT / "ablations" / "holdout_robustness"))
RESULTS_ROOT = project_root / "results"
# test -> folder under results/ (next to regional_fe_robustness/)
RESULT_DIRS = {
    "checks": "holdout_checks",
    "spatial_holdout": "regional_spatial_holdout",
    "temporal_holdout": "regional_temporal_holdout",
    "seed_stability": "seed_stability",
    "si": "paper/appendix",
}

EMPLOYMENT_CACHE = HOLDOUT_ROOT / "common" / "data" / "employment_muni_year_sni4d.parquet"
MAIN_RUN = "shared_cosine_v16"
REFERENCE_ROOT = HOLDOUT_ROOT / "common" / "data" / "reference_full"
REFERENCE_RUN = "v16_reference"

# period -> (embedding label, base years t), as in P04_entry.py
PERIODS = {
    "SNI92": ("sni92", range(1990, 2001)),
    "SNI02": ("sni02", range(2002, 2006)),
    "SNI07": ("sni07", range(2007, 2019)),
}
SD_MULT = 2.0  # P04 default: significance band of the entry/exit outcomes

OUTCOMES = {
    "entry":         {"col": "entry_industry"},
    "exit":          {"col": "exit_industry"},
    "startup_entry": {"col": "startup_entry"},
}

EFFECT_COLUMNS = ["period", "coef", "se", "pval", "ci_low", "ci_high", "n", "n_municipalities",
                  "rate", "density_sd"]


# ── Paths of the variants ─────────────────────────────────────────────────────

def spatial_variant(repeat: int, fold: int) -> tuple[Path, str]:
    return HOLDOUT_ROOT / "spatial_holdout" / "data" / f"r{repeat}_f{fold}", f"v16_sp_r{repeat}_f{fold}"


def temporal_variant(t0: int) -> tuple[Path, str]:
    return HOLDOUT_ROOT / "temporal_holdout" / "data" / f"T{t0}", f"v16_tw_T{t0}"


def is_trained(root: Path, run: str) -> bool:
    return (root / "embedding" / run / "training_provenance.json").exists()


def results_dir(test: str) -> Path:
    path = RESULTS_ROOT / RESULT_DIRS[test]
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_path(test: str, name: str) -> Path:
    """results/<folder of test>/shared_cosine_v16_<name>; the SI tables keep
    their bare name, as the other tables in results/paper/appendix/."""
    return results_dir(test) / (name if test == "si" else f"{MAIN_RUN}_{name}")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ── Embeddings and employment ─────────────────────────────────────────────────

class Embedding:
    """SNI92/02/07 4-digit embedding of one run, with its cosine matrices."""

    def __init__(self, name: str, root: Path, run: str):
        self.name, self.root, self.run = name, Path(root), run
        loaded = load_sni_embeddings(run, self.root / "embedding", verbose=False)
        self.codes = {label: codes for label, (_, codes) in loaded.items()}
        self.vecs = {label: vecs for label, (vecs, _) in loaded.items()}
        self.cos = {label: cosine_matrix(vecs) for label, vecs in self.vecs.items()}


def main_embedding() -> Embedding:
    return Embedding("v16", MAIN_DATA_ROOT, MAIN_RUN)


def reference_embedding() -> Embedding | None:
    if not is_trained(REFERENCE_ROOT, REFERENCE_RUN):
        print(f"  WARNING: {REFERENCE_RUN} not trained yet — reported without the retrain control.")
        return None
    return Embedding("v16_reference", REFERENCE_ROOT, REFERENCE_RUN)


class Employment:
    """Municipality × industry employment matrices M_t from the employment
    cache, in the row order of P04 (sorted municipalities) and the column
    order of the embedding's codes."""

    def __init__(self, path: Path = EMPLOYMENT_CACHE):
        if not path.exists():
            raise FileNotFoundError(f"{path} — run H0_employment_cache.py first.")
        df = pd.read_parquet(path)
        self.municipalities = sorted(df["municipality"].unique())
        self._by_year = {int(y): g for y, g in df.groupby("Year")}
        self._cache: dict[tuple[int, str], np.ndarray] = {}
        self.codes: dict[str, list[str]] = {}

    def set_codes(self, codes: dict[str, list[str]]) -> None:
        self.codes = codes

    def matrix(self, year: int, label: str) -> np.ndarray:
        key = (year, label)
        if key not in self._cache:
            self._cache[key] = empl_matrix(self._by_year[year], self.municipalities, year, self.codes[label])
        return self._cache[key]

    def density(self, year: int, label: str, emb: Embedding) -> np.ndarray:
        """D^{A,log} for every (municipality, industry) cell in `year`."""
        return employment_weighted_density(np.log1p(self.matrix(year, label)), emb.cos[label])


# ── Panels ────────────────────────────────────────────────────────────────────

class Panel:
    """A published v16 panel with the matrix position of every row."""

    def __init__(self, outcome: str, emp: Employment):
        self.outcome = outcome
        self.col = OUTCOMES[outcome]["col"]
        path = MAIN_DATA_ROOT / "processed" / "downstream_outcomes" / outcome / f"{MAIN_RUN}_panel.parquet"
        self.df = pd.read_parquet(path)
        muni_pos = {m: i for i, m in enumerate(emp.municipalities)}
        self.mi = self.df["municipality"].map(muni_pos).to_numpy()
        ci = np.full(len(self.df), -1)
        for period, (label, _) in PERIODS.items():
            mask = (self.df["period"] == period).to_numpy()
            pos = {c: i for i, c in enumerate(emp.codes[label])}
            ci[mask] = self.df.loc[mask, "industry"].map(pos).to_numpy()
        if np.isnan(self.mi.astype(float)).any() or (ci < 0).any():
            raise ValueError(f"{path}: municipality or industry not in the employment cache / embedding")
        self.mi = self.mi.astype(int)
        self.ci = ci.astype(int)
        self.groups = {(p, int(y)): rows for (p, y), rows in self.df.groupby(["period", "year"]).indices.items()}

    def density(self, emp: Employment, emb: Embedding, groups=None) -> np.ndarray:
        """Density of each row (NaN outside `groups`) from its own municipality's
        employment in its year and the embedding `emb`."""
        out = np.full(len(self.df), np.nan)
        for (period, year), rows in self.groups.items():
            if groups is not None and (period, year) not in groups:
                continue
            D = emp.density(year, PERIODS[period][0], emb)
            out[rows] = D[self.mi[rows], self.ci[rows]]
        return out

    def outcome_at_horizon(self, emp: Employment, period: str, t0: int, h: int) -> np.ndarray:
        """Outcome of the candidates of row (period, t0), realised at t0 + h
        (h = 1 reproduces the published outcome; start-up entry has no h > 1)."""
        rows = self.groups[(period, t0)]
        label = PERIODS[period][0]
        M_th = emp.matrix(t0 + h, label)
        if self.outcome == "entry":
            Y = entry_outcome(M_th)
        elif self.outcome == "exit":
            Y = exit_outcome(emp.matrix(t0, label), M_th)
        else:
            raise ValueError(f"no h-step outcome for {self.outcome}")
        return Y[self.mi[rows], self.ci[rows]].astype(float)


def load_main() -> tuple[Employment, Embedding]:
    """Employment cache with the v16 code order, and the v16 embedding."""
    emp = Employment()
    v16 = main_embedding()
    emp.set_codes(v16.codes)
    return emp, v16


def check_codes(emb: Embedding, reference: Embedding) -> None:
    for label, codes in reference.codes.items():
        if emb.codes[label] != codes:
            raise ValueError(f"{emb.name}: {label} codes differ from {reference.name}")


# ── Outcomes at other horizons (rules of P04_entry.py / P04_exit.py) ──────────

def significance_band(arcsinh_vals: np.ndarray, active_mask: np.ndarray, sd_mult: float):
    """Per municipality, ±sd_mult SD band (arcsinh scale) across its active
    industries — copied from P04_entry.py / P04_exit.py."""
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


def entry_outcome(M_t1: np.ndarray, sd_mult: float = SD_MULT) -> np.ndarray:
    """P04_entry.compute_entry_outcome: positive at t1 and within the band at t1."""
    arcsinh_t1 = np.arcsinh(M_t1)
    sig_lo, sig_hi = significance_band(arcsinh_t1, M_t1 > 0, sd_mult)
    return (M_t1 > 0) & (arcsinh_t1 >= sig_lo[:, None]) & (arcsinh_t1 <= sig_hi[:, None])


def exit_outcome(M_t: np.ndarray, M_t1: np.ndarray, sd_mult: float = SD_MULT) -> np.ndarray:
    """P04_exit.compute_exit_outcome: within the band at t and zero at t1."""
    arcsinh_t = np.arcsinh(M_t)
    sig_lo, sig_hi = significance_band(arcsinh_t, M_t > 0, sd_mult)
    return (M_t > 0) & (M_t1 == 0) & (arcsinh_t >= sig_lo[:, None]) & (arcsinh_t <= sig_hi[:, None])


# ── Estimation ────────────────────────────────────────────────────────────────

def _grouped_effects(frame: pd.DataFrame, outcome_col: str, annual: bool) -> tuple[dict, pd.DataFrame | None]:
    """As estimate_by_period / density_effect, with industry×fe_group FE."""
    fe = [("municipality", "year"), ("industry", "fe_group")]
    results, frames = {}, []
    for period in dict.fromkeys(frame["period"]):
        sub = frame[frame["period"] == period]
        r = fit(sub, outcome_col, ["density_std"], fe)
        results[period] = {k: (float(v[0]) if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        if annual:
            sub = sub.copy()
            years = sorted(sub["year"].unique())
            cols = []
            for t in years:
                sub[f"_d_{t}"] = sub["density_std"] * (sub["year"] == t)
                cols.append(f"_d_{t}")
            ry = fit(sub, outcome_col, cols, fe)
            n_by_year = sub.groupby("year").size()
            frames.append(pd.DataFrame({
                "year": years, "period": period, "coef": ry["coef"], "se": ry["se"], "t": ry["t"],
                "pval": ry["pval"], "ci_low": ry["ci_low"], "ci_high": ry["ci_high"], "df": ry["df"],
                "n": [int(n_by_year[t]) for t in years],
            }))
    return results, (pd.concat(frames, ignore_index=True) if annual else None)


def estimate(df: pd.DataFrame, outcome_col: str, density: np.ndarray, annual: bool = True,
             fe_group: np.ndarray | None = None, **labels) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Main specification on `df` with `density` as the regressor,
    standardised by its SD pooled over the sample (as in P04). Returns one row
    per period (labels + EFFECT_COLUMNS) and, if `annual`, the year-specific
    slopes. `fe_group` (one label per row: the embedding the row's density
    comes from) replaces the industry FE by industry×fe_group FE; without it
    the estimate is exactly the main specification."""
    frame = df[["period", "year", "municipality", "industry"]].copy()
    frame[outcome_col] = df[outcome_col].to_numpy(dtype=float)
    frame["density"] = np.asarray(density, dtype=float)
    if frame["density"].isna().any():
        raise ValueError("density has missing values in the estimation sample")
    sd = standardise_density(frame)
    if fe_group is not None:
        frame["fe_group"] = np.asarray(fe_group)
        results, year_df = _grouped_effects(frame, outcome_col, annual)
    elif annual:
        results, year_df = estimate_by_period(frame, outcome_col, verbose=False)
    else:
        results = {p: density_effect(frame[frame["period"] == p], outcome_col)
                   for p in dict.fromkeys(frame["period"])}
        year_df = None
    if year_df is not None:
        for key, value in reversed(labels.items()):
            year_df.insert(0, key, value)
    rows = [{**labels, "period": p, "coef": r["coef"], "se": r["se"], "pval": r["pval"],
             "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": r["n"],
             "n_municipalities": r["g_muni"], "rate": r["rate"], "density_sd": sd,
             "industry_fe": "industry" if fe_group is None else "industry x embedding"}
            for p, r in results.items()]
    return pd.DataFrame(rows), year_df


def print_effects(rows: pd.DataFrame, label: str) -> None:
    for _, r in rows.iterrows():
        print(f"    {label:<28} {r['period']}: β={r['coef'] * 100:+.3f} pp  SE={r['se'] * 100:.3f}  "
              f"N={int(r['n']):,}")
