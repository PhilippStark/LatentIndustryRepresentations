"""
P01_source_complementarity.py
=============================
Result 1: do the relatedness sources carry complementary information? A
model-free comparison of the raw sources (labor, input–output, ownership)
and the official NACE hierarchy, before any embedding is trained.

All sources are compared at the granularity they share, 2-digit divisions
(SNI07 by default). Labor and ownership relatedness (4-digit) are averaged
over all cross-division 4-digit pairs; IO relatedness is natively 2-digit;
the hierarchy enters as a same-section indicator. Low agreement between
sources is the expected finding: it motivates learning one representation
from all of them.

Analyses (one sources × sources matrix each)
--------------------------------------------
A  Spearman correlation of pairwise values over division pairs covered by both sources
B  Nearest-neighbour overlap: Jaccard overlap of each division's top --topk
   neighbours, averaged over divisions
D  Linear CKA between the sources' similarity matrices
E  Mutual information (kNN estimator, symmetrised) between pairwise values

Outputs: results/source_complementarity/P01_{a,b,d,e}_*_matrix.csv and a figure.
build_paper_tables.py restricts these to the sources used in a given run.

Usage
-----
  python scripts/evaluation/P01_source_complementarity.py
  python scripts/evaluation/P01_source_complementarity.py --topk 5 --sni-version SNI07
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

matplotlib.use("Agg")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import PROCESSED_DATA_DIR
from latent_industry.common.sni_codes import SNI_2digits_to_sections

RAW_SOURCE_FILES = {
    "labor": ("labor_relatedness.parquet", "phi"),
    "ownership": ("ownership_relatedness.parquet", "phi"),
}
NATIVE_2D_SOURCE_FILES = {
    "io": ("io_relatedness_{ver}.parquet", "sim"),
}
ALL_SOURCE_NAMES = list(RAW_SOURCE_FILES) + list(NATIVE_2D_SOURCE_FILES) + ["hierarchy"]


# ── Loading + alignment to 2-digit divisions ───────────────────────────────

def _load_4d_source_as_2d(filename: str, value_col: str, sni_version: str) -> pd.DataFrame:
    """4-digit sni_from/sni_to phi table -> mean-aggregated 2-digit division
    pairs, restricted to one coding vintage. Returns columns [a, b, value]
    with a < b (undirected, deduplicated)."""
    df = pd.read_parquet(PROCESSED_DATA_DIR / filename)
    df = df[df["coding"] == sni_version].copy()
    if df.empty:
        return pd.DataFrame(columns=["a", "b", "value"])
    df["div_from"] = df["sni_from"].astype(str).str[:2]
    df["div_to"] = df["sni_to"].astype(str).str[:2]
    df = df[df["div_from"] != df["div_to"]]
    agg = df.groupby(["div_from", "div_to"])[value_col].mean().reset_index()
    a = np.minimum(agg["div_from"], agg["div_to"])
    b = np.maximum(agg["div_from"], agg["div_to"])
    out = pd.DataFrame({"a": a, "b": b, "value": agg[value_col]})
    return out.groupby(["a", "b"])["value"].mean().reset_index()


def _load_native_2d_source(filename_template: str, value_col: str, sni_version: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / filename_template.format(ver=sni_version)
    if not path.exists():
        return pd.DataFrame(columns=["a", "b", "value"])
    df = pd.read_parquet(path)
    a = df["nace2d_from"].astype(str).str.zfill(2)
    b = df["nace2d_to"].astype(str).str.zfill(2)
    keep = a != b
    lo = np.minimum(a[keep], b[keep])
    hi = np.maximum(a[keep], b[keep])
    out = pd.DataFrame({"a": lo, "b": hi, "value": df.loc[keep, value_col].values})
    return out.groupby(["a", "b"])["value"].mean().reset_index()


def _hierarchy_source(divisions: list[str], sni_version: str) -> pd.DataFrame:
    """Binary same-section indicator — the only division-level signal the
    official hierarchy carries once you are already at the division digit
    (see module docstring)."""
    sec_map = SNI_2digits_to_sections(sni_version)
    rows = []
    for a, b in itertools.combinations(sorted(divisions), 2):
        sa, sb = sec_map.get(a), sec_map.get(b)
        if sa is None or sb is None:
            continue
        rows.append((a, b, float(sa == sb)))
    return pd.DataFrame(rows, columns=["a", "b", "value"])


def load_all_sources(sni_version: str) -> dict[str, pd.DataFrame]:
    sources = {}
    for name, (fname, col) in RAW_SOURCE_FILES.items():
        sources[name] = _load_4d_source_as_2d(fname, col, sni_version)
    for name, (fname_tmpl, col) in NATIVE_2D_SOURCE_FILES.items():
        sources[name] = _load_native_2d_source(fname_tmpl, col, sni_version)
    divisions = sorted(set().union(
        *[set(df["a"]) | set(df["b"]) for df in sources.values() if not df.empty]
    ))
    sources["hierarchy"] = _hierarchy_source(divisions, sni_version)
    return sources


def _paired_values(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    merged = df_a.merge(df_b, on=["a", "b"], suffixes=("_a", "_b"))
    return merged["value_a"].to_numpy(), merged["value_b"].to_numpy()


# ── The five pairwise metrics ───────────────────────────────────────────────

def correlation_metric(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    va, vb = _paired_values(df_a, df_b)
    if len(va) < 10 or np.ptp(va) == 0 or np.ptp(vb) == 0:
        return float("nan")
    return float(spearmanr(va, vb).statistic)


def nn_overlap_metric(df_a: pd.DataFrame, df_b: pd.DataFrame, topk: int) -> float:
    def neighbours(df: pd.DataFrame) -> dict[str, list[str]]:
        sym = pd.concat([
            df.rename(columns={"a": "src", "b": "dst"}),
            df.rename(columns={"b": "src", "a": "dst"}),
        ])
        out = {}
        for src, grp in sym.groupby("src"):
            out[src] = grp.sort_values("value", ascending=False)["dst"].head(topk).tolist()
        return out

    nbr_a, nbr_b = neighbours(df_a), neighbours(df_b)
    shared_nodes = set(nbr_a) & set(nbr_b)
    if not shared_nodes:
        return float("nan")
    jaccards = []
    for node in shared_nodes:
        sa, sb = set(nbr_a[node]), set(nbr_b[node])
        union = sa | sb
        jaccards.append(len(sa & sb) / len(union) if union else 0.0)
    return float(np.mean(jaccards))


def cka_metric(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    """Caveat: missing (a, b) pairs within the common division set are
    zero-filled (see to_matrix below) rather than dropped, since CKA needs
    both Gram matrices defined over the identical n x n index set. Sources
    with very different sparsity patterns (e.g. ownership's pairs vs.
    io's 3828) therefore share some structure purely from where each is
    zero — CKA values here run higher than the correlation/MI matrices for
    that reason and should be read as an upper bound, not compared at face
    value against the sparser pairwise metrics."""
    common = sorted((set(df_a["a"]) | set(df_a["b"])) & (set(df_b["a"]) | set(df_b["b"])))
    n = len(common)
    if n < 10:
        return float("nan")
    idx = {c: i for i, c in enumerate(common)}

    def to_matrix(df: pd.DataFrame) -> np.ndarray:
        m = np.full((n, n), np.nan)
        for r in df.itertuples():
            if r.a in idx and r.b in idx:
                i, j = idx[r.a], idx[r.b]
                m[i, j] = m[j, i] = r.value
        np.fill_diagonal(m, np.nanmax(m) if np.isfinite(np.nanmax(m)) else 0.0)
        return np.nan_to_num(m, nan=0.0)

    Ka, Kb = to_matrix(df_a), to_matrix(df_b)

    def hsic(K1: np.ndarray, K2: np.ndarray) -> float:
        h = np.eye(n) - np.ones((n, n)) / n
        k1c, k2c = h @ K1 @ h, h @ K2 @ h
        return float(np.sum(k1c * k2c))

    denom = np.sqrt(hsic(Ka, Ka) * hsic(Kb, Kb))
    return float(hsic(Ka, Kb) / denom) if denom > 0 else float("nan")


def mutual_information_metric(df_a: pd.DataFrame, df_b: pd.DataFrame, seed: int = 0) -> float:
    va, vb = _paired_values(df_a, df_b)
    if len(va) < 10:
        return float("nan")
    mi_ab = mutual_info_regression(va.reshape(-1, 1), vb, random_state=seed)[0]
    mi_ba = mutual_info_regression(vb.reshape(-1, 1), va, random_state=seed)[0]
    return float((mi_ab + mi_ba) / 2)


# ── Orchestration ───────────────────────────────────────────────────────────

def build_matrix(sources: dict[str, pd.DataFrame], metric_fn) -> pd.DataFrame:
    names = list(sources.keys())
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    for name in names:
        mat.loc[name, name] = 1.0
    for a, b in itertools.combinations(names, 2):
        val = metric_fn(sources[a], sources[b])
        mat.loc[a, b] = mat.loc[b, a] = val
    return mat


def plot_matrices(matrices: dict[str, pd.DataFrame], out_path: Path) -> None:
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (title, mat) in zip(axes, matrices.items()):
        im = ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1 if mat.to_numpy(dtype=float).min() < 0 else 0, vmax=1)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.iloc[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sni-version", default="SNI07", choices=["SNI92", "SNI02", "SNI07"])
    parser.add_argument("--topk", type=int, default=10, help="neighbourhood size for overlap metrics")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else project_root / "results" / "source_complementarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading sources at {args.sni_version} 2-digit division granularity …")
    sources = load_all_sources(args.sni_version)
    for name, df in sources.items():
        print(f"  {name:<12} {len(df):>5} division pairs, "
              f"{len(set(df['a']) | set(df['b'])) if not df.empty else 0} divisions covered")

    print("\nComputing correlation matrix (Spearman rho) …")
    corr = build_matrix(sources, correlation_metric)
    print("Computing nearest-neighbour overlap matrix (Jaccard@%d) …" % args.topk)
    nn_overlap = build_matrix(sources, lambda a, b: nn_overlap_metric(a, b, args.topk))
    print("Computing CKA matrix …")
    cka = build_matrix(sources, cka_metric)
    print("Computing mutual information matrix …")
    mi = build_matrix(sources, mutual_information_metric)

    matrices = {
        "A. Correlation (Spearman rho)": corr,
        "B. Nearest-neighbour overlap (Jaccard@k)": nn_overlap,
        "D. Centered Kernel Alignment (CKA)": cka,
        "E. Mutual information": mi,
    }
    for label, mat in matrices.items():
        stem = label.split(".")[0].strip().lower()
        mat.round(4).to_csv(out_dir / f"P01_{stem}_matrix.csv")
        print(f"  → {out_dir / f'P01_{stem}_matrix.csv'}")

    fig_path = out_dir / "P01_source_complementarity.png"
    plot_matrices(matrices, fig_path)

    print(
        "\nReading this report\n" + "=" * 70 + "\n"
        "Low correlation / low neighbourhood overlap / low CKA / low mutual "
        "information between sources is the expected and motivating finding "
        "here, not a data-quality problem: it is the empirical basis for "
        "training on a fused multilayer target (P02/P03) rather than any "
        "single relatedness measure. Consistently high agreement across all "
        "five sources would instead suggest the multi-source integration is "
        "unnecessary."
    )


if __name__ == "__main__":
    main()
