"""
build_paper_tables.py
=====================
Assembles the paper's Result 1–3 tables and figures in results/paper/ from
outputs other scripts have already written (nothing is recomputed):

1. Source complementarity (Result 1), restricted to the sources with
   non-zero training weight in {run}/training_provenance.json, using
   P01_source_complementarity.py's functions.
2. Reconstruction table (Result 2): baseline vs trained Spearman rho per
   evidence family, from P02's {run}_summary.csv, with a column marking
   whether the family is part of the training objective.
3. Retrieval table (Result 3, sections B/B2), from P03.
4. Copies of the hierarchy-preservation plot and the regional result plots
   into results/paper/ (growth and wage growth into results/paper/appendix/).

Usage
-----
  python scripts/evaluation/build_paper_tables.py --run-name shared_cosine_v16
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_industry.common.paths import EMBEDDING_DIR  # noqa: E402

from P01_source_complementarity import (  # noqa: E402
    ALL_SOURCE_NAMES, load_all_sources, build_matrix, plot_matrices,
    correlation_metric, nn_overlap_metric, cka_metric, mutual_information_metric,
)

DEFAULT_RUN_NAME = "shared_cosine_v16"

eval_dir  = project_root / "results" / "embedding_eval"
paper_dir = project_root / "results" / "paper"


# ── Which P01 sources did this run actually train on? ──────────────────────

def load_family_weights(run_name: str) -> dict:
    path = EMBEDDING_DIR / run_name / "training_provenance.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f).get("family_weights", {})


def active_sources(run_name: str) -> list[str]:
    """
    Sources with a non-zero training weight. Without a training_provenance.json
    (e.g. the baseline) every P01 source is kept.
    """
    weights = load_family_weights(run_name)
    if not weights:
        return list(ALL_SOURCE_NAMES)
    return [s for s in ALL_SOURCE_NAMES if weights.get(s) != 0.0]


# ── LaTeX / Markdown table writers ─────────────────────────────────────────

def _format_cell(v, float_fmt: str, na: str) -> str:
    if isinstance(v, str):
        return v
    return float_fmt.format(v) if pd.notna(v) else na


def write_latex_table(df: pd.DataFrame, out_path: Path, caption: str, label: str,
                       float_fmt: str = "{:.4f}") -> None:
    cols = df.columns.tolist()
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}",
        "\\toprule",
        " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = [_format_cell(row[c], float_fmt, "--") for c in cols]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_table(df: pd.DataFrame, out_path: Path, float_fmt: str = "{:.4f}") -> str:
    cols = df.columns.tolist()
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] + [":---:"] * (len(cols) - 1)) + "|",
    ]
    for _, row in df.iterrows():
        cells = [_format_cell(row[c], float_fmt, "n/a") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    out_path.write_text(md + "\n", encoding="utf-8")
    return md


def _mean_of(row: pd.Series, keys: list[str]) -> float:
    vals = [row[k] for k in keys if k in row.index and pd.notna(row[k])]
    return float(np.mean(vals)) if vals else float("nan")


# ── Rho table row layout (Result 2) ────────────────────────────────────────
# One row per evidence family. Labor/Ownership average over whichever
# SNI07/02/92 rho columns P02 computed (same granularity, different coding
# vintage). NACE's four hierarchy levels (4d/3d/2d/section) are different
# granularities and get one row each. Input-output gets two rows: the
# 2-digit hub-cosine rho (I_io_rho) and the aggregate-up rho over 4-digit
# codes (I_io_aggregate_rho, mean of 4d-4d cosines per division pair).
#
# A family excluded from training (lambda=0.0 in training_provenance.json)
# has no cached rho at all — P02 gates compute_key_metrics on
# is_family_excluded — so its row is dropped from the paper table entirely
# (see the pd.isna(b) and pd.isna(t) filter below) rather than printed as
# "--". These are the current run's OWN family names, in
# training_provenance.json vocabulary, for that filter to check against.
RHO_FAMILIES = [
    ("Labor",                  ["E_labor_rho_07", "E_labor_rho_02", "E_labor_rho_92"]),
    ("Ownership",               ["E2_ownership_rho_07", "E2_ownership_rho_02", "E2_ownership_rho_92"]),
    ("NACE 4-digit",            ["H_nace_rho_4d"]),
    ("NACE 3-digit",            ["H_nace_rho_3d"]),
    ("NACE 2-digit",            ["H_nace_rho_2d"]),
    ("NACE section",            ["H_nace_rho_section"]),
    ("Input-output (2-digit)",  ["I_io_rho"]),
    ("Input-output (4-digit)",  ["I_io_aggregate_rho"]),
    ("Hierarchy",               ["A_treedist_rho"]),
]


# ── Retrieval table row layout (Result 3, §B/B2) ───────────────────────────
RETRIEVAL_METRICS = [("MRR", "MRR"), ("MAP", "MAP"), ("R@1", "R1"), ("R@10", "R10")]
RETRIEVAL_DIRECTIONS = [
    ("SNI02 $\\rightarrow$ SNI07", "B", "02to07"),
    ("SNI92 $\\rightarrow$ SNI02", "B2", "92to02"),
]


def load_summary(run_name: str) -> pd.DataFrame:
    path = eval_dir / run_name / f"{run_name}_summary.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found — run P02_shared_geometry_reconstruction.py "
                          f"--run-name {run_name} first.")
    return pd.read_csv(path).set_index("run")


# Each build_resultN function is wrapped in its own try/except by main(), so
# a failure in one section is reported without skipping the others.

def build_result1(args) -> None:
    print("=" * 70)
    print("1. Source complementarity — restricted to sources used in this run")
    print("=" * 70)

    weights = load_family_weights(args.run_name)
    active = active_sources(args.run_name)
    excluded = [s for s in ALL_SOURCE_NAMES if s not in active]
    print(f"  family_weights: {weights or '(none found — no filtering applied)'}")
    print(f"  Active sources:   {active}")
    print(f"  Excluded sources: {excluded or 'none'}")

    all_sources = load_all_sources(args.sni_version)
    sources = {k: v for k, v in all_sources.items() if k in active}

    corr       = build_matrix(sources, correlation_metric)
    nn_overlap = build_matrix(sources, lambda a, b: nn_overlap_metric(a, b, args.topk))
    cka        = build_matrix(sources, cka_metric)
    mi         = build_matrix(sources, mutual_information_metric)

    matrices = {
        "A. Correlation (Spearman rho)": corr,
        "B. Nearest-neighbour overlap (Jaccard@k)": nn_overlap,
        "D. Centered Kernel Alignment": cka,
        "E. Mutual information": mi,
    }
    for label, mat in matrices.items():
        stem = label.split(".")[0].strip().lower()
        csv_path = paper_dir / f"result1_source_complementarity_{stem}_matrix.csv"
        mat.round(4).to_csv(csv_path)
        print(f"  → {csv_path}")

    fig_path = paper_dir / "result1_source_complementarity.png"
    plot_matrices(matrices, fig_path)


def build_result2(args) -> tuple[pd.Series, pd.Series, str]:
    """Returns (baseline, trained, trained_col) — result3 (retrieval table)
    reuses baseline/trained rather than reloading summary.csv itself."""
    print("\n" + "=" * 70)
    print("2. Rhos table (baseline text-prior vs. trained)")
    print("=" * 70)

    summary = load_summary(args.run_name)
    if args.run_name not in summary.index:
        raise SystemExit(
            f"{args.run_name!r} not found in this summary.csv's own 'run' column "
            f"(found: {list(summary.index)}). If this run was recently renamed/promoted "
            f"(e.g. via a filesystem mv), the CSV's internal 'run' column still holds "
            f"whatever name was passed to P02_shared_geometry_reconstruction.py at the "
            f"time it was written — fix that column's value (not just the filename), "
            f"or rerun P02 under the current --run-name."
        )
    baseline = summary.loc["baseline"]
    trained  = summary.loc[args.run_name]

    # Column header just says "Trained" — the run name goes in the table
    # caption instead, since spelling it out as a header (e.g. "Trained
    # (shared_cosine_v15_textprior_pca_no_nace_trade_innovation)") doesn't
    # fit in an actual paper table.
    trained_col = "Trained"
    rho_rows = []
    _skipped_rho_rows = []
    for family, keys in RHO_FAMILIES:
        b, t = _mean_of(baseline, keys), _mean_of(trained, keys)
        # Restrict the paper table to sources this run actually evaluated —
        # an excluded family (lambda=0.0) has no cached rho at all (P02
        # gates its computation on is_family_excluded), so b/t are both NaN
        # and the row would otherwise print as a bare "--" line.
        if pd.isna(b) and pd.isna(t):
            _skipped_rho_rows.append(family)
            continue
        delta = (t - b) if pd.notna(b) and pd.notna(t) else float("nan")
        rho_rows.append({"Evidence family": family,
                          "Baseline (text-prior)": b, trained_col: t, "Delta": delta})
    rho_df = pd.DataFrame(rho_rows)
    if _skipped_rho_rows:
        print(f"  NOTE: {_skipped_rho_rows} were excluded from training (lambda=0.0) for this "
              f"run and never evaluated by P02 — omitted from the paper table.")

    rho_csv = paper_dir / "result2_rho_table.csv"
    rho_df.to_csv(rho_csv, index=False)
    rho_tex = paper_dir / "result2_rho_table.tex"
    write_latex_table(rho_df, rho_tex,
                       caption="Reconstruction of each evidence family: cosine-similarity vs. "
                               "source-relatedness Spearman $\\rho$, baseline text-prior encoder "
                               f"vs. the trained embedding ({args.run_name.replace('_', '\\_')}).",
                       label="tab:rho-table")
    rho_md = write_markdown_table(rho_df, paper_dir / "result2_rho_table.md")
    print(rho_md)
    print(f"  → {rho_csv}\n  → {rho_tex}\n  → {rho_tex.with_suffix('.md')}")
    return baseline, trained, trained_col


def build_result3(args, baseline: pd.Series, trained: pd.Series, trained_col: str) -> None:
    print("\n" + "=" * 70)
    print("3. Retrieval table (Result 3, §B/B2)")
    print("=" * 70)

    retrieval_rows = []
    for dir_label, prefix, suffix in RETRIEVAL_DIRECTIONS:
        for metric_label, metric_key in RETRIEVAL_METRICS:
            key = f"{prefix}_{metric_key}_{suffix}"
            if key not in baseline.index:
                continue
            b, t = baseline[key], trained[key]
            retrieval_rows.append({"Direction": dir_label, "Metric": metric_label,
                                    "Baseline (text-prior)": b, trained_col: t, "Delta": t - b})
    retrieval_df = pd.DataFrame(retrieval_rows)

    retrieval_csv = paper_dir / "result3_retrieval_table.csv"
    retrieval_df.to_csv(retrieval_csv, index=False)
    retrieval_tex = paper_dir / "result3_retrieval_table.tex"
    write_latex_table(retrieval_df, retrieval_tex,
                       caption="Cross-version retrieval (Result 3): baseline text-prior encoder "
                               f"vs. the trained embedding ({args.run_name.replace('_', '\\_')}).",
                       label="tab:retrieval-table")
    retrieval_md = write_markdown_table(retrieval_df, paper_dir / "result3_retrieval_table.md")
    print(retrieval_md)
    print(f"  → {retrieval_csv}\n  → {retrieval_tex}\n  → {retrieval_tex.with_suffix('.md')}")


def build_result4(args) -> None:
    """
    Copy the already-built paper plots into results/paper/ under fixed
    result{N}_ names: P03's hierarchy-preservation plot and the regional entry,
    exit and start-up entry plots (main text). Growth and wage-growth plots go
    to results/paper/appendix/.
    """
    print("\n" + "=" * 70)
    print("4. Copying remaining paper-ready plots")
    print("=" * 70)

    run = args.run_name
    missing: list[Path] = []
    to_copy = [
        (eval_dir / run / f"{run}_A_hierarchy_preservation.png", "result3_hierarchy_preservation.png"),
        # result3_opportunity_space.png is written directly by
        # scripts/paper/plot_opportunity_space.py.
        (project_root / "results" / "regional_entry" / f"{run}_entry.png", "result4_entry.png"),
        (project_root / "results" / "regional_exit" / f"{run}_exit.png", "result4_exit.png"),
        (project_root / "results" / "regional_startup_entry" / f"{run}_startup_entry.png", "result4_startup.png"),
    ]
    for src, dst_name in to_copy:
        if src.exists():
            dst = paper_dir / dst_name
            shutil.copy2(src, dst)
            print(f"  → {dst}  (from {src})")
        else:
            print(f"  NOT FOUND (skipped): {src}")
            missing.append(src)

    # Appendix material: the binarised growth / wage-growth outcomes
    # (growth > 0, wage_growth > 0 vs. density) from P04_growth.py and
    # P04_wage_growth.py.
    appendix_dir = paper_dir / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    to_copy_appendix = [
        (project_root / "results" / "regional_growth" / f"{run}_growth_binary.png", "appendix_growth_binary.png"),
        (project_root / "results" / "regional_wage_growth" / f"{run}_wage_growth_binary.png", "appendix_wage_growth_binary.png"),
    ]
    for src, dst_name in to_copy_appendix:
        if src.exists():
            dst = appendix_dir / dst_name
            shutil.copy2(src, dst)
            print(f"  → {dst}  (from {src})")
        else:
            print(f"  NOT FOUND (skipped): {src}")
            missing.append(src)

    print(f"\nAll paper outputs written to {paper_dir}/")
    if missing:
        raise SystemExit(f"{len(missing)} source file(s) not found (listed above with 'NOT FOUND') — "
                          f"the corresponding results/paper/ output was left untouched, not "
                          f"regenerated from a stale copy.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--sni-version", default="SNI07", choices=["SNI92", "SNI02", "SNI07"],
                         help="granularity for the source-complementarity matrices (P01 default)")
    parser.add_argument("--topk", type=int, default=10,
                         help="neighbourhood size for the NN-overlap metric (P01 default)")
    args = parser.parse_args()

    paper_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    baseline = trained = trained_col = None

    try:
        build_result1(args)
    except Exception as e:
        print(f"\n*** RESULT 1 FAILED: {e} ***")
        failures.append("result1")

    try:
        baseline, trained, trained_col = build_result2(args)
    except Exception as e:
        print(f"\n*** RESULT 2 FAILED: {e} ***")
        failures.append("result2")

    if baseline is not None:
        try:
            build_result3(args, baseline, trained, trained_col)
        except Exception as e:
            print(f"\n*** RESULT 3 FAILED: {e} ***")
            failures.append("result3")
    else:
        print("\n*** RESULT 3 SKIPPED — depends on result2's baseline/trained rows, "
              "which failed to load. ***")
        failures.append("result3 (skipped, depends on result2)")

    try:
        build_result4(args)
    except Exception as e:
        print(f"\n*** RESULT 4 FAILED: {e} ***")
        failures.append("result4")

    print("\n" + "=" * 70)
    if failures:
        print(f"DONE WITH FAILURES: {failures} — see the *** markers above. "
              f"results/paper/ may still contain STALE output for the failed section(s).")
        raise SystemExit(1)
    print(f"All 4 result sections succeeded — results/paper/ fully reflects "
          f"--run-name {args.run_name}.")


if __name__ == "__main__":
    main()
