"""
build_density_robustness_table.py
=================================
Writes results/paper/density_robustness.tex: the entry, exit and startup-formation
density effects under the four density specifications produced by
P04_entry.py / P04_exit.py / P04_startup_entry.py:

  D^A        --density-var ew  --no-log-employment-weights   (file suffix _ew_raw)
  D^{A,log}  --density-var ew  --log-employment-weights      (no suffix; main specification)
  D^G        --density-var geo --no-log-employment-weights   (_geo_raw)
  D^{G,log}  --density-var geo --log-employment-weights      (_geo_log)

Reads the scripts' {run}{suffix}_summary.csv files; runs no regressions.

Usage
-----
  python scripts/evaluation/build_density_robustness_table.py --run-name shared_cosine_v16
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", required=True)
args = parser.parse_args()
run = args.run_name

# (variant label, OUT_NAME suffix, is the main/bolded spec)
VARIANTS = [
    ("$D^{\\mathrm{A}}$", "_ew_raw", False),
    ("$D^{\\mathrm{A,log}}$ (main)", "", True),
    ("$D^{\\mathrm{G}}$", "_geo_raw", False),
    ("$D^{\\mathrm{G,log}}$", "_geo_log", False),
]
OUTCOMES = [("Entry", "regional_entry", "entry_industry"),
            ("Exit", "regional_exit", "exit_industry"),
            ("Startup formation", "regional_startup_entry", "startup_entry")]
PERIODS = ["SNI92", "SNI02", "SNI07"]


def stars(pval: float) -> str:
    if pval < 0.01:
        return "^{***}"
    if pval < 0.05:
        return "^{**}"
    if pval < 0.1:
        return "^{*}"
    return ""


rows_by_outcome: dict[str, dict[str, dict]] = {}
missing: list[Path] = []

for outcome_label, subdir, col in OUTCOMES:
    rows_by_outcome[outcome_label] = {}
    for _, suffix, _ in VARIANTS:
        path = project_root / "results" / subdir / f"{run}{suffix}_summary.csv"
        if not path.exists():
            missing.append(path)
            continue
        summary = pd.read_csv(path)
        for _, row in summary.iterrows():
            key = (row["period"], suffix)
            rows_by_outcome[outcome_label][key] = {
                "coef": row[f"{col}_coef"], "se": row[f"{col}_se"], "pval": row[f"{col}_pval"],
            }

if missing:
    print("NOT FOUND (table will show '--' for these cells):")
    for p in missing:
        print(f"  {p}")

lines = [
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Robustness of regional density effects to the aggregation rule and",
    "employment-weighting scheme. Each cell reports the coefficient (in",
    "percentage points) on standardized regional opportunity density from a",
    "separate linear probability model of the row's outcome with",
    "municipality$\\times$year and industry fixed effects, estimated per",
    "classification period; standard errors clustered by municipality and",
    "industry in parentheses. $D^{\\mathrm{A,log}}$ (main specification, used",
    "in the body of the paper) is the log-employment-weighted arithmetic mean;",
    "$D^{\\mathrm{A}}$, $D^{\\mathrm{G}}$, and $D^{\\mathrm{G,log}}$ are robustness",
    "specifications. $^{*}p<0.1$, $^{**}p<0.05$, $^{***}p<0.01$.}",
    "\\label{tab:density_robustness}",
    "\\begin{tabular}{llcccc}",
    "\\toprule",
    "Outcome & Period & $D^{\\mathrm{A}}$ & $D^{\\mathrm{A,log}}$ (main) & $D^{\\mathrm{G}}$ & $D^{\\mathrm{G,log}}$ \\\\",
    "\\midrule",
]

for i, (outcome_label, _, _) in enumerate(OUTCOMES):
    lines.append(f"\\multirow{{3}}{{*}}{{{outcome_label}}}")
    for period in PERIODS:
        cells = []
        for _, suffix, is_main in VARIANTS:
            cell_data = rows_by_outcome[outcome_label].get((period, suffix))
            if cell_data is None:
                cells.append("--")
                continue
            coef_pp = cell_data["coef"] * 100
            se_pp = cell_data["se"] * 100
            star_str = stars(cell_data["pval"])
            value = f"{coef_pp:.2f}{star_str}"
            if is_main:
                value = f"\\mathbf{{{value}}}"
            cells.append(f"${value}$ ({se_pp:.2f})")
        lines.append(f" & {period} & " + " & ".join(cells) + " \\\\")
    if i < len(OUTCOMES) - 1:
        lines.append("\\addlinespace")

lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

out_path = project_root / "results" / "paper" / "density_robustness.tex"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n→ {out_path}")

if missing:
    sys.exit(1)
