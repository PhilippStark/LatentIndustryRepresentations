"""
make_regional_results_table.py
==============================
SI table of the main regional results: for industry entry, industry exit and
startup formation, per classification period, the number of observations,
the baseline rate and the effect of a one-standard-deviation increase in
opportunity density (municipality×year and industry fixed effects, standard
errors clustered by municipality and industry), with its 95% confidence
interval and as a percentage of the baseline rate.

Inputs   results/regional_{entry,exit,startup_entry}/{run}_summary.csv (P04 scripts)
Output   results/paper/appendix/regional_results.tex

Usage
-----
  python scripts/paper/make_regional_results_table.py --run-name shared_cosine_v16
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import RESULTS_DIR

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--run-name", default="shared_cosine_v16")
args = parser.parse_args()

OUTCOMES = [("Industry entry", "regional_entry", "entry_industry"),
            ("Industry exit", "regional_exit", "exit_industry"),
            ("Startup formation", "regional_startup_entry", "startup_entry")]

lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\small",
    r"\caption{Opportunity density and regional outcomes. Linear probability models per "
    r"classification period with municipality$\times$year and industry fixed effects; standard "
    r"errors clustered by municipality and industry, confidence intervals and $p$-values from a "
    r"$t$ distribution with $\min(G_{\text{municipality}}, G_{\text{industry}})-1$ degrees of "
    r"freedom. Coefficients in percentage points per standard deviation of opportunity density "
    r"(pooled over periods); all $p<0.001$. Baseline: share of candidate observations with the outcome.}",
    r"\label{tab:regional_results}",
    r"\begin{tabular}{llrrrrr}",
    r"\toprule",
    r"Outcome & Period & $N$ & Baseline (\%) & Coefficient (pp) & 95\% CI & \% of baseline \\",
    r"\midrule",
]
for k, (title, folder, col) in enumerate(OUTCOMES):
    s = pd.read_csv(RESULTS_DIR / folder / f"{args.run_name}_summary.csv")
    for j, (_, r) in enumerate(s.iterrows()):
        first = rf"\multirow{{{len(s)}}}{{*}}{{{title}}}" if j == 0 else ""
        n = f"{int(r['n_obs']):,}".replace(",", r"\,")
        lines.append(
            f"{first} & {r['period']} & {n} & {r[f'{col}_rate'] * 100:.2f} & "
            f"${r[f'{col}_coef'] * 100:.2f}$ & "
            f"[{r[f'{col}_ci_low'] * 100:.2f}, {r[f'{col}_ci_high'] * 100:.2f}] & "
            f"{r[f'{col}_pct_of_baseline']:.0f} \\\\"
        )
    if k < len(OUTCOMES) - 1:
        lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
out = RESULTS_DIR / "paper" / "appendix" / "regional_results.tex"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print(f"  → {out}")
