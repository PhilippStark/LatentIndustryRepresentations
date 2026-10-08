"""
R02_fixed_effects_robustness.py
===============================
Robustness of the regional density effects (entry, exit, start-up entry) to
the fixed-effects specification. The main specification is compared with
two alternatives (latent_industry.regional.fixed_effects):

  main      municipality×year + industry FE (main specification)
  additive  municipality + year + industry FE
  muni_ind  municipality×industry + year FE

All three absorb the fixed effects exactly and cluster standard errors
two-way by municipality and industry. Each is estimated per classification
period (one density slope), and with year-specific density slopes
(density × base-year indicators) inside the same model. Year-specific slopes
are reported for the main and additive specifications. Under
municipality×industry fixed effects they are not comparable to the period
slope (only the common level is identified from changes within
municipality–industry cells; differences between years also use
cross-sectional variation), so this specification is shown by its period
estimates.

Runs on the cached panels only (no microdata needed).

Inputs  (LATENT_DATA_ROOT, default LUNARC; set it to the repository's data/)
  processed/downstream_outcomes/{entry,exit,startup_entry}/{run}{suffix}_panel.parquet
Outputs
  results/regional_fe_robustness/{run}{suffix}_fe_robustness_period.csv
  results/regional_fe_robustness/{run}{suffix}_fe_robustness_year.csv
  results/paper/appendix/fe_robustness{suffix}.tex
  results/paper/appendix/fe_robustness_annual{suffix}.png

Usage
-----
  python scripts/regional/R02_fixed_effects_robustness.py
  python scripts/regional/R02_fixed_effects_robustness.py --panel-suffix _geo_log
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import PROCESSED_DATA_DIR, RESULTS_DIR
from latent_industry.regional.fixed_effects import (
    SPECS, annual_density_effects, density_effect, standardise_density,
)

OUTCOMES = [
    # (panel folder, outcome column, label)
    ("entry",         "entry_industry", "Entry"),
    ("exit",          "exit_industry",  "Exit"),
    ("startup_entry", "startup_entry",  "Start-up entry"),
]
SPEC_ORDER = ["main", "additive", "muni_ind"]
SPEC_STYLE = {   # colour, legend label, x-offset of the annual markers
    "main":     ("#1f4e79", r"Municipality$\times$year + industry FE (main)", 0.0),
    "additive": ("#e07a1f", "Municipality + year + industry FE", 0.3),
    "muni_ind": ("#8c8c8c", r"Municipality$\times$industry + year FE", 0.44),
}
TABLE_LABELS = {
    "main": r"Muni$\times$year + industry",
    "additive": "Muni + year + industry",
    "muni_ind": r"Muni$\times$industry + year",
}

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--run-name", default="shared_cosine_v16")
parser.add_argument("--panel-suffix", default="",
                    help="Density variant of the panel (_ew_raw, _geo_log, _geo_raw); "
                         "empty = main density D^{A,log}.")
args = parser.parse_args()

out_dir = RESULTS_DIR / "regional_fe_robustness"
appendix_dir = RESULTS_DIR / "paper" / "appendix"
out_dir.mkdir(parents=True, exist_ok=True)
appendix_dir.mkdir(parents=True, exist_ok=True)
tag = f"{args.run_name}{args.panel_suffix}"


# ═══════════════════════════════════════════════════════════════════════════════
# Estimation
# ═══════════════════════════════════════════════════════════════════════════════

period_rows: list[dict] = []
year_frames: list[pd.DataFrame] = []
for folder, outcome, label in OUTCOMES:
    path = PROCESSED_DATA_DIR / "downstream_outcomes" / folder / f"{tag}_panel.parquet"
    print(f"\n{'=' * 70}\n{label}: {path}")
    panel = pd.read_parquet(path)
    standardise_density(panel)
    for period in dict.fromkeys(panel["period"]):
        sub = panel[panel["period"] == period]
        for spec in SPEC_ORDER:
            r = density_effect(sub, outcome, spec=spec)
            period_rows.append({"outcome": label, "period": period, "spec": spec,
                                **{k: r[k] for k in ("coef", "se", "t", "df", "pval", "ci_low", "ci_high",
                                                     "n", "g_muni", "g_ind", "rate", "pct_of_baseline")}})
            if spec != "muni_ind":
                yr = annual_density_effects(sub, outcome, spec=spec)
                yr.insert(0, "spec", spec)
                yr.insert(0, "period", period)
                yr.insert(0, "outcome", label)
                year_frames.append(yr)
            print(f"  {period} {spec:<9} β={r['coef'] * 100:+7.3f} pp  "
                  f"95% CI [{r['ci_low'] * 100:+.3f}, {r['ci_high'] * 100:+.3f}]  "
                  f"p={r['pval']:.1e}  N={r['n']:,}")

period_df = pd.DataFrame(period_rows)
year_df = pd.concat(year_frames, ignore_index=True)
period_df.to_csv(out_dir / f"{tag}_fe_robustness_period.csv", index=False)
year_df.to_csv(out_dir / f"{tag}_fe_robustness_year.csv", index=False)
print(f"\n  → {out_dir / f'{tag}_fe_robustness_period.csv'}\n  → {out_dir / f'{tag}_fe_robustness_year.csv'}")


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX table: coefficients (percentage points) with 95% CI
# ═══════════════════════════════════════════════════════════════════════════════

def cell(r: pd.Series, bold: bool) -> str:
    coef = f"{r['coef'] * 100:.2f}"
    coef = rf"\mathbf{{{coef}}}" if bold else coef
    return rf"${coef}$ [{r['ci_low'] * 100:.2f}, {r['ci_high'] * 100:.2f}]"


lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\small",
    r"\caption{Regional density effects under alternative fixed-effects "
    r"specifications. Each cell reports the coefficient (percentage points per "
    r"standard deviation of density $D^{\mathrm{A,log}}$) from a linear "
    r"probability model estimated per classification period, with 95\% "
    r"confidence intervals in brackets. Fixed effects are absorbed exactly; "
    r"standard errors are clustered by municipality and industry, and intervals "
    r"use a $t$ distribution with $\min(G_{\text{municipality}}, "
    r"G_{\text{industry}})-1$ degrees of freedom. Column 1 is the main "
    r"specification.}",
    r"\label{tab:fe_robustness}",
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r" & & (1) & (2) & (3) \\",
    "Outcome & Period & " + " & ".join(TABLE_LABELS[s] for s in SPEC_ORDER) + r" \\",
    r"\midrule",
]
for i, (_, _, label) in enumerate(OUTCOMES):
    block = period_df[period_df["outcome"] == label]
    periods = list(dict.fromkeys(block["period"]))
    for j, period in enumerate(periods):
        row = block[block["period"] == period].set_index("spec")
        first = rf"\multirow{{{len(periods)}}}{{*}}{{{label}}}" if j == 0 else ""
        cells = " & ".join(cell(row.loc[s], bold=(s == "main")) for s in SPEC_ORDER)
        lines.append(f"{first} & {period} & {cells} \\\\")
    if i < len(OUTCOMES) - 1:
        lines.append(r"\addlinespace")
n_line = " & ".join(
    f"{int(period_df[period_df['spec'] == s]['n'].sum()):,}".replace(",", r"\,") for s in SPEC_ORDER)
lines += [
    r"\midrule",
    rf"\multicolumn{{2}}{{l}}{{Observations (all models)}} & {n_line} \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
tex_path = appendix_dir / f"fe_robustness{args.panel_suffix}.tex"
tex_path.write_text("\n".join(lines) + "\n")
print(f"  → {tex_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure: year-specific effects with 95% CI, all specifications
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(16, 4.6), sharex=True)
for ax, (_, _, label) in zip(axes, OUTCOMES):
    block = year_df[year_df["outcome"] == label]
    # Municipality×industry: period estimate and 95% CI drawn across the period's years.
    color, name, _ = SPEC_STYLE["muni_ind"]
    pblock = period_df[(period_df["outcome"] == label) & (period_df["spec"] == "muni_ind")]
    for k, (_, r) in enumerate(pblock.iterrows()):
        years = block.loc[block["period"] == r["period"], "year"]
        x0, x1 = years.min() - 0.3, years.max() + 0.7
        ax.fill_between([x0, x1], r["ci_low"] * 100, r["ci_high"] * 100, color=color, alpha=0.18, lw=0,
                        label=f"{name} (period estimate)" if k == 0 else None)
        ax.plot([x0, x1], [r["coef"] * 100] * 2, color=color, lw=1.4)
    for spec in ["main", "additive"]:
        color, name, offset = SPEC_STYLE[spec]
        b = block[block["spec"] == spec].sort_values("year")
        ax.errorbar(b["year"] + offset, b["coef"] * 100,
                    yerr=[(b["coef"] - b["ci_low"]) * 100, (b["ci_high"] - b["coef"]) * 100],
                    fmt="o", ms=3.2, lw=0.9, capsize=0, color=color, label=name, zorder=3)
    ax.axhline(0, color="black", lw=0.6)
    for boundary in (2001, 2006):   # SNI92 | SNI02 | SNI07 (transition years excluded)
        ax.axvline(boundary, color="#888888", lw=0.6, ls=":")
    ax.set_title(label, fontsize=12)
    ax.set_xlabel("Base year $t$")
    ax.grid(axis="y", lw=0.3, alpha=0.6)
axes[0].set_ylabel("Effect per 1-SD density (pp), 95% CI")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig_path = appendix_dir / f"fe_robustness_annual{args.panel_suffix}.png"
fig.savefig(fig_path, dpi=300)
print(f"  → {fig_path}")
print("\nDone.")
