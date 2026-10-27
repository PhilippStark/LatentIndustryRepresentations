"""
plot_multiple_regression.py
===========================
Paper figure and SI table for the multiple regressions of the regional
outcomes on opportunity density and one relational-view density at a time
(estimated by scripts/regional/R01_single_source_density.py).

Each model, per outcome and classification period:

  y_jmt = β_O D^O_jmt + β_V D^V_jmt + α_mt + γ_j + ε_jmt

with opportunity density D^O, the density D^V of one relational view (or the
text initialization), municipality×year and industry fixed effects, and
standard errors clustered by municipality and industry. Both densities are
standardised within the period.

Inputs   <LATENT_DATA_ROOT>/robustness/single_source_density/multiple_regression{suffix}.csv
Outputs  results/paper/result4_multiple_regression{suffix}.{png,pdf}
             main-text figure: industry entry, industry exit, startup formation
             (16-inch canvas like Figs. 5–7; place at full page width)
         results/paper/appendix/multiple_regression_growth{suffix}.{png,pdf}
             SI figure: positive employment growth, positive wage growth
         results/paper/appendix/multiple_regression{suffix}.tex
         results/paper/appendix/multiple_regression_growth{suffix}.tex
             SI tables, all views, outcomes and periods

Usage
-----
  python scripts/paper/plot_multiple_regression.py
  python scripts/paper/plot_multiple_regression.py --spec additive
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import LUNARC_DATA_ROOT, RESULTS_DIR

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--spec", default="main", choices=["main", "additive", "muni_ind"])
parser.add_argument("--dpi", type=int, default=600)
args = parser.parse_args()
suffix = "" if args.spec == "main" else f"_{args.spec}"

VIEWS = {   # view key in the CSV → label (manuscript terminology), top to bottom
    "labor": "Labour mobility",
    "ownership": "Common ownership",
    "io": "Input–output relatedness",
    "hierarchy": "Industry hierarchy",
    "baseline": "Text initialization",
}
VIEW_LABELS_2L = {   # two-line axis labels leave more width for the panels
    "labor": "Labour\nmobility", "ownership": "Common\nownership", "io": "Input–output\nrelatedness",
    "hierarchy": "Industry\nhierarchy", "baseline": "Text\ninitialization",
}
MAIN_OUTCOMES = {"entry": "Industry entry", "exit": "Industry exit", "startup_entry": "Startup formation"}
SI_OUTCOMES = {"growth": "Positive employment growth", "wage_growth": "Positive wage growth"}
PERIODS = {"SNI92": "#1f77b4", "SNI02": "#ff7f0e", "SNI07": "#2ca02c"}   # as in Figs. 5–7
SPEC_TEXT = {
    "main": r"municipality$\times$year and industry fixed effects",
    "additive": "municipality, year and industry fixed effects",
    "muni_ind": r"municipality$\times$industry and year fixed effects",
}

df = pd.read_csv(LUNARC_DATA_ROOT / "robustness" / "single_source_density" / f"multiple_regression{suffix}.csv")
for col in [c for c in df.columns if c.endswith(("_coef", "_ci_low", "_ci_high"))]:
    df[col] = df[col] * 100   # percentage points


# ═══════════════════════════════════════════════════════════════════════════════
# Figure: one full-width panel per outcome (true size, PNAS page width 17.8 cm)
# ═══════════════════════════════════════════════════════════════════════════════

# Drawn like Figs. 5–7 (regional_figures.py): a 16-inch-wide canvas scaled to the page
# width in the manuscript (17.8 cm, factor ≈ 0.44). Font sizes on the canvas are chosen
# so that all text prints at ≥ 6 pt (PNAS minimum): 14 pt on the canvas ≈ 6.1 pt printed.
CANVAS_W = 16.0
FONT = 14               # tick labels, view labels, annotations, axis label, legend
FONT_TITLE = 16         # panel titles
PERIOD_OFFSET = {"SNI92": -0.29, "SNI02": 0.0, "SNI07": 0.29}
DENSITY_OFFSET = {"opportunity": -0.07, "view": 0.07}
FIG_H = 7.0             # inches on the 16-inch canvas (≈ 7.8 cm at page width)


def plot_outcomes(outcomes: dict[str, str], out: Path) -> None:
    present = [o for o in outcomes if o in set(df["outcome"])]
    if not present:
        print(f"  (no estimates for {list(outcomes)} yet — {out.name} skipped)")
        return
    view_keys = list(VIEWS)
    fig, axes = plt.subplots(1, len(present), figsize=(CANVAS_W * len(present) / 3, FIG_H),
                             sharey=True, squeeze=False)
    for ax, outcome in zip(axes[0], present):
        sub = df[df["outcome"] == outcome]
        for y, view in enumerate(view_keys):
            if y % 2 == 0:
                ax.axhspan(y - 0.45, y + 0.45, color="#f2f2f2", zorder=0, lw=0)
            for period, color in PERIODS.items():
                row = sub[(sub["view"] == view) & (sub["period"] == period)]
                if row.empty:   # e.g. no input–output source exists for SNI92
                    ax.text(0.02, y + PERIOD_OFFSET[period], f"no {period} data",
                            transform=ax.get_yaxis_transform(), ha="left", va="center",
                            fontsize=FONT, color=color, style="italic")
                    continue
                row = row.iloc[0]
                for name, marker in [("opportunity", "o"), ("view", "s")]:
                    c = row[f"{name}_coef"]
                    ax.errorbar(c, y + PERIOD_OFFSET[period] + DENSITY_OFFSET[name],
                                xerr=[[c - row[f"{name}_ci_low"]], [row[f"{name}_ci_high"] - c]],
                                fmt=marker, color=color, ms=6 if marker == "o" else 5.5, lw=1.4,
                                capsize=2.5, capthick=1.4, mew=1.4,
                                mfc=color if name == "opportunity" else "white", zorder=3)
        ax.axvline(0, color="black", lw=0.8, zorder=1)
        ax.set_title(outcomes[outcome], fontsize=FONT_TITLE)
        ax.tick_params(axis="both", labelsize=FONT)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(view_keys) - 0.5, -0.5)
    axes[0, 0].set_yticks(range(len(view_keys)))
    axes[0, 0].set_yticklabels([VIEW_LABELS_2L[v] for v in view_keys])
    fig.supxlabel("Coefficient per one-standard-deviation increase in density "
                  "(percentage points), 95% CI", fontsize=FONT, y=0.088)
    handles = [Line2D([], [], color=c, marker="o", ls="", ms=6, label=p) for p, c in PERIODS.items()]
    handles += [
        Line2D([], [], color="#555555", marker="o", ls="", ms=6, label="Opportunity density"),
        Line2D([], [], color="#555555", marker="s", mfc="white", ls="", ms=5.5, mew=1.4,
               label="Relational-view density"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=FONT,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0.72 / fig.get_figheight(), 1, 1), w_pad=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  → {out}.png / .pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# SI tables: every view, outcome and period
# ═══════════════════════════════════════════════════════════════════════════════

def cell(row: pd.Series, name: str) -> str:
    return f"${row[f'{name}_coef']:.2f}$ [{row[f'{name}_ci_low']:.2f}, {row[f'{name}_ci_high']:.2f}]"


def write_table(outcomes: dict[str, str], out: Path, label: str, what: str) -> None:
    present = [o for o in outcomes if o in set(df["outcome"])]
    if not present:
        print(f"  (no estimates for {list(outcomes)} yet — {out.name} skipped)")
        return
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Multiple regressions of " + what + r" on opportunity density and one "
        r"relational-view density. Each row is one linear probability model, estimated per "
        r"classification period, with both densities (each standardised within the period) and "
        + SPEC_TEXT[args.spec] + r"; standard errors are clustered by municipality and industry. "
        r"Coefficients in percentage points per standard deviation, 95\% confidence intervals in "
        r"brackets. Labour mobility and common ownership enter as the row-normalised relations used "
        r"in training; input--output relatedness is not available for SNI92.}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lllcc}",
        r"\toprule",
        r"Outcome & Period & Relational view & Opportunity density & Relational-view density \\",
        r"\midrule",
    ]
    for k, outcome in enumerate(present):
        first_outcome = True
        for period in PERIODS:
            block = df[(df["outcome"] == outcome) & (df["period"] == period)].set_index("view")
            views = [v for v in VIEWS if v in block.index]
            for j, view in enumerate(views):
                o = outcomes[outcome] if first_outcome else ""
                p = period if j == 0 else ""
                lines.append(f"{o} & {p} & {VIEWS[view].replace('–', '--')} & "
                             f"{cell(block.loc[view], 'opportunity')} & {cell(block.loc[view], 'view')} \\\\")
                first_outcome = False
            if period != list(PERIODS)[-1]:
                lines.append(r"\addlinespace[2pt]")
        if k < len(present) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"  → {out}")


paper, appendix = RESULTS_DIR / "paper", RESULTS_DIR / "paper" / "appendix"
tag = suffix.replace("_", "-")
plot_outcomes(MAIN_OUTCOMES, paper / f"result4_multiple_regression{suffix}")
plot_outcomes(SI_OUTCOMES, appendix / f"multiple_regression_growth{suffix}")
write_table(MAIN_OUTCOMES, appendix / f"multiple_regression{suffix}.tex",
            f"tab:multiple_regression{tag}", "industry entry, industry exit and startup formation")
write_table(SI_OUTCOMES, appendix / f"multiple_regression_growth{suffix}.tex",
            f"tab:multiple_regression_growth{tag}", "positive employment and wage growth")
