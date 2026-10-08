"""
Estimate, save and plot the density regressions of one regional outcome
(shared by P04_entry.py, P04_exit.py, P04_startup_entry.py and the
binarised-outcome appendix of P04_growth.py / P04_wage_growth.py).

Writes
  {summary_path}  one row per classification period: N, municipalities, base
                  rate, coefficient, SE, p-value, 95% CI, partial r, effect
                  as % of the base rate (columns prefixed with the outcome)
  {year_path}     year-specific slopes with 95% CIs
  {fig_path}      the three-panel result figure (regional_figures)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from latent_industry.regional.fixed_effects import SPECS, estimate_by_period
from latent_industry.regional.regional_figures import plot_density_effect_summary


def report_density_regressions(
    panel: pd.DataFrame,
    outcome: str,
    outcome_label: str,
    effect_label: str,
    color: str,
    fig_path: Path,
    summary_path: Path,
    year_path: Path,
    summary_extra: dict | None = None,
    density_col: str = "density_std",
    plot_density_col: str = "density",
    spec: str = "main",
) -> tuple[dict, pd.DataFrame]:
    """`density_col` is the (standardised) regressor; `plot_density_col` the
    unstandardised density shown in the figure's decile panel."""
    print(f"\nDensity → {outcome}: {SPECS[spec]['label']}, SE clustered by municipality and industry")
    results, year_df = estimate_by_period(panel, outcome, density_col, spec)
    periods_order = list(results)

    rows = []
    for period in periods_order:
        r = results[period]
        rows.append({
            **(summary_extra or {}),
            "period": period,
            "spec": spec,
            "n_obs": r["n"],
            "n_municipalities": r["g_muni"],
            "n_industries": r["g_ind"],
            f"{outcome}_rate": r["rate"],
            f"{outcome}_coef": r["coef"],
            f"{outcome}_se": r["se"],
            f"{outcome}_pval": r["pval"],
            f"{outcome}_ci_low": r["ci_low"],
            f"{outcome}_ci_high": r["ci_high"],
            f"{outcome}_r": r["partial_r"],
            f"{outcome}_pct_of_baseline": r["pct_of_baseline"],
        })
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    year_df.to_csv(year_path, index=False)

    plot_panel = panel if plot_density_col == "density" else \
        panel.drop(columns=["density"], errors="ignore").rename(columns={plot_density_col: "density"})
    plot_density_effect_summary(plot_panel, year_df, results, periods_order, outcome,
                                outcome_label, effect_label, color, fig_path)
    print(f"  → {summary_path}\n  → {year_path}\n  → {fig_path}")
    return results, year_df
