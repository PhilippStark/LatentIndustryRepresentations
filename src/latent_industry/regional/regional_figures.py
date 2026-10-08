"""
Shared paper-figure plotting for the regional density → outcome regressions
(P04_entry.py, P04_exit.py, P04_startup_entry.py, and the binarized-outcome
appendix sections of P04_growth.py / P04_wage_growth.py).

One shared design so a formatting fix only has to happen once:

  plot_density_effect_summary()  — the main 3-panel figure (rate-by-decile,
    effect-over-time, effect-by-period) used once per script for its primary
    binary outcome.

  plot_density_effect_by_tier()  — the small standalone figure for a firm-
    size-stratified outcome (entry/exit only): one point+CI per period per
    size tier.

Both color periods consistently via period_color_map() so a color always
means the same SNI era within one figure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def period_color_map(periods_order: list[str]) -> dict[str, str]:
    """One color per SNI era, off the default matplotlib cycle — shared so
    Panels A and C (and the tier figure) always agree on which color is
    which period within a figure."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {p: cycle[k % len(cycle)] for k, p in enumerate(periods_order)}


def plot_density_effect_summary(
    panel: pd.DataFrame,
    year_df: pd.DataFrame,
    results: dict[str, dict],
    periods_order: list[str],
    outcome: str,
    outcome_label: str,
    effect_label: str,
    color: str,
    fig_path: Path,
) -> Path:
    """
    3-panel figure for a single binary outcome regressed on 1-SD-standardised
    density, industry + muni×year FE, per SNI era.

    results[period] must have: coef, se, ci_low, ci_high, rate, pct_of_baseline
    (all in raw probability units — this function converts to percentage
    points for display). year_df must have columns year, coef, ci_low,
    ci_high for this outcome (already filtered to it, or containing only it).
    """
    period_colors = period_color_map(periods_order)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A. Outcome rate by density decile, computed and shown SEPARATELY per
    # period (pooling deciles across periods would mix density distributions
    # from different classification vintages), plus a dashed line at each
    # period's own overall mean rate for reference.
    ax = axes[0]
    width = 0.7 / len(periods_order)
    for k, period_name in enumerate(periods_order):
        sub = panel[panel["period"] == period_name]
        color_p = period_colors[period_name]
        decile_cat = pd.qcut(sub["density"], q=10, duplicates="drop")
        decile = decile_cat.cat.codes + 1
        dec = sub[outcome].groupby(decile).mean()
        offset = (k - (len(periods_order) - 1) / 2) * width
        ax.bar(dec.index + offset, dec.values, width=width, label=period_name, color=color_p)
        ax.axhline(float(sub[outcome].mean()), color=color_p, linewidth=1.3,
                   linestyle="--", alpha=0.85)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Opportunity-density decile (1 = low, 10 = high)")
    ax.set_ylabel(outcome_label)
    ax.set_title("A. Rate by density")
    ax.legend(fontsize=8, title="period")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # B. Year-by-year coefficients, in percentage points (displayed as "%").
    ax = axes[1]
    # One segment per classification period, so line and band do not bridge
    # the excluded transition years.
    segments = [year_df[year_df["period"] == p] for p in periods_order] \
        if "period" in year_df.columns else [year_df]
    for seg in segments:
        yr = seg["year"] + 0.5
        ax.fill_between(yr, seg["ci_low"] * 100, seg["ci_high"] * 100, alpha=0.2, color=color, lw=0)
        ax.plot(yr, seg["coef"] * 100, "o-", color=color, linewidth=1.6, markersize=4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    for x_line in (2001.5, 2006.5):
        ax.axvline(x_line, color="grey", linewidth=1, linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel(effect_label)
    ax.set_title("B. Effect over time")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    # Limits from the CI band itself (plus zero), with headroom.
    b_y0 = min(0.0, float(year_df["ci_low"].min()) * 100)
    b_y1 = max(0.0, float(year_df["ci_high"].max()) * 100)
    b_pad = 0.08 * (b_y1 - b_y0)
    ax.set_ylim(b_y0 - b_pad, b_y1 + b_pad)

    # C. Per-period coefficient comparison — points with 95% CI, colored by
    # period (matching panel A), annotated with β (pp) and β as a % of that
    # period's base rate. Base rate itself isn't repeated here since panel A
    # already shows it (the dashed reference lines).
    ax = axes[2]
    x = np.arange(len(periods_order))
    point_info = []
    for pi, period_name in enumerate(periods_order):
        res = results[period_name]
        color_p = period_colors[period_name]
        coef_pp, ci_lo_pp, ci_hi_pp = res["coef"] * 100, res["ci_low"] * 100, res["ci_high"] * 100
        yerr = [[coef_pp - ci_lo_pp], [ci_hi_pp - coef_pp]]
        ax.errorbar(x[pi], coef_pp, yerr=yerr, fmt="o", color=color_p, ecolor=color_p,
                    elinewidth=1.4, capsize=4, markersize=7, markeredgecolor="white",
                    markeredgewidth=0.6)
        point_info.append((x[pi], coef_pp, ci_lo_pp, ci_hi_pp, res))
    ax.set_xticks(x)
    ax.set_xticklabels(periods_order)
    # Extra side margin so the leftmost/rightmost point's annotation text
    # (wider than the point itself) doesn't run into the y-axis or the
    # panel's right edge.
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Classification period")
    ax.set_ylabel(effect_label)
    ax.set_title("C. Effect by period")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    ax.relim()
    ax.autoscale_view()
    y0, y1 = ax.get_ylim()
    pad = 0.12 * (y1 - y0)
    # Stagger same-side labels (alternating a larger offset) so points whose
    # effects land close together — e.g. two periods within ~0.1pp of each
    # other — don't have their labels collide horizontally.
    above_rank, below_rank = 0, 0
    for xpos, coef_pp, ci_lo_pp, ci_hi_pp, res in point_info:
        above = coef_pp >= 0
        rank = above_rank if above else below_rank
        if above:
            above_rank += 1
        else:
            below_rank += 1
        offset_pts = 10 + (24 if rank % 2 == 1 else 0)
        label = f"{coef_pp:+.2f} pp\n({res['pct_of_baseline']:+.0f}% of base rate)"
        # Offset in points (not data units) so the label sits a fixed,
        # legible distance from the point regardless of the panel's data
        # range — avoids ballooning the y-axis to make room for text.
        ax.annotate(label, xy=(xpos, ci_hi_pp if above else ci_lo_pp),
                    xytext=(0, offset_pts if above else -offset_pts), textcoords="offset points",
                    ha="center", va="bottom" if above else "top", fontsize=8)
    ax.set_ylim(y0 - 2.4 * pad, y1 + 2.4 * pad)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_density_effect_by_tier(
    tier_results: dict[str, dict[str, dict]],
    periods_order: list[str],
    tiers: list[str],
    xlabel: str,
    effect_label: str,
    title: str,
    fig_path: Path,
) -> Path:
    """
    Small standalone figure: β (percentage points) by size tier, grouped by
    period, points with 95% CI. tier_results[period][tier] must have coef,
    ci_low, ci_high (raw probability units).
    """
    period_colors = period_color_map(periods_order)
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.7 / len(periods_order)
    xt = np.arange(len(tiers))
    for k, period_name in enumerate(periods_order):
        offset = (k - (len(periods_order) - 1) / 2) * width
        color = period_colors[period_name]
        coefs_pp = [tier_results[period_name][t]["coef"] * 100 for t in tiers]
        ci_lo_pp = [tier_results[period_name][t]["ci_low"] * 100 for t in tiers]
        ci_hi_pp = [tier_results[period_name][t]["ci_high"] * 100 for t in tiers]
        yerr = [[c - lo for c, lo in zip(coefs_pp, ci_lo_pp)],
                [hi - c for c, hi in zip(coefs_pp, ci_hi_pp)]]
        ax.errorbar(xt + offset, coefs_pp, yerr=yerr, fmt="o", color=color, ecolor=color,
                    elinewidth=1.4, capsize=4, markersize=7, markeredgecolor="white",
                    markeredgewidth=0.6, label=period_name)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(xt)
    ax.set_xticklabels(tiers)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(effect_label)
    ax.set_title(title)
    ax.legend(fontsize=8, title="period")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig_path
