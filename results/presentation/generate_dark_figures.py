"""
generate_dark_figures.py
========================
Dark-background versions of six paper figures for presentations
(result1_source_complementarity, result3_opportunity_space,
result4_single_source_density_comparison, result4_entry, result4_exit,
result4_startup). Replots from cached outputs; only the t-SNE of the
shared_cosine_v16 embeddings is refit (seconds).

Inputs
------
  results/paper/result1_source_complementarity_{a,b,d,e}_matrix.csv
  EMBEDDING_DIR/shared_cosine_v16/embeddings_sni{07,02,92}.npz
  results/regional_{entry,exit,startup_entry}/shared_cosine_v16_{summary,year_coefs}.csv
  PROCESSED_DATA_DIR/downstream_outcomes/{entry,exit,startup_entry}/shared_cosine_v16_panel.parquet
  single_source_density_comparison.csv (R01_single_source_density.py)

Usage
-----
  python results/presentation/generate_dark_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import (  # noqa: E402
    EMBEDDING_DIR, METADATA_DIR, PROCESSED_DATA_DIR, LUNARC_DATA_ROOT,
)
from latent_industry.common.sni_codes import SNI_2digits_to_sections  # noqa: E402
from latent_industry.regional.density import load_sni_embeddings  # noqa: E402

RUN_NAME = "shared_cosine_v16"
SEED = 42

paper_dir = project_root / "results" / "paper"
out_dir = project_root / "results" / "presentation"
out_dir.mkdir(parents=True, exist_ok=True)

LUNARC_SINGLE_SOURCE_DIR = LUNARC_DATA_ROOT / "robustness" / "single_source_density"

# ═══════════════════════════════════════════════════════════════════════════
# Dark theme
# ═══════════════════════════════════════════════════════════════════════════

FG = "#e8e8e8"          # primary foreground: text, zero-lines, axis spines
MUTED = "#9a9a9a"        # secondary foreground: reference/divider lines
GRID = "#4d4d4d"         # gridlines
LEGEND_BG = "#141414"    # legend box background (slightly lifted off pure black)

plt.rcParams.update({
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": FG,
    "axes.titlecolor": FG,
    "text.color": FG,
    "xtick.color": "#c0c0c0",
    "ytick.color": "#c0c0c0",
    "grid.color": GRID,
    "legend.facecolor": LEGEND_BG,
    "legend.edgecolor": MUTED,
    "legend.labelcolor": FG,
})


def savefig_dark(fig, path: Path, **kwargs) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="black", **kwargs)
    print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Result 1 — source complementarity matrices
# ═══════════════════════════════════════════════════════════════════════════

def build_result1() -> None:
    print("=" * 70)
    print("Result 1 — source complementarity (dark)")
    print("=" * 70)

    stems = [
        ("a", "A. Correlation (Spearman rho)"),
        ("b", "B. Nearest-neighbour overlap (Jaccard@k)"),
        ("d", "D. Centered Kernel Alignment"),
        ("e", "E. Mutual information"),
    ]
    matrices = {}
    for stem, label in stems:
        csv_path = paper_dir / f"result1_source_complementarity_{stem}_matrix.csv"
        if not csv_path.exists():
            print(f"  SKIPPED — {csv_path} not found.")
            continue
        matrices[label] = pd.read_csv(csv_path, index_col=0)

    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (title, mat) in zip(axes, matrices.items()):
        arr = mat.to_numpy(dtype=float)
        vmin = -1 if arr.min() < 0 else 0
        im = ax.imshow(arr, cmap="RdBu_r", vmin=vmin, vmax=1)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.iloc[i, j]
                if np.isfinite(v):
                    # Cell text sits on the RdBu_r colormap, not the black
                    # figure background — kept black (as in the light-mode
                    # original) rather than switched to FG, since the
                    # colormap's own cell colors (not the page) are what it
                    # needs to contrast against.
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_edgecolor(MUTED)
        cbar.ax.tick_params(colors="#c0c0c0")
    fig.tight_layout()
    savefig_dark(fig, out_dir / "result1_source_complementarity.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Result 3 — opportunity space (t-SNE by section)
# ═══════════════════════════════════════════════════════════════════════════

def build_result3() -> None:
    print("\n" + "=" * 70)
    print("Result 3 — opportunity space t-SNE (dark)")
    print("=" * 70)

    MAX_PER_VERSION = 600
    rng_np = np.random.default_rng(SEED)

    def subsample(codes, vecs, n):
        if len(codes) <= n:
            return codes, vecs
        idx = rng_np.choice(len(codes), n, replace=False)
        return [codes[i] for i in idx], vecs[idx]

    sni_emb = load_sni_embeddings(RUN_NAME, EMBEDDING_DIR)
    vecs_07, codes_07 = sni_emb["sni07"]
    vecs_02, codes_02 = sni_emb["sni02"]
    vecs_92, codes_92 = sni_emb["sni92"]

    # Same call order as P03_emergent_structure.py (07, then 02, then 92) —
    # the shared rng advances sequentially, so the order matters for
    # reproducing the same subsample.
    sc_07, sv_07 = subsample(codes_07, vecs_07, MAX_PER_VERSION)
    sc_02, sv_02 = subsample(codes_02, vecs_02, MAX_PER_VERSION)
    sc_92, sv_92 = subsample(codes_92, vecs_92, MAX_PER_VERSION)

    all_vecs = np.vstack([sv_07, sv_02, sv_92])
    print(f"  Running t-SNE on {len(all_vecs)} codes …")
    tsne = TSNE(n_components=2, perplexity=40, random_state=SEED, max_iter=1000, learning_rate="auto")
    xy = tsne.fit_transform(all_vecs)

    all_codes = sc_07 + sc_02 + sc_92
    all_vers = ["SNI07"] * len(sc_07) + ["SNI02"] * len(sc_02) + ["SNI92"] * len(sc_92)

    section_map_07 = SNI_2digits_to_sections("SNI07")
    section_map_02 = SNI_2digits_to_sections("SNI02")
    section_map_92 = SNI_2digits_to_sections("SNI92")

    def get_sec(code: str, ver: str) -> str:
        if ver == "SNI07":
            return section_map_07.get(code[:2], "?")
        if ver == "SNI02":
            return section_map_02.get(code[:2], "?")
        return section_map_92.get(code[:2], "?")

    all_secs = [get_sec(c, v) for c, v in zip(all_codes, all_vers)]

    def load_sec_names(year_file: str) -> dict:
        p = METADATA_DIR / f"mapping_Sni_section_{year_file}.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {}

    sec_names = {}
    for year_file in ("2007", "2002", "1992"):
        for letter, name in load_sec_names(year_file).items():
            sec_names.setdefault(letter, name)

    def section_label(letter: str) -> str:
        name = sec_names.get(letter, "")
        if not name or name == "Unknown":
            return letter
        if len(name) > 80 and ";" in name:
            name = name.split(";")[0]
        return f"{letter} — {name}"

    unique_secs = sorted({s for s in all_secs if s != "?"})
    n_secs = len(unique_secs)

    # Brighter, higher-value hues than the light-mode original (which used
    # value=0.62 for contrast against a WHITE page) — on a black page the
    # same logic inverts: high value/near-full brightness is what pops.
    hues = [mcolors.hsv_to_rgb([(i / n_secs) % 1.0, 0.85, 0.95]) for i in range(n_secs)]
    sec_color_map = {s: hues[i] for i, s in enumerate(unique_secs)}

    ver_marker = {"SNI07": "o", "SNI02": "s", "SNI92": "^"}
    ver_size = {"SNI07": 26, "SNI02": 26, "SNI92": 30}

    fig, ax = plt.subplots(figsize=(18, 14))
    all_vers_arr = np.array(all_vers)
    for ver in ["SNI07", "SNI02", "SNI92"]:
        mask = all_vers_arr == ver
        v_xy = xy[mask]
        v_secs = [s for s, m in zip(all_secs, mask) if m]
        colors = [sec_color_map.get(s, (0.55, 0.55, 0.55)) for s in v_secs]
        ax.scatter(v_xy[:, 0], v_xy[:, 1], c=colors,
                   marker=ver_marker[ver], s=ver_size[ver],
                   alpha=0.9, edgecolors="black", linewidths=0.4, zorder=2)

    sec_handles = [mpatches.Patch(color=sec_color_map[s], label=section_label(s)) for s in unique_secs]
    ver_handles = [
        plt.Line2D([0], [0], marker=ver_marker[v], color="none", markerfacecolor="#bbbbbb",
                   markeredgecolor="black", markersize=8, label=v)
        for v in ["SNI07", "SNI02", "SNI92"]
    ]

    leg1 = ax.legend(handles=sec_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     fontsize=8, title="Section", ncol=1, framealpha=0.9,
                     facecolor=LEGEND_BG, edgecolor=MUTED, labelcolor=FG)
    leg1.get_title().set_color(FG)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=ver_handles, loc="lower left", bbox_to_anchor=(1.01, 0.0),
                     fontsize=9, title="Version", framealpha=0.9,
                     facecolor=LEGEND_BG, edgecolor=MUTED, labelcolor=FG)
    leg2.get_title().set_color(FG)

    ax.axis("off")
    plt.tight_layout()
    savefig_dark(fig, out_dir / "result3_opportunity_space.png",
                 bbox_extra_artists=(leg1, leg2))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Result 4 — entry / exit / startup (density-effect summary, from cache)
# ═══════════════════════════════════════════════════════════════════════════

OUT_NAME = RUN_NAME  # main specification (municipality×year + industry FE)

RESULT4_OUTCOMES = [
    dict(name="entry", results_subdir="regional_entry", processed_subdir="entry",
         outcome_col="entry_industry", outcome_label="Observed entry rate",
         effect_label="Effect on entry probability per 1-SD density (in %)",
         color="crimson", out_png="result4_entry.png"),
    dict(name="exit", results_subdir="regional_exit", processed_subdir="exit",
         outcome_col="exit_industry", outcome_label="Observed exit rate",
         effect_label="Effect on exit probability per 1-SD density (in %)",
         color="steelblue", out_png="result4_exit.png"),
    dict(name="startup", results_subdir="regional_startup_entry", processed_subdir="startup_entry",
         outcome_col="startup_entry", outcome_label="Observed start-up rate",
         effect_label="Effect on start-up probability per 1-SD density (in %)",
         color="seagreen", out_png="result4_startup.png"),
]


def period_color_map(periods_order: list[str]) -> dict[str, str]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {p: cycle[k % len(cycle)] for k, p in enumerate(periods_order)}


def plot_density_effect_summary_dark(panel, year_df, results, periods_order,
                                      outcome, outcome_label, effect_label, color, fig_path):
    period_colors = period_color_map(periods_order)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A. Rate by density decile
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
        ax.axhline(float(sub[outcome].mean()), color=color_p, linewidth=1.3, linestyle="--", alpha=0.85)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Relatedness-density decile (1 = low, 10 = high)")
    ax.set_ylabel(outcome_label)
    ax.set_title("A. Rate by density")
    ax.legend(fontsize=8, title="period")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # B. Effect over time
    ax = axes[1]
    yr = year_df["year"] + 0.5
    ax.fill_between(yr, year_df["ci_low"] * 100, year_df["ci_high"] * 100, alpha=0.25, color=color)
    ax.plot(yr, year_df["coef"] * 100, "o-", color=color, linewidth=1.6, markersize=4)
    ax.axhline(0, color=FG, linewidth=0.8, linestyle="--")
    for x_line in (2001.5, 2006.5):
        ax.axvline(x_line, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel(effect_label)
    ax.set_title("B. Effect over time")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.relim()
    ax.autoscale_view()
    b_y0, b_y1 = ax.get_ylim()
    b_pad = 0.08 * (b_y1 - b_y0)
    ax.set_ylim(b_y0 - b_pad, b_y1 + b_pad)

    # C. Effect by period
    ax = axes[2]
    x = np.arange(len(periods_order))
    point_info = []
    for pi, period_name in enumerate(periods_order):
        res = results[period_name]
        color_p = period_colors[period_name]
        coef_pp, ci_lo_pp, ci_hi_pp = res["coef"] * 100, res["ci_low"] * 100, res["ci_high"] * 100
        yerr = [[coef_pp - ci_lo_pp], [ci_hi_pp - coef_pp]]
        ax.errorbar(x[pi], coef_pp, yerr=yerr, fmt="o", color=color_p, ecolor=color_p,
                    elinewidth=1.4, capsize=4, markersize=7, markeredgecolor=FG,
                    markeredgewidth=0.6)
        point_info.append((x[pi], coef_pp, ci_lo_pp, ci_hi_pp, res))
    ax.set_xticks(x)
    ax.set_xticklabels(periods_order)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.axhline(0, color=FG, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Classification period")
    ax.set_ylabel(effect_label)
    ax.set_title("C. Effect by period")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    ax.relim()
    ax.autoscale_view()
    y0, y1 = ax.get_ylim()
    pad = 0.12 * (y1 - y0)
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
        ax.annotate(label, xy=(xpos, ci_hi_pp if above else ci_lo_pp),
                    xytext=(0, offset_pts if above else -offset_pts), textcoords="offset points",
                    ha="center", va="bottom" if above else "top", fontsize=8, color=FG)
    ax.set_ylim(y0 - 2.4 * pad, y1 + 2.4 * pad)

    plt.tight_layout()
    savefig_dark(fig, fig_path)
    plt.close(fig)


def build_result4_outcomes() -> None:
    print("\n" + "=" * 70)
    print("Result 4 — entry / exit / startup density effects (dark)")
    print("=" * 70)

    for spec in RESULT4_OUTCOMES:
        panel_path = PROCESSED_DATA_DIR / "downstream_outcomes" / spec["processed_subdir"] / f"{OUT_NAME}_panel.parquet"
        year_path = project_root / "results" / spec["results_subdir"] / f"{OUT_NAME}_year_coefs.csv"
        summary_path = project_root / "results" / spec["results_subdir"] / f"{OUT_NAME}_summary.csv"
        missing = [p for p in (panel_path, year_path, summary_path) if not p.exists()]
        if missing:
            print(f"  SKIPPED {spec['name']} — missing cached input(s): {missing}")
            continue

        print(f"\n  {spec['name']}: loading cached panel/year/summary …")
        panel = pd.read_parquet(panel_path)
        year_df = pd.read_csv(year_path)
        summary = pd.read_csv(summary_path)
        periods_order = list(dict.fromkeys(panel["period"]))

        outcome = spec["outcome_col"]
        results = {}
        for _, row in summary.iterrows():
            coef, se = float(row[f"{outcome}_coef"]), float(row[f"{outcome}_se"])
            results[row["period"]] = {
                "coef": coef, "se": se, "pval": float(row[f"{outcome}_pval"]),
                "partial_r": float(row[f"{outcome}_r"]),
                "pct_of_baseline": float(row[f"{outcome}_pct_of_baseline"]),
                "rate": float(row[f"{outcome}_rate"]),
                "ci_low": float(row[f"{outcome}_ci_low"]), "ci_high": float(row[f"{outcome}_ci_high"]),
            }

        plot_density_effect_summary_dark(
            panel, year_df, results, periods_order, outcome,
            spec["outcome_label"], spec["effect_label"], spec["color"],
            out_dir / spec["out_png"],
        )


# ═══════════════════════════════════════════════════════════════════════════
# Result 4 (single-source density robustness) — boxplot comparison
# ═══════════════════════════════════════════════════════════════════════════

SOURCE_ORDER = ["baseline", "hierarchy", "io", "ownership", "labor", "main_run"]
SOURCE_LABELS = {"main_run": "Opportunity\nSpace"}
OUTCOME_TITLES = {"entry": "Entry", "exit": "Exit", "startup_entry": "Start-up"}
OUTCOME_YLABELS = {"entry": "Effect on entry (partial r)", "exit": "Effect on exit (partial r)",
                    "startup_entry": "Effect on start-up (partial r)"}


def build_result4_single_source_density() -> None:
    print("\n" + "=" * 70)
    print("Result 4 — single-source density robustness boxplot (dark)")
    print("=" * 70)

    csv_path = LUNARC_SINGLE_SOURCE_DIR / "single_source_density_comparison.csv"
    if not csv_path.exists():
        print(f"  SKIPPED — {csv_path} not found.")
        return
    comparison_df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, outcome_name in zip(axes, OUTCOME_TITLES):
        sub = comparison_df[comparison_df["outcome"] == outcome_name]
        box_data, labels = [], []
        for source_name in SOURCE_ORDER:
            vals = sub.loc[sub["source"] == source_name, "partial_r"].dropna().to_numpy()
            if len(vals) == 0:
                continue
            box_data.append(vals)
            labels.append(SOURCE_LABELS.get(source_name, source_name))

        ax.boxplot(
            box_data, tick_labels=labels, showmeans=True,
            boxprops=dict(color=FG), whiskerprops=dict(color=FG), capprops=dict(color=FG),
            medianprops=dict(color="#ffd166"),
            meanprops=dict(marker="D", markerfacecolor="#06d6a0", markeredgecolor=FG, markersize=6),
            flierprops=dict(markeredgecolor=FG, markersize=4),
        )
        for i, vals in enumerate(box_data, start=1):
            jitter = np.random.default_rng(0).normal(0, 0.03, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color="white", s=14, zorder=3, alpha=0.85)
        ax.axhline(0, color=FG, linewidth=0.8, linestyle="--")
        ax.set_title(OUTCOME_TITLES[outcome_name])
        ax.set_ylabel(OUTCOME_YLABELS[outcome_name])
        ax.tick_params(axis="x", labelsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.relim()
        ax.autoscale_view()
        y0, y1 = ax.get_ylim()
        pad = 0.08 * (y1 - y0)
        ax.set_ylim(y0 - pad, y1 + pad)
    plt.tight_layout()
    savefig_dark(fig, out_dir / "result4_single_source_density_comparison.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    build_result1()
    build_result3()
    build_result4_outcomes()
    build_result4_single_source_density()
    print(f"\nDone. Dark-mode figures written to {out_dir}/")


if __name__ == "__main__":
    main()
