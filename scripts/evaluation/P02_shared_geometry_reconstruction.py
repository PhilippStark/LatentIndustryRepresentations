"""
P02_shared_geometry_reconstruction.py
=====================================
Result 2: does the one shared embedding geometry reconstruct each relational
view it was trained on? Compares cosine similarity in the trained space with
each source's own relatedness, and with the baseline: the untrained text
encoder (sentence-transformers/all-MiniLM-L6-v2, 384-d). Training starts from
the same encoder's output reduced to 64-d by PCA, so the baseline is the text
prior before that reduction.

Sections
--------
E   Labor: Spearman rho between cosine similarity and labor relatedness (per SNI version)
E2  Ownership: same, against ownership relatedness
F   Section silhouette of SNI07 4-digit codes
I   Input-output: rho against IO relatedness at the 2-digit level
K   Source-wise KL divergence between each source's row-normalised relatedness
    and the embedding's cosine-softmax at one fixed temperature (--tau)
G   Summary table of all scalar metrics, baseline vs trained
K2  Per-family source table and dot plot

Hierarchy preservation and cross-version retrieval (P03's sections A/B/B2)
are recomputed here for the summary table.

Outputs: results/embedding_eval/{run}/ (figures, {run}_summary.csv,
{run}_K_source_table.csv, log).

Usage (main run)
----------------
  python scripts/evaluation/P02_shared_geometry_reconstruction.py --run-name shared_cosine_v16
"""

import argparse
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score, silhouette_samples

from _shared_eval_context import (
    SEED, PROCESSED_DATA_DIR,
    add_common_args, build_base_context, load_baseline_and_hub_embeddings,
    load_crosswalks, open_log,
    compute_key_metrics, kl_source_metrics, aggregate_up_io_metric,
)

parser = argparse.ArgumentParser()
add_common_args(parser)
parser.add_argument("--tau", type=float, default=0.05,
                    help="Fixed softmax temperature for section K's source-wise KL evaluation, "
                         "used for both the baseline and the trained embeddings (training "
                         "itself learns one temperature per target).")
args, _ = parser.parse_known_args()

ctx = build_base_context(args)
load_baseline_and_hub_embeddings(ctx)
load_crosswalks(ctx)

# Unpack onto locals for readability, mirroring the original module-level globals.
codes_07, codes_02, codes_92 = ctx.codes_07, ctx.codes_02, ctx.codes_92
idx_07, idx_02, idx_92       = ctx.idx_07, ctx.idx_02, ctx.idx_92
vecs_07, vecs_02, vecs_92    = ctx.vecs_07, ctx.vecs_02, ctx.vecs_92
bl_vecs_07, bl_vecs_02, bl_vecs_92 = ctx.bl_vecs_07, ctx.bl_vecs_02, ctx.bl_vecs_92
vecs_3d, vecs_2d, vecs_sec       = ctx.vecs_3d, ctx.vecs_2d, ctx.vecs_sec
bl_vecs_3d, bl_vecs_2d, bl_vecs_sec = ctx.bl_vecs_3d, ctx.bl_vecs_2d, ctx.bl_vecs_sec
idx_3d, idx_2d, idx_sec     = ctx.idx_3d, ctx.idx_2d, ctx.idx_sec
sni07, sni02, sni92         = ctx.sni07, ctx.sni02, ctx.sni92
section_map_07              = ctx.section_map_07
eval_dir                    = ctx.eval_dir
true_targets, true_targets_92 = ctx.true_targets, ctx.true_targets_92

# ── Redirect stdout to tee (log file + terminal) ───────────────────────────────
_tee, _log_file = open_log(eval_dir, args.run_name, "P02")
sys.stdout = _tee

# ══════════════════════════════════════════════════════════════════════════════
# E. Labor flow consistency
#    Spearman rank correlation between pairwise cosine similarity and phi.
#    Uses 4-digit atom embeddings; falls back gracefully if file absent.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("E. Labor flow consistency  (cosine sim vs Neffke-Henning phi)")
print("═" * 60)

labor_path = PROCESSED_DATA_DIR / "labor_relatedness.parquet"
ctx.labor_path = labor_path

metrics_summary: dict[str, float] = {}

if labor_path.exists():
    labor_df = pd.read_parquet(labor_path)
    ctx.labor_df = labor_df

    # One panel per version; only within-version labor flows (no cross-coding rows).
    # coding label in the parquet matches the SCB coding period used to build it.
    available_codings = set(labor_df["coding"].unique())
    ver_cfg = [
        ("SNI07", idx_07, vecs_07, "SNI07"),
        ("SNI02", idx_02, vecs_02, "SNI02"),
        ("SNI92", idx_92, vecs_92, "SNI92"),
    ]

    # Only draw as many panels as there are available codings (skip absent ones)
    active_cfg = [(c, ix, v, lbl) for c, ix, v, lbl in ver_cfg if c in available_codings]
    n_panels = max(len(active_cfg), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    any_results = False

    ax_iter = iter(axes)
    for coding, idx, vecs, label in ver_cfg:
        subset = labor_df[labor_df["coding"] == coding]
        phi_vals, sim_vals = [], []
        for _, row in subset.iterrows():
            s, t = str(row["sni_from"]), str(row["sni_to"])
            if s in idx and t in idx:
                phi_vals.append(float(row["phi"]))
                sim_vals.append(float(vecs[idx[s]] @ vecs[idx[t]]))

        print(f"\n  {label}")
        if coding not in available_codings:
            print(f"  Skipped: coding '{coding}' not found in labor_relatedness.parquet")
            continue
        ax = next(ax_iter)
        if len(phi_vals) >= 10:
            rho, pval = spearmanr(sim_vals, phi_vals)
            print(f"  Pairs evaluated:         {len(phi_vals):,}")
            print(f"  Spearman rho (sim~phi):  {rho:.4f}  (p={pval:.2e})")
            metrics_summary[f"E_labor_rho_{label[-2:].lower()}"] = round(rho, 4)
            any_results = True

            ax.hexbin(phi_vals, sim_vals, gridsize=35, cmap="Blues", mincnt=1)
            ax.set_xlabel("Neffke-Henning phi")
            ax.set_ylabel("Cosine similarity")
            ax.set_title(f"{label}  ρ={rho:.3f}  n={len(phi_vals):,}")
            plt.colorbar(ax.collections[0], ax=ax, label="pair count")
        else:
            ax.set_visible(False)
            print(f"  Skipped: too few matching pairs ({len(phi_vals)})")

    if any_results:
        fig.suptitle("Labor flow consistency — Neffke-Henning phi vs cosine similarity",
                     fontsize=12, y=1.01)
        plt.tight_layout()
        out = eval_dir / f"{args.run_name}_E_labor_consistency.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  → {out}")
    plt.close()
else:
    print(f"  {labor_path.name} not found — run build_labor_relatedness.py first.")


# ══════════════════════════════════════════════════════════════════════════════
# E2. Ownership flow consistency
#    Spearman rank correlation between pairwise cosine similarity and
#    ownership-cooccurrence phi. Same construction and per-version split as E.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("E2. Ownership flow consistency  (cosine sim vs ownership co-occurrence phi)")
print("═" * 60)

ownership_path = PROCESSED_DATA_DIR / "ownership_relatedness.parquet"
ctx.ownership_path = ownership_path
ver_cfg_own = [
    ("SNI07", idx_07, vecs_07, "SNI07"),
    ("SNI02", idx_02, vecs_02, "SNI02"),
    ("SNI92", idx_92, vecs_92, "SNI92"),
]

if ownership_path.exists():
    ownership_df = pd.read_parquet(ownership_path)
    ctx.ownership_df = ownership_df

    available_codings_own = set(ownership_df["coding"].unique())
    active_cfg_own = [(c, ix, v, lbl) for c, ix, v, lbl in ver_cfg_own if c in available_codings_own]
    n_panels_own = max(len(active_cfg_own), 1)
    fig, axes = plt.subplots(1, n_panels_own, figsize=(6 * n_panels_own, 5))
    if n_panels_own == 1:
        axes = [axes]
    any_results_own = False

    ax_iter_own = iter(axes)
    for coding, idx, vecs, label in ver_cfg_own:
        subset = ownership_df[ownership_df["coding"] == coding]
        phi_vals, sim_vals = [], []
        for _, row in subset.iterrows():
            s, t = str(row["sni_from"]), str(row["sni_to"])
            if s in idx and t in idx:
                phi_vals.append(float(row["phi"]))
                sim_vals.append(float(vecs[idx[s]] @ vecs[idx[t]]))

        print(f"\n  {label}")
        if coding not in available_codings_own:
            print(f"  Skipped: coding '{coding}' not found in ownership_relatedness.parquet")
            continue
        ax = next(ax_iter_own)
        if len(phi_vals) >= 10:
            rho, pval = spearmanr(sim_vals, phi_vals)
            print(f"  Pairs evaluated:         {len(phi_vals):,}")
            print(f"  Spearman rho (sim~phi):  {rho:.4f}  (p={pval:.2e})")
            metrics_summary[f"E2_ownership_rho_{label[-2:].lower()}"] = round(rho, 4)
            any_results_own = True

            ax.hexbin(phi_vals, sim_vals, gridsize=35, cmap="Greens", mincnt=1)
            ax.set_xlabel("Ownership co-occurrence phi")
            ax.set_ylabel("Cosine similarity")
            ax.set_title(f"{label}  ρ={rho:.3f}  n={len(phi_vals):,}")
            plt.colorbar(ax.collections[0], ax=ax, label="pair count")
        else:
            ax.set_visible(False)
            print(f"  Skipped: too few matching pairs ({len(phi_vals)})")

    if any_results_own:
        fig.suptitle("Ownership flow consistency — co-occurrence phi vs cosine similarity",
                     fontsize=12, y=1.01)
        plt.tight_layout()
        out = eval_dir / f"{args.run_name}_E2_ownership_consistency.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  → {out}")
    plt.close()
else:
    print(f"  {ownership_path.name} not found — run build_ownership_relatedness.py first.")


# ══════════════════════════════════════════════════════════════════════════════
# F. Global structure — Section Silhouette
#    Silhouette score of SNI07 4-digit atoms labeled by NACE section.
#    Higher = sections are geometrically compact and well-separated.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("F. Global structure — Section Silhouette (SNI07 4d)")
print("═" * 60)

sec_labels = [section_map_07.get(c[:2], "?") for c in codes_07]
known_mask = np.array([s != "?" for s in sec_labels])

if known_mask.sum() > 10:
    vecs_known = vecs_07[known_mask]
    labs_known = [s for s, m in zip(sec_labels, known_mask) if m]

    sil = silhouette_score(vecs_known, labs_known, metric="cosine")
    n_sections = len(set(labs_known))
    print(f"  Codes:    {known_mask.sum()}  |  Sections: {n_sections}")
    print(f"  Silhouette (cosine):  {sil:.4f}  "
          f"[range -1→1; higher = cleaner section geometry]")
    metrics_summary["F_section_silhouette"] = round(sil, 4)

    sample_sil = silhouette_samples(vecs_known, labs_known, metric="cosine")
    per_sec = {}
    for sec, s in zip(labs_known, sample_sil):
        per_sec.setdefault(sec, []).append(s)
    print("\n  Per-section mean silhouette:")
    for sec in sorted(per_sec):
        print(f"    {sec}: {np.mean(per_sec[sec]):.4f}  (n={len(per_sec[sec])})")
else:
    print("  Not enough labeled codes for Silhouette.")


# ══════════════════════════════════════════════════════════════════════════════
# I. IO relation consistency
#    Spearman rank correlation between blended input-output relatedness
#    ("sim" in io_relatedness_SNI07.parquet, the column the io view uses) and
#    trained cosine similarity at the 2-digit division level.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("I. IO relation consistency  (cosine sim vs blended IO relatedness)")
print("═" * 60)

io_path = PROCESSED_DATA_DIR / "io_relatedness_SNI07.parquet"
ctx.io_path = io_path
if io_path.exists():
    io_df = pd.read_parquet(io_path)
    ctx.io_df = io_df
    io_gt, io_sim = [], []
    for r in io_df.itertuples():
        a, b = str(r.nace2d_from).zfill(2), str(r.nace2d_to).zfill(2)
        if a in idx_2d and b in idx_2d and a != b:
            io_gt.append(float(r.sim))
            io_sim.append(float(vecs_2d[idx_2d[a]] @ vecs_2d[idx_2d[b]]))

    if len(io_gt) >= 10:
        rho, pval = spearmanr(io_sim, io_gt)
        print(f"  Pairs evaluated:        {len(io_gt):,}")
        print(f"  Spearman rho (sim~io):  {rho:.4f}  (p={pval:.2e})")
        metrics_summary["I_io_rho"] = round(float(rho), 4)

        _agg = aggregate_up_io_metric(vecs_07, codes_07, io_df)
        if _agg:
            print(f"  Spearman rho (aggregate-up sim~io):  {_agg['I_io_aggregate_rho']:.4f}  "
                  f"(mean of 4d-4d cosines per division pair)")
            metrics_summary.update(_agg)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hexbin(io_gt, io_sim, gridsize=35, cmap="Purples", mincnt=1)
        ax.set_xlabel("Blended IO relatedness (sim)")
        ax.set_ylabel("Trained cosine similarity (2d divisions)")
        ax.set_title(f"IO relation consistency  ρ={rho:.3f}  n={len(io_gt):,}")
        plt.tight_layout()
        out_path = eval_dir / f"{args.run_name}_I_io_consistency.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  → {out_path}")
        plt.close()
    else:
        print(f"  Skipped: too few matching pairs ({len(io_gt)})")
else:
    print(f"  {io_path.name} not found — run build_input_output_relatedness.py first.")

# ══════════════════════════════════════════════════════════════════════════════
# K. Source-wise KL evaluation
#    For every evidence family s, KL(P^(s) || Q) with Q the cosine-softmax at
#    the fixed temperature --tau (default 0.05), the same for the baseline and
#    the trained embeddings; the baseline-vs-trained delta is printed in
#    section G. Training learns its own temperature per target (bounded to
#    [e^-1, e]), so these values are not the training loss.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print(f"K. Source-wise KL evaluation  (tau={args.tau})")
print("═" * 60)

k_trained = kl_source_metrics(ctx, vecs_07, vecs_02, vecs_92, vecs_3d, vecs_2d, vecs_sec, args.tau)
if k_trained:
    for key in sorted(k_trained):
        print(f"  {key:<30}  {k_trained[key]}")
    metrics_summary.update(k_trained)
else:
    print("  No source files available for KL evaluation.")


# ══════════════════════════════════════════════════════════════════════════════
# G. Multi-signal summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("G. Multi-signal summary")
print("═" * 60)

# compute_key_metrics recomputes every scalar metric (including P03's A/B/B2)
# on the trained embeddings; the values accumulated by the sections above are
# overlaid (identical by construction).
_seeded = compute_key_metrics(
    ctx, vecs_07, vecs_02, vecs_92, vecs_3d, vecs_2d, vecs_sec,
    args.tau, rng=random.Random(SEED),
)
_seeded.update(metrics_summary)
metrics_summary = _seeded

col_w = 32

# Baseline comparison: the same metrics for the untrained text encoder
print("  Computing baseline metrics for comparison …")
baseline_summary = compute_key_metrics(
    ctx, bl_vecs_07, bl_vecs_02, bl_vecs_92, bl_vecs_3d, bl_vecs_2d, bl_vecs_sec,
    args.tau, rng=random.Random(SEED),
)

all_keys = sorted(set(metrics_summary) | set(baseline_summary))
print(f"\n  {'Metric':<{col_w}}  {'Baseline':>9}  {'Trained':>10}  {'Δ':>8}")
print(f"  {'-'*col_w}  {'-'*9}  {'-'*10}  {'-'*8}")
for k in all_keys:
    bl_v = baseline_summary.get(k, float("nan"))
    ft_v = metrics_summary.get(k, float("nan"))
    if np.isnan(bl_v) or np.isnan(ft_v):
        delta_s = "   n/a"
    else:
        delta_s = f"{ft_v - bl_v:+.4f}"
    bl_s = f"{bl_v:.4f}" if not np.isnan(bl_v) else "    n/a"
    ft_s = f"{ft_v:.4f}" if not np.isnan(ft_v) else "     n/a"
    print(f"  {k:<{col_w}}  {bl_s:>9}  {ft_s:>10}  {delta_s:>8}")

summary_df = pd.DataFrame([
    {"run": "baseline",     **{k: baseline_summary.get(k, float("nan")) for k in all_keys}},
    {"run": args.run_name,  **{k: metrics_summary.get(k, float("nan"))  for k in all_keys}},
])
summary_path = eval_dir / f"{args.run_name}_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\n  → {summary_path}")

# ══════════════════════════════════════════════════════════════════════════════
# K2. Consolidated Figure-2 source table + dot plot
#    One row per evidence family: KL_raw, KL_trained, relative KL reduction,
#    plus the matching Spearman-rho-style metric already computed in E/I/B.
# ══════════════════════════════════════════════════════════════════════════════

FAMILY_TABLE = [
    ("Labor (SNI07)",     "K_labor_KL_07",           "E_labor_rho_07"),
    ("Labor (SNI02)",     "K_labor_KL_02",           "E_labor_rho_02"),
    ("Labor (SNI92)",     "K_labor_KL_92",           "E_labor_rho_92"),
    ("Ownership (SNI07)", "K_ownership_KL_07",       "E2_ownership_rho_07"),
    ("Ownership (SNI02)", "K_ownership_KL_02",       "E2_ownership_rho_02"),
    ("Ownership (SNI92)", "K_ownership_KL_92",       "E2_ownership_rho_92"),
    ("Input-output",      "K_io_KL",                 "I_io_rho"),
    ("Concordance 02→07", "K_concordance_KL_02to07", "B_MRR_02to07"),
    ("Concordance 92→02", "K_concordance_KL_92to02", "B2_MRR_92to02"),
    # Hierarchy: no KL (a row distribution does not represent tree structure); the
    # tree-distance rho above (Section A) is the source-specific metric instead.
    ("Hierarchy",         None,                      "A_treedist_rho"),
]

fig2_rows = []
for family, kl_key, metric_key in FAMILY_TABLE:
    kl_raw     = baseline_summary.get(kl_key) if kl_key else None
    kl_trained = metrics_summary.get(kl_key) if kl_key else None
    metric_raw     = baseline_summary.get(metric_key)
    metric_trained = metrics_summary.get(metric_key)
    if kl_raw is None and metric_raw is None:
        continue
    rel_reduction = (
        (kl_raw - kl_trained) / kl_raw
        if kl_raw is not None and kl_trained is not None and kl_raw != 0
        else float("nan")
    )
    fig2_rows.append({
        "family": family,
        "KL_raw": kl_raw if kl_raw is not None else float("nan"),
        "KL_trained": kl_trained if kl_trained is not None else float("nan"),
        "relative_KL_reduction": round(rel_reduction, 4),
        "source_metric": metric_key,
        "source_metric_raw": metric_raw,
        "source_metric_trained": metric_trained,
    })

if fig2_rows:
    fig2_df = pd.DataFrame(fig2_rows)
    fig2_path = eval_dir / f"{args.run_name}_K_source_table.csv"
    fig2_df.to_csv(fig2_path, index=False)
    print(f"\n  Figure-2 source table → {fig2_path}")

    fig2_df_sorted = fig2_df.dropna(subset=["relative_KL_reduction"]).sort_values("relative_KL_reduction")
    fig, ax = plt.subplots(figsize=(8, 0.5 * max(len(fig2_df_sorted), 1) + 1.5))
    colors = ["crimson" if v < 0 else "steelblue" for v in fig2_df_sorted["relative_KL_reduction"]]
    ax.scatter(fig2_df_sorted["relative_KL_reduction"], fig2_df_sorted["family"],
               color=colors, s=80, zorder=3)
    ax.axvline(0, color="gray", linewidth=1, zorder=1)
    ax.set_xlabel("Relative KL reduction  (KL_raw − KL_trained) / KL_raw")
    ax.set_title(f"Source-wise information preservation — {args.run_name}\n"
                 "positive = integrated training improved this source's representation")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    dotplot_path = eval_dir / f"{args.run_name}_K_kl_dotplot.png"
    fig.savefig(dotplot_path, dpi=150, bbox_inches="tight")
    print(f"  Dot plot            → {dotplot_path}")
    plt.close()
else:
    print("\n  No families had both baseline and trained KL — skipping Figure-2 table.")

print(f"\nAll outputs written to {eval_dir}/")
print("Done.")

sys.stdout = sys.__stdout__
_log_file.close()
