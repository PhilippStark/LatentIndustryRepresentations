"""
P02_shared_geometry_reconstruction.py
=====================================
Result 2: does the one shared embedding geometry reconstruct each relational
view it was trained on? Compares cosine similarity in the trained space with
each source's own relatedness, and with the untrained text-encoder baseline
(sentence-transformers/all-MiniLM-L6-v2, the model used for the text-prior
initialisation).

Sections
--------
E   Labor: Spearman rho between cosine similarity and labor relatedness (per SNI version)
E2  Ownership: same, against ownership relatedness
F   Section silhouette of SNI07 4-digit codes
H   NACE text: rho between cosine similarity and NACE-description similarity
    (4-digit, 3-digit, 2-digit and section level)
I   Input-output: rho against IO relatedness at the 2-digit level
J   Repulsion and fused-target fidelity; only if diffusion_matrix_{graph}.npz
    exists (not produced by the current pipeline, so skipped for the main run)
K   Source-wise KL divergence between each view's target and the
    embedding's similarity distribution
G   Summary table of all scalar metrics, baseline vs trained
K2  Per-family source table and dot plot (feeds build_paper_tables.py)

Families with zero training weight (in training_provenance.json) are still
evaluated, as a generalisation check.

Outputs: results/embedding_eval/{run}/ (figures, {run}_summary.csv,
{run}_K_source_table.csv, log).

Usage (main run)
----------------
  python scripts/evaluation/P02_shared_geometry_reconstruction.py \
      --run-name shared_cosine_v16 --graph-name layers_v1
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
    relation_consistency, compute_key_metrics, kl_source_metrics,
    _HUB_LABEL_BY_SUFFIX, _SKIP_J2_REGIMES, is_family_excluded,
)

parser = argparse.ArgumentParser()
add_common_args(parser)  # includes --graph-name: H/I/J/K read graph_edges_{name}/diffusion_matrix_{name}
parser.add_argument("--tau", type=float, default=0.05,
                    help="Softmax temperature for section K's source-wise KL evaluation. "
                         "The same temperature is used for the baseline and trained "
                         "embeddings.")
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
_hub_from_cache              = ctx._hub_from_cache
_training_regime             = ctx.training_regime

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

if labor_path.exists() and not is_family_excluded(ctx, "labor"):
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
elif is_family_excluded(ctx, "labor"):
    print("  SKIPPED — labor was excluded from training (lambda=0.0 in training_provenance.json).")
else:
    print(f"  {labor_path.name} not found — run build_labor_relatedness.py first.")


# ══════════════════════════════════════════════════════════════════════════════
# E2. Ownership flow consistency
#    Spearman rank correlation between pairwise cosine similarity and
#    ownership-cooccurrence phi. Same construction and per-version split as E;
#    not split per training version yet upstream (see build_ownership_relatedness.py),
#    but the parquet itself already carries all three codings.
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

if ownership_path.exists() and not is_family_excluded(ctx, "ownership"):
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
elif is_family_excluded(ctx, "ownership"):
    print("  SKIPPED — ownership was excluded from training (lambda=0.0 in training_provenance.json).")
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
# H. NACE relation consistency (nace_4d / nace_3d / nace_2d / nace_sec)
#    Spearman rank correlation between pairwise cosine similarity in the
#    ORIGINAL NACE-description embedding (the nace_text view's edge weights in
#    graph_edges_{name}.parquet) and in the TRAINED
#    model's own embedding space, for the same code pairs. Tests whether the
#    larger embedding model's semantic structure survived into the smaller
#    fine-tuned sentence-transformer at every hierarchy level, not just 4d.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("H. NACE relation consistency (nace_4d / nace_3d / nace_2d / nace_sec)")
print("═" * 60)

graph_edges_path = PROCESSED_DATA_DIR / f"graph_edges_{args.graph_name}.parquet"
graph_nodes_path = PROCESSED_DATA_DIR / f"graph_nodes_{args.graph_name}.parquet"
ctx.graph_edges_path = graph_edges_path
ctx.graph_nodes_path = graph_nodes_path

if graph_edges_path.exists() and graph_nodes_path.exists() and not is_family_excluded(ctx, "nace_text"):
    graph_nodes_df = pd.read_parquet(graph_nodes_path)
    graph_edges_df = pd.read_parquet(graph_edges_path)
    node_lookup = dict(zip(graph_nodes_df.index, graph_nodes_df["code"]))
    ctx.graph_edges_df = graph_edges_df
    ctx.node_lookup = node_lookup

    nace_levels = [
        ("nace_4d",  idx_07,  vecs_07,  "4d"),
        ("nace_3d",  idx_3d,  vecs_3d,  "3d"),
        ("nace_2d",  idx_2d,  vecs_2d,  "2d"),
        ("nace_sec", idx_sec, vecs_sec, "section"),
    ]

    fig, axes = plt.subplots(1, len(nace_levels), figsize=(6 * len(nace_levels), 5))
    for ax, (etype, idx_map, vecs_l, lbl) in zip(axes, nace_levels):
        gt, sim = relation_consistency(etype, idx_map, vecs_l, graph_edges_df, node_lookup)
        print(f"\n  {etype} ({lbl})")
        hub_label = _HUB_LABEL_BY_SUFFIX.get(lbl)
        from_cache = hub_label is not None and _hub_from_cache.get(hub_label, False)
        metric_key = f"H_nace_rho_{lbl}_agg" if from_cache else f"H_nace_rho_{lbl}"
        if len(gt) >= 10:
            rho, pval = spearmanr(sim, gt)
            print(f"  Pairs evaluated:          {len(gt):,}")
            print(f"  Spearman rho (sim~nace):  {rho:.4f}  (p={pval:.2e})")
            if from_cache:
                print(f"  ({hub_label} read from the aggregation cache: compares the centroid "
                      f"of a division's codes with the base model's encoding of the division "
                      f"description; reported as {metric_key})")
            metrics_summary[metric_key] = round(float(rho), 4)
            ax.hexbin(gt, sim, gridsize=35, cmap="Greens", mincnt=1)
            ax.set_xlabel("Original NACE-embedding cosine similarity")
            ax.set_ylabel("Fine-tuned cosine similarity")
            ax.set_title(f"{etype}  ρ={rho:.3f}  n={len(gt):,}")
        else:
            ax.set_visible(False)
            print(f"  Skipped: too few matching pairs ({len(gt)})")

    fig.suptitle("NACE relation consistency — original embedding vs fine-tuned similarity",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = eval_dir / f"{args.run_name}_H_nace_consistency.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  → {out_path}")
    plt.close()
elif is_family_excluded(ctx, "nace_text"):
    print("  SKIPPED — nace_text was excluded from training (lambda=0.0 in training_provenance.json).")
else:
    print(f"  {graph_edges_path.name} / {graph_nodes_path.name} not found — run build_layer_transition_matrices.py first.")

# ══════════════════════════════════════════════════════════════════════════════
# I. IO relation consistency
#    Spearman rank correlation between blended input-output relatedness
#    ("sim" in io_relatedness_SNI07.parquet, the column the io view uses) and
#    fine-tuned cosine similarity at the 2-digit division level.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("I. IO relation consistency  (cosine sim vs blended IO relatedness)")
print("═" * 60)

io_path = PROCESSED_DATA_DIR / "io_relatedness_SNI07.parquet"
ctx.io_path = io_path
if io_path.exists() and not is_family_excluded(ctx, "io"):
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

        from _shared_eval_context import aggregate_up_io_metric
        _agg = aggregate_up_io_metric(vecs_07, codes_07, io_df)
        if _agg:
            print(f"  Spearman rho (aggregate-up sim~io):  {_agg['I_io_aggregate_rho']:.4f}  "
                  f"(mean of 4d-4d cosines per division pair)")
            metrics_summary.update(_agg)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hexbin(io_gt, io_sim, gridsize=35, cmap="Purples", mincnt=1)
        ax.set_xlabel("Blended IO relatedness (sim)")
        ax.set_ylabel("Fine-tuned cosine similarity (2d divisions)")
        ax.set_title(f"IO relation consistency  ρ={rho:.3f}  n={len(io_gt):,}")
        plt.tight_layout()
        out_path = eval_dir / f"{args.run_name}_I_io_consistency.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  → {out_path}")
        plt.close()
    else:
        print(f"  Skipped: too few matching pairs ({len(io_gt)})")
elif is_family_excluded(ctx, "io"):
    print("  SKIPPED — io was excluded from training (lambda=0.0 in training_provenance.json).")
else:
    print(f"  {io_path.name} not found — run build_input_output_relatedness.py first.")

# ══════════════════════════════════════════════════════════════════════════════
# J. Repulsion consistency & training-target fidelity
#    Requires diffusion_matrix_{name}.npz (a single fused diffusion target,
#    S and S_neg); the current graph pipeline does not write it, so J is
#    skipped for the main run.
#    J1: are repulsion pairs (S_neg) less similar than other SNI07 pairs?
#    J2: did training converge to its own assigned target? Spearman rho
#        between the diffusion target S (row-normalised — what the KL loss
#        was fit to) and the fine-tuned model's own cosine similarity, pooled
#        over every nonzero target pair across all three SNI versions. Unlike
#        every other section here, this checks fit quality, not domain
#        validity — a model can score well here and still have learned a
#        badly-calibrated target (that's what A-I are for).
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("J. Repulsion consistency & training-target fidelity")
print("═" * 60)

diffusion_path = PROCESSED_DATA_DIR / f"diffusion_matrix_{args.graph_name}.npz"
ctx.diffusion_path = diffusion_path
if diffusion_path.exists():
    dmat = np.load(diffusion_path, allow_pickle=True)
    S, S_neg = dmat["S"], dmat["S_neg"]
    dm_codes, dm_versions = dmat["codes"], dmat["versions"]
    ctx.S, ctx.S_neg, ctx.dm_codes, ctx.dm_versions = S, S_neg, dm_codes, dm_versions

    from _shared_eval_context import _align_to_diffusion_matrix
    dm_vecs, dm_valid = _align_to_diffusion_matrix(ctx, vecs_07, vecs_02, vecs_92, dm_codes, dm_versions)
    print(f"  Aligned {int(dm_valid.sum()):,}/{len(dm_codes):,} diffusion-matrix nodes to fine-tuned embeddings")

    # ── J1: repulsion ──────────────────────────────────────────────────────
    is_07 = dm_versions == "SNI07"
    sub_idx = np.where(is_07 & dm_valid)[0]
    if len(sub_idx) > 10:
        sub_vecs = dm_vecs[sub_idx]
        sim_mat  = sub_vecs @ sub_vecs.T
        neg_sub  = S_neg[np.ix_(sub_idx, sub_idx)]
        iu, ju   = np.triu_indices(len(sub_idx), k=1)
        neg_w, sim_w = neg_sub[iu, ju], sim_mat[iu, ju]
        repel_mask = neg_w > 0
        n_repel = int(repel_mask.sum())
        if n_repel >= 10:
            rho, pval = spearmanr(sim_w[repel_mask], neg_w[repel_mask])
            repel_mean = float(sim_w[repel_mask].mean())
            other_mean = float(sim_w[~repel_mask].mean())
            print(f"\n  J1. Repulsion pairs (S_neg > 0): {n_repel:,}")
            print(f"  Mean fine-tuned sim  (repulsion pairs):   {repel_mean:.4f}")
            print(f"  Mean fine-tuned sim  (other SNI07 pairs): {other_mean:.4f}")
            print(f"  Spearman rho (sim ~ repulsion weight):    {rho:.4f}  (p={pval:.2e})  "
                  f"[expect < 0: stronger repulsion → lower similarity]")
            metrics_summary["J1_repulsion_rho"] = round(float(rho), 4)
            metrics_summary["J1_repulsion_gap"] = round(other_mean - repel_mean, 4)

            fig, ax = plt.subplots(figsize=(8, 4))
            bins = np.linspace(-0.1, 1.0, 50)
            ax.hist(sim_w[~repel_mask], bins=bins, density=True, alpha=0.55,
                    color="steelblue", label=f"Other pairs (n={int((~repel_mask).sum()):,})")
            ax.hist(sim_w[repel_mask],  bins=bins, density=True, alpha=0.55,
                    color="crimson",    label=f"Repulsion pairs (n={n_repel:,})")
            ax.axvline(other_mean, color="steelblue", linestyle="--", linewidth=1.5)
            ax.axvline(repel_mean, color="crimson",   linestyle="--", linewidth=1.5)
            ax.set_xlabel("Fine-tuned cosine similarity")
            ax.set_ylabel("Density")
            ax.set_title(f"Repulsion consistency (SNI07 4d)  ρ={rho:.3f}")
            ax.legend(fontsize=9)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
            plt.tight_layout()
            out_path = eval_dir / f"{args.run_name}_J1_repulsion_consistency.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"  → {out_path}")
            plt.close()
        else:
            print(f"  J1. Skipped: too few repulsion pairs ({n_repel})")
    else:
        print("  J1. Skipped: too few valid SNI07 nodes in diffusion matrix")

    # ── J2: training-target fidelity ───────────────────────────────────────
    if _training_regime in _SKIP_J2_REGIMES:
        print(f"\n  J2. N/A — regime ({_training_regime!r}) is not trained on a single "
              f"fused diffusion target.")
    else:
        valid_idx = np.where(dm_valid)[0]
        if len(valid_idx) > 10:
            v_vecs = dm_vecs[valid_idx]
            v_sim  = v_vecs @ v_vecs.T
            v_S    = S[np.ix_(valid_idx, valid_idx)]
            nz_i, nz_j = np.nonzero(v_S)
            keep = nz_i != nz_j
            nz_i, nz_j = nz_i[keep], nz_j[keep]
            if len(nz_i) > 10:
                rho, pval = spearmanr(v_sim[nz_i, nz_j], v_S[nz_i, nz_j])
                print(f"\n  J2. Training-target fidelity (S vs fine-tuned sim)")
                print(f"  Nonzero target pairs evaluated: {len(nz_i):,}")
                print(f"  Spearman rho (sim ~ target S):  {rho:.4f}  (p={pval:.2e})")
                metrics_summary["J2_target_fidelity_rho"] = round(float(rho), 4)
            else:
                print("  J2. Skipped: too few nonzero target entries")
        else:
            print("  J2. Skipped: too few valid nodes")
else:
    print(f"  {diffusion_path.name} not found — section J skipped.")


# ══════════════════════════════════════════════════════════════════════════════
# K. Source-wise KL evaluation
#    For every evidence family s, KL(P^(s) || Q) at the trainer's own tau —
#    the raw-vs-trained delta (printed/saved in section G's Δ table, since
#    this dict feeds both metrics_summary here and compute_key_metrics for
#    the baseline pass) directly measures whether integrated training
#    improved or hurt each source, in the training loss's own units.
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

# Baseline comparison (always computed; delta = 0 if --baseline was set)
print("  Computing baseline metrics for comparison …")
baseline_summary = compute_key_metrics(
    ctx, bl_vecs_07, bl_vecs_02, bl_vecs_92, bl_vecs_3d, bl_vecs_2d, bl_vecs_sec,
    args.tau, rng=random.Random(SEED),
)

all_keys = sorted(set(metrics_summary) | set(baseline_summary))
print(f"\n  {'Metric':<{col_w}}  {'Baseline':>9}  {'Fine-tuned':>10}  {'Δ':>8}")
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
#    plus the matching Spearman-rho-style metric already computed in E/H/I/B/J2.
# ══════════════════════════════════════════════════════════════════════════════

FAMILY_TABLE = [
    ("Labor (SNI07)",     "K_labor_KL_07",           "E_labor_rho_07"),
    ("Labor (SNI02)",     "K_labor_KL_02",           "E_labor_rho_02"),
    ("Labor (SNI92)",     "K_labor_KL_92",           "E_labor_rho_92"),
    ("Ownership (SNI07)", "K_ownership_KL_07",       "E2_ownership_rho_07"),
    ("Ownership (SNI02)", "K_ownership_KL_02",       "E2_ownership_rho_02"),
    ("Ownership (SNI92)", "K_ownership_KL_92",       "E2_ownership_rho_92"),
    ("NACE 4-digit",      "K_nace_KL_4d",            "H_nace_rho_4d"),
    ("NACE 3-digit",      "K_nace_KL_3d",            "H_nace_rho_3d"),
    ("NACE 2-digit",      "K_nace_KL_2d",            "H_nace_rho_2d"),
    ("NACE section",      "K_nace_KL_section",       "H_nace_rho_section"),
    ("Input-output",      "K_io_KL",                 "I_io_rho"),
    ("Concordance 02→07", "K_concordance_KL_02to07", "B_MRR_02to07"),
    ("Concordance 92→02", "K_concordance_KL_92to02", "B2_MRR_92to02"),
    ("Fused diffusion",   "K_fused_target_KL",       "J2_target_fidelity_rho"),
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
