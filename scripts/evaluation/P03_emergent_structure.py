"""
P03_emergent_structure.py
=========================
Result 3: emergent structure of the trained space, beyond any single source.

Sections
--------
A   Hierarchy preservation: cosine similarity by shared-prefix level
    (same group > same division > same section > different section)
B   Cross-version retrieval SNI02 → SNI07 (Recall@k, MRR against the SCB
    correspondence table)
B2  Cross-version retrieval SNI92 → SNI02
C   Case study: neighbourhood of a code that splits across versions (peat extraction)
D   t-SNE of all codes coloured by version, and the cosine similarity of
    SCB-aligned vs random cross-version pairs

The paper's opportunity-space figure (t-SNE coloured by section) is produced
by scripts/paper/plot_opportunity_space.py.

Outputs: results/embedding_eval/{run}/ (figures and log).

Usage (main run)
----------------
  python scripts/evaluation/P03_emergent_structure.py --run-name shared_cosine_v16
"""

import argparse
import random
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from _shared_eval_context import (
    SEED,
    add_common_args, build_base_context, load_crosswalks, open_log,
    exhaustive_pairs_by_level, levels, level_labels, average_precision, Ks,
)

parser = argparse.ArgumentParser()
# P03 does not read the graph files, so no --graph-name.
add_common_args(parser, graph_name=False)
args, _ = parser.parse_known_args()

ctx = build_base_context(args)
load_crosswalks(ctx)

# Unpack onto locals for readability, mirroring the original module-level globals.
codes_07, codes_02, codes_92 = ctx.codes_07, ctx.codes_02, ctx.codes_92
idx_07, idx_02, idx_92       = ctx.idx_07, ctx.idx_02, ctx.idx_92
vecs_07, vecs_02, vecs_92    = ctx.vecs_07, ctx.vecs_02, ctx.vecs_92
sni07, sni02, sni92          = ctx.sni07, ctx.sni02, ctx.sni92
section_map_07               = ctx.section_map_07
section_map_02               = ctx.section_map_02
section_map_92               = ctx.section_map_92
eval_dir                     = ctx.eval_dir
mapping_02_07, true_targets     = ctx.mapping_02_07, ctx.true_targets
mapping_92_02, true_targets_92  = ctx.mapping_92_02, ctx.true_targets_92

# Single advancing RNG, mirroring the original script's module-level `rng` —
# A's diff_section/same_section sampling and D's aligned-vs-random random
# pairing are its only two consumers, in that order, exactly as before.
rng = random.Random(SEED)

# ── Redirect stdout to tee (log file + terminal) ───────────────────────────────
_tee, _log_file = open_log(eval_dir, args.run_name, "P03")
sys.stdout = _tee

# ══════════════════════════════════════════════════════════════════════════════
# A. Hierarchy preservation
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("A. Hierarchy preservation (SNI07)")
print("═" * 60)

# Note: vecs_07 is L2-normalised, so dot product = cosine similarity.
# Atoms are 4-digit → finest level is same_3d (within NACE group).
level_sims = {}
for lv in levels:
    sims = exhaustive_pairs_by_level(codes_07, vecs_07, lv, idx_07=idx_07,
                                      section_map_07=section_map_07, rng=rng)
    level_sims[lv] = sims
    print(f"  {level_labels[lv].replace(chr(10),' '):<38}  n={len(sims):5d}  mean sim={np.mean(sims):.4f}  ±{np.std(sims):.4f}")

means = [np.mean(level_sims[lv]) for lv in levels]
is_monotone = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
print(f"\n  Hierarchy monotone (3d > 2d > section): {'YES ✓' if is_monotone else 'NO — model needs more training'}")

# ── Tree-distance correlation — the "Hierarchy" row's source-specific metric
# (ordinal, not KL). Spearman rho between cosine
# DISTANCE and hierarchy tree distance (1=same 3d group … 4=different
# section), pooled across all four levels so no single level dominates.
same_section_sims = exhaustive_pairs_by_level(codes_07, vecs_07, "same_section", idx_07=idx_07,
                                               section_map_07=section_map_07, rng=rng)
treedist_pairs = (
    [(1, s) for s in level_sims["same_3d"]]
    + [(2, s) for s in level_sims["same_2d"]]
    + [(3, s) for s in same_section_sims]
    + [(4, s) for s in level_sims["diff_section"]]
)
_A_treedist_rho = None
if len(treedist_pairs) >= 10:
    td, sim_td = zip(*treedist_pairs)
    treedist_rho, treedist_p = spearmanr(1.0 - np.array(sim_td), np.array(td))
    print(f"  Tree-distance correlation (cosine distance ~ tree distance):  "
          f"rho={treedist_rho:.4f}  (p={treedist_p:.2e}, n={len(treedist_pairs)})")
    _A_treedist_rho = round(float(treedist_rho), 4)

# Paper figure: short "A./B." headlines only, no overall suptitle — the
# monotonicity call (is_monotone, printed above) and the "closest → farthest"
# level ordering belong in the paper's figure caption/log, not baked into
# the image as alarmist "training failure" text.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# A. Distribution of pairwise cosine similarity, by hierarchy level
data = [level_sims[lv] for lv in levels]
axes[0].boxplot(data, tick_labels=[level_labels[lv] for lv in levels],
                patch_artist=True, notch=False,
                boxprops=dict(facecolor="steelblue", alpha=0.6),
                medianprops=dict(color="crimson", linewidth=2),
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
axes[0].set_xlabel("Hierarchy level (closest → farthest)")
axes[0].set_ylabel("Pairwise cosine similarity")
axes[0].set_title("A. Similarity distribution by hierarchy level")
axes[0].yaxis.grid(True, linestyle="--", alpha=0.4); axes[0].set_axisbelow(True)

# B. Mean similarity per level — should decrease monotonically with tree
# distance if the embedding preserves the official SNI hierarchy ordinally.
x = np.arange(len(levels))
stds = [np.std(level_sims[lv]) for lv in levels]
colors = ["#2c7bb6", "#fdae61", "#d7191c"]
axes[1].bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=4, edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels([level_labels[lv] for lv in levels])
axes[1].set_xlabel("Hierarchy level (closest → farthest)")
axes[1].set_ylabel("Mean cosine similarity")
axes[1].set_title("B. Mean similarity by hierarchy level")
axes[1].yaxis.grid(True, linestyle="--", alpha=0.4); axes[1].set_axisbelow(True)
counts = [len(level_sims[lv]) for lv in levels]
for xi, (m, c) in enumerate(zip(means, counts)):
    axes[1].text(xi, m + stds[xi] + 0.001, f"n={c}", ha="center", fontsize=8, color="dimgray")

plt.tight_layout()
fig.savefig(eval_dir / f"{args.run_name}_A_hierarchy_preservation.png", dpi=150, bbox_inches="tight")
print(f"  → {eval_dir / f'{args.run_name}_A_hierarchy_preservation.png'}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# B. Cross-version retrieval: Recall@k and MRR
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("B. Cross-version retrieval (SNI02 → SNI07)")
print("═" * 60)

recall_at_k = {k: [] for k in Ks}
mrr_scores  = []
map_scores  = []

# Precompute SNI07 matrix for bulk retrieval
V07 = vecs_07  # (n_07, dim)

for src_code, targets in true_targets.items():
    q = vecs_02[[idx_02[src_code]]]        # (1, dim)
    sims = (q @ V07.T).ravel()             # cosine similarity to all SNI07 codes
    ranked_idx = np.argsort(-sims)         # descending
    ranked_codes = [codes_07[i] for i in ranked_idx]

    # Reciprocal rank of first hit
    for rank, c in enumerate(ranked_codes, start=1):
        if c in targets:
            mrr_scores.append(1.0 / rank)
            break
    else:
        mrr_scores.append(0.0)

    map_scores.append(average_precision(ranked_codes, targets))

    for k in Ks:
        hit = any(c in targets for c in ranked_codes[:k])
        recall_at_k[k].append(float(hit))

print(f"  SNI02 codes evaluated: {len(true_targets)}")
print(f"  Mean Reciprocal Rank (MRR):  {np.mean(mrr_scores):.4f}")
print(f"  Mean Average Precision (MAP): {np.mean(map_scores):.4f}  "
      f"(scores against all true partners, not just the first hit)")
for k in Ks:
    print(f"  Recall@{k:<3d}:                 {np.mean(recall_at_k[k]):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar([f"R@{k}" for k in Ks], [np.mean(recall_at_k[k]) for k in Ks],
            color="steelblue", alpha=0.8, edgecolor="white")
axes[0].set_ylim(0, 1); axes[0].set_ylabel("Recall")
axes[0].set_title("Recall@k  (SNI02 query → SNI07 targets)")
axes[0].yaxis.grid(True, linestyle="--", alpha=0.4); axes[0].set_axisbelow(True)
axes[0].axhline(np.mean(mrr_scores), color="crimson", linestyle="--", linewidth=1.5,
                label=f"MRR={np.mean(mrr_scores):.3f}")
axes[0].legend()

axes[1].hist(mrr_scores, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
axes[1].set_xlabel("Reciprocal rank of first hit")
axes[1].set_ylabel("Number of SNI02 codes")
axes[1].set_title("MRR distribution")
axes[1].yaxis.grid(True, linestyle="--", alpha=0.4); axes[1].set_axisbelow(True)
plt.tight_layout()
fig.savefig(eval_dir / f"{args.run_name}_B_retrieval_metrics.png", dpi=150, bbox_inches="tight")
print(f"  → {eval_dir / f'{args.run_name}_B_retrieval_metrics.png'}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# B2. Cross-version retrieval: SNI92 → SNI02
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("B2. Cross-version retrieval (SNI92 → SNI02)")
print("═" * 60)

recall_at_k_92 = {k: [] for k in Ks}
mrr_scores_92  = []
map_scores_92  = []

V02 = vecs_02  # (n_02, dim)

for src_code, targets in true_targets_92.items():
    q = vecs_92[[idx_92[src_code]]]        # (1, dim)
    sims = (q @ V02.T).ravel()             # cosine similarity to all SNI02 codes
    ranked_idx = np.argsort(-sims)         # descending
    ranked_codes = [codes_02[i] for i in ranked_idx]

    # Reciprocal rank of first hit
    for rank, c in enumerate(ranked_codes, start=1):
        if c in targets:
            mrr_scores_92.append(1.0 / rank)
            break
    else:
        mrr_scores_92.append(0.0)

    map_scores_92.append(average_precision(ranked_codes, targets))

    for k in Ks:
        hit = any(c in targets for c in ranked_codes[:k])
        recall_at_k_92[k].append(float(hit))

print(f"  SNI92 codes evaluated: {len(true_targets_92)}")
print(f"  Mean Reciprocal Rank (MRR):  {np.mean(mrr_scores_92):.4f}")
print(f"  Mean Average Precision (MAP): {np.mean(map_scores_92):.4f}")
for k in Ks:
    print(f"  Recall@{k:<3d}:                 {np.mean(recall_at_k_92[k]):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar([f"R@{k}" for k in Ks], [np.mean(recall_at_k_92[k]) for k in Ks],
            color="steelblue", alpha=0.8, edgecolor="white")
axes[0].set_ylim(0, 1); axes[0].set_ylabel("Recall")
axes[0].set_title("Recall@k  (SNI92 query → SNI02 targets)")
axes[0].yaxis.grid(True, linestyle="--", alpha=0.4); axes[0].set_axisbelow(True)
axes[0].axhline(np.mean(mrr_scores_92), color="crimson", linestyle="--", linewidth=1.5,
                label=f"MRR={np.mean(mrr_scores_92):.3f}")
axes[0].legend()

axes[1].hist(mrr_scores_92, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
axes[1].set_xlabel("Reciprocal rank of first hit")
axes[1].set_ylabel("Number of SNI92 codes")
axes[1].set_title("MRR distribution (SNI92 → SNI02)")
axes[1].yaxis.grid(True, linestyle="--", alpha=0.4); axes[1].set_axisbelow(True)
plt.tight_layout()
fig.savefig(eval_dir / f"{args.run_name}_B2_retrieval_92_02.png", dpi=150, bbox_inches="tight")
print(f"  → {eval_dir / f'{args.run_name}_B2_retrieval_92_02.png'}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# C. Case studies: codes that split/merge across versions
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("C. Case studies — splits across versions")
print("═" * 60)

CASE_STUDIES = [
    # (SNI02_4d_code, description_snippet)
    ("1030", "Peat extraction"),
    ("7487", "Other business activities n.e.c."),
    ("3663", "Other manufacturing n.e.c."),
]


def print_case(sni02_code: str, label: str) -> None:
    if sni02_code not in idx_02:
        print(f"  {sni02_code}: not in SNI02 embedding index — skip")
        return

    q     = vecs_02[[idx_02[sni02_code]]]
    sims  = (q @ V07.T).ravel()
    top5  = np.argsort(-sims)[:10]
    true  = true_targets.get(sni02_code, set())

    print(f"\n  SNI02 {sni02_code}: {sni02.get(sni02_code, '?')}")
    print(f"  True SNI07 targets ({len(true)}):")
    for t in sorted(true):
        sec = section_map_07.get(t[:2], "?")
        rank = int(np.where(np.argsort(-sims) == idx_07[t])[0][0]) + 1 if t in idx_07 else -1
        print(f"    {t} [{sec}] rank={rank:4d}  {sni07.get(t, '?')[:60]}")

    print(f"  Top-10 nearest SNI07 codes:")
    for i in top5:
        c   = codes_07[i]
        sec = section_map_07.get(c[:2], "?")
        tag = " ← TRUE" if c in true else ""
        print(f"    {c} [{sec}] sim={sims[i]:.4f}  {sni07.get(c, '?')[:55]}{tag}")


for code, label in CASE_STUDIES:
    print_case(code, label)


# ── Visualise the peat neighbourhood ──────────────────────────────────────────

peat_02  = "1030"   # SNI02 4d: peat extraction
peat_07s = ["0892", "0990", "2015"]   # SNI07 4d targets

if peat_02 in idx_02 and all(c in idx_07 for c in peat_07s):
    q       = vecs_02[[idx_02[peat_02]]]
    sims_q  = (q @ V07.T).ravel()
    top_n   = 20
    top_idx = np.argsort(-sims_q)[:top_n]

    # Collect all vectors: query + top-N SNI07 neighbours
    labels_plot = [f"SNI02\n{peat_02}\n(query)"]
    vecs_plot   = [vecs_02[idx_02[peat_02]]]
    colors_plot = ["red"]
    markers_p   = ["*"]

    for i in top_idx:
        c      = codes_07[i]
        is_tgt = c in peat_07s
        labels_plot.append(f"{'→ ' if is_tgt else ''}{c}")
        vecs_plot.append(vecs_07[i])
        colors_plot.append("darkorange" if is_tgt else "steelblue")
        markers_p.append("D" if is_tgt else "o")

    mat = np.vstack(vecs_plot)
    tsne_local = TSNE(n_components=2, perplexity=min(5, len(mat) - 1),
                      random_state=SEED, max_iter=1000)
    xy = tsne_local.fit_transform(mat)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (x, y) in enumerate(xy):
        ax.scatter(x, y, color=colors_plot[i], marker=markers_p[i],
                   s=200 if markers_p[i] != "o" else 80,
                   zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(labels_plot[i], (x, y),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7.5, color=colors_plot[i])

    ax.set_title(
        f"Neighbourhood of SNI02 {peat_02}  ({sni02.get(peat_02, 'peat extraction')})\n"
        "Red ★ = query  |  Orange ◆ = true SNI07 targets  |  Blue = nearest neighbours"
    )
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(eval_dir / f"{args.run_name}_C_peat_neighbourhood.png", dpi=150, bbox_inches="tight")
    print(f"\n  → {eval_dir / f'{args.run_name}_C_peat_neighbourhood.png'}")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# D. t-SNE visualisation — all codes, all versions
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("D. t-SNE visualisation")
print("═" * 60)

# Subsample for speed if large
MAX_PER_VERSION = 600
rng_np = np.random.default_rng(SEED)

def subsample(codes, vecs, n):
    if len(codes) <= n:
        return codes, vecs
    idx = rng_np.choice(len(codes), n, replace=False)
    return [codes[i] for i in idx], vecs[idx]

sc_07, sv_07 = subsample(codes_07, vecs_07, MAX_PER_VERSION)
sc_02, sv_02 = subsample(codes_02, vecs_02, MAX_PER_VERSION)
sc_92, sv_92 = subsample(codes_92, vecs_92, MAX_PER_VERSION)

all_vecs_tsne   = np.vstack([sv_07, sv_02, sv_92])
version_labels  = (["SNI07"] * len(sc_07) + ["SNI02"] * len(sc_02) + ["SNI92"] * len(sc_92))
all_codes_tsne  = sc_07 + sc_02 + sc_92

print(f"  Running t-SNE on {len(all_vecs_tsne)} codes …")
tsne = TSNE(n_components=2, perplexity=40, random_state=SEED, max_iter=1000, learning_rate="auto")
xy   = tsne.fit_transform(all_vecs_tsne)

# ── Plot 1: coloured by version ────────────────────────────────────────────────
ver_colors = {"SNI07": "#2c7bb6", "SNI02": "#d7191c", "SNI92": "#1a9641"}
fig, ax = plt.subplots(figsize=(10, 8))
for ver, col in ver_colors.items():
    mask = np.array([v == ver for v in version_labels])
    ax.scatter(xy[mask, 0], xy[mask, 1], c=col, s=10, alpha=0.45,
               label=ver, edgecolors="none")

# Annotate peat codes if they landed in the subsample
peat_all = [("02", peat_02)] + [("07", c) for c in peat_07s]
for ver_tag, code in peat_all:
    version = f"SNI{ver_tag.replace('07','07').replace('02','02')}"
    full_ver = "SNI07" if ver_tag == "07" else "SNI02"
    sc_list  = sc_07 if ver_tag == "07" else sc_02
    if code in sc_list:
        pos = sc_list.index(code) + (len(sc_07) if ver_tag == "02" else 0)
        ax.scatter(xy[pos, 0], xy[pos, 1], c="gold", s=150, marker="*",
                   edgecolors="black", linewidths=0.7, zorder=5)
        ax.annotate(f"{full_ver}/{code}", (xy[pos, 0], xy[pos, 1]),
                    fontsize=8, xytext=(6, 3), textcoords="offset points")

patches = [mpatches.Patch(color=c, label=v) for v, c in ver_colors.items()]
patches.append(mpatches.Patch(color="gold", label="Peat example"))
ax.legend(handles=patches, loc="upper right")
ax.set_title("t-SNE of SNI code embeddings — coloured by version\n(★ = peat split example)")
ax.axis("off")
plt.tight_layout()
fig.savefig(eval_dir / f"{args.run_name}_D_tsne_by_version.png", dpi=150, bbox_inches="tight")
print(f"  → {eval_dir / f'{args.run_name}_D_tsne_by_version.png'}")
plt.close()

# ── Plot 2: cosine similarity distribution — aligned pairs vs random ───────────
print("\n  Aligned vs random cosine similarity distribution …")

aligned_sims = []
for _, row in mapping_02_07.iterrows():
    src, tgt = row["SNI02"], row["SNI07"]
    if src in idx_02 and tgt in idx_07:
        sim = float(cosine_similarity(vecs_02[[idx_02[src]]], vecs_07[[idx_07[tgt]]])[0, 0])
        aligned_sims.append(sim)

n_rand = len(aligned_sims)
rand_sims = []
for _ in range(n_rand):
    s = rng.choice(codes_02)
    t = rng.choice(codes_07)
    if s in idx_02 and t in idx_07:
        sim = float(cosine_similarity(vecs_02[[idx_02[s]]], vecs_07[[idx_07[t]]])[0, 0])
        rand_sims.append(sim)

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(-0.1, 1.0, 50)
ax.hist(rand_sims,    bins=bins, alpha=0.55, color="steelblue", label=f"Random pairs (n={len(rand_sims)})", density=True)
ax.hist(aligned_sims, bins=bins, alpha=0.55, color="crimson",   label=f"SCB-aligned pairs (n={len(aligned_sims)})", density=True)
ax.axvline(np.mean(rand_sims),    color="steelblue", linestyle="--", linewidth=1.5, label=f"mean random={np.mean(rand_sims):.3f}")
ax.axvline(np.mean(aligned_sims), color="crimson",   linestyle="--", linewidth=1.5, label=f"mean aligned={np.mean(aligned_sims):.3f}")
ax.set_xlabel("Cosine similarity (SNI02 vs SNI07)")
ax.set_ylabel("Density")
ax.set_title("Cross-version similarity: SCB-aligned pairs vs random pairs")
ax.legend(fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
plt.tight_layout()
fig.savefig(eval_dir / f"{args.run_name}_D_aligned_vs_random_similarity.png", dpi=150, bbox_inches="tight")
print(f"  → {eval_dir / f'{args.run_name}_D_aligned_vs_random_similarity.png'}")
plt.close()

print(f"\nAll outputs written to {eval_dir}/")
print("Done.")

sys.stdout = sys.__stdout__
_log_file.close()
