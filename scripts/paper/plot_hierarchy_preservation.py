"""
plot_hierarchy_preservation.py
==============================
Paper figure (single column) and table: preservation of the official industry
hierarchy in the learned space. For all pairs of SNI 2007 four-digit
industries, cosine similarity by hierarchy level:

  same three-digit group
  same two-digit division, different group
  same section, different division
  different section

Inputs   EMBEDDING_DIR/{run}/embeddings_sni07.npz
Outputs  results/paper/result3_hierarchy_preservation.{png,pdf}   single-column figure
         results/paper/result3_hierarchy_table.tex                 table alternative

The figure is drawn at its printed size: one PNAS column (8.7 cm) wide,
7 pt text.

Usage
-----
  python scripts/paper/plot_hierarchy_preservation.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import EMBEDDING_DIR, RESULTS_DIR
from latent_industry.common.sni_codes import SNI_2digits_to_sections

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--run-name", default="shared_cosine_v16")
parser.add_argument("--dpi", type=int, default=600)
args = parser.parse_args()

TIERS = [  # (label in figure, label in table, colour)
    ("Same three-digit group", "Same three-digit group", "#1f4e79"),
    ("Same division,\nother group", "Same division, different group", "#3c7fc4"),
    ("Same section,\nother division", "Same section, different division", "#86b6e0"),
    ("Different section", "Different section", "#c9d9ea"),
]

data = np.load(EMBEDDING_DIR / args.run_name / "embeddings_sni07.npz", allow_pickle=True)
vecs = data["embeddings"].astype(np.float64)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
codes = [str(c) for c in data["codes"]]
section = SNI_2digits_to_sections("SNI07")

iu = np.triu_indices(len(codes), 1)
sim = (vecs @ vecs.T)[iu]
d3 = np.array([c[:3] for c in codes])
d2 = np.array([c[:2] for c in codes])
sec = np.array([section[c[:2]] for c in codes])
a, b = iu
tier = np.select([d3[a] == d3[b], d2[a] == d2[b], sec[a] == sec[b]], [0, 1, 2], default=3)
groups = [sim[tier == t] for t in range(len(TIERS))]
for (label, _, _), g in zip(TIERS, groups):
    print(f"  {label.replace(chr(10), ' '):<34} pairs {len(g):>7,}  mean {g.mean():.3f}  "
          f"sd {g.std():.3f}  median {np.median(g):.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure: one horizontal box per level (whiskers 5th–95th percentile), mean marked
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
fig, ax = plt.subplots(figsize=(8.7 / 2.54, 2.1))
positions = np.arange(len(TIERS))[::-1]
bp = ax.boxplot(groups, positions=positions, vert=False, widths=0.6, whis=(5, 95),
                showfliers=False, patch_artist=True,
                medianprops=dict(color="black", lw=0.9), whiskerprops=dict(lw=0.7),
                capprops=dict(lw=0.7), boxprops=dict(lw=0.7))
for patch, (_, _, color) in zip(bp["boxes"], TIERS):
    patch.set_facecolor(color)
    patch.set_edgecolor("black")
for y, g in zip(positions, groups):
    ax.plot(g.mean(), y, marker="D", color="white", mec="black", ms=4, mew=0.7, zorder=4)
ax.set_yticks(positions)
ax.set_yticklabels([t[0] for t in TIERS])
ax.axvline(0, color="black", lw=0.5, ls="--")
ax.set_xlabel("Cosine similarity of SNI 2007 industry pairs")
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
out = RESULTS_DIR / "paper" / "result3_hierarchy_preservation"
fig.savefig(out.with_suffix(".png"), dpi=args.dpi)
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)
print(f"  → {out}.png / .pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Table alternative
# ═══════════════════════════════════════════════════════════════════════════════

lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\small",
    r"\caption{Preservation of the official industry hierarchy. Cosine similarity between all "
    r"pairs of SNI 2007 four-digit industries in the learned representation, by the closest "
    r"shared level of the classification.}",
    r"\label{tab:hierarchy}",
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"Hierarchy level & Pairs & Mean & SD & Median \\",
    r"\midrule",
]
for (_, label, _), g in zip(TIERS, groups):
    n = f"{len(g):,}".replace(",", r"\,")
    lines.append(f"{label} & {n} & {g.mean():.3f} & {g.std():.3f} & {np.median(g):.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
tex = RESULTS_DIR / "paper" / "result3_hierarchy_table.tex"
tex.write_text("\n".join(lines) + "\n")
print(f"  → {tex}")
