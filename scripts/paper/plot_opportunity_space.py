"""
plot_opportunity_space.py
=========================
Paper figure: the learned opportunity space. A two-dimensional t-SNE
projection of the SNI 2007 4-digit industry embeddings, coloured by SNI 2007
section.

One run writes, from the same 2-D layout:

  result3_opportunity_space.{png,pdf}
      Overview at A4 text width (16 cm): the map with free space around it
      for close-up insets, and the section legend underneath.
  result3_opportunity_space_detail.{png,pdf}
      Detail sheet for cutting close-ups: the same layout on a much larger
      canvas (dots keep their size, so they lie far apart), every dot
      labelled with its SNI 2007 code. The PDF is vector and can be cropped
      at any zoom.
  result3_opportunity_space_locator.png, ..._detail (grid only in these)
      Overview and detail sheet with the same lettered grid, to find where a
      crop from the detail sheet sits on the overview.
  result3_opportunity_space_codes.csv
      Every code with its section, description and map position.
  result3_opportunity_space_neighbours.{csv,txt}
      Every code with its ten nearest neighbours in the embedding (cosine
      similarity), each written as "(code) name", for the close-up callouts.

Layout: t-SNE (cosine metric) of the 64-d embeddings, rotated so that its
widest spread is horizontal (t-SNE axes carry no meaning). Pairs of points
closer than MIN_SEPARATION are then pushed apart slightly; the shift is a
fraction of a dot on the overview. The detail sheet spreads dense clusters
further (--detail-separation, more horizontally than vertically since labels
are wide) inside the same frame, so that every code label sits next to its
own dot; its dots are therefore slightly displaced relative to the overview.

Inputs   EMBEDDING_DIR/{run}/embeddings_sni07.npz
         metadata/mapping_Sni_4digits_2007.json, metadata/mapping_Sni_section_2007.json
Outputs  results/paper/result3_opportunity_space*

Usage
-----
  python scripts/paper/plot_opportunity_space.py
  LATENT_DATA_ROOT=$PWD/data python scripts/paper/plot_opportunity_space.py
"""

import argparse
import json
import string
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.manifold import TSNE

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import EMBEDDING_DIR, METADATA_DIR, RESULTS_DIR
from latent_industry.common.sni_codes import SNI_2digits_to_sections

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--run-name", default="shared_cosine_v16")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--perplexity", type=float, default=30, help="t-SNE perplexity.")
parser.add_argument("--dpi", type=int, default=600, help="Resolution of the overview PNG.")
parser.add_argument("--detail-dpi", type=int, default=200, help="Resolution of the detail-sheet PNG.")
parser.add_argument("--detail-scale", type=float, default=0.85,
                    help="Detail sheet: inches per layout unit (the overview uses about 0.18).")
parser.add_argument("--detail-separation", type=float, default=0.5,
                    help="Detail sheet: minimum vertical distance between dots in layout units "
                         "(horizontally --detail-aspect times more); dense clusters spread out so labels fit.")
parser.add_argument("--detail-aspect", type=float, default=3.0,
                    help="Detail sheet: horizontal / vertical spacing ratio (labels are wider than tall).")
parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "paper")
args = parser.parse_args()

STEM = "result3_opportunity_space"
MIN_SEPARATION = 0.22       # layout units; about one dot diameter on the detail sheet
PAGE_WIDTH_IN = 16 / 2.54   # A4 text width (21 cm minus 2 × 2.5 cm margins)
DOT_SIZE = {"overview": 9, "detail": 400}        # marker area in pt²
DOT_EDGE_WIDTH = {"overview": 0.25, "detail": 1.3}
N_NEIGHBOURS = 10
LABEL_FONTSIZE = 20         # detail-sheet code labels; crops are scaled down ~5x into the insets
LABEL_WEIGHT = "bold"
GRID_STEP = 5.0             # layout units between locator grid lines

# Muted colours in the style of the reference figure. The largest sections get
# the most distinct hues; every pair differs by CIELAB ΔE > 16.
SECTION_COLORS = {
    "A": "#5DA349", "B": "#8C6A48", "C": "#E07A1F", "D": "#E9B730", "E": "#A9C25A",
    "F": "#B5473A", "G": "#2A9D8F", "H": "#3C7FC4", "I": "#F4A582", "J": "#1F3A7A",
    "K": "#7A58A8", "L": "#CDB98E", "M": "#86CBE0", "N": "#D47FAE", "O": "#62707C",
    "P": "#A9DDA3", "Q": "#C9375E", "R": "#BBA0D6", "S": "#8E8E52", "T": "#B3B3B3",
    "U": "#2B2B2B",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data and layout
# ═══════════════════════════════════════════════════════════════════════════════

data = np.load(EMBEDDING_DIR / args.run_name / "embeddings_sni07.npz", allow_pickle=True)
codes = [str(c) for c in data["codes"]]
vecs = data["embeddings"].astype(np.float64)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

sec_of_div = SNI_2digits_to_sections("SNI07")
sections = np.array([sec_of_div.get(c[:2], "") for c in codes], dtype=object)
with open(METADATA_DIR / "mapping_Sni_4digits_2007.json") as f:
    descriptions = json.load(f)
print(f"  {len(codes)} SNI 2007 codes, {int(np.sum(sections == ''))} without a section")

print("  t-SNE …")
xy = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed, metric="cosine",
          max_iter=1000, learning_rate="auto", init="pca").fit_transform(vecs)
xy -= xy.mean(axis=0)
_, _, vt = np.linalg.svd(xy, full_matrices=False)
xy = xy @ vt.T


def separate(points: np.ndarray, min_dist: float, iterations: int = 200) -> np.ndarray:
    """Push apart pairs closer than min_dist (each point moves half the deficit)."""
    p = points.copy()
    for _ in range(iterations):
        pairs = cKDTree(p).query_pairs(min_dist, output_type="ndarray")
        if len(pairs) == 0:
            break
        d = p[pairs[:, 1]] - p[pairs[:, 0]]
        dist = np.linalg.norm(d, axis=1)
        d[dist == 0] = np.random.default_rng(0).normal(size=(int(np.sum(dist == 0)), 2))
        dist = np.maximum(dist, 1e-9)
        push = (d / dist[:, None]) * ((min_dist - dist) / 2 * 1.05)[:, None]
        np.add.at(p, pairs[:, 0], -push)
        np.add.at(p, pairs[:, 1], push)
    return p


xy_sep = separate(xy, MIN_SEPARATION)
shift = np.linalg.norm(xy_sep - xy, axis=1)
print(f"  separation: {int(np.sum(shift > 1e-9))} points moved, max shift {shift.max():.3f} units "
      f"(median nearest-neighbour distance {np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]):.3f})")
xy = xy_sep

pad = 1.0
xlim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad)
ylim = (xy[:, 1].min() - pad, xy[:, 1].max() + pad)
span_x, span_y = xlim[1] - xlim[0], ylim[1] - ylim[0]

# Draw large sections first so that small ones stay visible on top.
sizes = pd.Series(sections).value_counts()
order = np.argsort([-sizes[s] for s in sections], kind="stable")
colors = np.array([SECTION_COLORS.get(s, "#cccccc") for s in sections], dtype=object)

# ── positions and nearest neighbours (embedding space) for callouts ─────────
def label(code: str) -> str:
    return f"({code}) {descriptions.get(code, '')}"


sim = vecs @ vecs.T
np.fill_diagonal(sim, -np.inf)
args.out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"code": codes, "section": sections,
              "description": [descriptions.get(c, "") for c in codes],
              "x": np.round(xy[:, 0], 4), "y": np.round(xy[:, 1], 4)}
             ).to_csv(args.out_dir / f"{STEM}_codes.csv", index=False)
nn_rows, nn_text = [], []
for i, c in enumerate(codes):
    nn = np.argsort(-sim[i])[:N_NEIGHBOURS]
    nn_rows.append({"industry": label(c),
                    **{f"neighbour_{k}": label(codes[j]) for k, j in enumerate(nn, start=1)}})
    nn_text += [label(c)] + [f"  {k}. {label(codes[j])}" for k, j in enumerate(nn, start=1)] + [""]
pd.DataFrame(nn_rows).to_csv(args.out_dir / f"{STEM}_neighbours.csv", index=False)
(args.out_dir / f"{STEM}_neighbours.txt").write_text("\n".join(nn_text), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

with open(METADATA_DIR / "mapping_Sni_section_2007.json") as f:
    section_names = json.load(f)
section_names.update({
    "D": "Electricity, gas, steam and air conditioning",
    "E": "Water supply, sewerage and waste management",
    "G": "Wholesale and retail trade",
    "I": "Accommodation and food services",
    "M": "Professional, scientific and technical activities",
    "N": "Administrative and support services",
    "O": "Public administration and defence",
    "Q": "Human health and social work",
    "T": "Households as employers",
    "U": "Extraterritorial organisations",
})

plt.rcParams.update({"font.family": "DejaVu Sans"})


def draw_dots(ax, kind: str, pos: np.ndarray) -> None:
    ax.scatter(pos[order, 0], pos[order, 1], c=list(colors[order]), s=DOT_SIZE[kind], alpha=0.95,
               zorder=3, edgecolors="white", linewidths=DOT_EDGE_WIDTH[kind])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_grid(ax, fontsize: float) -> None:
    """Lettered locator grid (columns A, B, …; rows 1, 2, … from the top)."""
    # Cell edges cover the whole map; cells cut by the map border are labelled
    # at the centre of their visible part.
    xs = np.arange(np.floor(xlim[0] / GRID_STEP), np.ceil(xlim[1] / GRID_STEP) + 1) * GRID_STEP
    ys = np.arange(np.ceil(ylim[1] / GRID_STEP), np.floor(ylim[0] / GRID_STEP) - 1, -1) * GRID_STEP
    style = dict(color="#9a9a9a", lw=0.6, ls="--", zorder=1)
    for x in xs[1:-1]:
        ax.axvline(x, **style)
    for y in ys[1:-1]:
        ax.axhline(y, **style)
    text = dict(fontsize=fontsize, color="#555555", ha="center", va="center", zorder=2)
    for i, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
        for j, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
            cx = (max(x0, xlim[0]) + min(x1, xlim[1])) / 2
            cy = (min(y0, ylim[1]) + max(y1, ylim[0])) / 2
            ax.text(cx, cy, f"{string.ascii_uppercase[i]}{j + 1}", alpha=0.35, **text)


# ═══════════════════════════════════════════════════════════════════════════════
# Overview
# ═══════════════════════════════════════════════════════════════════════════════

# True-size layout (inches): the figure is exactly the page text width, so all
# sizes are final. The map is centred with free space on the sides and above
# it for the close-up insets; the section legend runs underneath in three
# columns.
FIG_W = PAGE_WIDTH_IN
MAP_W, MAP_H = 4.6, 2.95                      # map box (equal aspect inside)
SIDE_MARGIN = (FIG_W - MAP_W) / 2             # ≈ 0.85 in on each side
TOP_MARGIN = 0.6
LEGEND_H = 1.1                                # legend block incl. title
LEGEND_GAP = 0.45                             # between map and legend
FIG_H = TOP_MARGIN + MAP_H + LEGEND_GAP + LEGEND_H
MAP_BOX = [SIDE_MARGIN / FIG_W, (LEGEND_H + LEGEND_GAP) / FIG_H, MAP_W / FIG_W, MAP_H / FIG_H]
LEGEND_TOP = LEGEND_H / FIG_H
LEGEND_WRAP = 44                              # characters per legend line
LEGEND_FONTSIZE = 5.5


def overview(grid: bool) -> plt.Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(MAP_BOX)
    draw_dots(ax, "overview", xy)
    if grid:
        draw_grid(ax, fontsize=3.5)
    present = [s for s in SECTION_COLORS if s in set(sections)]
    handles = [
        Line2D([0], [0], marker="o", ls="", markersize=3.4, markerfacecolor=SECTION_COLORS[s],
               markeredgecolor="white", markeredgewidth=0.25,
               label=f"{s}  " + "\n     ".join(textwrap.wrap(section_names.get(s, s), LEGEND_WRAP)))
        for s in present
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, LEGEND_TOP), frameon=False,
               ncol=3, fontsize=LEGEND_FONTSIZE, handletextpad=0.2, labelspacing=0.35,
               columnspacing=1.2, title="Section (SNI 2007)", title_fontsize=6)
    return fig


for grid, name in [(False, STEM), (True, f"{STEM}_locator")]:
    fig = overview(grid)
    fig.savefig(args.out_dir / f"{name}.png", dpi=args.dpi)
    if not grid:
        fig.savefig(args.out_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  → {args.out_dir / name}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Detail sheet
# ═══════════════════════════════════════════════════════════════════════════════

def place_labels(points_pt: np.ndarray, texts: list[str], radius_pt: float,
                 fontsize: float) -> list[tuple[float, float, str, str]]:
    """Greedy label placement: try eight positions around each dot and take the
    first that overlaps neither another label nor another dot (in points)."""
    char_w, height = 0.68 * fontsize, 1.0 * fontsize   # bold digits are wider
    gap = radius_pt + 1.5
    candidates = [  # (dx, dy, ha, va) relative to the dot centre
        (gap, 0, "left", "center"), (-gap, 0, "right", "center"),
        (0, gap, "center", "bottom"), (0, -gap, "center", "top"),
        (gap * 0.75, gap * 0.75, "left", "bottom"), (gap * 0.75, -gap * 0.75, "left", "top"),
        (-gap * 0.75, gap * 0.75, "right", "bottom"), (-gap * 0.75, -gap * 0.75, "right", "top"),
    ]

    def box(x, y, w, ha, va):
        x0 = x if ha == "left" else x - w if ha == "right" else x - w / 2
        y0 = y if va == "bottom" else y - height if va == "top" else y - height / 2
        return np.array([x0, y0, x0 + w, y0 + height])

    dots = np.column_stack([points_pt - radius_pt, points_pt + radius_pt])
    placed: list[np.ndarray] = []
    out = []
    n_fallback = 0

    def overlap(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    tree = cKDTree(points_pt)
    for i in np.argsort(-np.array([len(tree.query_ball_point(p, 40)) for p in points_pt])):
        w = char_w * len(texts[i])
        near = [j for j in tree.query_ball_point(points_pt[i], w + 3 * gap) if j != i]
        near_labels = [q for q in placed if abs(q[0] - points_pt[i, 0]) < 3 * w and abs(q[1] - points_pt[i, 1]) < 3 * w]
        best, best_cost = None, np.inf
        for dx, dy, ha, va in candidates:   # first free position, else the least-overlapping one
            b = box(points_pt[i, 0] + dx, points_pt[i, 1] + dy, w, ha, va)
            cost = sum(overlap(b, dots[j]) for j in near) + sum(overlap(b, q) for q in near_labels)
            if cost < best_cost:
                best, best_cost = (b, dx, dy, ha, va), cost
            if cost == 0:
                break
        chosen = best
        n_fallback += best_cost > 0
        placed.append(chosen[0])
        out.append((i, chosen[1], chosen[2], chosen[3], chosen[4]))
    print(f"  detail labels: {n_fallback} of {len(texts)} could not avoid overlap")
    return out


# Detail positions: the overview layout with dense clusters pushed further apart
# so that dots and labels fit. After each spreading pass the layout is fitted
# back into the overview's bounding box, so the detail sheet keeps the same
# frame and aspect ratio as the overview.
stretch = np.array([1 / args.detail_aspect, 1.0])     # separate in a squashed frame, then undo
lo, hi = xy.min(axis=0), xy.max(axis=0)
xy_detail = xy.copy()
for _ in range(10):
    xy_detail = separate(xy_detail * stretch, args.detail_separation, iterations=500) / stretch
    d_lo, d_hi = xy_detail.min(axis=0), xy_detail.max(axis=0)
    xy_detail = lo + (xy_detail - d_lo) * (hi - lo) / (d_hi - d_lo)
shift = np.linalg.norm(xy_detail - xy, axis=1)
print(f"  detail separation: {int(np.sum(shift > 1e-9))} points moved, max shift {shift.max():.3f} units; "
      f"closest pair (spread metric) {cKDTree(xy_detail * stretch).query(xy_detail * stretch, k=2)[0][:, 1].min():.3f}")


def detail(grid: bool) -> plt.Figure:
    width, height = span_x * args.detail_scale, span_y * args.detail_scale
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    draw_dots(ax, "detail", xy_detail)
    if grid:
        draw_grid(ax, fontsize=40)
    pt_per_unit = args.detail_scale * 72
    points_pt = np.column_stack([(xy_detail[:, 0] - xlim[0]) * pt_per_unit,
                                 (xy_detail[:, 1] - ylim[0]) * pt_per_unit])
    radius_pt = np.sqrt(DOT_SIZE["detail"]) / 2
    for i, dx, dy, ha, va in place_labels(points_pt, codes, radius_pt, LABEL_FONTSIZE):
        ax.annotate(codes[i], xy_detail[i], xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=LABEL_FONTSIZE, fontweight=LABEL_WEIGHT, color="#222222", zorder=4)
    return fig


for grid, name in [(False, f"{STEM}_detail"), (True, f"{STEM}_locator_detail")]:
    fig = detail(grid)
    fig.savefig(args.out_dir / f"{name}.png", dpi=args.detail_dpi)
    if not grid:
        fig.savefig(args.out_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  → {args.out_dir / name}.png")
print(f"  → {args.out_dir / STEM}_codes.csv\n  → {args.out_dir / STEM}_neighbours.csv / .txt")
