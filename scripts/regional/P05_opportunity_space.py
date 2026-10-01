"""
P05_opportunity_space.py
=========================
Builds the paper's companion interactive figure: a t-SNE map of every SNI
industry (all three classification vintages, embedded jointly in ONE t-SNE
fit so positions are directly comparable across eras), with a dropdown to
pick a municipality (by name) and a slider to pick a year (1990-2019). For
the selected (municipality, year), every industry the municipality ALREADY
HAS is colored by its relatedness density to the rest of that municipality's
existing employment structure — the same D^{A,log} baseline definition used
by P04_entry.py / P04_exit.py / etc. (log-employment-weighted mean cosine
similarity, leave-one-out) — i.e. how well-embedded that existing industry
is in the local mix. Industries the municipality does NOT yet have are
shown as flat gray dots (no density on the colour scale). A background
field (client-side kernel-smoothed heatmap, see computeField() in the JS)
interpolates present industries' density across the whole map for a
continuous "coherence landscape" read, with the individual dots on top for
per-industry hover.

Density is still computed for EVERY industry, present or not (unlike the
P04_*.py candidate-entry panels, which only keep industries ABSENT at year
t — that's the regression's outcome universe) — the leave-one-out formula
gives even an absent industry a well-defined value, and it's kept in the
embedded data / shown on hover for context, just not on the colour scale
or in the field (an absent industry's "closeness to the existing mix" is a
different, also-interesting question, but not what this view is for).

t-SNE runs on the FULL code set per version (no 600/version subsample, unlike
P03_emergent_structure.py's D plot / the existing *_D_tsne_interactive.html)
so the published map is complete.

Output is one self-contained HTML with a hand-written JS layer (not
Plotly's built-in slider/animation frames, which only animate a single
dimension and can't compose an independent dropdown + slider): the
per-(period, year, municipality, industry) density/presence arrays are
quantized (density -> uint16, presence -> uint8) and embedded client-side
as base64; dropdown/slider changes slice the right rows and call
Plotly.restyle() directly. Written to
{INTERACTIVE_PLOTS_DIR}/{run_name}_opportunity_space.html — several MB,
same "doesn't belong in the git-tracked results/ tree" rule as the other
interactive plots (see latent_industry.common.paths).

This script reads the full individual-level SCB microdata
(individ_cleaned_1990-2019.parquet) and loops the density computation over
all 30 years — run it via a SLURM job (see
CPU_regional_opportunity_space.sh), not directly in an interactive shell.

Usage
-----
  python scripts/regional/P05_opportunity_space.py
  python scripts/regional/P05_opportunity_space.py --run-name shared_cosine_v16
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from sklearn.manifold import TSNE

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import (
    EMBEDDING_DIR, METADATA_DIR, INTERACTIVE_PLOTS_DIR, PROCESSED_DATA_DIR,
)
from latent_industry.common.municipalities import harmonization_table, load_municipality_crosswalk
from latent_industry.common.sni_codes import SNI_2digits_to_sections
from latent_industry.regional.density import (
    load_sni_embeddings, cosine_matrix, employment_weighted_density,
    empl_matrix as _shared_empl_matrix,
)

DATA_ROOT    = Path("/lunarc/nobackup/projects/lu2025-17-39")
INDIVID_PATH = DATA_ROOT / "SCB" / "Preprocessed" / "individ_cleaned_1990-2019.parquet"
EMBED_BASE   = EMBEDDING_DIR

SEED             = 42
DEFAULT_RUN_NAME = "shared_cosine_v16"
PRESENT_OPACITY  = 1.0
ABSENT_OPACITY   = 0.6    # industries the municipality doesn't have yet: flat gray, not on the density scale
ABSENT_COLOR     = "#c7c7c7"
COLORSCALE       = "Viridis"
DEFAULT_MUNI     = "0180"  # Stockholm, if present in the crosswalk — else first alphabetically
DEFAULT_YEAR     = 2019
GRID_N           = 70     # background field resolution (GRID_N x GRID_N), one shared grid for all periods
GRID_MARGIN      = 0.08   # fractional padding around the joint t-SNE bounding box
FIELD_BANDWIDTH_FRAC = 1 / 25  # kernel bandwidth as a fraction of the grid's bounding-box diagonal

# Actual calendar years each classification vintage's employment data covers
# (not the P04_*.py candidate-panel ranges, which stop one year early since
# those need a t+1 outcome — this script has no such constraint).
PERIOD_SPECS = [
    dict(key="sni92", label="SNI92", year_file="1992", years=list(range(1990, 2002))),
    dict(key="sni02", label="SNI02", year_file="2002", years=list(range(2002, 2007))),
    dict(key="sni07", label="SNI07", year_file="2007", years=list(range(2007, 2020))),
]

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                    help="Embedding run (from train_sentence_embedding.py --run-name).")
parser.add_argument("--tsne-perplexity", type=float, default=40.0)
parser.add_argument("--from-cache", default=False, action="store_true",
                    help="Skip the SCB microdata load, the per-year density computation, and "
                         "the t-SNE fit entirely — rebuild the HTML directly from this run's "
                         "already-written {run}_cache.npz/.json (written by a normal full run "
                         "at the end of section 5). For iterating on the HTML/JS template "
                         "without repeating the expensive data load + density + t-SNE pass.")
args, _ = parser.parse_known_args()

INTERACTIVE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR  = PROCESSED_DATA_DIR / "downstream_outcomes" / "opportunity_space"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_NPZ  = CACHE_DIR / f"{args.run_name}_cache.npz"
CACHE_JSON = CACHE_DIR / f"{args.run_name}_cache.json"

print(f"Run: {args.run_name}   from_cache={args.from_cache}")

crosswalk         = load_municipality_crosswalk()
muni_names_by_code = crosswalk.target_names  # {canonical 4-digit code: name}, 290 entries


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


if args.from_cache:
    # ─────────────────────────────────────────────────────────────────────────
    # Rebuild PERIOD_SPECS entirely from the cache written by a prior full run
    # (see the `else` branch below) — skips the SCB microdata load, all 30
    # density-matrix passes, and the t-SNE fit. For iterating on the HTML/JS
    # template (section 6 onward) without repeating the expensive part.
    # ─────────────────────────────────────────────────────────────────────────
    for _p in (CACHE_NPZ, CACHE_JSON):
        if not _p.exists():
            raise SystemExit(f"--from-cache requires {_p} — run this script without "
                              f"--from-cache once first.")
    print(f"\n--from-cache: rebuilding from {CACHE_NPZ.name} / {CACHE_JSON.name} — skipping "
          f"the SCB microdata load, density computation, and t-SNE fit entirely.")

    with open(CACHE_JSON) as f:
        cache_meta = json.load(f)
    all_municipalities = cache_meta["all_municipalities"]
    npz = np.load(CACHE_NPZ)

    PERIOD_SPECS = cache_meta["periods"]  # list of dicts: key, label, codes, hover, years
    for spec in PERIOD_SPECS:
        k = spec["key"]
        spec["density_b64"]  = b64(npz[f"{k}_density_q"])
        spec["presence_b64"] = b64(npz[f"{k}_presence"])
        spec["xy"]           = npz[f"{k}_xy"]
    print(f"  {len(all_municipalities)} municipalities, "
          + ", ".join(f"{s['label']}: {len(s['codes'])} codes" for s in PERIOD_SPECS))

else:
    MUNI_XWALK = harmonization_table(crosswalk)

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Load individual data, aggregate to (municipality, year, industry)
    #    — identical join/aggregation logic to P04_entry.py section 1.
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\nLoading {INDIVID_PATH} …")
    raw = pl.read_parquet(INDIVID_PATH, columns=["Year", "Sni", "WorkplaceMunicipality"])

    base = (
        raw
        .filter(pl.col("Sni").is_not_null() & (pl.col("Sni") != 0))
        .filter(pl.col("WorkplaceMunicipality").is_not_null())
        .with_columns([
            pl.col("Sni").cast(pl.Utf8).str.zfill(5).str.slice(0, 4).alias("sni_4d"),
            pl.col("WorkplaceMunicipality").cast(pl.Utf8).str.zfill(4).alias("source_code"),
        ])
        .filter(pl.col("sni_4d") != "0000")
        .join(MUNI_XWALK, on=["Year", "source_code"], how="inner")
    )

    empl = (
        base.group_by(["municipality", "Year", "sni_4d"])
        .agg(pl.len().alias("workers"))
        .sort(["municipality", "Year", "sni_4d"])
        .to_pandas()
    )
    all_municipalities = sorted(empl["municipality"].unique())
    print(f"  {len(empl):,} (municipality, year, industry) rows, "
          f"{len(all_municipalities)} municipalities")

    def empl_matrix(year: int, codes: list[str]) -> np.ndarray:
        return _shared_empl_matrix(empl, all_municipalities, year, codes)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Embeddings, descriptions, section maps — FULL code sets, no subsample
    # ═══════════════════════════════════════════════════════════════════════

    sni_embeddings = load_sni_embeddings(args.run_name, EMBED_BASE)

    for spec in PERIOD_SPECS:
        vecs, codes = sni_embeddings[spec["key"]]
        spec["vecs"]  = vecs
        spec["codes"] = codes

    with open(METADATA_DIR / "mapping_Sni_4digits_1992.json") as f: desc_92 = json.load(f)
    with open(METADATA_DIR / "mapping_Sni_4digits_2002.json") as f: desc_02 = json.load(f)
    with open(METADATA_DIR / "mapping_Sni_4digits_2007.json") as f: desc_07 = json.load(f)
    DESCRIPTIONS = {"sni92": desc_92, "sni02": desc_02, "sni07": desc_07}

    def load_section_names(year_file: str) -> dict[str, str]:
        p = METADATA_DIR / f"mapping_Sni_section_{year_file}.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {}

    for spec in PERIOD_SPECS:
        spec["desc"]      = DESCRIPTIONS[spec["key"]]
        spec["sec_map"]   = SNI_2digits_to_sections(spec["label"])   # 2-digit -> letter
        spec["sec_names"] = load_section_names(spec["year_file"])    # letter -> name

    def hover_text(spec: dict, code: str) -> str:
        desc = spec["desc"].get(code, "")
        if len(desc) > 90:
            desc = desc[:90] + "…"
        letter   = spec["sec_map"].get(code[:2], "?")
        sec_name = spec["sec_names"].get(letter, "")
        sec_label = f"{letter} — {sec_name}" if sec_name else letter
        return f"<b>{code}</b> ({spec['label']})<br>{desc}<br>Section {sec_label}"

    for spec in PERIOD_SPECS:
        spec["hover"] = [hover_text(spec, c) for c in spec["codes"]]

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Joint t-SNE — all codes from all three versions in ONE fit, so a
    #    position means the same thing across eras (matches
    #    P03_emergent_structure.py's D plot, just without the 600/version cap).
    # ═══════════════════════════════════════════════════════════════════════

    all_vecs  = np.vstack([spec["vecs"] for spec in PERIOD_SPECS])
    n_total   = len(all_vecs)
    print(f"\nRunning joint t-SNE on {n_total} codes (full set, no subsampling) …")
    tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, random_state=SEED,
                max_iter=1000, learning_rate="auto")
    xy = tsne.fit_transform(all_vecs)

    offset = 0
    for spec in PERIOD_SPECS:
        n = len(spec["codes"])
        spec["xy"] = xy[offset:offset + n]
        offset += n
    print("  done.")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Density + presence for EVERY industry (not just absent ones), every
    #    year each classification vintage was in use. D^{A,log} baseline:
    #    employment-weighted mean cosine similarity, log(1+employment)
    #    weights, leave-one-out — same definition as P04_entry.py's default.
    # ═══════════════════════════════════════════════════════════════════════

    for spec in PERIOD_SPECS:
        cos_sim = cosine_matrix(spec["vecs"])
        years   = spec["years"]
        n_years, n_muni, n_codes = len(years), len(all_municipalities), len(spec["codes"])
        print(f"\n{spec['label']}: density for {n_years} years × {n_muni} municipalities "
              f"× {n_codes} industries …")

        density  = np.zeros((n_years, n_muni, n_codes), dtype=np.float32)
        presence = np.zeros((n_years, n_muni, n_codes), dtype=np.uint8)
        for yi, year in enumerate(years):
            M_t = empl_matrix(year, spec["codes"])
            M_t_w = np.log1p(M_t)
            density[yi]  = employment_weighted_density(M_t_w, cos_sim)
            presence[yi] = (M_t > 0).astype(np.uint8)
            print(f"  {year}", end="  ", flush=True)
        print()

        spec["density"]  = density
        spec["presence"] = presence
        print(f"  density range: [{density.min():.4f}, {density.max():.4f}]  "
              f"mean presence rate: {presence.mean():.4%}")

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Quantize for embedding: density -> uint16 (per-period min-max scale,
    #    so the colorbar/quantization stays fixed while browsing years/munis
    #    within one period), presence -> uint8. Cache the quantized arrays +
    #    t-SNE coords (npz) and the small per-code metadata (json) so a
    #    template-only fix downstream doesn't require repeating sections 1-4.
    #    Base64-encode for the HTML payload.
    # ═══════════════════════════════════════════════════════════════════════

    npz_out = {}
    for spec in PERIOD_SPECS:
        d = spec["density"]
        p_mask = spec["presence"].astype(bool)
        dmin, dmax = float(d.min()), float(d.max())
        scale = (dmax - dmin) or 1.0
        q = np.clip(((d - dmin) / scale) * 65535, 0, 65535).astype("<u2")  # little-endian uint16

        spec["density_min"]  = dmin  # full range (present + absent) — used for uint16 dequantization
        spec["density_max"]  = dmax
        # Present-only range — used for the colour scale, so it isn't stretched
        # by absent industries' (never displayed) density values.
        if p_mask.any():
            spec["color_min"] = float(d[p_mask].min())
            spec["color_max"] = float(d[p_mask].max())
        else:
            spec["color_min"], spec["color_max"] = dmin, dmax
        spec["density_b64"]  = b64(q)
        spec["presence_b64"] = b64(spec["presence"])

        npz_out[f"{spec['key']}_density_q"]  = q
        npz_out[f"{spec['key']}_presence"]   = spec["presence"]
        npz_out[f"{spec['key']}_xy"]         = spec["xy"].astype(np.float32)

    np.savez_compressed(CACHE_NPZ, **npz_out)
    with open(CACHE_JSON, "w") as f:
        json.dump({
            "all_municipalities": all_municipalities,
            "periods": [
                {
                    "key": spec["key"], "label": spec["label"], "years": spec["years"],
                    "codes": spec["codes"], "hover": spec["hover"],
                    "density_min": spec["density_min"], "density_max": spec["density_max"],
                    "color_min": spec["color_min"], "color_max": spec["color_max"],
                }
                for spec in PERIOD_SPECS
            ],
        }, f)
    print(f"\n  → cache: {CACHE_NPZ}  +  {CACHE_JSON}  "
          f"(rerun with --from-cache to iterate on the HTML template only)")

    print(f"\nEmbedded payload size: "
          f"{sum(len(s['density_b64']) + len(s['presence_b64']) for s in PERIOD_SPECS) / 1e6:.1f} MB "
          f"(base64, before HTML/plotly.js overhead)")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Municipality dropdown data (names, not codes) + the shared background-field
#    grid — ONE grid (same x/y coordinates) across all three periods, built from
#    the joint t-SNE's combined bounding box, so the field doesn't jump around
#    when the slider crosses a period boundary. Only the field's z VALUES change
#    per (period, municipality, year) selection, computed client-side from that
#    snapshot's PRESENT industries only (kernel-weighted average of their
#    density, not a point-count KDE — see computeField() in the JS below).
# ═══════════════════════════════════════════════════════════════════════════════

muni_options = sorted(
    ({"code": m, "name": muni_names_by_code.get(m, m)} for m in all_municipalities),
    key=lambda d: d["name"],
)
default_muni = DEFAULT_MUNI if DEFAULT_MUNI in all_municipalities else all_municipalities[0]

all_xy = np.vstack([spec["xy"] for spec in PERIOD_SPECS])
x_min, x_max = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
y_min, y_max = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
x_pad = GRID_MARGIN * ((x_max - x_min) or 1.0)
y_pad = GRID_MARGIN * ((y_max - y_min) or 1.0)
grid_gx = np.linspace(x_min - x_pad, x_max + x_pad, GRID_N).tolist()
grid_gy = np.linspace(y_min - y_pad, y_max + y_pad, GRID_N).tolist()
grid_diag = float(np.hypot(x_max - x_min, y_max - y_min))
grid_bandwidth = grid_diag * FIELD_BANDWIDTH_FRAC


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Base Plotly figure. Trace 0 is the background field (Heatmap, always
#    visible, restyled to a freshly kernel-smoothed z grid on every
#    dropdown/slider change — see computeField() in the JS below). Traces
#    1..3 are one scatter per period, static x/y/hover; marker color/opacity
#    are placeholders, immediately overwritten by the JS on page load (same
#    update path the dropdown/slider use). Present industries get a literal
#    Viridis colour string computed in JS (so they can share a trace with
#    flat-gray absent industries — Plotly's own numeric colorscale can't mix
#    the two within one trace); the field trace still uses Plotly's native
#    numeric colorscale (it only ever averages present industries' density),
#    so it's also this figure's single shared colorbar.
# ═══════════════════════════════════════════════════════════════════════════════

field_trace = go.Heatmap(
    x=grid_gx, y=grid_gy,
    z=[[None] * len(grid_gx) for _ in range(len(grid_gy))],
    colorscale=COLORSCALE,
    zmin=PERIOD_SPECS[0]["color_min"], zmax=PERIOD_SPECS[0]["color_max"],
    zsmooth="best",
    opacity=0.85,
    showscale=True,
    colorbar=dict(title="Relatedness<br>density<br>(present)", thickness=14, len=0.7),
    hoverinfo="skip",
)

traces = [field_trace]
for i, spec in enumerate(PERIOD_SPECS):
    n = len(spec["codes"])
    traces.append(go.Scatter(
        x=spec["xy"][:, 0], y=spec["xy"][:, 1],
        mode="markers",
        marker=dict(
            size=7,
            color=[ABSENT_COLOR] * n,
            opacity=1.0,
            line=dict(width=0.3, color="white"),
        ),
        customdata=[[0.0, "—"]] * n,
        text=spec["hover"],
        hovertemplate="%{text}<br>Density: %{customdata[0]:.4f}<br>Present: %{customdata[1]}<extra></extra>",
        name=spec["label"],
        visible=(i == 0),
        showlegend=False,
    ))

fig = go.Figure(data=traces)
fig.update_layout(
    title="Regional opportunity space",
    autosize=True,
    height=760,
    margin=dict(l=20, r=20, t=90, b=20),
    plot_bgcolor="white",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
)

plot_div_html = fig.to_html(include_plotlyjs=True, full_html=False, div_id="opp-plot")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Assemble the full HTML: controls (dropdown + slider) + the plot div +
#    embedded data payload + hand-written JS driving Plotly.restyle().
# ═══════════════════════════════════════════════════════════════════════════════

payload = {
    "periods": [
        {
            # "index": this period's SCATTER trace position in `traces` (offset
            # by 1 — trace 0 is the always-visible field). The JS's
            # Plotly.restyle() calls need this to target the right trace.
            "index": i + 1,
            "key": spec["key"], "label": spec["label"], "years": spec["years"],
            "n_codes": len(spec["codes"]), "n_muni": len(all_municipalities),
            "density_min": spec["density_min"], "density_max": spec["density_max"],
            "color_min": spec["color_min"], "color_max": spec["color_max"],
            "density_b64": spec["density_b64"], "presence_b64": spec["presence_b64"],
        }
        for i, spec in enumerate(PERIOD_SPECS)
    ],
    "municipalities": muni_options,
    "muni_index": {m: i for i, m in enumerate(all_municipalities)},
    "present_opacity": PRESENT_OPACITY,
    "absent_opacity": ABSENT_OPACITY,
    "absent_color": ABSENT_COLOR,
    "default_muni": default_muni,
    "default_year": DEFAULT_YEAR,
    "grid": {"gx": grid_gx, "gy": grid_gy, "bandwidth": grid_bandwidth},
}

CONTROLS_CSS = """
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; padding: 16px 24px; }
  #controls { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; margin-bottom: 8px; }
  #controls label { font-size: 13px; font-weight: 600; color: #333; margin-right: 8px; }
  #muni-select { font-size: 14px; padding: 4px 8px; min-width: 220px; }
  #year-slider { width: 320px; }
  #year-label { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 3.5em; display: inline-block; }
  #legend-note { font-size: 12px; color: #666; margin-top: 4px; }
"""

CONTROLS_HTML = f"""
<div id="controls">
  <div>
    <label for="muni-select">Municipality</label>
    <select id="muni-select"></select>
  </div>
  <div>
    <label for="year-slider">Year</label>
    <input id="year-slider" type="range" min="1990" max="2019" step="1" value="{DEFAULT_YEAR}">
    <span id="year-label">{DEFAULT_YEAR}</span>
  </div>
</div>
<div id="legend-note">
  Colour (dots + background field) = relatedness density among industries the municipality
  ALREADY HAS that year (D<sup>A,log</sup> baseline) — how well-embedded each existing industry is
  in the rest of the local structure. Gray dots = not yet present that year (no density shown).
  The background field is a smoothed interpolation across present industries only.
</div>
"""

JS = r"""
<script>
const DATA = __PAYLOAD_JSON__;

function b64ToUint16(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Uint16Array(bytes.buffer);
}
function b64ToUint8(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

const decoded = {};
DATA.periods.forEach(p => {
  decoded[p.key] = {
    density: b64ToUint16(p.density_b64),
    presence: b64ToUint8(p.presence_b64),
  };
});

// Plotly.js's own Viridis stops — matched exactly so the literal marker
// colors (present industries) line up with the field trace's native
// numeric colorscale (which drives the shared colorbar).
const VIRIDIS = [
  [0, "#440154"], [0.0627, "#48186a"], [0.1255, "#472d7b"], [0.1882, "#424086"],
  [0.251, "#3b528b"], [0.3137, "#33638d"], [0.3765, "#2c728e"], [0.4392, "#26828e"],
  [0.502, "#21918c"], [0.5647, "#1fa088"], [0.6275, "#28ae80"], [0.6902, "#3fbc73"],
  [0.7529, "#5ec962"], [0.8157, "#84d44b"], [0.8784, "#addc30"], [0.9412, "#d8e219"],
  [1, "#fde725"],
].map(([t, hex]) => [t, [
  parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16),
]]);

function viridisColor(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < VIRIDIS.length - 1; i++) {
    const [t0, c0] = VIRIDIS[i], [t1, c1] = VIRIDIS[i + 1];
    if (t <= t1) {
      const f = (t - t0) / ((t1 - t0) || 1);
      return `rgb(${Math.round(c0[0] + f * (c1[0] - c0[0]))},`
           + `${Math.round(c0[1] + f * (c1[1] - c0[1]))},`
           + `${Math.round(c0[2] + f * (c1[2] - c0[2]))})`;
    }
  }
  const [, last] = VIRIDIS[VIRIDIS.length - 1];
  return `rgb(${last[0]},${last[1]},${last[2]})`;
}

// Kernel-weighted (Nadaraya-Watson) smoothing of present-industry density
// values onto the shared grid — NOT a point-count KDE: a grid cell's value
// is a weighted AVERAGE of nearby present industries' density, weighted by
// a Gaussian falloff with distance. Cells far from any present industry
// (total weight below the cutoff) are left null (transparent).
function computeField(xs, ys, vals, gx, gy, bandwidth) {
  const nx = gx.length, ny = gy.length;
  const twoBw2 = 2 * bandwidth * bandwidth;
  const WEIGHT_CUTOFF = 0.05;
  const z = new Array(ny);
  for (let j = 0; j < ny; j++) {
    const row = new Array(nx);
    const gyj = gy[j];
    for (let i = 0; i < nx; i++) {
      const gxi = gx[i];
      let wsum = 0, vsum = 0;
      for (let k = 0; k < xs.length; k++) {
        const dx = xs[k] - gxi, dy = ys[k] - gyj;
        const w = Math.exp(-(dx * dx + dy * dy) / twoBw2);
        wsum += w;
        vsum += w * vals[k];
      }
      row[i] = wsum > WEIGHT_CUTOFF ? vsum / wsum : null;
    }
    z[j] = row;
  }
  return z;
}

const gd = document.getElementById("opp-plot");
const muniSelect = document.getElementById("muni-select");
const yearSlider = document.getElementById("year-slider");
const yearLabel  = document.getElementById("year-label");

DATA.municipalities.forEach(m => {
  const opt = document.createElement("option");
  opt.value = m.code;
  opt.textContent = m.name;
  if (m.code === DATA.default_muni) opt.selected = true;
  muniSelect.appendChild(opt);
});

function periodForYear(year) {
  return DATA.periods.find(p => p.years[0] <= year && year <= p.years[p.years.length - 1]);
}

let activePeriodKey = null;

function updatePlot() {
  const year = parseInt(yearSlider.value, 10);
  const muniCode = muniSelect.value;
  const period = periodForYear(year);
  const yearIdx = period.years.indexOf(year);
  const muniIdx = DATA.muni_index[muniCode];
  const nCodes = period.n_codes;
  const rowBase = (yearIdx * period.n_muni + muniIdx) * nCodes;

  const qArr = decoded[period.key].density;
  const pArr = decoded[period.key].presence;
  const dmin = period.density_min, dmax = period.density_max;
  const scale = (dmax - dmin) || 1;
  const cmin = period.color_min, cmax = period.color_max;
  const cscale = (cmax - cmin) || 1;

  const colorStr   = new Array(nCodes);
  const opacityArr = new Array(nCodes);
  const customdata = new Array(nCodes);
  const rawDensity = new Array(nCodes);
  let nPresent = 0;
  for (let i = 0; i < nCodes; i++) {
    const q = qArr[rowBase + i];
    const val = dmin + (q / 65535) * scale;
    rawDensity[i] = val;
    const present = pArr[rowBase + i] === 1;
    if (present) {
      colorStr[i] = viridisColor((val - cmin) / cscale);
      opacityArr[i] = DATA.present_opacity;
      nPresent++;
    } else {
      colorStr[i] = DATA.absent_color;
      opacityArr[i] = DATA.absent_opacity;
    }
    customdata[i] = [val, present ? "yes" : "no"];
  }

  if (period.key !== activePeriodKey) {
    const visible = new Array(1 + DATA.periods.length).fill(false);
    visible[0] = true;                    // field trace always visible
    visible[period.index] = true;
    Plotly.restyle(gd, { visible: visible }, visible.map((_, i) => i));
    activePeriodKey = period.key;
  }

  Plotly.restyle(gd, {
    "marker.color": [colorStr],
    "marker.opacity": [opacityArr],
    customdata: [customdata],
  }, [period.index]);

  // Background field: kernel-smoothed from this snapshot's PRESENT
  // industries only, reusing the x/y already rendered on the scatter trace
  // (no need to re-embed coordinates in DATA).
  const xs = gd.data[period.index].x, ys = gd.data[period.index].y;
  const presentXs = [], presentYs = [], presentVals = [];
  for (let i = 0; i < nCodes; i++) {
    if (pArr[rowBase + i] === 1) {
      presentXs.push(xs[i]);
      presentYs.push(ys[i]);
      presentVals.push(rawDensity[i]);
    }
  }
  const zGrid = computeField(presentXs, presentYs, presentVals, DATA.grid.gx, DATA.grid.gy, DATA.grid.bandwidth);
  Plotly.restyle(gd, { z: [zGrid], zmin: [cmin], zmax: [cmax] }, [0]);

  const muniName = DATA.municipalities.find(m => m.code === muniCode).name;
  Plotly.relayout(gd, {
    title: {
      text: `${muniName} — ${year}  |  ${nPresent}/${nCodes} industries present`
          + `<br><sup>${period.label} classification  ·  hover a dot for code, description, `
          + `density and presence</sup>`,
    },
  });
  yearLabel.textContent = year;
}

muniSelect.addEventListener("change", updatePlot);
yearSlider.addEventListener("input", updatePlot);
yearSlider.value = DATA.default_year;
updatePlot();
</script>
"""

full_html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Regional opportunity space — {args.run_name}</title>
<style>{CONTROLS_CSS}</style>
</head>
<body>
{CONTROLS_HTML}
{plot_div_html}
{JS.replace("__PAYLOAD_JSON__", json.dumps(payload))}
</body>
</html>
"""

out_path = INTERACTIVE_PLOTS_DIR / f"{args.run_name}_opportunity_space.html"
out_path.write_text(full_html, encoding="utf-8")
print(f"\n  → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

print("\nDone.")
