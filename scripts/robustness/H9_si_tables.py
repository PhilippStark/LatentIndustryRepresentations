"""
H9_si_tables.py
===============
Collects the outputs of H3–H5 into compact supplementary tables (.csv and
LaTeX .tex). Coefficients are effects of a one-SD increase in density, in
percentage points, with standard errors in parentheses. Tables whose inputs
are missing are skipped.

Outputs (results/paper/appendix/)
---------------------------------
  holdout_spatial_crossfit   outcome × period: v16, v16_reference, v16 with
                             industry×fold FE, cross-fitted (industry×fold FE;
                             repeat 1; range over repeats if there are several)
  holdout_temporal_window    outcome × test row (2016–2018, pooled): v16,
                             v16_reference, window embedding
  holdout_temporal_horizon   outcome × h (pooled over windows): v16, window
  seed_stability_regional    outcome × period: v16, v16_reference, seeds (mean ± SD [min, max])
  seed_stability_geometry    P02 metric: v16, v16_reference, seeds (mean ± SD)

Usage
-----
  python scripts/robustness/H9_si_tables.py
"""

import pandas as pd

from _holdout_common import OUTCOMES, result_path, results_dir

GEOMETRY_METRICS = {
    "B_R1_02to07": "Retrieval R@1, SNI02 → SNI07",
    "B_MRR_02to07": "Retrieval MRR, SNI02 → SNI07",
    "B2_R1_92to02": "Retrieval R@1, SNI92 → SNI02",
    "A_treedist_rho": "Hierarchy: tree-distance ρ",
    "E_labor_rho_07": "Labor consistency ρ (SNI07)",
    "E2_ownership_rho_07": "Ownership consistency ρ (SNI07)",
    "I_io_rho": "IO consistency ρ",
    "F_section_silhouette": "Section silhouette (SNI07)",
}

out_dir = results_dir("si")


def pp(coef: float, se: float | None = None) -> str:
    return f"{coef * 100:+.3f}" + (f" ({se * 100:.3f})" if se is not None and pd.notna(se) else "")


def save(table: pd.DataFrame, name: str, caption: str) -> None:
    table.to_csv(result_path("si", f"{name}.csv"), index=False)
    latex = table.to_latex(index=False, escape=True, column_format="l" * len(table.columns), caption=caption,
                           label=f"tab:{name}")
    result_path("si", f"{name}.tex").write_text(latex, encoding="utf-8")
    print(f"  → {out_dir / name}.csv/.tex")


def effect(rows: pd.DataFrame, **match) -> str:
    sel = rows
    for key, value in match.items():
        sel = sel[sel[key].astype(str) == str(value)]
    return pp(sel["coef"].iloc[0], sel["se"].iloc[0]) if len(sel) else ""


# ── Spatial ───────────────────────────────────────────────────────────────────
records = []
for outcome in OUTCOMES:
    path = result_path("spatial_holdout", f"crossfit_{outcome}_summary.csv")
    if not path.exists():
        continue
    s = pd.read_csv(path)
    crossfits = sorted(v for v in s["variant"].unique()
                       if v.startswith("crossfit_r") and not v.endswith("_shared_fe"))
    for period in s["period"].unique():
        cf = s[(s["period"] == period) & s["variant"].isin(crossfits)]
        rec = {"Outcome": outcome, "Period": period,
               "v16": effect(s, variant="v16", period=period),
               "v16 retrained": effect(s, variant="v16_reference", period=period),
               "v16, industry×fold FE": effect(s, variant=crossfits[0].replace("crossfit", "v16_fold_fe"),
                                               period=period),
               "Cross-fitted": effect(s, variant=crossfits[0], period=period)}
        if len(crossfits) > 1:
            rec[f"Range over {len(crossfits)} fold draws"] = f"[{pp(cf['coef'].min())}, {pp(cf['coef'].max())}]"
        records.append(rec)
if records:
    save(pd.DataFrame(records), "holdout_spatial_crossfit",
         "Spatial hold-out: density effects with each municipality's density built from an embedding "
         "trained without that municipality's labor flows and ownership links "
         "(municipality×year and industry×fold fixed effects).")

# ── Temporal ──────────────────────────────────────────────────────────────────
records = []
for outcome in OUTCOMES:
    summary = result_path("temporal_holdout", f"window_{outcome}_summary.csv")
    years = result_path("temporal_holdout", f"window_{outcome}_year_coefs.csv")
    if not summary.exists():
        continue
    s, y = pd.read_csv(summary), pd.read_csv(years)
    for row in ["pooled"] + sorted(y["year"].unique()):
        src, match = (s, {}) if row == "pooled" else (y, {"year": row})
        records.append({"Outcome": outcome, "Test row t": row,
                        **{label: effect(src, variant=v, **match)
                           for label, v in (("v16", "v16"), ("v16 retrained", "v16_reference"),
                                            ("Embedding ≤ t−1", "window"))}})
if records:
    save(pd.DataFrame(records), "holdout_temporal_window",
         "Temporal hold-out: density in year t built with an embedding trained on register years up to t−1 "
         "(SNI07 period).")

records = []
for outcome in OUTCOMES:
    path = result_path("temporal_holdout", f"horizon_{outcome}.csv")
    if not path.exists():
        continue
    h = pd.read_csv(path)
    pooled = h[h["t0"].astype(str) == "pooled"]
    for horizon in sorted(h["h"].unique()):
        sel = pooled[pooled["h"] == horizon]
        src = sel if len(sel) else h[h["h"] == horizon]  # a single window has no pooled row
        records.append({"Outcome": outcome, "h": horizon,
                        "v16": effect(src, variant="v16"), "Embedding ≤ T0": effect(src, variant="window")})
if records:
    save(pd.DataFrame(records), "holdout_temporal_horizon",
         "Forecast horizon: density at T0 (embedding trained up to T0) and outcomes realised at T0 + h, "
         "pooled over windows (SNI07 period).")

# ── Seeds ─────────────────────────────────────────────────────────────────────
path = result_path("seed_stability", "regional_summary.csv")
if path.exists():
    s = pd.read_csv(path)
    records = [{"Outcome": r["outcome"], "Period": r["period"], "σ": r["sigma"],
                "v16": pp(r.get("v16_coef"), r.get("v16_se")),
                "v16 retrained": pp(r["v16_reference_coef"], r["v16_reference_se"]) if "v16_reference_coef" in r and pd.notna(r["v16_reference_coef"]) else "",
                f"Seeds (n={int(r['n_seeds'])}), mean ± SD [min, max]":
                    f"{r['coef_mean'] * 100:+.3f} ± {r['coef_sd'] * 100:.3f} [{pp(r['coef_min'])}, {pp(r['coef_max'])}]"}
               for _, r in s.iterrows()]
    save(pd.DataFrame(records), "seed_stability_regional",
         "Seed stability: density effects across training runs from perturbed initialisations.")

path = result_path("seed_stability", "geometry_summary.csv")
if path.exists():
    g = pd.read_csv(path)
    g = g[g["metric"].isin(GEOMETRY_METRICS)]
    records = [{"Metric": GEOMETRY_METRICS[r["metric"]], "σ": r["sigma"],
                "v16": f"{r['v16']:.4f}", "v16 retrained": f"{r['v16_reference']:.4f}",
                f"Seeds (n={int(r['n_seeds'])}), mean ± SD": f"{r['mean']:.4f} ± {r['sd']:.4f}"}
               for _, r in g.iterrows()]
    save(pd.DataFrame(records), "seed_stability_geometry",
         "Seed stability: evaluation metrics of the shared geometry across training runs.")

print("Done.")
