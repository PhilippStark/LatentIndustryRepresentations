"""
H4_evaluate_temporal.py
=======================
Temporal hold-out with sliding windows: embedding v16_tw_T{T0} is trained
only on register years ≤ T0 (temporal_holdout/data/T{T0}/). SNI07 period
only. Panel row t: density from M_t, outcome realised at t+1.

Test 1 — temporal hold-out ("window")
-------------------------------------
Embedding ≤ T0, panel row t = T0 + 1: density from the local employment of
T0 + 1 with the T0 geometry, the row's published outcome. The test rows of
all windows are stacked into one panel and estimated with year-specific
slopes; v16 and v16_reference on exactly the same rows. Each test year has
its own window embedding, so the industry FE are interacted with the year
(municipality×year + industry×year FE, for every variant; see
_holdout_common.estimate). "window_shared_fe" keeps the shared industry FE
(attenuated: documentation only).

Test 2 — forecast horizon ("horizon")
-------------------------------------
Embedding ≤ T0, density from M_{T0}, candidates of row T0, outcome realised
at T0 + h (h = 1: the published outcome; h = 2, 3: the entry/exit rules of
P04 applied at T0 + h; start-up entry h = 1 only), T0 + h ≤ 2019. Estimated
per (T0, h) (one embedding: main specification) and pooled over T0 for each
h with industry×T0 FE (every variant); v16 at the same lag for comparison.

Outputs (results/regional_temporal_holdout/, prefixed shared_cosine_v16_)
-------------------------------------------------------------------------
  window_{outcome}_summary.csv, window_{outcome}_year_coefs.csv
  horizon_{outcome}.csv  (t0 = "pooled" for the stacked windows)

Usage
-----
  python scripts/robustness/H4_evaluate_temporal.py
"""

import argparse

import numpy as np
import pandas as pd

from _holdout_common import (
    OUTCOMES, Embedding, Panel, check_codes, estimate, is_trained, load_main, print_effects,
    reference_embedding, result_path, results_dir, temporal_variant,
)

PERIOD = "SNI07"
LAST_ROW, LAST_REGISTER_YEAR = 2018, 2019

parser = argparse.ArgumentParser()
parser.add_argument("--windows", type=int, nargs="+", default=[2015, 2016, 2017, 2018])
parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
parser.add_argument("--outcomes", nargs="+", default=list(OUTCOMES))
args = parser.parse_args()

out_dir = results_dir("temporal_holdout")
emp, v16 = load_main()
reference = reference_embedding()

windows = {}
for t0 in args.windows:
    root, run = temporal_variant(t0)
    if not is_trained(root, run):
        print(f"T0 = {t0}: {run} not trained — skipped.")
        continue
    windows[t0] = Embedding(run, root, run)
    check_codes(windows[t0], v16)
if not windows:
    raise SystemExit("No trained window.")
print(f"Windows: {sorted(windows)}")

for outcome in args.outcomes:
    col = OUTCOMES[outcome]["col"]
    print(f"\n{'=' * 60}\n{outcome}\n{'=' * 60}")
    panel = Panel(outcome, emp)
    df = panel.df
    v16_density = df["density"].to_numpy(dtype=float)
    ref_density = panel.density(emp, reference) if reference is not None else None

    # ── Test 1: row T0 + 1 with the T0 embedding ──────────────────────────────
    test_rows, window_density = [], []
    for t0, emb in sorted(windows.items()):
        if t0 + 1 > LAST_ROW:
            continue
        group = (PERIOD, t0 + 1)
        rows = panel.groups[group]
        test_rows.append(rows)
        window_density.append(panel.density(emp, emb, groups={group})[rows])
    summaries, years = [], []
    if test_rows:
        rows = np.concatenate(test_rows)
        sub = df.iloc[rows]
        year_group = sub["year"].to_numpy()
        variants = {"window": (np.concatenate(window_density), year_group),
                    "window_shared_fe": (np.concatenate(window_density), None),
                    "v16": (v16_density[rows], year_group)}
        if ref_density is not None:
            variants["v16_reference"] = (ref_density[rows], year_group)
        print("Test 1 (row T0 + 1):")
        for name, (d, group) in variants.items():
            effects, year_df = estimate(sub, col, d, annual=True, fe_group=group, outcome=outcome, variant=name)
            print_effects(effects, name)
            summaries.append(effects)
            years.append(year_df)
        pd.concat(summaries).to_csv(result_path("temporal_holdout", f"window_{outcome}_summary.csv"), index=False)
        pd.concat(years).to_csv(result_path("temporal_holdout", f"window_{outcome}_year_coefs.csv"), index=False)

    # ── Test 2: density at T0 against outcomes at T0 + h ──────────────────────
    horizons = [h for h in args.horizons if h == 1 or outcome != "startup_entry"]
    horizon_rows = []
    print("Test 2 (density at T0, outcome at T0 + h):")
    for h in horizons:
        stacked = {"window": [], "v16": []}
        frames, t0_group = [], []
        for t0, emb in sorted(windows.items()):
            if t0 + h > LAST_REGISTER_YEAR:
                continue
            group = (PERIOD, t0)
            rows = panel.groups[group]
            sub = df.iloc[rows].copy()
            if h > 1:
                sub[col] = panel.outcome_at_horizon(emp, PERIOD, t0, h)
            variants = {"window": panel.density(emp, emb, groups={group})[rows], "v16": v16_density[rows]}
            for name, d in variants.items():
                effects, _ = estimate(sub, col, d, annual=False, outcome=outcome, t0=t0, h=h, variant=name)
                horizon_rows.append(effects)
                stacked[name].append(d)
            frames.append(sub)
            t0_group.append(np.full(len(sub), t0))
        if len(frames) > 1:
            pooled = pd.concat(frames)
            for name, parts in stacked.items():
                effects, _ = estimate(pooled, col, np.concatenate(parts), annual=False,
                                      fe_group=np.concatenate(t0_group),
                                      outcome=outcome, t0="pooled", h=h, variant=name)
                print_effects(effects, f"h={h} pooled {name}")
                horizon_rows.append(effects)
    pd.concat(horizon_rows).to_csv(result_path("temporal_holdout", f"horizon_{outcome}.csv"), index=False)
    print(f"  → {out_dir}/*_{outcome}*.csv")

print("\nDone.")
