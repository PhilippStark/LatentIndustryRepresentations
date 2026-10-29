"""
H3_evaluate_spatial.py
======================
Spatial hold-out: density effects when each municipality's density is
built from its own local employment and an embedding trained without any
labor-flow or ownership information from that municipality.

Variants
--------
  v16              published embedding (density as in the published panels)
  v16_reference    v16 pipeline retrained with the same code (retrain-only control)
  crossfit_r{r}    each municipality's density from the embedding of the fold
                   that held it out (v16_sp_r{r}_f{fold}, spatial_holdout/data/);
                   municipality×year + industry×fold FE
  crossfit_r{r}_shared_fe   the same density with the shared industry FE of
                   the main specification (attenuated: documentation only)
  v16_fold_fe_r{r} v16 density with the FE of crossfit_r{r} (same-specification
                   benchmark)

Estimates (density standardised within each sample)
----------------------------------------------------
  crossfit    full panels, all periods: period slopes and year-specific slopes
  per_fold    test municipalities of one fold: fold embedding vs v16 vs
              reference (one embedding: main specification)
  by_stratum  municipalities of one size stratum: crossfit (industry×fold FE)
              vs v16

The cross-fitted panel stacks the densities of five embeddings. Each
embedding sets its own density level for every industry, so the industry FE
are interacted with the fold (see _holdout_common.estimate). Because every
municipality is in one fold, all FE are then fold-specific and the pooled
slope is a precision-weighted combination of the five per-fold slopes.

Outputs (results/regional_spatial_holdout/, prefixed shared_cosine_v16_)
------------------------------------------------------------------------
  crossfit_{outcome}_summary.csv, crossfit_{outcome}_year_coefs.csv,
  per_fold_{outcome}.csv, by_stratum_{outcome}.csv

Usage
-----
  python scripts/robustness/H3_evaluate_spatial.py            # every complete repeat
  python scripts/robustness/H3_evaluate_spatial.py --repeats 1
"""

import argparse

import numpy as np
import pandas as pd

from _holdout_common import (
    OUTCOMES, Embedding, Panel, check_codes, estimate, is_trained, load_main, print_effects,
    reference_embedding, result_path, results_dir, spatial_variant,
)

parser = argparse.ArgumentParser()
parser.add_argument("--repeats", type=int, default=None,
                    help="Use repeats 1..N (default: every repeat whose folds are all trained).")
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--outcomes", nargs="+", default=list(OUTCOMES))
args = parser.parse_args()

out_dir = results_dir("spatial_holdout")
emp, v16 = load_main()
reference = reference_embedding()

repeats = []
r = 1
while result_path("spatial_holdout", f"folds_r{r}.csv").exists() and (args.repeats is None or r <= args.repeats):
    if all(is_trained(*spatial_variant(r, k)) for k in range(1, args.k + 1)):
        repeats.append(r)
    else:
        print(f"Repeat {r}: not every fold is trained — skipped.")
    r += 1
if not repeats:
    raise SystemExit("No complete repeat (run H1_spatial_folds.py and train every fold first).")
print(f"Repeats: {repeats}")

fold_maps, fold_embeddings = {}, {}
for r in repeats:
    folds = pd.read_csv(result_path("spatial_holdout", f"folds_r{r}.csv"), dtype={"municipality": str})
    fold_maps[r] = folds.set_index("municipality")
    for k in range(1, args.k + 1):
        root, run = spatial_variant(r, k)
        fold_embeddings[r, k] = Embedding(run, root, run)
        check_codes(fold_embeddings[r, k], v16)

for outcome in args.outcomes:
    col = OUTCOMES[outcome]["col"]
    print(f"\n{'=' * 60}\n{outcome}\n{'=' * 60}")
    panel = Panel(outcome, emp)
    df = panel.df
    dens = {"v16": df["density"].to_numpy(dtype=float)}
    if reference is not None:
        dens["v16_reference"] = panel.density(emp, reference)

    fold_of, stratum_of = {}, {}
    for r in repeats:
        fold_of[r] = df["municipality"].map(fold_maps[r]["fold"]).to_numpy()
        stratum_of[r] = df["municipality"].map(fold_maps[r]["stratum"]).to_numpy()
        if np.isnan(fold_of[r].astype(float)).any():
            raise ValueError(f"repeat {r}: a panel municipality has no fold")
        crossfit = np.full(len(df), np.nan)
        for k in range(1, args.k + 1):
            in_fold = fold_of[r] == k
            crossfit[in_fold] = panel.density(emp, fold_embeddings[r, k])[in_fold]
        dens[f"crossfit_r{r}"] = crossfit

    # 1. cross-fitted full panel
    runs = [(name, d, None) for name, d in dens.items() if not name.startswith("crossfit_r")]
    for r in repeats:
        runs += [(f"crossfit_r{r}", dens[f"crossfit_r{r}"], fold_of[r]),
                 (f"crossfit_r{r}_shared_fe", dens[f"crossfit_r{r}"], None),
                 (f"v16_fold_fe_r{r}", dens["v16"], fold_of[r])]
    summaries, years = [], []
    for name, d, group in runs:
        rows, year_df = estimate(df, col, d, annual=True, fe_group=group, outcome=outcome, variant=name)
        print_effects(rows, name)
        summaries.append(rows)
        years.append(year_df)
    pd.concat(summaries).to_csv(result_path("spatial_holdout", f"crossfit_{outcome}_summary.csv"), index=False)
    pd.concat(years).to_csv(result_path("spatial_holdout", f"crossfit_{outcome}_year_coefs.csv"), index=False)

    # 2. per fold (test municipalities of the fold only)
    per_fold = []
    for r in repeats:
        for k in range(1, args.k + 1):
            mask = fold_of[r] == k
            sub = df[mask]
            variants = {"holdout": dens[f"crossfit_r{r}"], "v16": dens["v16"]}
            if "v16_reference" in dens:
                variants["v16_reference"] = dens["v16_reference"]
            for name, d in variants.items():
                rows, _ = estimate(sub, col, d[mask], annual=False,
                                   outcome=outcome, repeat=r, fold=k, variant=name)
                per_fold.append(rows)
        print(f"  per fold: repeat {r} done")
    pd.concat(per_fold).to_csv(result_path("spatial_holdout", f"per_fold_{outcome}.csv"), index=False)

    # 3. by size stratum (strata do not depend on the repeat)
    by_stratum = []
    strata = stratum_of[repeats[0]]
    for s in sorted(np.unique(strata)):
        mask = strata == s
        sub = df[mask]
        variants = {"v16": (dens["v16"], None),
                    **{f"crossfit_r{r}": (dens[f"crossfit_r{r}"], fold_of[r]) for r in repeats}}
        for name, (d, group) in variants.items():
            rows, _ = estimate(sub, col, d[mask], annual=False, fe_group=None if group is None else group[mask],
                               outcome=outcome, stratum=int(s), variant=name)
            by_stratum.append(rows)
    pd.concat(by_stratum).to_csv(result_path("spatial_holdout", f"by_stratum_{outcome}.csv"), index=False)
    print(f"  → {out_dir}/*_{outcome}*.csv")

print("\nDone.")
