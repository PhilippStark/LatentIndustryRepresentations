# Hold-out and seed robustness of `shared_cosine_v16` (SI)

Three supplementary tests of the main run:

| Test | Question | Results |
|---|---|---|
| Spatial hold-out | Does the density effect stay the same in municipalities whose labor flows and ownership links were never seen in training? | `results/regional_spatial_holdout/` |
| Temporal hold-out and forecast horizon | Does it stay the same in years after the training data ends? How far ahead does density predict entry and exit? | `results/regional_temporal_holdout/` |
| Seed and initialisation stability | How much do the geometry and the regional estimates vary across training runs? | `results/seed_stability/` |

Only the embedding changes. Density is always built from the **local
employment** of the municipality and year in the regression (D^{A,log}, the
main specification), and the regression is the main specification
(municipality×year + industry FE, SE two-way clustered by municipality and
industry). The reported quantity is the coefficient on standardised density.
Every table also reports `v16_reference`, the main pipeline retrained with
this code, so the hold-out effect is separated from retraining drift (v16 is
not bitwise reproducible across nodes).

## Which training inputs carry municipal or year information

| View | Source | Municipal | Years |
|---|---|---|---|
| labor | individual register, worker flows t→t+1, Poisson expected-flow covariates | yes | 1990–2019 |
| ownership | individual register, industry pairs within an OrgNr per year | yes | 1990–2019 |
| io | national public IO tables | no | 2008–2019 |
| hierarchy, crosswalk transitions, text prior | metadata and descriptions | no | no |

The spatial hold-out rebuilds labor and ownership without the test
municipalities (`--exclude-municipalities`); the temporal hold-out rebuilds
labor, ownership and IO up to T0 (`--max-year`).

**Timing.** A panel row of year t has density from M_t and an outcome
realised at t+1 (as in P04 and the manuscript). "Embedding up to T0" means
every training input uses register years ≤ T0.

**Fixed effects when embeddings are stacked.** The spatial cross-fit and the
stacked temporal windows combine densities from several embeddings in one
regression. Each embedding sets its own density level for every industry, so
the industry FE are interacted with the embedding (industry×fold,
industry×year, industry×T0); with a shared industry FE these level
differences attenuate β. Regressions on a single embedding use the main
specification unchanged (`_holdout_common.estimate(fe_group=...)`).

## Scripts

| Script | Does | Needs |
|---|---|---|
| `_holdout_common.py` | paths and variant naming; employment cache → M_t; D^{A,log} for any embedding; `estimate()`; h-step outcomes | — |
| `H0_employment_cache.py` | employment cache from the register (once) + checks that density, h = 1 outcomes and published estimates are reproduced | microdata (cache only) |
| `H1_spatial_folds.py` | size-stratified folds (5 folds × 3 repeats; Stockholm, Göteborg, Malmö in different folds) | cache |
| `H2_check_reference.py` | reference retrain vs published inputs, bitwise trainer check, seed-alone Δcos, retrain drift | reference runs |
| `H3_evaluate_spatial.py` | cross-fitted, per-fold and per-stratum coefficients | fold runs |
| `H4_evaluate_temporal.py` | window (test 1) and forecast horizon (test 2) coefficients | window runs |
| `H5_evaluate_seeds.py` | coefficient and P02-metric spread over seeds; k-NN overlap | seed runs + P02 |
| `H9_si_tables.py` | SI tables | H3–H5 output |

The evaluation scripts (H3–H5, H9) re-use the published v16 panels and
replace only `density`; they need no microdata.

Patched pipeline scripts (all flags off by default, so the main run is
unchanged):

| File | Flag | Flag off |
|---|---|---|
| `scripts/data/build_labor_relatedness.py`, `build_ownership_relatedness.py` | `--exclude-municipalities FILE` (CSV with a `municipality` column; helper `scripts/data/_municipality_exclusion.py`) | code path and metadata unchanged |
| `scripts/training/train_shared_cosine_embedding.py` | `--init-noise SIGMA`: Z0 ← normalize(Z0 + SIGMA·ε), ε ~ N(0, I/d) from `--seed` | no random number drawn; H2 checks bitwise identity |

## Data locations

Variant data (relatedness, graphs, embeddings) contain restricted register
aggregates and stay on LUNARC, under `HOLDOUT_ROOT`
(default `<MAIN_DATA_ROOT>/ablations/holdout_robustness`):

```
common/data/                     employment cache, reference_full/ (v16_reference)
spatial_holdout/data/r{r}_f{k}/  run v16_sp_r{r}_f{k}
temporal_holdout/data/T{T0}/     run v16_tw_T{T0}
seed_stability/data/s{SS}_n{σ}/  run v16_seed_s{SS}_n{σ}
```

Each variant folder is a complete `LATENT_DATA_ROOT`, so the pipeline
scripts run on it unchanged. Set `LATENT_MAIN_DATA_ROOT` and `HOLDOUT_ROOT`
if the data move. Only aggregate tables are written to `results/`.

## How to run

The whole chain took about 2 h 15 min on one lu48 node (33 trainings).
Variants whose embedding exists can be skipped, so the chain is resumable.

```bash
export MAIN=/lunarc/nobackup/projects/lu2025-17-39/DataLatentIndustryRepresentations
export HOLDOUT_ROOT=$MAIN/ablations/holdout_robustness LATENT_MAIN_DATA_ROOT=$MAIN
REF=$HOLDOUT_ROOT/common/data/reference_full
graph() { python scripts/graph/build_layer_transition_matrices.py --name layers_v1
          python scripts/graph/build_layer_diffusion_targets.py --name layers_v1; }
train() { local run=$1; shift
          python scripts/training/train_shared_cosine_embedding.py --name layers_v1 --run-name "$run" "$@"; }

# 0. Employment cache and checks
python scripts/robustness/H0_employment_cache.py

# 1. Reference retrain (published IO tables copied into $REF/processed first)
export LATENT_DATA_ROOT=$REF
python scripts/data/build_labor_relatedness.py
python scripts/data/build_ownership_relatedness.py
graph
train v16_reference --init-noise 0
train v16_reference_repotrainer                    # flag omitted: H2 checks it is bitwise identical
train v16_reference_seed43 --seed 43 --init-noise 0
python scripts/robustness/H2_check_reference.py

# 2. Temporal windows, T0 = 2015..2018 (raw/io copied into the variant root first)
export LATENT_DATA_ROOT=$HOLDOUT_ROOT/temporal_holdout/data/T$T0
python scripts/data/build_labor_relatedness.py --max-year $T0
python scripts/data/build_ownership_relatedness.py --max-year $T0
python scripts/data/build_input_output_relatedness.py --max-year $T0
graph; train v16_tw_T$T0 --init-noise 0

# 3. Seed variants, S = 1..10, SIGMA = 0.5 (processed/ symlinked to $REF/processed)
export LATENT_DATA_ROOT=$HOLDOUT_ROOT/seed_stability/data/s${SS}_n$SIGMA
train v16_seed_s${SS}_n$SIGMA --seed $S --init-noise $SIGMA
python scripts/evaluation/P02_shared_geometry_reconstruction.py --run-name v16_seed_s${SS}_n$SIGMA
#   then move results/embedding_eval/<run>/ to $LATENT_DATA_ROOT/embedding_eval/ (same for v16_reference)

# 4. Spatial folds, r = 1..3, k = 1..5 (published IO tables copied into processed/ first)
python scripts/robustness/H1_spatial_folds.py --repeats 3
FOLD=results/regional_spatial_holdout/shared_cosine_v16_folds_r${R}_test_f${K}.csv
export LATENT_DATA_ROOT=$HOLDOUT_ROOT/spatial_holdout/data/r${R}_f${K}
python scripts/data/build_labor_relatedness.py --exclude-municipalities $FOLD
python scripts/data/build_ownership_relatedness.py --exclude-municipalities $FOLD
graph; train v16_sp_r${R}_f${K} --init-noise 0

# 5. Evaluation and SI tables
unset LATENT_DATA_ROOT
python scripts/robustness/H3_evaluate_spatial.py
python scripts/robustness/H4_evaluate_temporal.py
python scripts/robustness/H5_evaluate_seeds.py
python scripts/robustness/H9_si_tables.py
```

Never `import` the relatedness builders: they run on import and would
overwrite `processed/` of whatever `LATENT_DATA_ROOT` points to.

## Outputs

All files in the three test folders and in `results/holdout_checks/` are
prefixed `shared_cosine_v16_`.

| Folder | Files |
|---|---|
| `results/holdout_checks/` | `H0_checks.json`, `H2_reference_checks.json` |
| `results/regional_spatial_holdout/` | `folds_r{r}.csv`, `folds_r{r}_test_f{k}.csv`; `crossfit_{outcome}_summary.csv`, `crossfit_{outcome}_year_coefs.csv`, `per_fold_{outcome}.csv`, `by_stratum_{outcome}.csv` |
| `results/regional_temporal_holdout/` | `window_{outcome}_summary.csv`, `window_{outcome}_year_coefs.csv`, `horizon_{outcome}.csv` |
| `results/seed_stability/` | `regional_{outcome}_by_seed.csv`, `regional_summary.csv`, `geometry_metrics_by_seed.csv`, `geometry_summary.csv`, `neighbour_overlap.csv` |
| `results/paper/appendix/` | SI tables (`.csv`, `.tex`): `holdout_spatial_crossfit`, `holdout_temporal_window`, `holdout_temporal_horizon`, `seed_stability_regional`, `seed_stability_geometry` |
