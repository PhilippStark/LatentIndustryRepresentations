# shared_cosine_v16 — main paper run

The embedding all paper results are based on (`data/embedding/shared_cosine_v16/`).
Every script's defaults are this run's settings.

## Graph (`layers_v1`)

```
python scripts/data/build_labor_relatedness.py          # --min-labor-count 10 (default)
python scripts/data/build_ownership_relatedness.py
python scripts/data/build_input_output_relatedness.py
python scripts/graph/build_layer_transition_matrices.py --name layers_v1
python scripts/graph/build_layer_diffusion_targets.py --name layers_v1
```

- 2,614 nodes: 4-digit codes of SNI92/SNI02/SNI07 and their 3-digit, 2-digit
  and section hubs.
- A labor pair needs at least 10 pooled job movers (both directions, whole
  period). Kept pairs: SNI92 39,291 / SNI02 24,168 / SNI07 51,977.

## Diffusion targets (17)

| view | scales (t) | targets |
|---|---|---|
| labor | 1, 2, 4 | 9 (split by SNI version) |
| hierarchy | 1, 2, 4 | 3 (t = 1 is the ordinal 4-digit tier target) |
| ownership | 1, 4 | 2 |
| io | 1 | 1 (native 2-digit layer) |
| transition | 1 | 2 (SNI92↔SNI02, SNI02↔SNI07) |

## Training

```
python scripts/training/train_shared_cosine_embedding.py --name layers_v1 --run-name shared_cosine_v16
```

which is, spelled out:

```
--embedding-dim 64 --text-prior-model sentence-transformers/all-MiniLM-L6-v2 \
--lambda-labor 0.30 --lambda-io 0.25 --lambda-hierarchy 0.10 \
--lambda-transition 0.10 --lambda-ownership 0.125 \
--saturation-frac 0.2 --reweight-stall-ratio 2.0 --reweight-max-ratio 20 \
--reweight-decay 0.9 --reweight-stall-min-improvement 0.1 \
--lr 0.01 --epochs 500 --seed 42 --eval-every 25 --log-every 100
```

- Z starts from each node's SCB description encoded with all-MiniLM-L6-v2
  (384-d), reduced to 64-d by PCA (76.6% of variance) and re-normalised. No
  loss term ties Z to this start afterwards.
- A view's weight is divided among its targets in proportion to the inverse
  of each target's null loss. With these weights hierarchy_t2 and
  hierarchy_t4 receive about 0.0006 each; hierarchy acts almost entirely
  through the ordinal target hierarchy_t1.
- Each target has a learnable temperature τ bounded to [e^-1, e].
- Single seed (42).

## Reproducibility

Trained on CPU (LUNARC, lu48 partition), seed 42. Training is deterministic on
a given machine but not bitwise across machines: small floating-point
differences grow over 500 epochs. A retrain on other hardware changes the P02
evaluation metrics by at most 0.012 (SNI02→SNI07 R@1), most by less than
0.005. The paper's results use the released embeddings.

## Evaluation

```
python scripts/evaluation/P01_source_complementarity.py
python scripts/evaluation/P02_shared_geometry_reconstruction.py --run-name shared_cosine_v16
python scripts/evaluation/P03_emergent_structure.py --run-name shared_cosine_v16
python scripts/evaluation/build_paper_tables.py --run-name shared_cosine_v16
```

P02's baseline is the untrained all-MiniLM-L6-v2 encoding (384-d), i.e. the
text prior before the PCA to 64-d. R01's "baseline" density is the 64-d PCA
text prior Z actually started from.

## Regional evaluation

`P04_entry.py`, `P04_exit.py`, `P04_startup_entry.py` (log-employment-weighted
density, each script's default), `P05_opportunity_space.py`,
`R01_single_source_density.py` and `R02_fixed_effects_robustness.py`. Growth
and wage growth are appendix-only.

## Robustness

- `R02_fixed_effects_robustness.py`: alternative fixed effects
  (`regional_fe_robustness/`).
- `scripts/robustness/` (details in its README): spatial hold-out (5 folds × 3
  repeats, labor and ownership rebuilt without the test municipalities;
  `regional_spatial_holdout/`), temporal hold-out (embeddings trained on
  register years ≤ T0 = 2015–2018, test rows and forecast horizon;
  `regional_temporal_holdout/`), and seed stability (10 seeds, `--init-noise
  0.5`; `seed_stability/`). Checks of the pipeline in `holdout_checks/`, SI
  tables in `paper/appendix/holdout_*` and `paper/appendix/seed_stability_*`.
  The variant embeddings stay on LUNARC (`ablations/holdout_robustness/`).
