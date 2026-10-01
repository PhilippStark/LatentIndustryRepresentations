# shared_cosine_v16 — main paper run

This is the current main/production embedding run for the project. The name
deliberately carries no parameter details (unlike its predecessor,
`shared_cosine_v15_textprior_pca_no_nace_trade_innovation`) — this file is
the authoritative record of what it actually is.

Promoted to main on 2026-08-31, superseding `shared_cosine_v15_textprior_pca_
no_nace_trade_innovation` (moved to `DataLatentIndustryRepresentations/
ablations/embedding/` as a preserved comparison baseline, same name, still
runnable/queryable there).

## Graph

- `--name layers_v1` (`build_layer_transition_matrices.py`), same node
  universe as v14/v15 (2,614 nodes).
- **Labor edge threshold raised 3 → 10**: `build_labor_relatedness.py
  --min-labor-count 10` (previously hardcoded at 3, no CLI flag — the flag
  was added for this change). A labor pair now needs ≥10 pooled worker
  movers (both directions, full period) to be kept at all, versus ≥3
  before. Kept pairs: SNI92 39,291 / SNI02 24,168 / SNI07 51,977.
- **trade/innovation evidence families removed from the codebase entirely**
  (2026-08-31) — not just excluded via zero lambda as in v15, but deleted:
  `build_trade_layer`/`build_innovation_layer` and their CLI flags no
  longer exist. They contributed exactly zero training gradient in v15
  anyway (`--lambda-trade 0.0 --lambda-innovation 0.0`), so removing them
  changes nothing about what was actually trained.

## Diffusion scales (`build_layer_diffusion_targets.py --expanded`)

| family | scales (t) | note |
|---|---|---|
| labor | 1, 2, 4 | widened from v15's t=1 only (9 targets: 3 versions × 3 scales) |
| hierarchy | 1, 2, 4 | widened from v15's t=(1,4) |
| nace_text | 1, 2, 4 | zero training weight (see lambdas below) — scale choice is moot |
| ownership | 1, 4 | unchanged production default (`EXPANDED_DEFAULT_SCALES`) |
| io | 1 | hard-pinned (`IO_NATIVE_SCALES`) — native 2d-hub layer, not overridable |
| transition | 1 | hard-pinned (`TRANSITION_PAIR_SCALES`) — direct correspondence, not overridable |

## Training (`train_shared_cosine_embedding.py`)

```
--name layers_v1 --run-name shared_cosine_v16 \
--text-prior-init --text-prior-pca --embedding-dim 64 \
--lambda-hierarchy 0.10 --lambda-transition 0.10 --lambda-labor 0.30 \
--lambda-io 0.25 --lambda-nace_text 0.0 --lambda-ownership 0.125 \
--dynamic-reweight --reweight-stall-ratio 2.0 --saturation-frac 0.2 \
--eval-every 25 --epochs 500 --seed 42 --log-every 100
```

- Z warm-started from each node's own NACE text description
  (`sentence-transformers/all-MiniLM-L6-v2`, 384d → PCA to 64d) instead of
  random noise — no ongoing anchor loss ties Z back to this afterward.
- `--lambda-ownership 0.125` is explicit here.

## Regional evaluation

Core analysis only: `P04_entry.py`, `P04_exit.py`, `P04_startup_entry.py`
(`--density-var ew --log-employment-weights`, both already each script's
default). Growth/wage-growth are appendix-only and not part of this run's
primary results.

## Provenance

- Built and trained by `CPU_scale_ablation_v15_t124816.sh` (job 3551807,
  2026-08-28), under the working name
  `shared_cosine_v15_textprior_pca_no_nace_trade_innovation_scale_t124_labormin10`.
- Renamed and promoted to `shared_cosine_v16` on 2026-08-31; its embedding,
  evaluation, and regional-outcome artifacts were moved from
  `DataLatentIndustryRepresentations/ablations/` into the main
  `DataLatentIndustryRepresentations/` tree and `results/` under the new
  name at that point.
- Note for reproducibility: during this same job, a dangling-symlink bug in
  an earlier draft of the ablation script caused production's `layers_v1`
  graph files (`labor_relatedness.parquet`, `layers_layers_v1.npz`,
  `graph_edges/nodes/negative_edges_layers_v1.parquet`) to be silently
  overwritten in place with this run's threshold=10 build — i.e. this is
  also why production no longer has a threshold=3 copy of these files
  on disk. The original threshold=3 build is still fully reproducible
  (`build_labor_relatedness.py` with no `--min-labor-count` flag, default
  3, followed by `build_layer_transition_matrices.py`) if ever needed for
  comparison.
