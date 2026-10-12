# Code audit: removals for the shared\_cosine\_v16 main run

Oct 12, 2026 · @Phil

## Scope

This audit lists code and outputs in LatentIndustryRepresentations that the shared\_cosine\_v16 main run does not use. Line numbers refer to commit `16dbc5c`.

The main run executes, in order:

1. `build_labor_relatedness`, `build_ownership_relatedness`, `build_input_output_relatedness`
2. `build_layer_transition_matrices --name layers_v1`
3. `build_layer_diffusion_targets`
4. `train_shared_cosine_embedding` with the v16 flags
5. `P01`, `P02`, `P03`
6. Regional `P04`, `P05`, `R01`

Everything below is either outside that path or only kept alive by zero-weight views.

## A. Dead concepts: remove completely

Fourteen groups of code and outputs are not part of the main run and can be deleted outright.

### 1. Train/val/test split

- [ ] `src/latent_industry/common/splits.py`: the whole file.
- [ ] `layers.py` L170–186 (`build_node_concept_groups`).
- [ ] `build_layer_transition_matrices.py`: L73–75 (`--split-*` flags), L101–113 (assigning the split), L154–156 (split entries in the provenance file).
- [ ] The `split` column in `graph_edges_*.parquet`, and its mention in `data/README.md` L33.

### 2. Repulsion / `S_neg` / fused diffusion matrix

- [ ] `graph/repulsion.py`: the whole file.
- [ ] `build_layer_transition_matrices.py` L95–96.
- [ ] `artifacts.py` L29–31 and L61 (`S_neg` is written into `layers_*.npz` and returned by `load_layers`).
- [ ] P02 section J, L401–503. It looks for `diffusion_matrix_*.npz`, which nothing produces.
- [ ] `_shared_eval_context.py`:
  - L495–570: `_align_to_diffusion_matrix`, `_SKIP_J2_REGIMES`, `repulsion_and_fidelity_metrics`.
  - L717–732: `K_fused`.
  - L858.
  - `ctx.training_regime`, L205–217. Its only purpose is to skip J2.
- [ ] The `"regime"` field in the training provenance file, and `metadata/README.md` L42.

### 3. The nace\_text view

nace\_text has weight 0. It is also what makes the loss NaN and leaves `best_loss` at Infinity.

- [ ] `layers.py` L43–52 (`NaceEmbeddingContext`), L270–447 (centering, loading, encoding, `_level_pairs`, `build_nace_layer`) and L474 (where it is called).
- [ ] `build_layer_transition_matrices.py` L47–51 (`--nace-embedding-file`, `--nace-sharpen`), and the `graph_negative_edges_*.parquet` output.
- [ ] nace\_text in the training code:
  - `operator_data.py` L21. This also auto-generates `--lambda-nace_text`.
  - `build_layer_diffusion_targets.py` L344 (`LAYER_NAMES`).
  - `operator_diagnostics.py` L258.
- [ ] Evaluation: P02 section H (L279–346) and `_shared_eval_context.py` L436–453, L691, L848.
- [ ] `nodes.py` L233–236 (the `sni07_3d/2d/sec` aliases). The NACE builder is their only user.

Keep the MiniLM text prior used to initialise Z, and the P02 baseline. Both are separate from this view.

### 4. Trade/innovation remnants

No code is left, but the outputs still carry them.

- [ ] `training_provenance.json` and `loss_metric_correlation.csv`.
- [ ] P02 metrics sections E3 and I2.
- [ ] `data/README.md` L31, which says `layers_layers_v1.npz` contains trade and innovation views. If so, the stored graph and target npz files still carry them.
- [ ] The comment in `build_paper_tables.py` L218.

### 5. `best_loss`

- [ ] `operator_trainer.py` L177, L233–234 and the return value.
- [ ] `train_shared_cosine_embedding.py` L242, L249, L304.

### 6. Alternative observation models and losses

- [ ] `operator_observation_models.py` L24–66 and L95–105: bilinear model, MLP model, registry, factory.
- [ ] `operator_losses.py` L43–45 (`mse_loss`) and the `--io-loss` flag (`train_shared_cosine_embedding.py` L115–116).
- [ ] `_KIND_BY_CLASS` and `model_kinds` in `operator_diagnostics.py` L32 and L44.

### 7. Leftovers from a fine-tuned sentence-transformer version

- [ ] `_shared_eval_context.py` L148–171 (`load_model`, with its "FINE-TUNED" branch and `model_final`).
- [ ] `--model-dir` and `DEFAULT_MODEL_DIR` (L53, L93–94, L200–203).
- [ ] The encode-on-the-fly fallback in `load_or_encode`.
- [ ] The "Fine-tuned" column header in P02 section G, and the wording in the P02 H/I comments.

### 8. Combined transition layer

- [ ] `layers.py` L146–167 is built and saved, then thrown away again (`layer_diffusion.py` L261). The two pair layers are what training actually uses. Build only those, in stage 1.

### 9. Lifting IO to 4-digit, and raw edge tables

- [ ] `layer_diffusion.py` L27–102 (`build_membership_matrix`, `lift_to_finer_domain`, `_edges_to_matrix`, `lift_io_layer`). Keep `_mask_4d_sni07`: the ordinal hierarchy tiers use it.
- [ ] `artifacts.load_raw_edges`.
- [ ] Their only user is `diagnose_scale_redundancy.py`. Archive that script or delete it.

### 10. Graph tables nobody reads

- [ ] `graph_edges_*` and `graph_nodes_*.parquet`, `save_graph_tables`, `Layer.edges`, `_symmetric_edge_table`. Their only readers are P02 H (nace) and the diagnostic script.
- [ ] `Layer.avail` is saved but never used by training, which recomputes active rows in `operator_data.py` L129.

### 11. Unused diagnostics and outputs

- [ ] `operator_diagnostics.py` L53–133: `load_training_state`, `layer_reconstruction_report`, `_row_reconstruction_metrics`, `DEFAULT_TOP_KS`.
- [ ] `operator_state.pt`: nothing reads it.
- [ ] `ownership_flows_by_year.parquet` and `io_flows_by_year.parquet`: nothing reads them.

`symmetric_flow_phi` in `flow_metrics.py` is used by ownership and IO, so keep it.

### 12. Flags that do nothing

- [ ] `--expanded` (`build_layer_diffusion_targets.py` L349–350), plus the error message in `artifacts.py` L121 that still tells you to pass it.
- [ ] `--scales-transition` and `--scales-io`, which only print a warning (L365–372).

### 13. Never referenced anywhere

- [ ] `CONFIG_PATH` (`paths.py` L18).
- [ ] `format_p` (`fixed_effects.py` L186).
- [ ] `_pairwise_lookup_source` (`R01` L266).

### 14. Stale result files

These were produced by older code than what is in the repo now.

- [ ] `results/embedding_eval/shared_cosine_v16/*_P02_metrics.txt` contains sections E3, I2 and J, which no longer exist in the code.
- [ ] `results/source_complementarity/`: `P01_diffusion_overlap_matrix.csv` (includes nace\_text) and `P01_e_matrix.csv` are not produced by the current P01, and `P01_c_matrix.csv` is missing. The paper uses `results/paper/result1_*`, so this folder can go.

## B. Options the main run does not use

These are not artifacts, but remove them if the code should be main-run-only. If you keep them, set the defaults to the v16 values.

- [ ] Random initialisation and the non-PCA branch of the Z initialisation (`train_shared_cosine_embedding.py` L145–158, L215–216).
- [ ] Defaults that differ from v16: `--saturation-frac 0`, dynamic reweighting off, `--reweight-stall-ratio 5.0` (v16 uses 2.0), equal `DEFAULT_LAMBDAS`. As things stand, running the script with no flags does not reproduce the main run.
- [ ] `is_family_excluded` and the "SKIPPED" branches in P02. These are ablation machinery.
- [ ] Fallback IO column names (`layers.py` L244).

## C. Code that is used but documented wrongly

These pieces run in the main run, but their help text, comments or docstrings describe something else.

| Where | Says | Actually does |
| --- | --- | --- |
| `train_shared_cosine_embedding.py` L117–121, L182 (`--lambda` help, log message) | Family weight is "split evenly" across its targets | Splits by inverse null loss (`operator_data.py` L140–154) |
| `train_shared_cosine_embedding.py` L91–94 (`--dynamic-reweight` help) | Normalises "instead of its epoch-1 loss" | Divides by the epoch-1 loss (`operator_trainer.py` L28–34) |
| P02 section K, L507 | Uses "the trainer's own tau" | Uses a fixed 0.05; training learns τ per target within \[0.37, 2.72\] |
| P02 docstring L25–26 | Zero-weight families are still evaluated | The code skips them |
| `flow_metrics.py` docstring | Labor uses `symmetric_flow_phi` | Labor uses Poisson observed/expected |
| `build_layer_diffusion_targets.py` main-run usage line (`--scales 1 2 4`) | Reads as if the scales apply to every view | Ownership is not in `LAYER_NAMES`, so it always gets (1, 4) |
| R01 vs P02 "baseline" | Same name | R01: 64-d PCA text prior. P02: raw 384-d MiniLM |
| P02 L170–172 | Ownership is "not split per training version yet" | It is split per version |

## After cleaning

Removing all of the above should leave the v16 embeddings unchanged, up to floating-point noise. None of the removed targets carries weight or gradient, and none of them uses random numbers.

- [ ] Rebuild `layers_v1` and the diffusion targets with the cleaned code.
- [ ] Rerun the v16 training command.
- [ ] Compare the old and new `embeddings_*.npz` to confirm they match.
- [ ] Rerun P01–P03, so the provenance file, `loss_metric_correlation.csv` and the metrics files are free of the old remnants.

Three points from the earlier review concern method rather than leftover code, and can be decided separately:

- Only one seed (42) was run.
- The τ bound limits how well the transition targets can be fitted.
- Hierarchy t2/t4 get about 0.0006 weight each, so they are effectively off.
