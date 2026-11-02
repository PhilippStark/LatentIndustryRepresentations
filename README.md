# Latent Industry Representations
This project develops latent industry representations to study regional path development by integrating several relational sources: labour mobility, common ownership, input–output relations, the classification hierarchy, and sentence embeddings of the official industry descriptions. It further introduces an opportunity space framework that links these latent industry structures to regional economic composition, enabling the analysis of industry emergence, decline, and regional diversification within a shared latent space.

## Data
The learned industry embeddings, the matrices representing the relational sources, and the evaluation datasets are in [`data/`](data/README.md). Set `LATENT_DATA_ROOT` to that folder to run the code against them.

## Code
Python 3.12; dependencies in `requirements.txt`. The main run is `shared_cosine_v16` on the graph `layers_v1` (full configuration in [`data/embedding/shared_cosine_v16/training_provenance.json`](data/embedding/shared_cosine_v16/training_provenance.json)). Every script's defaults are the main run's settings.

```bash
export LATENT_DATA_ROOT="$(pwd)/data"

# 1. Relational sources (restricted SCB register data; public IO tables)
python scripts/data/build_labor_relatedness.py
python scripts/data/build_ownership_relatedness.py
python scripts/data/build_input_output_relatedness.py

# 2. Industry graph: one transition matrix per source, then the diffusion targets
python scripts/graph/build_layer_transition_matrices.py --name layers_v1
python scripts/graph/build_layer_diffusion_targets.py --name layers_v1

# 3. Shared embedding
python scripts/training/train_shared_cosine_embedding.py --name layers_v1 --run-name shared_cosine_v16

# 4. Evaluation (Results 1–3) and paper tables
python scripts/evaluation/P01_source_complementarity.py
python scripts/evaluation/P02_shared_geometry_reconstruction.py --run-name shared_cosine_v16
python scripts/evaluation/P03_emergent_structure.py --run-name shared_cosine_v16
python scripts/evaluation/build_paper_tables.py --run-name shared_cosine_v16

# 5. Regional analysis (restricted register data; P04_* also run on the released panels with --from-cache)
python scripts/regional/P04_entry.py      # likewise P04_exit.py, P04_startup_entry.py
python scripts/regional/P05_opportunity_space.py
python scripts/regional/R01_single_source_density.py --run-name shared_cosine_v16
python scripts/regional/R02_fixed_effects_robustness.py

# 6. Hold-out and seed robustness (SI; retrains on restricted data, see scripts/robustness/README.md)
python scripts/robustness/H3_evaluate_spatial.py      # likewise H4_evaluate_temporal.py, H5_evaluate_seeds.py
python scripts/robustness/H9_si_tables.py
```

Steps 2–4 run on the released files in `data/` without access to the restricted data.

| Folder | Content |
|---|---|
| `src/latent_industry/` | Shared code: SNI codes and municipalities (`common/`), industry graph and diffusion targets (`graph/`), embedding model and training (`training/`), regional densities and fixed-effects estimation (`regional/`) |
| `scripts/` | Entry points, in pipeline order as above; `scripts/paper/` builds the paper's figures and tables; `scripts/robustness/` holds the spatial/temporal hold-out and seed-stability checks ([README](scripts/robustness/README.md)) |
| `metadata/` | SNI code descriptions, version crosswalks, municipality tables |
| `results/` | Evaluation output, regional estimates and robustness checks (`regional_fe_robustness/`, `regional_spatial_holdout/`, `regional_temporal_holdout/`, `seed_stability/`, `holdout_checks/`), and the paper's figures and tables (`results/paper/`) |

## License and citation
The code is released under the [MIT License](LICENSE). To cite the code and data, use [`CITATION.cff`](CITATION.cff) (GitHub: "Cite this repository").
