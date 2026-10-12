# Latent Industry Representations
This project develops latent industry representations to study regional path development by integrating multiple forms of industry relatedness, including worker mobility, semantic similarity, hierarchical structures, and input–output relations. It further introduces an opportunity space framework that links these latent industry structures to regional economic composition, enabling the analysis of industry emergence, decline, and regional diversification within a shared latent space.

## Data
The learned industry embeddings, the matrices representing the relational views, and the evaluation datasets are in [`data/`](data/README.md). Set `LATENT_DATA_ROOT` to that folder to run the code against them.

## Code
Python 3.12; dependencies in `requirements.txt`. The main run is `shared_cosine_v16` on the graph `layers_v1` (details in [`results/shared_cosine_v16_SPECS.md`](results/shared_cosine_v16_SPECS.md)). Every script's defaults are the main run's settings.

```bash
export LATENT_DATA_ROOT="$(pwd)/data"

# 1. Relatedness sources (restricted SCB register data; public IO tables)
python scripts/data/build_labor_relatedness.py
python scripts/data/build_ownership_relatedness.py
python scripts/data/build_input_output_relatedness.py

# 2. Industry graph: one transition matrix per view, then the diffusion targets
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
```

Steps 2–4 run on the released files in `data/` without access to the restricted data.

| Folder | Content |
|---|---|
| `src/latent_industry/` | Shared code: SNI codes and municipalities (`common/`), industry graph and diffusion targets (`graph/`), embedding model and training (`training/`), regional densities and fixed-effects estimation (`regional/`) |
| `scripts/` | Entry points, in pipeline order as above; `scripts/paper/` builds the paper's figures and tables |
| `metadata/` | SNI code descriptions, version crosswalks, municipality tables |
| `results/` | Evaluation output, regional estimates, and the paper's figures and tables (`results/paper/`) |
