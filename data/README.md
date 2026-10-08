# Data

Data supporting the analyses in the paper: the learned industry embeddings,
the matrices representing the relational views, and the evaluation datasets.
All files belong to the paper's main run, `shared_cosine_v16` (graph
`layers_v1`; see `results/shared_cosine_v16_SPECS.md`).

The layout mirrors the project's data root (`embedding/`, `processed/`), so
the code reads these files directly when pointed at this folder:

```bash
export LATENT_DATA_ROOT="$(pwd)/data"   # see src/latent_industry/common/paths.py
```

## Contents

### Learned industry embeddings — `embedding/shared_cosine_v16/`

| File | Content |
|---|---|
| `embeddings_sni92.npz`, `embeddings_sni02.npz`, `embeddings_sni07.npz` | 64-d industry embeddings per classification version. Keys: `embeddings` (n × 64, float), `codes` (n, 4-digit SNI code). |
| `embeddings_sni07_section.npz`, `embeddings_sni07_2d.npz`, `embeddings_sni07_3d.npz` | Embeddings of the SNI 2007 hierarchy nodes (21 sections, 88 2-digit and 272 3-digit codes). |
| `operator_state.pt` | Trained model state (PyTorch). |
| `training_provenance.json` | Full training configuration (loss types, weights, margins, seed). |
| `hub_coherence.csv`, `loss_metric_correlation.csv` | Training diagnostics. |

### Relational views — `processed/`

| File | Content |
|---|---|
| `layers_layers_v1.npz` | Row-stochastic transition matrices of the relational views over the shared 2,614-node graph (4-digit codes of SNI92/02/07 plus hierarchy hubs). Per view `<v>`: `<v>__P` (2614 × 2614), `<v>__avail`, `<v>__domain`. Views: labor, nace_text, io, hierarchy, transition, ownership, trade, innovation (trade, innovation and nace_text carry zero training weight in v16). |
| `layer_diffusion_expanded_layers_v1.npz` | The 24 multi-scale diffusion targets the v16 embedding was trained on (`<view>_t<scale>__P`, 2614 × 2614). Target list in `target_ids`. |
| `graph_nodes_layers_v1.parquet`, `graph_edges_layers_v1.parquet`, `graph_negative_edges_layers_v1.parquet` | Node universe and edge lists of the multi-relational graph (`src`, `dst`, `weight`, `type`, `layer`, `split`). |
| `labor_relatedness.parquet` | Labor-flow relatedness between 4-digit industries (`sni_from`, `sni_to`, `phi`, `coding`). Industry pairs with fewer than 10 pooled job movers are excluded. |
| `ownership_relatedness.parquet` (+ `.metadata.json`) | Ownership (corporate group) relatedness between 4-digit industries. |
| `io_relatedness_SNI07.parquet` | Input–output relatedness between 2-digit NACE Rev. 2 industries (SCB and Eurostat tables). |
| `io_relatedness_SNI02.parquet` | Input–output relatedness between 2-digit NACE Rev. 1.1 industries (WIOD tables). |
| `nace_embeddings_openai.npz` | Text embeddings of the official NACE Rev. 2 descriptions, input to the `nace_text` view. |

### Evaluation datasets — `processed/downstream_outcomes/`

Industry × municipality × year panels for the regional evaluation (290
municipalities, base years *t* = 1990–2018 with outcomes at *t*+1, split into
SNI92/SNI02/SNI07 classification periods).

| Folder | Outcome column | Definition |
|---|---|---|
| `entry/` | `entry_industry` | Candidates: industries with zero employment in the municipality at *t*. 1 if employment reaches a non-negligible level at *t*+1 |
| `exit/` | `exit_industry` | Candidates: industries present in the municipality at *t*. 1 if employment drops to zero by *t*+1 |
| `startup_entry/` | `startup_entry` | Same candidates as `entry/`. 1 if a genuinely new firm (SCB firm-demography definition, following Grillitsch 2021) is founded in the cell at *t*+1 |

Each panel has the columns `period`, `year`, `municipality`, `industry`,
`density` and the outcome. `density` is the relatedness density of the
industry to the municipality's existing employment structure in the v16
embedding space. File variants differ only in the density specification:

| File | Density specification |
|---|---|
| `shared_cosine_v16_panel.parquet` | Baseline: employment-weighted mean cosine similarity, log(1+employment) weights |
| `shared_cosine_v16_ew_raw_panel.parquet` | Employment-weighted, raw employment weights |
| `shared_cosine_v16_geo_log_panel.parquet` | Geometric-mean analogue, log employment weights |
| `shared_cosine_v16_geo_raw_panel.parquet` | Geometric-mean analogue, raw employment weights |

The paper's regressions are linear probability models of the outcome on
standardised `density` per classification period, with municipality×year and
industry fixed effects and standard errors clustered by municipality and
industry (`src/latent_industry/regional/fixed_effects.py`). They can be
re-estimated from these panels alone:

```bash
LATENT_DATA_ROOT=$PWD/data python scripts/regional/P04_entry.py --from-cache
LATENT_DATA_ROOT=$PWD/data python scripts/regional/R02_fixed_effects_robustness.py
```

## Restricted source data

The embeddings, relational views and evaluation panels are aggregates derived
from Swedish register data (Statistics Sweden) and public input–output and
classification tables. They contain no individual- or firm-level records. The
underlying raw data cannot be publicly shared due to legal, contractual and
data-access restrictions. Researchers seeking access to the restricted data
should apply directly to the respective data providers under their
established access procedures. Scripts that build these files from the raw
data (`scripts/data/`, `scripts/graph/`, `scripts/regional/P04_*.py` without
`--from-cache`) therefore need that access.
