"""
Repository-relative paths shared by all scripts.

METADATA_DIR (git-tracked) holds small reference tables: SNI code mappings,
section maps and cross-version correspondences. Data produced by the
pipeline or cached from external sources lives under LUNARC_DATA_ROOT
(raw/, processed/, embedding/). Set LATENT_DATA_ROOT to override that root,
e.g. to the repository's data/ folder, which mirrors its layout for the
released files.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
METADATA_DIR = PROJECT_ROOT / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data root: LUNARC project storage by default; override with the
# LATENT_DATA_ROOT environment variable (e.g. the repository's data/ folder).
LUNARC_DATA_ROOT = Path(os.environ.get(
    "LATENT_DATA_ROOT",
    "/lunarc/nobackup/projects/lu2025-17-39/DataLatentIndustryRepresentations",
))
RAW_DATA_DIR = LUNARC_DATA_ROOT / "raw"          # cached/downloaded external source files
PROCESSED_DATA_DIR = LUNARC_DATA_ROOT / "processed"  # pipeline-generated artifacts
EMBEDDING_DIR = LUNARC_DATA_ROOT / "embedding"   # trained embeddings, one folder per run
