"""
build_layer_transition_matrices.py
==================================
Stage 1 of the graph pipeline: builds the node index of the shared industry
graph (4-digit codes of SNI92/SNI02/SNI07 plus their 3-digit, 2-digit and
section hubs) and one row-stochastic transition matrix per relational view
(labor, ownership, io, hierarchy, transition_SNI92_SNI02,
transition_SNI02_SNI07; see src/latent_industry/graph/layers.py).

Inputs: the relatedness tables from scripts/data/ (labor, IO, ownership) and
the SNI mappings/crosswalks in metadata/.

Outputs (<LATENT_DATA_ROOT>/processed/)
---------------------------------------
  layers_{name}.npz          per-view P / domain (src/latent_industry/graph/artifacts.py)
  layer_sources_{name}.json  paths and SHA-256 of the input relatedness files

Usage (main run)
----------------
  python scripts/graph/build_layer_transition_matrices.py --name layers_v1
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import PROCESSED_DATA_DIR
from latent_industry.graph import artifacts
from latent_industry.graph.layers import build_all_layers
from latent_industry.graph.nodes import NodeIndex

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Version name used in all output filenames.")
parser.add_argument(
    "--labor-relatedness-file",
    default="labor_relatedness.parquet",
    help="Absolute path or filename under PROCESSED_DATA_DIR.",
)
parser.add_argument(
    "--io-relatedness-file-sni07",
    default="io_relatedness_SNI07.parquet",
    help="Absolute path or filename under PROCESSED_DATA_DIR (SCB+Eurostat, 2008-2019).",
)
parser.add_argument(
    "--io-relatedness-file-sni02",
    default="io_relatedness_SNI02.parquet",
    help="Absolute path or filename under PROCESSED_DATA_DIR (WIOD, 1995/2000/2005). "
         "No --io-relatedness-file-sni92: no IO source reaches back that far.",
)
parser.add_argument(
    "--ownership-relatedness-file",
    default="ownership_relatedness.parquet",
    help="Absolute path or filename under PROCESSED_DATA_DIR (see build_ownership_relatedness.py).",
)
args = parser.parse_args()

print(f"\n{'=' * 60}\nLayer transition matrices  |  name={args.name}\n{'=' * 60}\n")

ni = NodeIndex()
print(ni.summary())

layers = build_all_layers(
    ni,
    labor_relatedness_file=args.labor_relatedness_file,
    io_relatedness_files={
        "SNI07": args.io_relatedness_file_sni07,
        "SNI02": args.io_relatedness_file_sni02,
    },
    ownership_relatedness_file=args.ownership_relatedness_file,
)

print("\nSaving …")
artifacts.save_layers(args.name, layers)


def _resolve(path_or_name: str) -> Path:
    path = Path(path_or_name)
    return path if path.is_absolute() else PROCESSED_DATA_DIR / path


def _file_record(path_or_name: str) -> dict:
    path = _resolve(path_or_name)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


provenance = {
    "name": args.name,
    "labor_relatedness": _file_record(args.labor_relatedness_file),
    "ownership_relatedness": _file_record(args.ownership_relatedness_file),
    "io_relatedness": {
        "SNI07": _file_record(args.io_relatedness_file_sni07),
        "SNI02": _file_record(args.io_relatedness_file_sni02),
    },
}
(PROCESSED_DATA_DIR / f"layer_sources_{args.name}.json").write_text(
    json.dumps(provenance, indent=2), encoding="utf-8"
)
print(f"  layers_{args.name}.npz  ({len(layers)} views, {ni.N} nodes)")
print(f"  layer_sources_{args.name}.json")
print("\nDone. Run build_layer_diffusion_targets.py --name " + args.name + " next.")
