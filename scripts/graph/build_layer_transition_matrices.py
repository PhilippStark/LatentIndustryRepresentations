"""build_layer_transition_matrices.py — stage 1 of the graph pipeline, run
ONCE per --name.

Builds the node index and one row-stochastic transition matrix per
independent evidence layer (labor, nace_text, io, hierarchy, transition,
ownership — see src/latent_industry/graph/layers.py), plus the repulsion matrix
(verified "officially excludes" pairs). None of this depends on any
diffusion/combination parameter, so it doesn't need rebuilding just because
you're retuning how the layers get combined — see
scripts/graph/build_multilayer_diffusion.py (stage 2) for that.

Usage
-----
  python scripts/graph/build_layer_transition_matrices.py --name multilayer_v1

Outputs (under <LUNARC_DATA_ROOT>/processed/, see src/latent_industry/graph/artifacts.py)
-------
  layers_{name}.npz                     per-layer P/avail/domain + S_neg
  graph_nodes_{name}.parquet            full node table
  graph_edges_{name}.parquet            every layer's raw edges, with a leakage-safe
                                         train/val/test split column
  graph_negative_edges_{name}.parquet   diagnostic-only negative NACE cosine pairs
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import PROCESSED_DATA_DIR
from latent_industry.common.splits import split_edge_table
from latent_industry.graph import artifacts
from latent_industry.graph.layers import build_all_layers, build_node_concept_groups
from latent_industry.graph.nodes import NodeIndex
from latent_industry.graph.repulsion import build_repulsion_matrix

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Version name used in all output filenames.")
parser.add_argument("--nace-embedding-file", default="nace_embeddings_openai.npz",
                     help="Precomputed NACE embeddings in metadata/ (see build_nace_embeddings.py).")
parser.add_argument("--nace-sharpen", type=float, default=1.0,
                     help="NACE cosine similarity power applied before row-normalising (1.0 = no-op). "
                          "See layers.build_nace_layer's docstring for why only this one layer needs it.")
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
parser.add_argument("--split-seed", type=int, default=0)
parser.add_argument("--split-train-frac", type=float, default=0.7)
parser.add_argument("--split-val-frac", type=float, default=0.15)
args = parser.parse_args()

print(f"\n{'=' * 60}\nLayer transition matrices  |  name={args.name}\n{'=' * 60}\n")

ni = NodeIndex()
print(ni.summary())

layers, nace_ctx = build_all_layers(
    ni,
    args.nace_embedding_file,
    nace_sharpen=args.nace_sharpen,
    labor_relatedness_file=args.labor_relatedness_file,
    io_relatedness_files={
        "SNI07": args.io_relatedness_file_sni07,
        "SNI02": args.io_relatedness_file_sni02,
    },
    ownership_relatedness_file=args.ownership_relatedness_file,
)

print("\nRepulsion …")
S_neg = build_repulsion_matrix(nace_ctx, ni.target_ids, ni.target_pos)

print("\nAssembling edge table …")
edges_df = pd.concat([layer.edges for layer in layers.values()],
                      ignore_index=True) if layers else pd.DataFrame()
if not edges_df.empty:
    concept_groups = build_node_concept_groups(ni)
    edges_df = split_edge_table(edges_df, "src", "dst", group=concept_groups,
                                 seed=args.split_seed, train_frac=args.split_train_frac,
                                 val_frac=args.split_val_frac)
    print("  Edge counts by layer:")
    for lname, cnt in edges_df["layer"].value_counts().items():
        print(f"    {lname:<12}: {cnt:,}")
    print("  Edge counts by type (nace_text splits into its 4 levels here):")
    for tname, cnt in edges_df["type"].value_counts().items():
        print(f"    {tname:<12}: {cnt:,}")
    print(f"  Split (seed={args.split_seed}): " +
          "  ".join(f"{s}={cnt:,}" for s, cnt in edges_df["split"].value_counts().items()))

negative_edges_df = nace_ctx.negative_edges

print("\nSaving …")
artifacts.save_layers(args.name, ni, layers, S_neg)
artifacts.save_graph_tables(args.name, ni, edges_df, negative_edges_df)


def _resolve(path_or_name: str) -> Path:
    path = Path(path_or_name)
    return path if path.is_absolute() else PROCESSED_DATA_DIR / path


def _file_record(path_or_name: str) -> dict:
    path = _resolve(path_or_name)
    record = {"path": str(path), "exists": path.exists()}
    if path.exists():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        record.update({"bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        metadata_path = path.with_suffix(".metadata.json")
        record["source_metadata_path"] = str(metadata_path)
        if metadata_path.exists():
            record["source_metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            record["source_metadata"] = None
    return record


provenance = {
    "name": args.name,
    "labor_relatedness": _file_record(args.labor_relatedness_file),
    "io_relatedness": {
        "SNI07": _file_record(args.io_relatedness_file_sni07),
        "SNI02": _file_record(args.io_relatedness_file_sni02),
    },
    "nace_embedding_file": args.nace_embedding_file,
    "nace_sharpen": args.nace_sharpen,
    "split_seed": args.split_seed,
    "split_train_frac": args.split_train_frac,
    "split_val_frac": args.split_val_frac,
}
(PROCESSED_DATA_DIR / f"layer_sources_{args.name}.json").write_text(
    json.dumps(provenance, indent=2), encoding="utf-8"
)
print(f"  layers_{args.name}.npz")
print(f"  graph_nodes_{args.name}.parquet            ({len(ni.nodes)} nodes)")
print(f"  graph_edges_{args.name}.parquet            ({len(edges_df):,} edges)")
print(f"  graph_negative_edges_{args.name}.parquet   ({len(negative_edges_df):,} diagnostic pairs)")
print(f"  layer_sources_{args.name}.json")
print("\nDone. Run build_multilayer_diffusion.py --name " + args.name + " next.")
