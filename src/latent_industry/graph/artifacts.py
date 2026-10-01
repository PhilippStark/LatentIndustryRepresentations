"""artifacts.py — save/load for the two pipeline stages' on-disk outputs.

Stage 1 (build_layer_transition_matrices.py, run once per `name`, cached):
  layers_{name}.npz                     one P/avail/domain per layer, + S_neg
  graph_nodes_{name}.parquet            full node table
  graph_edges_{name}.parquet            every layer's raw (pre-normalisation) edges
  graph_negative_edges_{name}.parquet   diagnostic-only negative NACE cosine pairs

Stage 2 (build_layer_diffusion_targets.py, reads stage 1's cache):
  layer_diffusion_expanded_{name}.npz   one P_l^(t)/domain/family per
                                         (layer, t) pair, each its own target —
                                         see
                                         layer_diffusion.build_layer_diffusion_targets_expanded.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from latent_industry.common.paths import PROCESSED_DATA_DIR
from latent_industry.graph.layers import Layer
from latent_industry.graph.nodes import NodeIndex


def _layers_path(name: str) -> Path:
    return PROCESSED_DATA_DIR / f"layers_{name}.npz"


def save_layers(name: str, ni: NodeIndex, layers: dict[str, Layer], S_neg: np.ndarray) -> None:
    payload = {"layer_names": np.array(sorted(layers)), "target_ids": np.array(ni.target_ids),
               "S_neg": S_neg.astype(np.float32)}
    for lname, layer in layers.items():
        payload[f"{lname}__P"] = layer.P.astype(np.float32)
        payload[f"{lname}__avail"] = layer.avail
        payload[f"{lname}__domain"] = layer.domain
    np.savez_compressed(_layers_path(name), **payload)


def load_layers(name: str) -> tuple[dict[str, Layer], np.ndarray, list[int]]:
    """Loads cached layer transition matrices for stage 2. Raises a clear
    error if stage 1 hasn't been run yet for this name — the old failure
    mode (silently reading a stale or wrong-name file) is what NAME
    threading in run_pipeline.py exists to prevent; this is the same
    protection at the layer-cache boundary."""
    path = _layers_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/graph/build_layer_transition_matrices.py "
            f"--name {name} first (stage 1, run once; this stage reads its cached output)."
        )
    data = np.load(path, allow_pickle=False)
    layer_names = [str(x) for x in data["layer_names"]]
    layers = {}
    for lname in layer_names:
        layers[lname] = Layer(
            name=lname,
            P=data[f"{lname}__P"].astype(np.float64),
            avail=data[f"{lname}__avail"],
            domain=data[f"{lname}__domain"],
            edges=pd.DataFrame(columns=["src", "dst", "weight", "type"]),  # provenance only, not cached here
        )
    return layers, data["S_neg"], data["target_ids"].tolist()


def save_graph_tables(name: str, ni: NodeIndex, edges_df: pd.DataFrame,
                       negative_edges_df: pd.DataFrame) -> None:
    ni.nodes.to_parquet(PROCESSED_DATA_DIR / f"graph_nodes_{name}.parquet", index=True)
    edges_df.to_parquet(PROCESSED_DATA_DIR / f"graph_edges_{name}.parquet", index=False)
    negative_edges_df.to_parquet(PROCESSED_DATA_DIR / f"graph_negative_edges_{name}.parquet", index=False)


def load_raw_edges(name: str, layer: str | None = None) -> pd.DataFrame:
    """Stage 1's pre-normalisation edge table (graph_edges_{name}.parquet),
    optionally filtered to one layer. Used by layer_diffusion.py's
    cross-granularity lift, which needs `io`'s *raw* weights — the cached
    `Layer.P` in layers_{name}.npz has already been row-normalised, and
    lifting must happen before that normalisation (math doc §2.1)."""
    path = PROCESSED_DATA_DIR / f"graph_edges_{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/graph/build_layer_transition_matrices.py "
            f"--name {name} first (stage 1; this reads its cached edge table)."
        )
    edges = pd.read_parquet(path)
    return edges[edges["layer"] == layer].reset_index(drop=True) if layer is not None else edges


def _layer_diffusion_expanded_path(name: str) -> Path:
    return PROCESSED_DATA_DIR / f"layer_diffusion_expanded_{name}.npz"


def save_layer_diffusion_targets_expanded(
    name: str, targets: dict[str, np.ndarray], domains: dict[str, np.ndarray],
    families: dict[str, str], layer_scales: dict[str, tuple[int, ...]],
) -> None:
    """One entry per `target_id = f"{layer}_t{t}"` — see
    `layer_diffusion.build_layer_diffusion_targets_expanded`. `families`
    (target_id -> layer name) is saved as data, not just provenance: it's
    read back by `operator_data.load_operator_data_expanded` to group
    per-target loss weights by layer."""
    target_ids = sorted(targets)
    payload = {
        "target_ids": np.array(target_ids),
        "families_json": np.array(json.dumps(families)),
        "layer_scales_json": np.array(json.dumps({l: list(s) for l, s in layer_scales.items()})),
    }
    for target_id in target_ids:
        payload[f"{target_id}__P"] = targets[target_id].astype(np.float32)
        payload[f"{target_id}__domain"] = domains[target_id]
    np.savez_compressed(_layer_diffusion_expanded_path(name), **payload)


def load_layer_diffusion_targets_expanded(
    name: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """Returns (targets, domains, families), all keyed by target_id
    (f"{layer}_t{t}") — see save_layer_diffusion_targets_expanded."""
    path = _layer_diffusion_expanded_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/graph/build_layer_diffusion_targets.py "
            f"--name {name} --expanded first."
        )
    data = np.load(path, allow_pickle=False)
    target_ids = [str(x) for x in data["target_ids"]]
    targets = {t: data[f"{t}__P"].astype(np.float64) for t in target_ids}
    domains = {t: data[f"{t}__domain"] for t in target_ids}
    families = json.loads(str(data["families_json"]))
    return targets, domains, families


