"""
artifacts.py — save/load of the graph pipeline's outputs (PROCESSED_DATA_DIR).

Stage 1 (scripts/graph/build_layer_transition_matrices.py):
  layers_{name}.npz                     P / domain per view

Stage 2 (scripts/graph/build_layer_diffusion_targets.py):
  layer_diffusion_expanded_{name}.npz   P_l^t / domain / family per (view, t)
"""

import json
from pathlib import Path

import numpy as np

from latent_industry.common.paths import PROCESSED_DATA_DIR
from latent_industry.graph.layers import Layer


def _layers_path(name: str) -> Path:
    return PROCESSED_DATA_DIR / f"layers_{name}.npz"


def save_layers(name: str, layers: dict[str, Layer]) -> None:
    payload = {"layer_names": np.array(sorted(layers))}
    for lname, layer in layers.items():
        payload[f"{lname}__P"] = layer.P.astype(np.float32)
        payload[f"{lname}__domain"] = layer.domain
    np.savez_compressed(_layers_path(name), **payload)


def load_layers(name: str) -> dict[str, Layer]:
    """
    Load the cached view transition matrices of stage 1; raises if stage 1 has
    not been run for `name`.
    """
    path = _layers_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/graph/build_layer_transition_matrices.py "
            f"--name {name} first."
        )
    data = np.load(path, allow_pickle=False)
    return {
        lname: Layer(name=lname, P=data[f"{lname}__P"].astype(np.float64), domain=data[f"{lname}__domain"])
        for lname in (str(x) for x in data["layer_names"])
    }


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
            f"--name {name} first."
        )
    data = np.load(path, allow_pickle=False)
    target_ids = [str(x) for x in data["target_ids"]]
    targets = {t: data[f"{t}__P"].astype(np.float64) for t in target_ids}
    domains = {t: data[f"{t}__domain"] for t in target_ids}
    families = json.loads(str(data["families_json"]))
    return targets, domains, families
