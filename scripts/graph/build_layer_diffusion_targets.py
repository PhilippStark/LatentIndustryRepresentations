"""
build_layer_diffusion_targets.py
================================
Stage 2 of the graph pipeline: turns each relational view's transition
matrix P_l (from build_layer_transition_matrices.py) into multi-scale
diffusion targets P_l^(t), one separate training target per (view, t)
(src/latent_industry/graph/layer_diffusion.build_layer_diffusion_targets_expanded).
Views are never mixed or averaged across scales.

Scales
------
  --scales                 t values for every view without its own override
                           (default EXPANDED_DEFAULT_SCALES = (1, 4))
  --scales-<view>          override for one view; labor is split by version
                           (labor_SNI92 / labor_SNI02 / labor_SNI07) and
                           --scales-labor applies to all three
  transition               split into transition_SNI92_SNI02 and
                           transition_SNI02_SNI07, fixed at t = 1 (override with
                           --scales-transition-92-02 / --scales-transition-02-07)
  io                       native 2-digit layer, fixed at t = 1 (not diffused)
  ownership                EXPANDED_DEFAULT_SCALES = (1, 4)

Output: <LATENT_DATA_ROOT>/processed/layer_diffusion_expanded_{name}.npz

Usage (main run; 24 targets)
----------------------------
  python scripts/graph/build_layer_diffusion_targets.py --name layers_v1 \
      --scales 1 2 4 --scales-labor 1 2 4
"""

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import PROCESSED_DATA_DIR
from latent_industry.graph import artifacts
from latent_industry.graph.layer_diffusion import (
    EXPANDED_DEFAULT_SCALES, IO_NATIVE_SCALES, TRANSITION_PAIR_SCALES,
    build_layer_diffusion_targets_expanded,
)
from latent_industry.graph.nodes import NodeIndex

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

LAYER_NAMES = ["labor", "nace_text", "io", "hierarchy", "transition"]

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True,
                     help="Same --name passed to build_layer_transition_matrices.py.")
parser.add_argument("--expanded", action="store_true",
                     help="Accepted and ignored — per-(layer, t) targets are the only mode.")
parser.add_argument("--scales", type=int, nargs="+", default=list(EXPANDED_DEFAULT_SCALES),
                     help="Diffusion steps t, applied to any layer without its own "
                          "--scales-<layer> override. Default: EXPANDED_DEFAULT_SCALES.")
for _lname in LAYER_NAMES:
    parser.add_argument(f"--scales-{_lname}", type=int, nargs="+", default=None, dest=f"scales_{_lname}",
                         help=f"Override --scales for {_lname} only.")
parser.add_argument("--scales-transition-92-02", type=int, nargs="+", default=None,
                     dest="scales_transition_92_02",
                     help="Override TRANSITION_PAIR_SCALES=(1,) for transition_SNI92_SNI02.")
parser.add_argument("--scales-transition-02-07", type=int, nargs="+", default=None,
                     dest="scales_transition_02_07",
                     help="Override TRANSITION_PAIR_SCALES=(1,) for transition_SNI02_SNI07.")
args = parser.parse_args()

if args.scales_transition is not None:
    print("WARNING: --scales-transition is ignored (transition is split into "
          "transition_SNI92_SNI02/transition_SNI02_SNI07 — use --scales-transition-92-02/"
          "--scales-transition-02-07 instead).")

if args.scales_io is not None:
    print("WARNING: --scales-io is ignored (io uses its native, un-lifted, "
          "un-diffused 2d-hub layer pinned to IO_NATIVE_SCALES=(1,) — see the module docstring).")

layer_scales: dict[str, tuple[int, ...]] = {}
for lname in LAYER_NAMES:
    l_scales = getattr(args, f"scales_{lname}")
    layer_scales[lname] = tuple(l_scales) if l_scales is not None else tuple(args.scales)

t92_02 = tuple(args.scales_transition_92_02) if args.scales_transition_92_02 else TRANSITION_PAIR_SCALES
t02_07 = tuple(args.scales_transition_02_07) if args.scales_transition_02_07 else TRANSITION_PAIR_SCALES

print(f"\n{'=' * 60}\nLayer diffusion targets  |  name={args.name}\n{'=' * 60}\n")
print("Per-layer scales:")
for lname in LAYER_NAMES:
    if lname == "transition":
        print(f"  {'transition':<12}: split -> transition_SNI92_SNI02={t92_02}  "
              f"transition_SNI02_SNI07={t02_07}")
    elif lname == "io":
        print(f"  {'io':<12}: native 2d-hub, un-lifted  scales={IO_NATIVE_SCALES}")
    else:
        print(f"  {lname:<12}: {layer_scales[lname]}")

print("\nLoading cached layer transition matrices …")
layers, S_neg, target_ids = artifacts.load_layers(args.name)
n_nodes = next(iter(layers.values())).P.shape[0]
print(f"  {len(layers)} layers, {n_nodes} nodes")

ni = NodeIndex()  # rebuilt deterministically from the same metadata files stage 1 used

expanded_layer_scales = {k: v for k, v in layer_scales.items() if k not in ("transition", "io")}
if args.scales_transition_92_02 is not None:
    expanded_layer_scales["transition_SNI92_SNI02"] = tuple(args.scales_transition_92_02)
if args.scales_transition_02_07 is not None:
    expanded_layer_scales["transition_SNI02_SNI07"] = tuple(args.scales_transition_02_07)
# labor targets are keyed per version (labor_SNI92/labor_SNI02/labor_SNI07),
# so --scales-labor is applied to each of them.
if args.scales_labor is not None:
    for _ver in ("SNI92", "SNI02", "SNI07"):
        expanded_layer_scales[f"labor_{_ver}"] = tuple(args.scales_labor)

print("\nComputing per-(layer, t) diffusion targets …")
t0 = time.time()
targets, domains, families = build_layer_diffusion_targets_expanded(
    layers, ni, layer_scales=expanded_layer_scales,
)
print(f"  done ({time.time() - t0:.1f}s)")

for target_id in sorted(targets):
    P_t = targets[target_id]
    domain = domains[target_id]
    # Domain-restrict before counting active rows: labor's per-version
    # split (and any other shared-underlying-P view) reuses the SAME
    # full P for every version, so an undomained row-sum would count
    # every version's active rows, not just this target's own —
    # cosmetic-only (training already domain-restricts correctly via
    # OperatorData), but misleading to print unrestricted.
    active = (P_t.sum(axis=1) > 0) & domain
    print(f"  {target_id:<16} (family={families[target_id]:<12}): domain={int(domain.sum()):>5}  "
          f"active rows={int(active.sum()):>5}  shape={P_t.shape}")

print("\nSaving …")
provenance_scales = dict(expanded_layer_scales)
provenance_scales["transition_SNI92_SNI02"] = expanded_layer_scales.get(
    "transition_SNI92_SNI02", TRANSITION_PAIR_SCALES)
provenance_scales["transition_SNI02_SNI07"] = expanded_layer_scales.get(
    "transition_SNI02_SNI07", TRANSITION_PAIR_SCALES)
artifacts.save_layer_diffusion_targets_expanded(args.name, targets, domains, families, provenance_scales)
print(f"  layer_diffusion_expanded_{args.name}.npz")
print("\nDone. Run scripts/training/train_shared_cosine_embedding.py --name " + args.name + " next.")
