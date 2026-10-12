"""
build_layer_diffusion_targets.py
================================
Stage 2 of the graph pipeline: turns each relational view's transition
matrix P_l (from build_layer_transition_matrices.py) into multi-scale
diffusion targets P_l^(t), one separate training target per (view, t)
(src/latent_industry/graph/layer_diffusion.build_layer_diffusion_targets_expanded).
Views are never mixed or averaged across scales.

Scales (defaults = main run, 17 targets)
----------------------------------------
  labor        t = 1, 2, 4   (--scales-labor; split into labor_SNI92 /
                              labor_SNI02 / labor_SNI07, 9 targets)
  hierarchy    t = 1, 2, 4   (--scales-hierarchy; hierarchy_t1 is the ordinal
                              4-digit tier target)
  ownership    t = 1, 4      (--scales-ownership)
  io           t = 1         (native 2-digit layer, not diffused)
  transition   t = 1         (transition_SNI92_SNI02, transition_SNI02_SNI07)

Output: <LATENT_DATA_ROOT>/processed/layer_diffusion_expanded_{name}.npz

Usage (main run)
----------------
  python scripts/graph/build_layer_diffusion_targets.py --name layers_v1
"""

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.graph import artifacts
from latent_industry.graph.layer_diffusion import DEFAULT_LAYER_SCALES, build_layer_diffusion_targets_expanded
from latent_industry.graph.nodes import NodeIndex

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True,
                     help="Same --name passed to build_layer_transition_matrices.py.")
for _lname in ("labor", "hierarchy", "ownership"):
    parser.add_argument(f"--scales-{_lname}", type=int, nargs="+", default=list(DEFAULT_LAYER_SCALES[_lname]),
                         help=f"Diffusion steps t for {_lname}. Default: {DEFAULT_LAYER_SCALES[_lname]}.")
args = parser.parse_args()

layer_scales = dict(DEFAULT_LAYER_SCALES)
for _lname in ("labor", "hierarchy", "ownership"):
    layer_scales[_lname] = tuple(getattr(args, f"scales_{_lname}"))

print(f"\n{'=' * 60}\nLayer diffusion targets  |  name={args.name}\n{'=' * 60}\n")
print("Per-layer scales:")
for lname, scales in layer_scales.items():
    print(f"  {lname:<24}: {scales}")

print("\nLoading cached layer transition matrices …")
layers = artifacts.load_layers(args.name)
n_nodes = next(iter(layers.values())).P.shape[0]
print(f"  {len(layers)} layers, {n_nodes} nodes")

ni = NodeIndex()  # rebuilt deterministically from the same metadata files stage 1 used

print("\nComputing per-(layer, t) diffusion targets …")
t0 = time.time()
targets, domains, families = build_layer_diffusion_targets_expanded(layers, ni, layer_scales=layer_scales)
print(f"  done ({time.time() - t0:.1f}s)")

for target_id in sorted(targets):
    P_t = targets[target_id]
    domain = domains[target_id]
    # labor's per-version targets share one P, so count active rows within
    # the target's own domain only.
    active = (P_t.sum(axis=1) > 0) & domain
    print(f"  {target_id:<26} (family={families[target_id]:<11}): domain={int(domain.sum()):>5}  "
          f"active rows={int(active.sum()):>5}")

print("\nSaving …")
artifacts.save_layer_diffusion_targets_expanded(args.name, targets, domains, families, layer_scales)
print(f"  layer_diffusion_expanded_{args.name}.npz  ({len(targets)} targets)")
print("\nDone. Run scripts/training/train_shared_cosine_embedding.py --name " + args.name + " next.")
