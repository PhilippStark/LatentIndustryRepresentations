"""build_layer_diffusion_targets.py — per-(layer, t) diffusion targets for
the shared-cosine training regime. Reads the stage-1 layer cache
(build_layer_transition_matrices.py) and keeps every (layer, t) power
P_l^(t) as its own separate training target — see
src/latent_industry/graph/layer_diffusion.build_layer_diffusion_targets_expanded.
No supra-network, no coupling parameter, no cross-layer mixing, no
weight-averaging across scales (that earlier regime is archived).

Usage
-----
  python scripts/graph/build_layer_diffusion_targets.py --name multilayer_v1
  python scripts/graph/build_layer_diffusion_targets.py --name multilayer_v1 --scales 1 2 4
  python scripts/graph/build_layer_diffusion_targets.py --name multilayer_v1 --scales-labor 1

`--scales` sets which `t` values each layer gets, defaulting to
`EXPANDED_DEFAULT_SCALES=(1,4)` per diagnose_scale_redundancy.py's finding
that t=2/4/8 are highly mutually redundant for every layer while t=1 is not.
`--scales-<layer>` overrides it for one layer. `--expanded` is accepted and
ignored (this is the only mode).

`transition` is always split into
`transition_SNI92_SNI02`/`transition_SNI02_SNI07` (see
build_transition_pair_layers) and pinned to t=1 only — `--scales-transition`
is ignored (a warning is printed if passed); use
`--scales-transition-92-02`/`--scales-transition-02-07` instead if you need
to override that pin for an experiment.

`labor` is always split into `labor_SNI92`/
`labor_SNI02`/`labor_SNI07` (see build_labor_version_layers) — unlike
`transition`, `--scales-labor` (v7 gap fix) DOES propagate to all three
split names uniformly (there's no per-version pin to work around here,
just the split itself), so `--scales-labor 1` concentrates every version's
entire budget on t=1 as expected. Before this fix, `--scales-labor` was
silently ignored under `--expanded`: the override was keyed by `"labor"`,
but the expanded loop looks targets up by `"labor_SNI92"` etc, so it never
matched and every version silently fell back to `EXPANDED_DEFAULT_SCALES`.

`io` uses its NATIVE 2d-hub layer (NOT lifted to 4d —
`lift_io_layer` is not called), pinned to
`IO_NATIVE_SCALES=(1,)`, i.e. raw `io_relatedness.parquet`, undiffused.
Root-caused 2026-07-27: `P02_shared_geometry_reconstruction.py`'s
(formerly `evaluate_embeddings.py`'s) `I_io_rho` checks the
2d-hub embeddings directly, which the OLD lifted-to-4d `io_t1`/`io_t4`
targets never touched at all (their domain was 4d-SNI07-only) — training
harder on the lifted target could lower its own loss while `I_io_rho` got
worse, since the two were never optimising the same rows. `--scales-io` is
still accepted but ignored (warning printed if passed —
diffusing the native layer further would defeat the point of this fix).

Output: layer_diffusion_expanded_{name}.npz — one P_l^(t)/domain/family
per (layer, t) pair.
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
# v7 gap fix: --scales-labor was silently dropped —
# build_layer_diffusion_targets_expanded looks targets up by
# "labor_SNI92"/"labor_SNI02"/"labor_SNI07" (the per-version split), not
# "labor", so the override never matched and every version always fell
# back to EXPANDED_DEFAULT_SCALES regardless of what was passed here.
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
