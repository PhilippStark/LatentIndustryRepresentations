"""diagnose_scale_redundancy.py — the one diagnostic to run BEFORE building
the shared-cosine multi-target training pipeline (see the design note this
answers): if a layer's own diffusion targets at t in {1,2,4,8} turn out to
be highly correlated with each other, treating each scale as a separate
training target adds little real information and the retrieval/consistency
ceiling won't move just from splitting one weighted-average target into
several near-duplicate ones.

Computes each layer's own P_layer^t independently per scale (build_diffusion,
no weighting) and reports row-wise Spearman rho + cosine similarity between
every pair of scales, restricted to active rows and each layer's own domain
— exactly the comparison the "many per-layer targets" proposal's risk
section calls for, and cheap enough to run directly (dense matrix powers up
to t=8 on ~1,000-2,000 node domains, no training).

Usage
-----
  python scripts/evaluation/diagnose_scale_redundancy.py --name multilayer_v1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.graph import artifacts
from latent_industry.graph.layer_diffusion import build_diffusion, lift_io_layer
from latent_industry.graph.nodes import NodeIndex

SCALES = (1, 2, 4, 8)
MAX_ROWS_SAMPLED = 400  # diagnostic only — cap per-layer row count for speed


def _row_agreement(Pa: np.ndarray, Pb: np.ndarray, active_idx: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = active_idx if active_idx.size <= MAX_ROWS_SAMPLED else rng.choice(active_idx, MAX_ROWS_SAMPLED, replace=False)
    rhos, coss = [], []
    for i in idx:
        p, q = Pa[i], Pb[i]
        if np.ptp(p) > 1e-12 and np.ptp(q) > 1e-12:
            rhos.append(spearmanr(p, q).statistic)
        denom = np.linalg.norm(p) * np.linalg.norm(q)
        coss.append(float(p @ q / denom) if denom > 0 else 0.0)
    return (float(np.nanmean(rhos)) if rhos else float("nan")), float(np.mean(coss))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading cached layers for name={args.name} …")
    layers, _S_neg, _target_ids = artifacts.load_layers(args.name)
    ni = NodeIndex()
    io_raw_edges = artifacts.load_raw_edges(args.name, layer="io")

    effective_layers = dict(layers)
    effective_layers["io"] = lift_io_layer(io_raw_edges, ni)

    print(f"\n{'=' * 70}\nWithin-layer scale redundancy  |  scales={SCALES}\n{'=' * 70}")
    print("(row Spearman rho / row cosine, averaged over active rows, sampled "
          f"to <= {MAX_ROWS_SAMPLED} rows per layer for speed)\n")

    for lname, layer in sorted(effective_layers.items()):
        active_idx = np.nonzero(layer.avail)[0]
        print(f"\n{lname}  (domain={int(layer.domain.sum())}, active={active_idx.size})")
        if active_idx.size == 0:
            print("  no active rows, skipping")
            continue

        powers = {t: build_diffusion(layer, t) for t in SCALES}

        header = "        " + "".join(f"t={t:<10}" for t in SCALES)
        print(header)
        for ta in SCALES:
            row = f"  t={ta:<3} "
            for tb in SCALES:
                if tb <= ta:
                    row += " " * 12
                    continue
                rho, cos = _row_agreement(powers[ta], powers[tb], active_idx, rng)
                row += f"rho={rho:+.2f} "
            print(row)
        # also print cosine on its own line per pair, since near-one-hot rows
        # (transition) can make Spearman noisy/undefined while cosine stays informative
        print("  (cosine similarity, same pairs)")
        for ta in SCALES:
            row = f"  t={ta:<3} "
            for tb in SCALES:
                if tb <= ta:
                    row += " " * 12
                    continue
                _rho, cos = _row_agreement(powers[ta], powers[tb], active_idx, rng)
                row += f"cos={cos:+.2f} "
            print(row)

    print(f"\n{'=' * 70}\nReading this report\n{'=' * 70}")
    print(
        "High rho/cosine (>~0.9) between adjacent scales (e.g. t=1 vs t=2) for a "
        "layer means those two targets are nearly redundant — training against "
        "both adds little beyond one of them. Low/negative correlation between "
        "distant scales (e.g. t=1 vs t=8) means they carry genuinely different "
        "structure (local vs. global reachability) and are worth keeping as "
        "separate targets. A layer where every pair is highly correlated is the "
        "case the proposal's own risk section warns about: extra targets for "
        "that layer add little, and the retrieval ceiling likely won't move from "
        "splitting it alone."
    )


if __name__ == "__main__":
    main()
