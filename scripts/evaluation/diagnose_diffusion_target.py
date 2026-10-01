"""
diagnose_diffusion_target.py — row-entropy / sparsity / top-k-mass diagnostic
for a diffusion target (diffusion_matrix_{name}.npz), meant to be run BEFORE
language fine-tuning (see TODO.md §7).

train_sentence_embedding.py fits a sharp softmax (tau=0.05 by default) to
this target via KL divergence. If the target itself is broad/near-uniform,
the only way the model can match it is by flattening its own similarity
geometry — representation collapse that trivially satisfies the loss while
destroying the finer relatedness structure downstream evals measure. This
script reports, per active row: nonzero count, entropy (in bits, vs the
fully-uniform max), top-k mass, and effective support size (2^entropy) —
so that gets caught here instead of in a post-hoc eval regression.

sparsity_report() is also imported by scripts/evaluation/
diagnose_target_feasibility.py so both scripts share one implementation
instead of drifting apart.

Usage
-----
  python scripts/evaluation/diagnose_diffusion_target.py
  python scripts/evaluation/diagnose_diffusion_target.py --name routing_topk30
"""

import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
from latent_industry.common.paths import PROCESSED_DATA_DIR

# ── Defaults — edit when running directly in VS Code ─────────────────────────
DEFAULT_NAME = "routing"


def row_normalize_target(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero diagonal, clip negatives, row-normalise — identical to the
    re-enforcement train_sentence_embedding.py applies to handle any
    numerical drift from build_diffusion_targets.py. Returns (P, active_mask)."""
    S = S.astype(np.float64).copy()
    np.fill_diagonal(S, 0.0)
    S = np.clip(S, 0.0, None)
    row_sums = S.sum(axis=1, keepdims=True)
    active = row_sums[:, 0] > 0
    P = np.zeros_like(S)
    P[active] = S[active] / row_sums[active]
    return P, active


def _pct(x: np.ndarray) -> np.ndarray:
    return np.percentile(x, [5, 25, 50, 75, 95])


def sparsity_report(S: np.ndarray) -> dict:
    """Row-entropy / sparsity / top-k-mass diagnostics for a diffusion
    target matrix S (raw, pre-normalisation — this function handles the
    row-normalisation itself). Returns a flat dict of scalar summary
    statistics plus the per-row arrays needed for plotting, so callers can
    both print a report and reuse the numbers (see diagnose_target_
    feasibility.py)."""
    n = S.shape[0]
    P, active = row_normalize_target(S)

    n_possible_pairs = n * (n - 1)
    nnz = int((S > 0).sum())

    Pa = P[active]
    row_nnz = (Pa > 0).sum(axis=1)
    eps = 1e-15
    row_entropy = -(Pa * np.log2(Pa + eps)).sum(axis=1)
    max_entropy_uniform_full = np.log2(n - 1)

    sorted_desc = -np.sort(-Pa, axis=1)
    topk_mass = {k: sorted_desc[:, :k].sum(axis=1) for k in [1, 5, 10, 20]}
    eff_support = 2.0 ** row_entropy

    return {
        "n": n,
        "n_active_rows": int(active.sum()),
        "nnz_off_diagonal": nnz,
        "n_possible_pairs": n_possible_pairs,
        "pct_nonzero_pairs": 100 * nnz / n_possible_pairs,
        "row_nnz": row_nnz,
        "row_entropy": row_entropy,
        "max_entropy_uniform_full": max_entropy_uniform_full,
        "mean_entropy_ratio": float(row_entropy.mean() / max_entropy_uniform_full),
        "topk_mass": topk_mass,
        "eff_support": eff_support,
    }


def print_sparsity_report(name: str, report: dict) -> None:
    n = report["n"]
    print(f"\nActive rows: {report['n_active_rows']}/{n}")
    print(f"Nonzero off-diagonal entries: {report['nnz_off_diagonal']:,} / "
          f"{report['n_possible_pairs']:,}  ({report['pct_nonzero_pairs']:.1f}% of all ordered pairs)")

    print(f"\nPer-row nonzero count (out of {n - 1} possible):")
    print(f"  mean={report['row_nnz'].mean():.1f}  p5/p25/p50/p75/p95 = {_pct(report['row_nnz'])}")

    print(f"\nPer-row entropy (bits). Fully uniform over all {n - 1} other codes = "
          f"{report['max_entropy_uniform_full']:.2f} bits.")
    print(f"  mean={report['row_entropy'].mean():.2f}  p5/p25/p50/p75/p95 = {_pct(report['row_entropy'])}")
    print(f"  mean entropy / max possible entropy = {report['mean_entropy_ratio']:.3f}")

    print("\nTop-k mass per row (fraction of row's probability captured by its k largest entries):")
    for k, vals in report["topk_mass"].items():
        print(f"  top-{k:<3} mean={vals.mean():.4f}  p5/p25/p50/p75/p95 = {_pct(vals)}")

    print("\nEffective support size (2^entropy):")
    print(f"  mean={report['eff_support'].mean():.1f}  p5/p25/p50/p75/p95 = {_pct(report['eff_support'])}")
    print(f"  as fraction of n={n}: mean={report['eff_support'].mean() / n:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_NAME,
                        help="--name used when running build_diffusion_targets.py.")
    args, _ = parser.parse_known_args()

    path = PROCESSED_DATA_DIR / f"diffusion_matrix_{args.name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run build_diffusion_targets.py --name {args.name} first.")

    data = np.load(path, allow_pickle=True)
    S = data["S"]
    print(f"Loaded {path.name}: S {S.shape}")

    report = sparsity_report(S)
    print_sparsity_report(args.name, report)


if __name__ == "__main__":
    main()
