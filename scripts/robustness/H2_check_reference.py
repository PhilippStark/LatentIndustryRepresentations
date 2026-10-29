"""
H2_check_reference.py
=====================
Checks the reference retrain (<HOLDOUT_ROOT>/common/data/reference_full/),
i.e. the v16 pipeline rerun with the patched builders and trainer, all flags
off.

Checks (the script exits with an error if 3 fails; 1–2 are warnings, since
the published relatedness files may predate the current builder code — the
hold-outs are always compared with v16_reference, built by the same code)
---------------------------------------------------------------------------
  1. labor and ownership relatedness equal the published files
     (<MAIN_DATA_ROOT>/processed/; same pairs, max relative |Δphi| < 1e-9);
  2. layers and diffusion targets equal those of the 2026-10-12 audit rebuild
     (<MAIN_DATA_ROOT>/verification/audit_cleanup_2026-10-12/processed/);
  3. patched trainer with --init-noise 0 (v16_reference) and repository
     trainer (v16_reference_repotrainer) give bitwise identical embeddings.
Reported only:
  4. seed alone: max |Δcos| over SNI07 4-digit pairs between v16_reference
     (seed 42) and v16_reference_seed43;
  5. retrain drift: max and mean |Δcos| between v16_reference and the
     published v16 embedding.

Output: results/holdout_checks/shared_cosine_v16_H2_reference_checks.json

Usage
-----
  python scripts/robustness/H2_check_reference.py
"""

import sys

import numpy as np
import pandas as pd

from _holdout_common import (
    MAIN_DATA_ROOT, MAIN_RUN, REFERENCE_ROOT, REFERENCE_RUN, Embedding, result_path, write_json,
)

AUDIT_PROCESSED = MAIN_DATA_ROOT / "verification" / "audit_cleanup_2026-10-12" / "processed"
report: dict = {}
failures: list[str] = []
warnings: list[str] = []

# 1. relatedness tables
for name in ("labor_relatedness", "ownership_relatedness"):
    keys = ["coding", "sni_from", "sni_to"]
    ref = pd.read_parquet(REFERENCE_ROOT / "processed" / f"{name}.parquet").sort_values(keys).reset_index(drop=True)
    pub = pd.read_parquet(MAIN_DATA_ROOT / "processed" / f"{name}.parquet").sort_values(keys).reset_index(drop=True)
    same_pairs = len(ref) == len(pub) and ref[keys].equals(pub[keys])
    rel_diff = float(np.max(np.abs(ref["phi"] - pub["phi"]) / np.abs(pub["phi"]))) if same_pairs else float("nan")
    report[name] = {"rows_reference": len(ref), "rows_published": len(pub),
                    "same_pairs": bool(same_pairs), "phi_max_rel_diff": rel_diff}
    print(f"{name}: same pairs = {same_pairs}, max rel |Δphi| = {rel_diff:.2e}")
    if not same_pairs or not rel_diff < 1e-9:
        warnings.append(f"{name} differs from the published file")

# 2. layers and diffusion targets
for fname in ("layers_layers_v1.npz", "layer_diffusion_expanded_layers_v1.npz"):
    ref = np.load(REFERENCE_ROOT / "processed" / fname, allow_pickle=True)
    aud = np.load(AUDIT_PROCESSED / fname, allow_pickle=True)
    max_diff, mismatched = 0.0, []
    for key in sorted(set(ref.files) | set(aud.files)):
        if key not in ref.files or key not in aud.files:
            mismatched.append(key)
            continue
        a, b = ref[key], aud[key]
        if a.dtype.kind in "fc":
            if a.shape != b.shape:
                mismatched.append(key)
            else:
                max_diff = max(max_diff, float(np.max(np.abs(a - b))) if a.size else 0.0)
        elif not np.array_equal(a, b):
            mismatched.append(key)
    report[fname] = {"max_abs_diff": max_diff, "mismatched_keys": mismatched}
    print(f"{fname}: max |Δ| = {max_diff:.2e}, mismatched keys = {mismatched}")
    if mismatched or max_diff > 1e-9:
        warnings.append(f"{fname} differs from the audit rebuild")

# 3.–5. embeddings
reference = Embedding("v16_reference", REFERENCE_ROOT, REFERENCE_RUN)
repo_trainer = Embedding("v16_reference_repotrainer", REFERENCE_ROOT, "v16_reference_repotrainer")
seed43 = Embedding("v16_reference_seed43", REFERENCE_ROOT, "v16_reference_seed43")
v16 = Embedding("v16", MAIN_DATA_ROOT, MAIN_RUN)

bitwise = all(np.array_equal(reference.vecs[l], repo_trainer.vecs[l]) for l in reference.vecs)
report["patched_vs_repo_trainer_bitwise"] = bool(bitwise)
print(f"patched trainer (--init-noise 0) vs repository trainer: bitwise identical = {bitwise}")
if not bitwise:
    failures.append("patched trainer with --init-noise 0 is not bitwise identical to the repository trainer")


def cos_diff(a: Embedding, b: Embedding, label: str = "sni07") -> tuple[float, float]:
    iu = np.triu_indices(len(a.codes[label]), k=1)
    d = np.abs(a.cos[label][iu] - b.cos[label][iu])
    return float(d.max()), float(d.mean())


report["seed42_vs_seed43_cos"] = dict(zip(("max_abs_diff", "mean_abs_diff"), cos_diff(reference, seed43)))
report["reference_vs_published_v16_cos"] = dict(zip(("max_abs_diff", "mean_abs_diff"), cos_diff(reference, v16)))
print(f"seed 42 vs seed 43 (init-noise 0): {report['seed42_vs_seed43_cos']}")
print(f"v16_reference vs published v16:    {report['reference_vs_published_v16_cos']}")

report["failures"] = failures
report["warnings"] = warnings
write_json(result_path("checks", "H2_reference_checks.json"), report)
if warnings:
    print("\nWARNINGS:\n  " + "\n  ".join(warnings))
if failures:
    print("\nCHECKS FAILED:\n  " + "\n  ".join(failures))
    sys.exit(1)
print("\nAll checks passed.")
