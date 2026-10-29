"""
H5_evaluate_seeds.py
====================
Seed and initialisation stability. The v16 inputs are unchanged; each
variant (seed_stability/data/<variant>/, run v16_seed_<variant>) starts from
the text prior perturbed with --init-noise SIGMA drawn from --seed
(train_shared_cosine_embedding.py), because --seed alone changes almost
nothing (H2_check_reference.py, check 4).

Reported
--------
  regional   main-specification period slopes on the full published panels,
             density recomputed with each seed's embedding; mean, SD, min, max
             per outcome × period × SIGMA, next to v16 and v16_reference
  geometry   the P02 summary metrics of each seed (embedding_eval/ moved into
             the variant folder by the job), next to v16 and v16_reference
  neighbours SNI07 4-digit codes: mean Jaccard overlap of the 10 nearest
             neighbours and Pearson correlation of the pairwise cosines,
             seed vs v16, seed vs v16_reference and between seeds of one SIGMA

Outputs (results/seed_stability/, prefixed shared_cosine_v16_)
--------------------------------------------------------------
  regional_{outcome}_by_seed.csv, regional_summary.csv,
  geometry_metrics_by_seed.csv, geometry_summary.csv, neighbour_overlap.csv

Usage
-----
  python scripts/robustness/H5_evaluate_seeds.py
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from _holdout_common import (
    HOLDOUT_ROOT, MAIN_RUN, OUTCOMES, REFERENCE_ROOT, REFERENCE_RUN, REPO_ROOT, Embedding, Panel,
    check_codes, estimate, is_trained, load_main, reference_embedding, result_path, results_dir,
)

K_NEIGHBOURS = 10
LABEL = "sni07"

parser = argparse.ArgumentParser()
parser.add_argument("--outcomes", nargs="+", default=list(OUTCOMES))
args = parser.parse_args()

out_dir = results_dir("seed_stability")
emp, v16 = load_main()
reference = reference_embedding()

seeds = []  # (variant, seed, sigma, Embedding)
for root in sorted((HOLDOUT_ROOT / "seed_stability" / "data").glob("*")):
    run = f"v16_seed_{root.name}"
    if not is_trained(root, run):
        continue
    prov = json.loads((root / "embedding" / run / "training_provenance.json").read_text())
    emb = Embedding(run, root, run)
    check_codes(emb, v16)
    seeds.append((root.name, int(prov["seed"]), float(prov.get("init_noise", 0.0)), emb))
if not seeds:
    raise SystemExit("No trained seed variant.")
print(f"Seed variants: {[s[0] for s in seeds]}")

# ── Regional coefficients ─────────────────────────────────────────────────────
summary_rows = []
for outcome in args.outcomes:
    col = OUTCOMES[outcome]["col"]
    print(f"\n{outcome}:")
    panel = Panel(outcome, emp)
    by_seed = [estimate(panel.df, col, panel.df["density"].to_numpy(dtype=float), annual=False,
                        outcome=outcome, variant="v16", seed=np.nan, sigma=np.nan)[0]]
    if reference is not None:
        by_seed.append(estimate(panel.df, col, panel.density(emp, reference), annual=False,
                                outcome=outcome, variant="v16_reference", seed=42, sigma=0.0)[0])
    for name, seed, sigma, emb in seeds:
        rows, _ = estimate(panel.df, col, panel.density(emp, emb), annual=False,
                           outcome=outcome, variant=name, seed=seed, sigma=sigma)
        by_seed.append(rows)
        print(f"  {name} done")
    by_seed = pd.concat(by_seed)
    by_seed.to_csv(result_path("seed_stability", f"regional_{outcome}_by_seed.csv"), index=False)

    fixed = by_seed[by_seed["variant"].isin(["v16", "v16_reference"])]
    for (period, sigma), g in by_seed[~by_seed["variant"].isin(["v16", "v16_reference"])].groupby(["period", "sigma"]):
        row = {"outcome": outcome, "period": period, "sigma": sigma, "n_seeds": len(g),
               "coef_mean": g["coef"].mean(), "coef_sd": g["coef"].std(ddof=1),
               "coef_min": g["coef"].min(), "coef_max": g["coef"].max(), "se_mean": g["se"].mean()}
        for _, f in fixed[fixed["period"] == period].iterrows():
            row[f"{f['variant']}_coef"] = f["coef"]
            row[f"{f['variant']}_se"] = f["se"]
        summary_rows.append(row)
pd.DataFrame(summary_rows).to_csv(result_path("seed_stability", "regional_summary.csv"), index=False)

# ── Geometry metrics (P02 summary) ────────────────────────────────────────────
def p02_metrics(path, run):
    if not path.exists():
        print(f"  WARNING: {path} missing")
        return None
    df = pd.read_csv(path)
    return df[df["run"] == run].iloc[0].drop("run").astype(float)


metric_rows = []
v16_metrics = p02_metrics(REPO_ROOT / "results" / "embedding_eval" / MAIN_RUN / f"{MAIN_RUN}_summary.csv", MAIN_RUN)
ref_metrics = p02_metrics(REFERENCE_ROOT / "embedding_eval" / f"{REFERENCE_RUN}_summary.csv", REFERENCE_RUN)
for name, metrics, seed, sigma in [("v16", v16_metrics, np.nan, np.nan), ("v16_reference", ref_metrics, 42, 0.0)]:
    if metrics is not None:
        metric_rows.append({"variant": name, "seed": seed, "sigma": sigma, **metrics.to_dict()})
for name, seed, sigma, emb in seeds:
    metrics = p02_metrics(emb.root / "embedding_eval" / f"{emb.run}_summary.csv", emb.run)
    if metrics is not None:
        metric_rows.append({"variant": name, "seed": seed, "sigma": sigma, **metrics.to_dict()})
geometry = pd.DataFrame(metric_rows)
geometry.to_csv(result_path("seed_stability", "geometry_metrics_by_seed.csv"), index=False)

seed_geometry = geometry[~geometry["variant"].isin(["v16", "v16_reference"])]
metric_cols = [c for c in geometry.columns if c not in ("variant", "seed", "sigma")]
geo_summary = []
for sigma, g in seed_geometry.groupby("sigma"):
    for m in metric_cols:
        row = {"sigma": sigma, "metric": m, "n_seeds": len(g), "mean": g[m].mean(), "sd": g[m].std(ddof=1),
               "min": g[m].min(), "max": g[m].max()}
        for name, metrics in (("v16", v16_metrics), ("v16_reference", ref_metrics)):
            row[name] = metrics[m] if metrics is not None and m in metrics else np.nan
        geo_summary.append(row)
pd.DataFrame(geo_summary).to_csv(result_path("seed_stability", "geometry_summary.csv"), index=False)

# ── Neighbourhood stability (SNI07 4-digit) ───────────────────────────────────
def knn_sets(emb: Embedding) -> list[set]:
    cos = emb.cos[LABEL].copy()
    np.fill_diagonal(cos, -np.inf)
    return [set(row) for row in np.argsort(-cos, axis=1)[:, :K_NEIGHBOURS]]


def compare(a: Embedding, b: Embedding) -> tuple[float, float]:
    ka, kb = knn_sets(a), knn_sets(b)
    jaccard = float(np.mean([len(x & y) / len(x | y) for x, y in zip(ka, kb)]))
    iu = np.triu_indices(len(a.codes[LABEL]), k=1)
    corr = float(np.corrcoef(a.cos[LABEL][iu], b.cos[LABEL][iu])[0, 1])
    return jaccard, corr


overlap = []
anchors = [("v16", v16)] + ([("v16_reference", reference)] if reference is not None else [])
for name, seed, sigma, emb in seeds:
    for anchor_name, anchor in anchors:
        j, c = compare(emb, anchor)
        overlap.append({"variant": name, "sigma": sigma, "against": anchor_name, "knn_jaccard": j, "cos_corr": c})
if reference is not None:
    j, c = compare(reference, v16)
    overlap.append({"variant": "v16_reference", "sigma": 0.0, "against": "v16", "knn_jaccard": j, "cos_corr": c})
for sigma in sorted({s[2] for s in seeds}):
    group = [s for s in seeds if s[2] == sigma]
    pairs = [compare(a[3], b[3]) for a, b in itertools.combinations(group, 2)]
    if pairs:
        overlap.append({"variant": f"seed pairs ({len(pairs)})", "sigma": sigma, "against": "each other",
                        "knn_jaccard": float(np.mean([p[0] for p in pairs])),
                        "cos_corr": float(np.mean([p[1] for p in pairs]))})
pd.DataFrame(overlap).to_csv(result_path("seed_stability", "neighbour_overlap.csv"), index=False)
print(f"\n→ {out_dir}")
print("Done.")
