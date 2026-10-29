"""
H1_spatial_folds.py
===================
Size-stratified fold assignment of the 290 municipalities for the spatial
hold-out.

Design
------
Size is a municipality's mean yearly workplace employment over 1990–2019
(employment cache, H0_employment_cache.py); --strata quantile strata (1 =
smallest). Within each stratum the municipalities are shuffled with the fold
seed (= repeat number) and dealt round-robin into --k folds, the counter
running on across strata (largest first), so every fold gets the same number
of municipalities and each stratum is split as evenly as possible. Stockholm
(0180), Göteborg (1480) and Malmö (1280) are dealt first, so they always
land in different folds.

Outputs (results/regional_spatial_holdout/, prefixed shared_cosine_v16_)
------------------------------------------------------------------------
  folds_r{r}.csv            municipality, stratum, fold
  folds_r{r}_test_f{k}.csv  municipality — the --exclude-municipalities input
                            of fold k (scripts/data/build_*_relatedness.py)

Usage
-----
  python scripts/robustness/H1_spatial_folds.py --repeats 3
"""

import argparse

import numpy as np
import pandas as pd

from _holdout_common import EMPLOYMENT_CACHE, result_path, results_dir

LARGEST_CITIES = ["0180", "1480", "1280"]  # Stockholm, Göteborg, Malmö

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=5, help="Number of folds.")
parser.add_argument("--repeats", type=int, default=3, help="Fold assignments, with fold seeds 1..repeats.")
parser.add_argument("--strata", type=int, default=5, help="Number of size strata (quantiles).")
args = parser.parse_args()

empl = pd.read_parquet(EMPLOYMENT_CACHE)
size = empl.groupby(["municipality", "Year"])["workers"].sum().groupby("municipality").mean()
if len(size) != 290:
    raise ValueError(f"expected 290 municipalities, found {len(size)}")
missing = [m for m in LARGEST_CITIES if m not in size.index]
if missing:
    raise ValueError(f"municipality codes not in the cache: {missing}")
stratum = pd.Series(pd.qcut(size.rank(method="first"), args.strata, labels=False) + 1, index=size.index)

out_dir = results_dir("spatial_holdout")
for repeat in range(1, args.repeats + 1):
    rng = np.random.default_rng(repeat)
    order: list[str] = []
    for s in sorted(stratum.unique(), reverse=True):
        members = rng.permutation(sorted(stratum.index[stratum == s]))
        if s == stratum.max():
            members = LARGEST_CITIES + [m for m in members if m not in LARGEST_CITIES]
        order.extend(members)
    folds = pd.DataFrame({"municipality": order})
    folds["stratum"] = folds["municipality"].map(stratum).astype(int)
    folds["fold"] = np.arange(len(folds)) % args.k + 1
    folds = folds.sort_values("municipality").reset_index(drop=True)

    big_folds = folds.set_index("municipality").loc[LARGEST_CITIES, "fold"]
    assert big_folds.nunique() == len(LARGEST_CITIES)
    folds.to_csv(result_path("spatial_holdout", f"folds_r{repeat}.csv"), index=False)
    for fold in range(1, args.k + 1):
        folds.loc[folds["fold"] == fold, ["municipality"]].to_csv(
            result_path("spatial_holdout", f"folds_r{repeat}_test_f{fold}.csv"), index=False
        )
    table = pd.crosstab(folds["stratum"], folds["fold"])
    print(f"\nRepeat {repeat} (fold seed {repeat}): municipalities per stratum × fold\n{table}")
print(f"\n→ {out_dir}")
