"""
compare_muni_year_fe_spec.py
=================================
Compares regional density regressions' headline coefficients between the
published joint municipality×year FE spec (P04_entry.py/_exit.py/
_startup_entry.py/_growth.py/_wage_growth.py's long-standing behaviour,
{run_name}_summary.csv / {run_name}_period_summary.csv) and a separate,
additive municipality + year FE spec run under --out-suffix (the more common
two-way panel FE spec in the literature — see those scripts' own
--out-suffix flag, added precisely so an investigative run never overwrites
the published joint-FE outputs it is being compared against).

This is a draft/investigative tool, deliberately NOT wired into
build_paper_tables.py, build_density_robustness_table.py, or any other
"produce final results" script: run it AFTER the investigative --out-suffix
job (e.g. CPU_regional_separate_muni_year_fe_v16.sh) completes, to see
whether the separate-FE spec changes the headline density coefficients or
their significance before deciding whether to adopt it as the new default
(i.e. before dropping --out-suffix and rerunning as the published baseline).

Usage
-----
  python scripts/evaluation/compare_muni_year_fe_spec.py \
      --run-name shared_cosine_v16 --new-suffix _sepFE
"""

import argparse
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default="shared_cosine_v16")
parser.add_argument("--new-suffix", default="_sepFE",
                     help="--out-suffix used for the separate-FE investigative run.")
args = parser.parse_args()
run = args.run_name
suf = args.new_suffix

# (outcome label, results subdir, headline outcome column, key column)
SUMMARY_SPECS = [
    ("Entry",          "regional_entry",         "entry_industry", "period"),
    ("Exit",           "regional_exit",          "exit_industry",  "period"),
    ("Start-up entry", "regional_startup_entry", "startup_entry",  "period"),
]

rows: list[dict] = []
missing: list[Path] = []


def _compare_rows(label: str, old: pd.Series, new: pd.Series, period, coef, se, pval) -> dict:
    old_coef, new_coef = old[coef], new[coef]
    return {
        "outcome": label, "period": period,
        "coef_joint_muniYear":     old_coef, "se_joint_muniYear":     old[se], "pval_joint_muniYear":     old[pval],
        "coef_separate_muni_year": new_coef, "se_separate_muni_year": new[se], "pval_separate_muni_year": new[pval],
        "coef_pct_change": (new_coef - old_coef) / abs(old_coef) * 100 if old_coef != 0 else float("nan"),
        "sig_flip_at_5pct": (old[pval] < 0.05) != (new[pval] < 0.05),
    }


for label, subdir, col, key in SUMMARY_SPECS:
    old_path = project_root / "results" / subdir / f"{run}_summary.csv"
    new_path = project_root / "results" / subdir / f"{run}{suf}_summary.csv"
    if not old_path.exists():
        missing.append(old_path); continue
    if not new_path.exists():
        missing.append(new_path); continue
    old_df = pd.read_csv(old_path).set_index(key)
    new_df = pd.read_csv(new_path).set_index(key)
    for period in old_df.index:
        if period not in new_df.index:
            continue
        rows.append(_compare_rows(label, old_df.loc[period], new_df.loc[period], period,
                                   f"{col}_coef", f"{col}_se", f"{col}_pval"))

# growth / wage_growth: period_summary.csv — growth has one row per (period,
# density_var), filtered to the headline softmax density; wage_growth has
# just one row per period.
for label, subdir, density_filter in [
    ("Growth",      "regional_growth",      "softmax"),
    ("Wage growth", "regional_wage_growth", None),
]:
    old_path = project_root / "results" / subdir / f"{run}_period_summary.csv"
    new_path = project_root / "results" / subdir / f"{run}{suf}_period_summary.csv"
    if not old_path.exists():
        missing.append(old_path); continue
    if not new_path.exists():
        missing.append(new_path); continue
    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)
    if density_filter is not None:
        old_df = old_df[old_df["density_var"] == density_filter]
        new_df = new_df[new_df["density_var"] == density_filter]
    old_df = old_df.set_index("period")
    new_df = new_df.set_index("period")
    for period in old_df.index:
        if period not in new_df.index:
            continue
        rows.append(_compare_rows(label, old_df.loc[period], new_df.loc[period], period,
                                   "coef", "se", "pval"))

if missing:
    print("NOT FOUND (run the investigative --out-suffix job first if the "
          f"{suf} files are missing):")
    for p in missing:
        print(f"  {p}")

df = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print("\n" + "=" * 80)
print(f"Joint municipality×year FE  vs.  separate municipality + year FE  (run={run})")
print("=" * 80)
print(df.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

out_path = project_root / "results" / "paper" / "appendix" / f"fe_spec_comparison_{run}.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\n→ {out_path}")

if not df.empty and df["sig_flip_at_5pct"].any():
    print("\nWARNING: significance at 5% flips for at least one outcome/period under the "
          "separate-FE spec — inspect before adopting it as the new default.")
