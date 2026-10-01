"""
build_paper_table.py — assemble the Figure-2 paper table from one or more
P02_shared_geometry_reconstruction.py (formerly evaluate_embeddings.py)
runs' {run}_K_source_table.csv outputs.

P02_shared_geometry_reconstruction.py's Section K writes one row per fine-grained evidence
family (Labor SNI07/02/92, NACE 4d/3d/2d/section, IO, Concordance 02->07/
92->02, Fused diffusion, Hierarchy). The paper table collapses those into
one row per evidence family and adds a column per compared encoder:

  | Evidence family | Raw 384D | Integrated 384D | Integrated 768D | Source metric |

"Raw 384D" isn't a run of its own — it's the KL_raw column baked into every
run's own baseline pass (same fixed 384D MiniLM baseline regardless of which
trained model is being evaluated), so it's read off the first --runs entry
and sanity-checked for near-agreement against the rest.

Usage
-----
  python scripts/evaluation/build_paper_table.py \\
      --runs multilayer_v1 multilayer_v1_768d \\
      --labels "Integrated 384D" "Integrated 768D"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import PROCESSED_DATA_DIR  # noqa: F401  (keeps import style consistent)

eval_dir = project_root / "results" / "embedding_eval"

parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="+", required=True,
                    help="run-name(s) whose {run}_K_source_table.csv to read, in the order "
                         "they should appear as columns (e.g. multilayer_v1 multilayer_v1_768d).")
parser.add_argument("--labels", nargs="+", default=None,
                    help="Column header per run (same length/order as --runs). "
                         "Default: 'Integrated {run}'.")
parser.add_argument("--out", default=None,
                    help="Output path prefix (writes {out}.md and {out}.csv). "
                         "Default: results/embedding_eval/paper_source_table")
args = parser.parse_args()

labels = args.labels if args.labels else [f"Integrated {r}" for r in args.runs]
if len(labels) != len(args.runs):
    raise SystemExit(f"--labels has {len(labels)} entries but --runs has {len(args.runs)}")

out_prefix = Path(args.out) if args.out else eval_dir / "paper_source_table"

# ── Macro evidence families — collapse Section K's fine-grained rows ──────────
# See P02_shared_geometry_reconstruction.py's FAMILY_TABLE for the fine-grained row names.
MACRO_FAMILIES = {
    "Labor":           ["Labor (SNI07)", "Labor (SNI02)", "Labor (SNI92)"],
    "NACE semantics":  ["NACE 4-digit", "NACE 3-digit", "NACE 2-digit", "NACE section"],
    "Input-output":    ["Input-output"],
    "Concordance":     ["Concordance 02→07", "Concordance 92→02"],
    "Hierarchy":       ["Hierarchy"],
    "Fused diffusion": ["Fused diffusion"],
}
FAMILY_ORDER = ["Labor", "NACE semantics", "Input-output", "Concordance", "Hierarchy", "Fused diffusion"]


def load_run_table(run: str) -> pd.DataFrame:
    # 2026-07-29: evaluate_embeddings.py (now P02_shared_geometry_reconstruction.py)
    # writes into a per-run subfolder (results/embedding_eval/{run}/) instead of
    # one flat directory.
    path = eval_dir / run / f"{run}_K_source_table.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found — run P02_shared_geometry_reconstruction.py --run-name {run} first.")
    return pd.read_csv(path)


def macro_row(df: pd.DataFrame, members: list[str]) -> dict:
    sub = df[df["family"].isin(members)]
    kl_raw     = sub["KL_raw"].mean(skipna=True)
    kl_trained = sub["KL_trained"].mean(skipna=True)
    metric_trained = sub["source_metric_trained"].mean(skipna=True)
    metric_names = sub["source_metric"].dropna().unique().tolist()
    return {
        "KL_raw": kl_raw,
        "KL_trained": kl_trained,
        "metric_trained": metric_trained,
        "metric_names": metric_names,
    }


run_tables = {run: load_run_table(run) for run in args.runs}

# Sanity-check that KL_raw (the fixed 384D baseline) roughly agrees across runs
raw_refs = {}
for family, members in MACRO_FAMILIES.items():
    vals = [macro_row(run_tables[r], members)["KL_raw"] for r in args.runs]
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) > 1 and (max(vals) - min(vals)) > 0.05 * max(abs(v) for v in vals):
        print(f"WARNING: KL_raw for '{family}' disagrees across runs by >5% "
              f"({dict(zip(args.runs, vals))}) — different baseline text/model between runs?")
    raw_refs[family] = vals[0] if vals else float("nan")

rows = []
for family in FAMILY_ORDER:
    members = MACRO_FAMILIES[family]
    row = {"Evidence family": family, "Raw 384D": round(raw_refs[family], 4)}
    metric_names_all = set()
    for run, label in zip(args.runs, labels):
        mr = macro_row(run_tables[run], members)
        row[label] = round(mr["KL_trained"], 4) if not np.isnan(mr["KL_trained"]) else None
        row[f"{label} — source metric"] = (
            round(mr["metric_trained"], 4) if not np.isnan(mr["metric_trained"]) else None
        )
        metric_names_all.update(mr["metric_names"])
    row["Source metric"] = ", ".join(sorted(metric_names_all)) if metric_names_all else ""
    rows.append(row)

table_df = pd.DataFrame(rows)
out_prefix.parent.mkdir(parents=True, exist_ok=True)
table_df.to_csv(f"{out_prefix}.csv", index=False)

# ── Markdown table matching the plan's proposed shape ──────────────────────
md_cols = ["Evidence family", "Raw 384D"] + list(labels) + ["Source metric"]
lines = [
    "| " + " | ".join(md_cols) + " |",
    "|" + "|".join(["---"] + [":---:"] * (len(labels) + 1) + ["---"]) + "|",
]
for row in rows:
    cells = [row["Evidence family"], f"{row['Raw 384D']:.4f}" if row["Raw 384D"] == row["Raw 384D"] else "n/a"]
    for label in labels:
        v = row[label]
        cells.append(f"{v:.4f}" if v is not None else "n/a")
    cells.append(row["Source metric"])
    lines.append("| " + " | ".join(cells) + " |")
md = "\n".join(lines)

Path(f"{out_prefix}.md").write_text(md + "\n", encoding="utf-8")
print(md)
print(f"\n→ {out_prefix}.md")
print(f"→ {out_prefix}.csv  (full precision, plus per-run source-metric columns)")
