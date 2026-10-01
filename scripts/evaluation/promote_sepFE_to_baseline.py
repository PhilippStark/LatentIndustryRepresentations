"""
promote_sepFE_to_baseline.py
=================================
Promotes the separate municipality + year FE investigative run
(--out-suffix _sepFE, see P04_entry.py/_exit.py/_startup_entry.py/_growth.py/
_wage_growth.py/R01_single_source_density.py's own --out-suffix flag) to be
the published shared_cosine_v16 baseline, replacing the old joint
municipality×year FE spec everywhere.

For every results/regional_*/ file (and the LUNARC-side single-source-
density robustness check) matching "{run}_sepFE*", the existing
"{run}*" (--out-suffix "") target is ARCHIVED (moved, never deleted) to
results/archive/v16_joint_muniYear_FE/<same relative path>, then the
_sepFE file is COPIED (original left in place too) onto the target path —
so after this runs, plain {run}_* everywhere is now the separate-FE spec,
and every prior joint-FE number is still recoverable from the archive dir
(on top of git history for anything git-tracked).

Finally rebuilds results/paper/ from the newly-promoted baseline:
build_paper_tables.py, build_density_robustness_table.py (now unsuffixed,
so it overwrites the published density_robustness.tex), and a manual copy
of the single-source-density comparison plot (never git-tracked, see
R01_single_source_density.py's own docstring).

SAFETY: dry-run by default — prints the full plan and does nothing. Pass
--apply to actually move/copy files and rebuild results/paper/.

PREREQUISITE: every VARIANT below must have its _sepFE file present, or
this refuses to run (partial promotion would leave density_robustness.tex
built from a MIX of joint-FE and separate-FE numbers) — run
CPU_density_robustness_sepFE_v16.sh and
CPU_robustness_single_source_density_sepFE_v16.sh first if anything is
still missing.

Usage
-----
  python scripts/evaluation/promote_sepFE_to_baseline.py                 # dry run
  python scripts/evaluation/promote_sepFE_to_baseline.py --apply         # do it
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default="shared_cosine_v16")
parser.add_argument("--out-suffix", default="_sepFE")
parser.add_argument("--apply", action="store_true",
                     help="Actually move/copy files and rebuild results/paper/. "
                          "Without this flag, only prints the plan.")
parser.add_argument("--force", action="store_true",
                     help="Proceed even if one of the REQUIRED variants below is missing "
                          "(partial promotion — NOT recommended for a paper-ready baseline).")
args = parser.parse_args()
run, suf = args.run_name, args.out_suffix

LUNARC_SINGLE_SOURCE_DIR = Path(
    "/lunarc/nobackup/projects/lu2025-17-39/DataLatentIndustryRepresentations/"
    "robustness/single_source_density"
)

# Every {run}{suf}-prefixed file under these roots gets promoted (recursively,
# so results/regional_growth/maps/ is covered too).
SEARCH_DIRS = [
    project_root / "results" / "regional_entry",
    project_root / "results" / "regional_exit",
    project_root / "results" / "regional_startup_entry",
    project_root / "results" / "regional_growth",
    project_root / "results" / "regional_wage_growth",
    LUNARC_SINGLE_SOURCE_DIR,
]

# Required for a paper-ready promotion: the baseline (D^{A,log}) plus the 3
# density_robustness.tex variants for entry/exit/startup_entry, and the
# single-source density comparison.
REQUIRED_MARKERS = [
    project_root / "results" / "regional_entry" / f"{run}{suf}_summary.csv",
    project_root / "results" / "regional_exit" / f"{run}{suf}_summary.csv",
    project_root / "results" / "regional_startup_entry" / f"{run}{suf}_summary.csv",
    project_root / "results" / "regional_growth" / f"{run}{suf}_summary.csv",
    project_root / "results" / "regional_wage_growth" / f"{run}{suf}_period_summary.csv",
] + [
    project_root / "results" / subdir / f"{run}{density_suffix}{suf}_summary.csv"
    for subdir in ("regional_entry", "regional_exit", "regional_startup_entry")
    for density_suffix in ("_ew_raw", "_geo_raw", "_geo_log")
] + [
    LUNARC_SINGLE_SOURCE_DIR / f"single_source_density_comparison{suf}.csv",
]

missing_required = [p for p in REQUIRED_MARKERS if not p.exists()]
if missing_required and not args.force:
    print("REFUSING to promote — missing required _sepFE outputs (run "
          "CPU_density_robustness_sepFE_v16.sh / "
          "CPU_robustness_single_source_density_sepFE_v16.sh first):")
    for p in missing_required:
        print(f"  {p}")
    print("\n(pass --force to promote anyway, e.g. for a partial/staged promotion)")
    sys.exit(1)

archive_root = project_root / "results" / "archive" / "v16_joint_muniYear_FE"

plan: list[tuple[Path, Path, Path]] = []  # (sepFE_src, archive_dst, live_target)
for search_dir in SEARCH_DIRS:
    if not search_dir.exists():
        continue
    for src in sorted(search_dir.rglob(f"*{suf}*")):
        if not src.is_file():
            continue
        target = src.parent / src.name.replace(suf, "")
        if target == src:
            continue
        rel = target.relative_to(project_root) if project_root in target.parents else Path("robustness_single_source_density") / target.name
        archive_dst = archive_root / rel
        plan.append((src, archive_dst, target))

print(f"Promotion plan ({len(plan)} files):")
for src, archive_dst, target in plan:
    exists_note = "" if target.exists() else "  (no existing file to archive)"
    print(f"  {target}{exists_note}")
    print(f"    archived joint-FE version -> {archive_dst}")
    print(f"    <- copied from {src}")

if missing_required:
    print(f"\nWARNING: proceeding with --force despite {len(missing_required)} missing "
          "required file(s) listed above — this will be a PARTIAL promotion.")

if not args.apply:
    print("\nDRY RUN — nothing was changed. Re-run with --apply to execute this plan.")
    sys.exit(0)

for src, archive_dst, target in plan:
    if target.exists():
        archive_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(archive_dst))
    shutil.copy2(str(src), str(target))

print(f"\nPromoted {len(plan)} files. Joint-FE originals archived under {archive_root}/ "
      f"(git-tracked files also remain in git history).")

print("\n" + "=" * 70)
print("Rebuilding results/paper/ from the newly-promoted baseline")
print("=" * 70)

subprocess.run([sys.executable, str(project_root / "scripts" / "evaluation" / "build_paper_tables.py"),
                "--run-name", run], cwd=project_root, check=True)
subprocess.run([sys.executable, str(project_root / "scripts" / "evaluation" / "build_density_robustness_table.py"),
                "--run-name", run], cwd=project_root, check=True)

single_source_png = LUNARC_SINGLE_SOURCE_DIR / "single_source_density_comparison.png"
paper_png = project_root / "results" / "paper" / "result4_single_source_density_comparison.png"
if single_source_png.exists():
    shutil.copy2(single_source_png, paper_png)
    print(f"  → {paper_png}  (from {single_source_png})")
else:
    print(f"  NOT FOUND (skipped): {single_source_png}")

print("\nDone. Review results/paper/ (especially density_robustness.tex and "
      "result4_*.png) before committing.")
