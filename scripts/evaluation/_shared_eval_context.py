"""
_shared_eval_context.py
========================
Shared setup, argument parsing, and pure-function metric helpers for
scripts/evaluation/P02_shared_geometry_reconstruction.py (Result 2: does
one shared geometry reconstruct every source) and
scripts/evaluation/P03_emergent_structure.py (Result 3: emergent
structure) — the two files evaluate_embeddings.py (2285 lines) was split
into 2026-07-30 to match docs/Embedding validation.md's 6-part paper
results structure.

NOT a standalone entrypoint (leading underscore, no `if __name__ ==
"__main__"`) — import from it, don't run it directly.

P02 owns: E, E2, E3, F, H, I, I2, J, K narrative sections + G (multi-signal
summary) + K2 (consolidated source table/dot plot).
P03 owns: A, B, B2, C, D, D2 narrative sections + the aligned-vs-random
cosine-similarity histogram.

Both need: SNI 4d code/description loading, the fine-tuned model, the 4d
atom embeddings (vecs_07/02/92), the hierarchy-ordinal pair builder
(exhaustive_pairs_by_level / levels), and the cross-version crosswalks
(mapping_02_07/true_targets, mapping_92_02/true_targets_92) — P03's own
A/B/B2 narrative uses these directly, and P02's compute_key_metrics (used
to seed section G's "trained" column — see that file's own comment for
why) independently recomputes A/B/B2's scalars from any embedding set
passed to it, using the same functions.

Baseline embeddings (bl_vecs_*) and NACE-hierarchy-hub embeddings
(vecs_3d/2d/sec + bl_vecs_3d/2d/sec) are loaded by
`load_baseline_and_hub_embeddings()` rather than unconditionally here —
verified 2026-07-30 that P03's sections (the former lines 357-1123 of
evaluate_embeddings.py: A/B/B2/C/D/D2 + aligned-vs-random) never reference
bl_vecs_*/vecs_3d/2d/sec/idx_3d/2d/sec/_bl_st at all, so P03 calls
`build_base_context()` alone and skips an extra baseline
SentenceTransformer encode pass (plus three hierarchy-hub encode passes)
it would otherwise never use.

A note on randomness and the split: exhaustive_pairs_by_level's
"diff_section"/"same_section" levels draw a bounded random subsample (up
to max_n=5000) via an explicit `rng` parameter (not a hidden shared
module-level object, unlike the original script). In the original
monolithic script, a single `random.Random(42)` was shared and
sequentially advanced across the whole run, so section G's
`compute_key_metrics(bl_vecs_07, ...)` call for `baseline_summary`
happened to draw from whatever position section A and D's own draws had
already left the RNG at — an accidental coupling between unrelated
sections, not a deliberate property of the "pure function of its inputs"
`compute_key_metrics` is documented to be. Now that A (P03) and
compute_key_metrics (P02) run in separate processes, that accidental
cross-section coupling can no longer be reproduced (nor should it be —
it was never a meaningful dependency). Each call site below is given its
own freshly-seeded `random.Random(SEED)`, so `A_sim_same_3d`/`A_sim_same_2d`
(exhaustive enumerations, well under the 5000 cap for the current SNI07
atom set — never RNG-dependent at all) match the original run exactly,
and `A_sim_diff_section`/`A_treedist_rho`/`A_hierarchy_monotone` match
exactly for the TRAINED column (P02's first and only fresh-RNG use there
reproduces P03/original's own first-and-only fresh-RNG A section
bit-for-bit) — but the BASELINE row's diff_section-derived values may
differ in the last decimal place or two from the original run's baseline
row, since original's baseline call inherited RNG state from sections
that no longer run in the same process. See P02's own module docstring
for the verification result.
"""

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.sni_codes import SNI_2digits_to_sections
from latent_industry.common.paths import METADATA_DIR, PROCESSED_DATA_DIR, EMBEDDING_DIR

metadata_dir = METADATA_DIR
results_dir  = project_root / "results"
embed_dir    = EMBEDDING_DIR

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED       = 42

DEFAULT_RUN_NAME  = "multilayer_v1"
DEFAULT_MODEL_DIR = str(EMBEDDING_DIR / DEFAULT_RUN_NAME / "model_final")
DEFAULT_BASELINE  = False


# ── Tee: write to both stdout and a log file ───────────────────────────────────

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, text):
        for f in self.files: f.write(text); f.flush()
    def flush(self):
        for f in self.files: f.flush()


def open_log(eval_dir: Path, run_name: str, suffix: str) -> tuple["Tee", object]:
    """Redirect sys.stdout to a Tee(terminal, logfile). Returns (tee, file) —
    caller is responsible for restoring sys.stdout and closing the file when
    done (see each P0x script's final lines)."""
    log_path = eval_dir / f"{run_name}_{suffix}_metrics.txt"
    log_file = open(log_path, "w")
    tee = Tee(sys.__stdout__, log_file)
    return tee, log_file


def _prefix_groups(codes: list[str], n: int) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in codes:
        groups.setdefault(c[:n], []).append(c)
    return groups


# ── CLI ─────────────────────────────────────────────────────────────────────

def add_common_args(parser, *, graph_name: bool = True):
    """--baseline / --model-dir / --run-name (+ --graph-name unless
    graph_name=False), with the exact defaults/help text
    evaluate_embeddings.py used pre-split. P02 needs --graph-name (H/I/J/K
    read graph_edges_{name}/graph_nodes_{name}/diffusion_matrix_{name});
    P03's sections (A/B/B2/C/D/D2) never reference args.graph_name
    (verified 2026-07-30 — no such reference in the former lines
    357-1123), so P03's own parser should call this with graph_name=False
    rather than carry a dead flag."""
    parser.add_argument("--baseline", action="store_true", default=DEFAULT_BASELINE,
                        help="Use raw BASE_MODEL instead of the fine-tuned model.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="Path to model_final directory.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                        help="Label for output files, e.g. 'multilayer_v1' or 'multilayer_v1_768d'.")
    if graph_name:
        parser.add_argument("--graph-name", default=DEFAULT_RUN_NAME,
                            help="--name used when running build_multilayer_diffusion.py — selects which "
                                 "graph_edges_{name}.parquet / graph_nodes_{name}.parquet / "
                                 "diffusion_matrix_{name}.npz to read for sections H/I/J/K. Defaults to "
                                 "--run-name's value so a no-args run always evaluates a model against "
                                 "its OWN graph — pass this explicitly if you're evaluating a run (e.g. "
                                 "a --base-model capacity comparison) whose graph-name differs from its "
                                 "run-name. 'multilayer_v1' is the current blessed graph/target; "
                                 "'routing'/'default' are retired pipelines kept only for historical "
                                 "comparison — see TODO.md §2/§5 and docs/conceptual_audit.md.")
    return parser


# ── Load SNI descriptions ──────────────────────────────────────────────────────

def load_4d(year: str) -> dict[str, str]:
    with open(metadata_dir / f"mapping_Sni_4digits_{year}.json") as f:
        d = json.load(f)
    d.pop("0000", None)
    return d


def _load_codes(filename: str, drop_key: str) -> dict[str, str]:
    with open(metadata_dir / filename) as f:
        d = json.load(f)
    d.pop(drop_key, None)
    return d


# ── Load or compute embeddings ─────────────────────────────────────────────────

# train_sentence_embedding.py actually saves the 4d targets under the plain
# "sni07"/"sni02"/"sni92" names (embed_and_save's label, not this file's
# lookup label) — so a straight `embeddings_{label}.npz` lookup for
# "sni07_4d" never hits that cache. Map each label to its legacy alias here
# once, instead of assuming a single naming convention every embedding
# producer must match — any regime's export (see e.g.
# training/operator_embedding.py) can use either name and still be found.
_LABEL_ALIASES = {"sni07_4d": "sni07", "sni02_4d": "sni02", "sni92_4d": "sni92"}


def load_or_encode(
    label: str, code_to_text: dict[str, str], model, *, baseline: bool, eff_embed_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    candidates = [label] + ([_LABEL_ALIASES[label]] if label in _LABEL_ALIASES else [])
    if not baseline:
        for candidate in candidates:
            npz_path = eff_embed_dir / f"embeddings_{candidate}.npz"
            if npz_path.exists():
                data  = np.load(npz_path, allow_pickle=True)
                codes = list(data["codes"])
                vecs  = data["embeddings"].astype(np.float32)
                print(f"  {label}: loaded {len(codes)} embeddings from disk ({npz_path.name})")
                return vecs, codes

    print(f"  {label}: encoding {len(code_to_text)} codes with model …")
    codes = sorted(code_to_text.keys())
    texts = [code_to_text[c] for c in codes]
    vecs  = model.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    return vecs.astype(np.float32), codes


def load_model(args):
    from sentence_transformers import SentenceTransformer
    if args.baseline:
        print(f"Mode: BASELINE (raw {BASE_MODEL})")
        return SentenceTransformer(BASE_MODEL)
    elif args.model_dir is not None:
        model_path = Path(args.model_dir)
        if model_path.exists():
            print(f"Mode: FINE-TUNED ({model_path})  [--model-dir override]")
            return SentenceTransformer(str(model_path))
        else:
            print(f"WARNING: {model_path} not found — falling back to baseline model.")
            return SentenceTransformer(BASE_MODEL)
    else:
        model_path = embed_dir / "model_final"
        if model_path.exists():
            print(f"Mode: FINE-TUNED ({model_path})")
            return SentenceTransformer(str(model_path))
        else:
            print(f"WARNING: {model_path} not found — falling back to baseline model.")
            return SentenceTransformer(BASE_MODEL)


# v7 §4.4: once operator_embedding.export_embeddings exports hub rows as the
# AGGREGATE of their 4d descendants (v7 §1.2) rather than raw free
# parameters, load_or_encode above already prefers that cached
# embeddings_sni07_{3d,2d,section}.npz over live text-encoding — no loading-
# path change needed. What DOES need to change is the metric name: a hub
# vector read from that cache means "centroid of a division's codes", not
# "the model's encoding of the division description" (what every other
# regime's H_nace_rho_3d/2d/section has always meant) — comparing the two
# under one name would silently compare different things across the change.
# Suffix only the labels that actually came from the aggregation cache.
_HUB_LABEL_BY_SUFFIX = {"3d": "sni07_3d", "2d": "sni07_2d", "section": "sni07_section"}


def build_base_context(args) -> SimpleNamespace:
    """Everything both P02 and P03 need before their own narrative sections
    run: eval_dir, SNI 4d descriptions/section maps, the fine-tuned (or
    baseline, if --baseline) model, and the 4d atom embeddings/codes/index.
    Does NOT load baseline-comparison or NACE-hub embeddings — call
    load_baseline_and_hub_embeddings(ctx, args) separately for those
    (P02-only; see this module's docstring)."""
    ctx = SimpleNamespace()
    ctx.args = args

    # One subfolder per run (2026-07-29) — results/embedding_eval/ had grown to
    # 100+ flat files across runs, since every run writes ~17 outputs and none
    # of the older ones get cleaned up. Filenames keep their {run_name}_ prefix
    # for anything (e.g. build_paper_table.py) that still greps/copies a single
    # file out of context.
    ctx.eval_dir = results_dir / "embedding_eval" / args.run_name
    ctx.eval_dir.mkdir(parents=True, exist_ok=True)

    ctx.sni07 = load_4d("2007")
    ctx.sni02 = load_4d("2002")
    ctx.sni92 = load_4d("1992")

    ctx.section_map_07 = SNI_2digits_to_sections("SNI07")
    ctx.section_map_02 = SNI_2digits_to_sections("SNI02")
    ctx.section_map_92 = SNI_2digits_to_sections("SNI92")

    # Determine effective embed_dir (may be overridden by --model-dir)
    if args.model_dir is not None:
        ctx.eff_embed_dir = Path(args.model_dir).parent
    else:
        ctx.eff_embed_dir = embed_dir

    # v7 §4.6: J2_target_fidelity scores fidelity to the FUSED multilayer_v1
    # diffusion target — meaningless (reads as a regression, isn't one) for a
    # regime that never trains on it. train_shared_cosine_embedding.py writes
    # its own regime into training_provenance.json; use that to skip J2 rather
    # than deleting it outright (legacy multilayer_v1/baseline runs, which never
    # write this file or write a different regime, keep exactly today's
    # behaviour).
    _provenance_path = ctx.eff_embed_dir / "training_provenance.json"
    ctx.training_regime = None
    ctx.family_weights: dict[str, float] = {}
    if _provenance_path.exists() and not args.baseline:
        try:
            _provenance = json.loads(_provenance_path.read_text())
            ctx.training_regime = _provenance.get("regime")
            ctx.family_weights = _provenance.get("family_weights", {})
        except (json.JSONDecodeError, OSError):
            ctx.training_regime = None

    ctx.model = load_model(args)

    print("\nLoading embeddings …")
    ctx.vecs_07, ctx.codes_07 = load_or_encode(
        "sni07_4d", ctx.sni07, ctx.model, baseline=args.baseline, eff_embed_dir=ctx.eff_embed_dir)
    ctx.vecs_02, ctx.codes_02 = load_or_encode(
        "sni02_4d", ctx.sni02, ctx.model, baseline=args.baseline, eff_embed_dir=ctx.eff_embed_dir)
    ctx.vecs_92, ctx.codes_92 = load_or_encode(
        "sni92_4d", ctx.sni92, ctx.model, baseline=args.baseline, eff_embed_dir=ctx.eff_embed_dir)

    ctx.idx_07 = {c: i for i, c in enumerate(ctx.codes_07)}
    ctx.idx_02 = {c: i for i, c in enumerate(ctx.codes_02)}
    ctx.idx_92 = {c: i for i, c in enumerate(ctx.codes_92)}

    return ctx


def load_baseline_and_hub_embeddings(ctx: SimpleNamespace) -> None:
    """P02-only (see module docstring): encodes the frozen BASE_MODEL
    baseline (bl_vecs_07/02/92) for section G's delta comparison, and the
    SNI07 hierarchy-hub descriptions (3d/2d/section — not fine-tuning
    targets, so no cached embeddings_sni07_3d.npz etc. unless
    operator_embedding's aggregation export wrote one; load_or_encode falls
    through to live-encoding otherwise) needed by section H and by
    compute_key_metrics. Mutates ctx in place."""
    args = ctx.args

    # ── Baseline embeddings (always encoded for delta comparison in section G) ─
    if args.baseline:
        ctx._bl_st = ctx.model   # primary is already BASE_MODEL; delta will be zero
    else:
        print(f"\nEncoding baseline embeddings ({BASE_MODEL}) for comparison …")
        from sentence_transformers import SentenceTransformer
        ctx._bl_st = SentenceTransformer(BASE_MODEL)

    ctx.bl_vecs_07 = ctx._bl_st.encode([ctx.sni07.get(c, c) for c in ctx.codes_07],
                                        batch_size=128, normalize_embeddings=True,
                                        show_progress_bar=False).astype(np.float32)
    ctx.bl_vecs_02 = ctx._bl_st.encode([ctx.sni02.get(c, c) for c in ctx.codes_02],
                                        batch_size=128, normalize_embeddings=True,
                                        show_progress_bar=False).astype(np.float32)
    ctx.bl_vecs_92 = ctx._bl_st.encode([ctx.sni92.get(c, c) for c in ctx.codes_92],
                                        batch_size=128, normalize_embeddings=True,
                                        show_progress_bar=False).astype(np.float32)

    # ── SNI07 hierarchy-hub descriptions (3d/2d/section) ───────────────────────
    ctx.desc_3d  = _load_codes("mapping_Sni_3digits_2007.json", "000")
    ctx.desc_2d  = _load_codes("mapping_Sni_2digits_2007.json", "00")
    ctx.desc_sec = _load_codes("mapping_Sni_section_2007.json", "Unknown")

    print("\nEncoding SNI07 hierarchy-hub descriptions (3d/2d/section) …")
    ctx.vecs_3d,  ctx.codes_3d  = load_or_encode(
        "sni07_3d", ctx.desc_3d, ctx.model, baseline=args.baseline, eff_embed_dir=ctx.eff_embed_dir)
    ctx.vecs_2d,  ctx.codes_2d  = load_or_encode(
        "sni07_2d", ctx.desc_2d, ctx.model, baseline=args.baseline, eff_embed_dir=ctx.eff_embed_dir)
    ctx.vecs_sec, ctx.codes_sec = load_or_encode(
        "sni07_section", ctx.desc_sec, ctx.model, baseline=args.baseline, eff_embed_dir=ctx.eff_embed_dir)
    ctx.idx_3d  = {c: i for i, c in enumerate(ctx.codes_3d)}
    ctx.idx_2d  = {c: i for i, c in enumerate(ctx.codes_2d)}
    ctx.idx_sec = {c: i for i, c in enumerate(ctx.codes_sec)}

    _hub_from_cache = {
        lbl: (ctx.eff_embed_dir / f"embeddings_{lbl}.npz").exists() and not args.baseline
        for lbl in _HUB_LABEL_BY_SUFFIX.values()
    }
    ctx._hub_from_cache = _hub_from_cache

    ctx.bl_vecs_3d  = ctx._bl_st.encode([ctx.desc_3d.get(c, c) for c in ctx.codes_3d],
                                         batch_size=128, normalize_embeddings=True,
                                         show_progress_bar=False).astype(np.float32)
    ctx.bl_vecs_2d  = ctx._bl_st.encode([ctx.desc_2d.get(c, c) for c in ctx.codes_2d],
                                         batch_size=128, normalize_embeddings=True,
                                         show_progress_bar=False).astype(np.float32)
    ctx.bl_vecs_sec = ctx._bl_st.encode([ctx.desc_sec.get(c, c) for c in ctx.codes_sec],
                                         batch_size=128, normalize_embeddings=True,
                                         show_progress_bar=False).astype(np.float32)


# ── Hierarchy-level pair sampling (section A; also used by compute_key_metrics) ─

levels = ["same_3d", "same_2d", "diff_section"]
level_labels = {
    "same_3d":      "Same 3d group\n(class siblings)",
    "same_2d":      "Same 2d division\ndiff group",
    "diff_section": "Different\nsection",
}


def exhaustive_pairs_by_level(
    codes: list[str], vecs: np.ndarray, level: str, *,
    idx_07: dict, section_map_07: dict, rng: random.Random, max_n: int = 5000,
) -> list[float]:
    """
    Collect cosine similarities for all code pairs at a given hierarchy level.

    Atoms are 4-digit, so the finest level is same_3d (codes sharing a 3-digit
    group prefix, i.e. siblings within one NACE class group).
    """
    sims: list[float] = []
    by_3d = _prefix_groups(codes, 3)
    by_2d = _prefix_groups(codes, 2)
    sm    = section_map_07

    if level == "same_3d":
        # Pairs sharing same 3-digit group — siblings at the finest level
        for grp in by_3d.values():
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    a, b = grp[i], grp[j]
                    sims.append(float(vecs[idx_07[a]] @ vecs[idx_07[b]]))
    elif level == "same_2d":
        # Same division, different group
        for grp in by_2d.values():
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    a, b = grp[i], grp[j]
                    if a[:3] != b[:3]:
                        sims.append(float(vecs[idx_07[a]] @ vecs[idx_07[b]]))
    elif level == "diff_section":
        attempts, cap = 0, max_n * 30
        while len(sims) < max_n and attempts < cap:
            attempts += 1
            a, b = rng.sample(codes, 2)
            sa, sb = sm.get(a[:2], "?"), sm.get(b[:2], "?")
            if sa != sb and sa != "?" and sb != "?":
                sims.append(float(vecs[idx_07[a]] @ vecs[idx_07[b]]))
    elif level == "same_section":
        # Same section, different 2-digit division — fills the gap between
        # same_2d and diff_section for the tree-distance correlation below.
        by_sec: dict[str, list[str]] = {}
        for c in codes:
            by_sec.setdefault(sm.get(c[:2], "?"), []).append(c)
        secs = [s for s in by_sec if s != "?" and len(by_sec[s]) > 1]
        attempts, cap = 0, max_n * 30
        while len(sims) < max_n and attempts < cap and secs:
            attempts += 1
            sec = rng.choice(secs)
            a, b = rng.sample(by_sec[sec], 2)
            if a[:2] != b[:2]:
                sims.append(float(vecs[idx_07[a]] @ vecs[idx_07[b]]))

    if len(sims) > max_n:
        rng.shuffle(sims)
        sims = sims[:max_n]
    return sims


# ── Cross-version retrieval helpers (sections B/B2; also used by compute_key_metrics) ─

Ks = [1, 5, 10, 20]


def average_precision(ranked_codes: list[str], true_targets_set: set[str]) -> float:
    """v7 §4.1: MRR only asks whether *some* correct partner ranks first —
    the wrong question for a many-to-many crosswalk (a model that perfectly
    satisfies one partner per code and abandons the rest scores MRR=1.0).
    Standard average precision over the FULL ranked list against every true
    partner: precision at each rank a true partner is found, averaged over
    all true partners (0.0 if none are ever found)."""
    if not true_targets_set:
        return 0.0
    hits = 0
    precisions = []
    for rank, c in enumerate(ranked_codes, start=1):
        if c in true_targets_set:
            hits += 1
            precisions.append(hits / rank)
    return float(np.mean(precisions)) if precisions else 0.0


def load_crosswalks(ctx: SimpleNamespace) -> None:
    """mapping_02_07/true_targets and mapping_92_02/true_targets_92 — needed
    by P03's own B/B2 narrative AND by compute_key_metrics (P02). Mutates
    ctx in place."""
    mapping_02_07 = pd.read_csv(metadata_dir / "mapping_02_07.csv", dtype=str)
    mapping_02_07["SNI02"] = mapping_02_07["SNI02"].str.zfill(5).str[:4]
    mapping_02_07["SNI07"] = mapping_02_07["SNI07"].str.zfill(5).str[:4]
    true_targets: dict[str, set] = {}
    for _, row in mapping_02_07.drop_duplicates().iterrows():
        src, tgt = row["SNI02"], row["SNI07"]
        if src in ctx.idx_02 and tgt in ctx.idx_07:
            true_targets.setdefault(src, set()).add(tgt)
    ctx.mapping_02_07 = mapping_02_07
    ctx.true_targets  = true_targets

    mapping_92_02 = pd.read_csv(metadata_dir / "mapping_92_02.csv")
    mapping_92_02["SNI92"] = mapping_92_02["SNI92"].astype(str).str.zfill(5).str[:4]
    mapping_92_02["SNI02"] = mapping_92_02["SNI02"].astype(str).str.zfill(5).str[:4]
    true_targets_92: dict[str, set] = {}
    for _, row in mapping_92_02.drop_duplicates().iterrows():
        src = row["SNI92"]
        tgt = row["SNI02"]
        if src in ctx.idx_92 and tgt in ctx.idx_02:
            true_targets_92.setdefault(src, set()).add(tgt)
    ctx.mapping_92_02  = mapping_92_02
    ctx.true_targets_92 = true_targets_92


# ── Shared helpers for sections H/I/J — pure functions of their inputs so
# compute_key_metrics (section G) can reuse them on baseline embeddings too ───

def relation_consistency(
    etype: str, idx_map: dict, vecs_l: np.ndarray, edges_df: pd.DataFrame, node_lookup: dict
) -> tuple[list, list]:
    """Ground-truth edge weight and matching fine-tuned cosine similarity for
    every `etype` edge whose endpoints resolve to codes in idx_map."""
    sub = edges_df[edges_df["type"] == etype]
    gt, sim = [], []
    for r in sub.itertuples():
        ca = node_lookup.get(int(r.src))
        cb = node_lookup.get(int(r.dst))
        if ca in idx_map and cb in idx_map:
            gt.append(float(r.weight))
            sim.append(float(vecs_l[idx_map[ca]] @ vecs_l[idx_map[cb]]))
    return gt, sim


def nace_consistency_metrics(
    ctx: SimpleNamespace,
    v07: np.ndarray, v3d: np.ndarray, v2d: np.ndarray, vsec: np.ndarray,
    edges_df: pd.DataFrame, node_lookup: dict,
) -> dict:
    out = {}
    for etype, idx_map, vecs_l, lbl in [
        ("nace_4d",  ctx.idx_07,  v07,  "4d"),
        ("nace_3d",  ctx.idx_3d,  v3d,  "3d"),
        ("nace_2d",  ctx.idx_2d,  v2d,  "2d"),
        ("nace_sec", ctx.idx_sec, vsec, "section"),
    ]:
        gt, sim = relation_consistency(etype, idx_map, vecs_l, edges_df, node_lookup)
        if len(gt) >= 10:
            rho, _ = spearmanr(sim, gt)
            out[f"H_nace_rho_{lbl}"] = round(float(rho), 4)
    return out


def io_consistency_metric(ctx: SimpleNamespace, v2d: np.ndarray, io_df: pd.DataFrame) -> dict:
    gt, sim = [], []
    for r in io_df.itertuples():
        a, b = str(r.nace2d_from).zfill(2), str(r.nace2d_to).zfill(2)
        if a in ctx.idx_2d and b in ctx.idx_2d and a != b:
            gt.append(float(r.sim))
            sim.append(float(v2d[ctx.idx_2d[a]] @ v2d[ctx.idx_2d[b]]))
    if len(gt) >= 10:
        rho, _ = spearmanr(sim, gt)
        return {"I_io_rho": round(float(rho), 4)}
    return {}




def aggregate_up_io_metric(v07: np.ndarray, codes07: list[str], io_df: pd.DataFrame) -> dict:
    """v7 §4.3: S̄_ab = mean_{i in a, j in b} cos(z_i, z_j), computed directly
    over 4d-SNI07 codes grouped by 2d division — a genuinely different
    number from I_io_rho (mean-of-pairwise-cosines vs. the cosine of the
    hub's own normalised mean), not a redundant recomputation. Per v7 §1.2
    (hub rows ARE the flat mean of their 4d descendants), the two should now
    converge; a persistent gap between them is the direct measure of
    hub/4d decoupling the todo calls out."""
    div_groups = _prefix_groups(codes07, 2)
    idx07 = {c: i for i, c in enumerate(codes07)}
    gt, sim = [], []
    for r in io_df.itertuples():
        a, b = str(r.nace2d_from).zfill(2), str(r.nace2d_to).zfill(2)
        if a == b or a not in div_groups or b not in div_groups:
            continue
        idx_a = [idx07[c] for c in div_groups[a]]
        idx_b = [idx07[c] for c in div_groups[b]]
        block = v07[idx_a] @ v07[idx_b].T
        gt.append(float(r.sim))
        sim.append(float(block.mean()))
    if len(gt) >= 10:
        rho, _ = spearmanr(sim, gt)
        return {"I_io_aggregate_rho": round(float(rho), 4)}
    return {}


def _align_to_diffusion_matrix(
    ctx: SimpleNamespace,
    v07: np.ndarray, v02: np.ndarray, v92: np.ndarray, dm_codes: np.ndarray, dm_versions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build an (n_dm, dim) embedding array in diffusion_matrix_{name}.npz's own
    node order, by looking each (code, version) up in the matching vecs_*/idx_*."""
    vecs_by_ver = {"SNI07": (v07, ctx.idx_07), "SNI02": (v02, ctx.idx_02), "SNI92": (v92, ctx.idx_92)}
    n_dm = len(dm_codes)
    dm_vecs  = np.zeros((n_dm, v07.shape[1]), dtype=np.float32)
    dm_valid = np.zeros(n_dm, dtype=bool)
    for i, (c, v) in enumerate(zip(dm_codes, dm_versions)):
        vv, ii = vecs_by_ver[str(v)]
        j = ii.get(str(c))
        if j is not None:
            dm_vecs[i] = vv[j]
            dm_valid[i] = True
    return dm_vecs, dm_valid


# v7 §4.6: J2_target_fidelity/K_fused_target_KL score fidelity to the FUSED
# multilayer_v1 diffusion target — meaningless (reads as a regression,
# isn't one) for a regime that never trains on it.
_SKIP_J2_REGIMES = {"shared_cosine_expanded"}


def is_family_excluded(ctx: SimpleNamespace, family: str) -> bool:
    """True if `family`'s training weight was set to exactly 0.0 in this
    run's training_provenance.json (ctx.family_weights, from
    build_base_context). Used to skip/annotate evaluation sections for a
    source deliberately excluded from training, instead of reporting a
    held-out-looking correlation/KL as if it were a normal trained-family
    result. False (never excluded) for baseline runs, legacy runs without a
    training_provenance.json, or any family not in family_weights at all —
    only an explicit exact-zero weight counts."""
    return ctx.family_weights.get(family) == 0.0


def repulsion_and_fidelity_metrics(
    ctx: SimpleNamespace,
    v07: np.ndarray, v02: np.ndarray, v92: np.ndarray,
    S: np.ndarray, S_neg: np.ndarray, dm_codes: np.ndarray, dm_versions: np.ndarray,
) -> dict:
    out = {}
    dm_vecs, dm_valid = _align_to_diffusion_matrix(ctx, v07, v02, v92, dm_codes, dm_versions)

    # J1: repulsion pairs (S_neg > 0) should be LESS similar than other SNI07 pairs.
    is_07 = dm_versions == "SNI07"
    sub_idx = np.where(is_07 & dm_valid)[0]
    if len(sub_idx) > 10:
        sub_vecs = dm_vecs[sub_idx]
        sim_mat  = sub_vecs @ sub_vecs.T
        neg_sub  = S_neg[np.ix_(sub_idx, sub_idx)]
        iu, ju   = np.triu_indices(len(sub_idx), k=1)
        neg_w, sim_w = neg_sub[iu, ju], sim_mat[iu, ju]
        repel_mask = neg_w > 0
        if int(repel_mask.sum()) >= 10:
            rho, _ = spearmanr(sim_w[repel_mask], neg_w[repel_mask])
            out["J1_repulsion_rho"] = round(float(rho), 4)
            out["J1_repulsion_gap"] = round(float(sim_w[~repel_mask].mean() - sim_w[repel_mask].mean()), 4)

    # J2: did training converge to its own assigned diffusion target S?
    if ctx.training_regime not in _SKIP_J2_REGIMES:
        valid_idx = np.where(dm_valid)[0]
        if len(valid_idx) > 10:
            v_vecs = dm_vecs[valid_idx]
            v_sim  = v_vecs @ v_vecs.T
            v_S    = S[np.ix_(valid_idx, valid_idx)]
            nz_i, nz_j = np.nonzero(v_S)
            keep = nz_i != nz_j
            nz_i, nz_j = nz_i[keep], nz_j[keep]
            if len(nz_i) > 10:
                rho, _ = spearmanr(v_sim[nz_i, nz_j], v_S[nz_i, nz_j])
                out["J2_target_fidelity_rho"] = round(float(rho), 4)

    return out


# ── Shared helpers for section K — source-wise KL evaluation ──────────────────
# For every evidence
# family s, build P^(s) = row-normalised(A^(s)) directly from that source's
# raw relation matrix (same construction as train_kl's target, just per
# source instead of fused), and evaluate against the SAME cosine-softmax
# Q the trainer optimises (identical tau). This tests the exact quantity
# training minimises, per source — unlike H/I/J's Spearman correlations,
# which only test rank agreement and can't be compared to a training loss.

def _softmax_rows(sim: np.ndarray, tau: float) -> np.ndarray:
    """softmax(sim / tau, axis=1) with the diagonal already expected to be
    -inf (self-similarity excluded, matching train_kl's cos_sim.fill_diagonal_)."""
    z = sim / tau
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def _row_kl(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """KL(P_i || Q_i) per row, safe for P_ij == 0 (0 * log(0/x) := 0)."""
    mask = P > 0
    terms = np.zeros_like(P)
    terms[mask] = P[mask] * np.log(P[mask] / np.clip(Q[mask], 1e-12, None))
    return terms.sum(axis=1)


def _row_kl_from_pairs(
    idx_map: dict, vecs: np.ndarray, pairs: list[tuple[str, str, float]],
    tau: float, symmetric: bool = True,
) -> dict:
    """Generic same-universe source KL evaluator: build A^(s) from raw
    (code_a, code_b, weight) triples, row-normalise to P over anchors with
    any outgoing mass, and compare against cosine-softmax Q from `vecs` in
    the same code universe. Returns {} if fewer than 10 anchors survive."""
    n = len(idx_map)
    if n == 0:
        return {}
    A = np.zeros((n, n), dtype=np.float64)
    for a, b, w in pairs:
        if a in idx_map and b in idx_map and a != b:
            i, j = idx_map[a], idx_map[b]
            wv = max(float(w), 0.0)
            A[i, j] = max(A[i, j], wv)
            if symmetric:
                A[j, i] = max(A[j, i], wv)
    row_sums = A.sum(axis=1)
    active = row_sums > 0
    n_active = int(active.sum())
    if n_active < 10:
        return {}
    P = A[active] / row_sums[active, None]

    sim = (vecs @ vecs.T).astype(np.float64)
    np.fill_diagonal(sim, -np.inf)
    Q = _softmax_rows(sim, tau)[active]

    kl_rows = _row_kl(P, Q)
    return {"KL": float(kl_rows.mean()), "n_anchors": n_active}


def _row_kl_concordance(
    idx_src: dict, vecs_src: np.ndarray, idx_tgt: dict, vecs_tgt: np.ndarray,
    true_targets_map: dict[str, set], tau: float,
) -> dict:
    """Cross-version concordance KL: P^(concordance)_i is uniform over i's
    true mapped targets (equal-weighted multi-positive, per plan §3.1's
    'concordance' row), Q_i is cosine-softmax from the source code to every
    target-version code (no diagonal exclusion — src/tgt are different
    versions, so there's no self-similarity to mask)."""
    kls = []
    for src_code, targets in true_targets_map.items():
        if src_code not in idx_src:
            continue
        tgt_idx = [idx_tgt[c] for c in targets if c in idx_tgt]
        if not tgt_idx:
            continue
        sim = (vecs_tgt @ vecs_src[idx_src[src_code]]).astype(np.float64)
        z = sim / tau
        z = z - z.max()
        ez = np.exp(z)
        Q = ez / ez.sum()
        p = 1.0 / len(tgt_idx)
        kl = float(sum(p * np.log(p / max(Q[j], 1e-12)) for j in tgt_idx))
        kls.append(kl)
    if len(kls) < 10:
        return {}
    return {"KL": float(np.mean(kls)), "n_anchors": len(kls)}


def kl_source_metrics(
    ctx: SimpleNamespace,
    v07: np.ndarray, v02: np.ndarray, v92: np.ndarray,
    v3d: np.ndarray, v2d: np.ndarray, vsec: np.ndarray, tau: float,
) -> dict:
    """All family-level KL(P_s || Q) results for one embedding bundle.
    Called once directly on the trained embeddings (section K) and once via
    compute_key_metrics on the baseline embeddings (section G), so the
    printed/saved Δ is always KL(raw) - KL(trained) at matched tau."""
    out: dict = {}

    if ctx.labor_path.exists() and not is_family_excluded(ctx, "labor"):
        for coding, idx_, v_, lbl in [("SNI07", ctx.idx_07, v07, "07"),
                                       ("SNI02", ctx.idx_02, v02, "02"),
                                       ("SNI92", ctx.idx_92, v92, "92")]:
            sub = ctx.labor_df[ctx.labor_df["coding"] == coding]
            pairs = [(str(r.sni_from), str(r.sni_to), float(r.phi)) for r in sub.itertuples()]
            res = _row_kl_from_pairs(idx_, v_, pairs, tau, symmetric=True)
            if res:
                out[f"K_labor_KL_{lbl}"] = round(res["KL"], 4)

    if ctx.ownership_path.exists() and not is_family_excluded(ctx, "ownership"):
        for coding, idx_, v_, lbl in [("SNI07", ctx.idx_07, v07, "07"),
                                       ("SNI02", ctx.idx_02, v02, "02"),
                                       ("SNI92", ctx.idx_92, v92, "92")]:
            sub = ctx.ownership_df[ctx.ownership_df["coding"] == coding]
            pairs = [(str(r.sni_from), str(r.sni_to), float(r.phi)) for r in sub.itertuples()]
            res = _row_kl_from_pairs(idx_, v_, pairs, tau, symmetric=True)
            if res:
                out[f"K_ownership_KL_{lbl}"] = round(res["KL"], 4)

    if ctx.graph_edges_path.exists() and ctx.graph_nodes_path.exists() and not is_family_excluded(ctx, "nace_text"):
        for etype, idx_map, vecs_l, lbl in [("nace_4d", ctx.idx_07, v07, "4d"),
                                             ("nace_3d", ctx.idx_3d, v3d, "3d"),
                                             ("nace_2d", ctx.idx_2d, v2d, "2d"),
                                             ("nace_sec", ctx.idx_sec, vsec, "section")]:
            sub = ctx.graph_edges_df[ctx.graph_edges_df["type"] == etype]
            pairs = [(ctx.node_lookup.get(int(r.src)), ctx.node_lookup.get(int(r.dst)), float(r.weight))
                     for r in sub.itertuples()]
            res = _row_kl_from_pairs(idx_map, vecs_l, pairs, tau, symmetric=True)
            if res:
                out[f"K_nace_KL_{lbl}"] = round(res["KL"], 4)

    if ctx.io_path.exists() and not is_family_excluded(ctx, "io"):
        pairs = [(str(r.nace2d_from).zfill(2), str(r.nace2d_to).zfill(2), float(r.sim))
                 for r in ctx.io_df.itertuples()]
        res = _row_kl_from_pairs(ctx.idx_2d, v2d, pairs, tau, symmetric=True)
        if res:
            out["K_io_KL"] = round(res["KL"], 4)

    res = _row_kl_concordance(ctx.idx_02, v02, ctx.idx_07, v07, ctx.true_targets, tau)
    if res:
        out["K_concordance_KL_02to07"] = round(res["KL"], 4)
    res = _row_kl_concordance(ctx.idx_92, v92, ctx.idx_02, v02, ctx.true_targets_92, tau)
    if res:
        out["K_concordance_KL_92to02"] = round(res["KL"], 4)

    # v7 §4.6: same regime guard as J2_target_fidelity_rho — KL-to-the-fused-
    # target is equally meaningless for a regime that never trains on it.
    if ctx.diffusion_path.exists() and ctx.training_regime not in _SKIP_J2_REGIMES:
        dm_vecs, dm_valid = _align_to_diffusion_matrix(ctx, v07, v02, v92, ctx.dm_codes, ctx.dm_versions)
        S_pos = np.clip(ctx.S, 0.0, None).astype(np.float64)
        np.fill_diagonal(S_pos, 0.0)
        row_sums = S_pos.sum(axis=1)
        active = (row_sums > 0) & dm_valid
        n_active = int(active.sum())
        if n_active >= 10:
            P_fused = S_pos[active] / row_sums[active, None]
            sim = (dm_vecs @ dm_vecs.T).astype(np.float64)
            np.fill_diagonal(sim, -np.inf)
            Q_fused = _softmax_rows(sim, tau)[active]
            kl_rows = _row_kl(P_fused, Q_fused)
            out["K_fused_target_KL"] = round(float(kl_rows.mean()), 4)
            out["K_fused_target_n_anchors"] = n_active

    return out


# ── Helper: compute all scalar metrics for any set of embeddings ──────────────
# Reads its fixed inputs (codes_*, idx_*, true_targets*, section_map_07,
# labor_path/ownership_path + their loaded _dfs,
# graph_edges_df/node_lookup, io_df, S/S_neg/dm_codes/dm_versions,
# training_regime) off `ctx` — safe to call on ANY (v07, v02, v92, v3d, v2d,
# vsec) bundle sharing the same code universe as ctx. `rng` defaults to a
# freshly-seeded random.Random(SEED) — see this module's docstring for why
# each call site should pass its own fresh one rather than share one across
# calls.

def compute_key_metrics(
    ctx: SimpleNamespace,
    v07: np.ndarray, v02: np.ndarray, v92: np.ndarray,
    v3d: np.ndarray, v2d: np.ndarray, vsec: np.ndarray,
    tau: float, rng: random.Random | None = None,
) -> dict:
    if rng is None:
        rng = random.Random(SEED)
    m: dict = {}

    # A: hierarchy preservation
    _ls = {lv: exhaustive_pairs_by_level(ctx.codes_07, v07, lv, idx_07=ctx.idx_07,
                                          section_map_07=ctx.section_map_07, rng=rng)
           for lv in levels}
    _mn = [np.mean(_ls[lv]) for lv in levels]
    m["A_hierarchy_monotone"] = int(all(_mn[i] >= _mn[i+1] for i in range(len(_mn)-1)))
    for lv in levels:
        m[f"A_sim_{lv}"] = round(float(np.mean(_ls[lv])), 4)

    _same_sec = exhaustive_pairs_by_level(ctx.codes_07, v07, "same_section", idx_07=ctx.idx_07,
                                           section_map_07=ctx.section_map_07, rng=rng)
    _td_pairs = (
        [(1, s) for s in _ls["same_3d"]] + [(2, s) for s in _ls["same_2d"]]
        + [(3, s) for s in _same_sec] + [(4, s) for s in _ls["diff_section"]]
    )
    if len(_td_pairs) >= 10:
        _td, _sim_td = zip(*_td_pairs)
        _td_rho, _ = spearmanr(1.0 - np.array(_sim_td), np.array(_td))
        m["A_treedist_rho"] = round(float(_td_rho), 4)

    # B: SNI02 → SNI07 retrieval
    _mrr, _map, _r1, _r10 = [], [], [], []
    for src, tgts in ctx.true_targets.items():
        ranked = [ctx.codes_07[i] for i in np.argsort(-(v02[[ctx.idx_02[src]]] @ v07.T).ravel())]
        for rk, c in enumerate(ranked, 1):
            if c in tgts: _mrr.append(1.0/rk); break
        else: _mrr.append(0.0)
        _map.append(average_precision(ranked, tgts))
        _r1.append(float(any(c in tgts for c in ranked[:1])))
        _r10.append(float(any(c in tgts for c in ranked[:10])))
    m.update(B_MRR_02to07=round(float(np.mean(_mrr)), 4),
             B_MAP_02to07=round(float(np.mean(_map)), 4),
             B_R1_02to07=round(float(np.mean(_r1)), 4),
             B_R10_02to07=round(float(np.mean(_r10)), 4))

    # B2: SNI92 → SNI02 retrieval
    _mrr2, _map2, _r1_2 = [], [], []
    for src, tgts in ctx.true_targets_92.items():
        ranked = [ctx.codes_02[i] for i in np.argsort(-(v92[[ctx.idx_92[src]]] @ v02.T).ravel())]
        for rk, c in enumerate(ranked, 1):
            if c in tgts: _mrr2.append(1.0/rk); break
        else: _mrr2.append(0.0)
        _map2.append(average_precision(ranked, tgts))
        _r1_2.append(float(any(c in tgts for c in ranked[:1])))
    m.update(B2_MRR_92to02=round(float(np.mean(_mrr2)), 4),
             B2_MAP_92to02=round(float(np.mean(_map2)), 4),
             B2_R1_92to02=round(float(np.mean(_r1_2)), 4))

    # E: labor rho
    if ctx.labor_path.exists() and not is_family_excluded(ctx, "labor"):
        _ldf = ctx.labor_df
        for coding, v_, idx_, lbl in [("SNI07", v07, ctx.idx_07, "07"),
                                       ("SNI02", v02, ctx.idx_02, "02"),
                                       ("SNI92", v92, ctx.idx_92, "92")]:
            sub = _ldf[_ldf["coding"] == coding]
            phi_, sim_ = [], []
            for _, row in sub.iterrows():
                s, t = str(row["sni_from"]), str(row["sni_to"])
                if s in idx_ and t in idx_:
                    phi_.append(float(row["phi"]))
                    sim_.append(float(v_[idx_[s]] @ v_[idx_[t]]))
            if len(phi_) >= 10:
                rho_, _ = spearmanr(sim_, phi_)
                m[f"E_labor_rho_{lbl}"] = round(float(rho_), 4)

    # E2: ownership rho
    if ctx.ownership_path.exists() and not is_family_excluded(ctx, "ownership"):
        _odf = ctx.ownership_df
        for coding, v_, idx_, lbl in [("SNI07", v07, ctx.idx_07, "07"),
                                       ("SNI02", v02, ctx.idx_02, "02"),
                                       ("SNI92", v92, ctx.idx_92, "92")]:
            sub = _odf[_odf["coding"] == coding]
            phi_, sim_ = [], []
            for _, row in sub.iterrows():
                s, t = str(row["sni_from"]), str(row["sni_to"])
                if s in idx_ and t in idx_:
                    phi_.append(float(row["phi"]))
                    sim_.append(float(v_[idx_[s]] @ v_[idx_[t]]))
            if len(phi_) >= 10:
                rho_, _ = spearmanr(sim_, phi_)
                m[f"E2_ownership_rho_{lbl}"] = round(float(rho_), 4)

    # F: section silhouette
    from sklearn.metrics import silhouette_score
    _sl = [ctx.section_map_07.get(c[:2], "?") for c in ctx.codes_07]
    _km = np.array([s != "?" for s in _sl])
    if _km.sum() > 10:
        m["F_section_silhouette"] = round(float(
            silhouette_score(v07[_km], [s for s, m_ in zip(_sl, _km) if m_], metric="cosine")), 4)

    # H: NACE relation consistency
    if ctx.graph_edges_path.exists() and ctx.graph_nodes_path.exists() and not is_family_excluded(ctx, "nace_text"):
        m.update(nace_consistency_metrics(ctx, v07, v3d, v2d, vsec, ctx.graph_edges_df, ctx.node_lookup))

    # I: IO relation consistency
    if ctx.io_path.exists() and not is_family_excluded(ctx, "io"):
        m.update(io_consistency_metric(ctx, v2d, ctx.io_df))
        m.update(aggregate_up_io_metric(v07, ctx.codes_07, ctx.io_df))

    # J: repulsion consistency & training-target fidelity
    if ctx.diffusion_path.exists():
        m.update(repulsion_and_fidelity_metrics(ctx, v07, v02, v92, ctx.S, ctx.S_neg, ctx.dm_codes, ctx.dm_versions))

    # K: source-wise KL(P^(s) || Q)
    m.update(kl_source_metrics(ctx, v07, v02, v92, v3d, v2d, vsec, tau))

    return m
