"""
_shared_eval_context.py
=======================
Shared setup, argument parsing and metric helpers for
P02_shared_geometry_reconstruction.py (Result 2) and
P03_emergent_structure.py (Result 3). Imported by those scripts; not run
directly.

Provides
--------
- add_common_args: --run-name.
- build_base_context: SNI descriptions and section maps and the run's 4-digit
  embeddings for SNI92/SNI02/SNI07 (EMBEDDING_DIR/{run}/embeddings_*.npz).
- load_baseline_and_hub_embeddings (P02 only): the baseline, i.e. the
  untrained text encoder (sentence-transformers/all-MiniLM-L6-v2, 384-d,
  without the PCA to 64-d used to initialise training), and the run's
  hierarchy-hub embeddings.
- load_crosswalks: the SCB correspondence tables SNI92→SNI02→SNI07.
- Metric helpers (hierarchy-level similarity pairs, retrieval, average
  precision) and compute_key_metrics, which recomputes every scalar metric
  for any embedding set so that baseline and trained embeddings are scored
  identically.

Sampled metrics take an explicit random.Random(SEED) so results do not
depend on the order in which sections run.
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

DEFAULT_RUN_NAME  = "shared_cosine_v16"


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

def add_common_args(parser):
    """Add --run-name."""
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                        help="Embedding run under EMBEDDING_DIR; also the label of the output folder.")
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


# ── Load embeddings ────────────────────────────────────────────────────────────

def load_embeddings(label: str, run_dir: Path) -> tuple[np.ndarray, list[str]]:
    """embeddings_{label}.npz written by training/operator_embedding.export_embeddings."""
    data = np.load(run_dir / f"embeddings_{label}.npz", allow_pickle=True)
    codes = list(data["codes"])
    vecs = data["embeddings"].astype(np.float32)
    print(f"  {label}: loaded {len(codes)} embeddings")
    return vecs, codes


def build_base_context(args) -> SimpleNamespace:
    """Everything both P02 and P03 need before their own narrative sections
    run: eval_dir, SNI 4d descriptions/section maps, and the run's 4d
    embeddings/codes/index. Baseline and hub embeddings are loaded separately
    by load_baseline_and_hub_embeddings (P02 only)."""
    ctx = SimpleNamespace()
    ctx.args = args

    # One output folder per run; filenames keep their {run_name}_ prefix.
    ctx.eval_dir = results_dir / "embedding_eval" / args.run_name
    ctx.eval_dir.mkdir(parents=True, exist_ok=True)

    ctx.sni07 = load_4d("2007")
    ctx.sni02 = load_4d("2002")
    ctx.sni92 = load_4d("1992")

    ctx.section_map_07 = SNI_2digits_to_sections("SNI07")
    ctx.section_map_02 = SNI_2digits_to_sections("SNI02")
    ctx.section_map_92 = SNI_2digits_to_sections("SNI92")

    ctx.run_dir = embed_dir / args.run_name

    print("\nLoading embeddings …")
    ctx.vecs_07, ctx.codes_07 = load_embeddings("sni07", ctx.run_dir)
    ctx.vecs_02, ctx.codes_02 = load_embeddings("sni02", ctx.run_dir)
    ctx.vecs_92, ctx.codes_92 = load_embeddings("sni92", ctx.run_dir)

    ctx.idx_07 = {c: i for i, c in enumerate(ctx.codes_07)}
    ctx.idx_02 = {c: i for i, c in enumerate(ctx.codes_02)}
    ctx.idx_92 = {c: i for i, c in enumerate(ctx.codes_92)}

    return ctx


def load_baseline_and_hub_embeddings(ctx: SimpleNamespace) -> None:
    """P02-only (see module docstring): encodes the baseline (frozen
    BASE_MODEL) for the SNI 4-digit codes and the SNI07 hub descriptions
    (3d/2d/section), and loads the run's hub embeddings, which are aggregates
    of their 4-digit descendants. Mutates ctx in place."""
    print(f"\nEncoding baseline embeddings ({BASE_MODEL}) …")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BASE_MODEL)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=128, normalize_embeddings=True,
                            show_progress_bar=False).astype(np.float32)

    ctx.bl_vecs_07 = encode([ctx.sni07.get(c, c) for c in ctx.codes_07])
    ctx.bl_vecs_02 = encode([ctx.sni02.get(c, c) for c in ctx.codes_02])
    ctx.bl_vecs_92 = encode([ctx.sni92.get(c, c) for c in ctx.codes_92])

    # ── SNI07 hierarchy hubs (3d/2d/section) ───────────────────────────────────
    ctx.desc_3d  = _load_codes("mapping_Sni_3digits_2007.json", "000")
    ctx.desc_2d  = _load_codes("mapping_Sni_2digits_2007.json", "00")
    ctx.desc_sec = _load_codes("mapping_Sni_section_2007.json", "Unknown")

    print("\nLoading SNI07 hierarchy-hub embeddings (3d/2d/section) …")
    ctx.vecs_3d,  ctx.codes_3d  = load_embeddings("sni07_3d", ctx.run_dir)
    ctx.vecs_2d,  ctx.codes_2d  = load_embeddings("sni07_2d", ctx.run_dir)
    ctx.vecs_sec, ctx.codes_sec = load_embeddings("sni07_section", ctx.run_dir)
    ctx.idx_3d  = {c: i for i, c in enumerate(ctx.codes_3d)}
    ctx.idx_2d  = {c: i for i, c in enumerate(ctx.codes_2d)}
    ctx.idx_sec = {c: i for i, c in enumerate(ctx.codes_sec)}

    ctx.bl_vecs_3d  = encode([ctx.desc_3d.get(c, c) for c in ctx.codes_3d])
    ctx.bl_vecs_2d  = encode([ctx.desc_2d.get(c, c) for c in ctx.codes_2d])
    ctx.bl_vecs_sec = encode([ctx.desc_sec.get(c, c) for c in ctx.codes_sec])


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
    """
    Average precision of a ranked list against all true partners (precision at
    the rank of each true partner, averaged over partners; 0.0 if none is
    found). Unlike MRR, this rewards retrieving every partner of a
    many-to-many correspondence, not just the first.
    """
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


# ── Shared helpers for section I — pure functions of their inputs so
# compute_key_metrics (section G) can reuse them on baseline embeddings too ───

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
    """
    Division-level similarity S̄_ab = mean over i in a, j in b of cos(z_i, z_j),
    computed from the 4-digit SNI07 embeddings grouped by 2-digit division, and
    its Spearman rho against IO relatedness. Complements I_io_rho, which uses
    the hub (division) embeddings directly.
    """
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


# ── Shared helpers for section K — source-wise KL evaluation ──────────────────
# For every evidence family s, build P^(s) = row-normalised(A^(s)) directly
# from that source's raw relation matrix, and evaluate it against a
# cosine-softmax Q at one fixed temperature (P02 --tau, default 0.05) for both
# the baseline and the trained embeddings. Training instead learns one
# temperature per target, bounded to [e^-1, e], and fits the diffusion
# targets, so these KL values are a comparison between embeddings, not the
# training loss itself.

def _softmax_rows(sim: np.ndarray, tau: float) -> np.ndarray:
    """softmax(sim / tau, axis=1) with the diagonal already expected to be
    -inf (self-similarity excluded, as in training)."""
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
    """
    Cross-version concordance KL: P_i is uniform over the true correspondence
    targets of source code i, Q_i is the cosine-softmax from i to every code of
    the target version (no diagonal to exclude, as source and target are
    different versions).
    """
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

    if ctx.labor_path.exists():
        for coding, idx_, v_, lbl in [("SNI07", ctx.idx_07, v07, "07"),
                                       ("SNI02", ctx.idx_02, v02, "02"),
                                       ("SNI92", ctx.idx_92, v92, "92")]:
            sub = ctx.labor_df[ctx.labor_df["coding"] == coding]
            pairs = [(str(r.sni_from), str(r.sni_to), float(r.phi)) for r in sub.itertuples()]
            res = _row_kl_from_pairs(idx_, v_, pairs, tau, symmetric=True)
            if res:
                out[f"K_labor_KL_{lbl}"] = round(res["KL"], 4)

    if ctx.ownership_path.exists():
        for coding, idx_, v_, lbl in [("SNI07", ctx.idx_07, v07, "07"),
                                       ("SNI02", ctx.idx_02, v02, "02"),
                                       ("SNI92", ctx.idx_92, v92, "92")]:
            sub = ctx.ownership_df[ctx.ownership_df["coding"] == coding]
            pairs = [(str(r.sni_from), str(r.sni_to), float(r.phi)) for r in sub.itertuples()]
            res = _row_kl_from_pairs(idx_, v_, pairs, tau, symmetric=True)
            if res:
                out[f"K_ownership_KL_{lbl}"] = round(res["KL"], 4)

    if ctx.io_path.exists():
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

    return out


# ── Helper: compute all scalar metrics for any set of embeddings ──────────────
# Reads its fixed inputs (codes_*, idx_*, true_targets*, section_map_07,
# labor_path/ownership_path/io_path + their loaded _dfs) off `ctx` — safe to
# call on ANY (v07, v02, v92, v3d, v2d,
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
    if ctx.labor_path.exists():
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
    if ctx.ownership_path.exists():
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

    # I: IO relation consistency
    if ctx.io_path.exists():
        m.update(io_consistency_metric(ctx, v2d, ctx.io_df))
        m.update(aggregate_up_io_metric(v07, ctx.codes_07, ctx.io_df))

    # K: source-wise KL(P^(s) || Q)
    m.update(kl_source_metrics(ctx, v07, v02, v92, v3d, v2d, vsec, tau))

    return m
