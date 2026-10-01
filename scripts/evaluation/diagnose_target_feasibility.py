"""
diagnose_target_feasibility.py — model-independent feasibility, source-
integrity, and leakage diagnostics for a diffusion target (see
docs/conceptual_audit.md). Run BEFORE any GPU fine-tuning.

No text encoder anywhere in this script — the "free vector ceiling" fit is
~1,600 x 384 free parameters optimized directly against the target via the
same forward-KL objective train_sentence_embedding.py uses; this is cheap on
CPU or a single GPU (minutes) and measures whether the target is even
representable by SOME rank-384 cosine geometry, independent of any sentence
encoder's capacity or text-information limits.

Neither build_input_output_relatedness.py nor build_diffusion_targets.py can
be imported as library modules — both are executable pipeline scripts with
top-level side effects (downloads, full graph builds). Where this script
needs logic from them (NACE centering, crosswalk-based concept groups), it
re-implements the small piece needed rather than importing the script.

Sections
--------
1. Source integrity        — IO conservation context (code_reference.csv
                              fan-out + io_relatedness.parquet sanity), NACE
                              sign distribution before/after centering.
2. Source agreement         — pairwise rank correlation between evidence
                              families that share common (code_a, code_b)
                              pairs, before any diffusion/fusion.
3. Fused-target feasibility  — sparsity/entropy (reuses diagnose_diffusion_
                              target.py's sparsity_report), asymmetry +
                              detailed-balance residual, best rank-k
                              reconstruction ceiling, and (optional, if
                              --name-no-topk is given) top-k truncation
                              fidelity.
4. Model-independent ceilings — free-vector fit + held-out (train/test)
                              labor and nace_4d rho.
5. Leakage-safe split report — split-size/coverage + a self-consistency
                              check on graph_edges_{name}.parquet's split
                              column, using an independently re-derived
                              concept-group mapping.

Usage
-----
  python scripts/evaluation/diagnose_target_feasibility.py --name routing_topk30
  python scripts/evaluation/diagnose_target_feasibility.py --name routing_topk30 --name-no-topk routing
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts" / "evaluation"))

from latent_industry.common.paths import METADATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from latent_industry.common.splits import union_find_groups, split_key_for
from diagnose_diffusion_target import sparsity_report, print_sparsity_report, row_normalize_target

metadata_dir = METADATA_DIR
results_dir = project_root / "results" / "target_feasibility"
results_dir.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files): self.files = files
    def write(self, text):
        for f in self.files: f.write(text); f.flush()
    def flush(self):
        for f in self.files: f.flush()


def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────

DEFAULT_NAME = "routing_topk30"

parser = argparse.ArgumentParser()
parser.add_argument("--name", default=DEFAULT_NAME,
                    help="--name used when running build_diffusion_targets.py.")
parser.add_argument("--name-no-topk", default=None,
                    help="Optional: name of a companion artifact built WITHOUT --target-topk, "
                         "for the top-k truncation fidelity check in section 3. Skipped if omitted.")
parser.add_argument("--nace-embedding-file", default="nace_embeddings_openai.npz",
                    help="Must match the file build_diffusion_targets.py was run with.")
parser.add_argument("--free-vec-epochs", type=int, default=300)
parser.add_argument("--free-vec-lr", type=float, default=0.05)
parser.add_argument("--free-vec-dim", type=int, default=384)
parser.add_argument("--free-vec-tau", type=float, default=0.05)
parser.add_argument("--rank-k", type=int, default=384)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out-tag", default="",
                    help="Appended to every output filename (e.g. '_768d'). Use this when "
                         "re-running against the SAME --name at a different --free-vec-dim/"
                         "--rank-k (a capacity-ceiling comparison) so the new run doesn't "
                         "overwrite the baseline diagnostics you're comparing against.")
args, _ = parser.parse_known_args()

out_stem = f"{args.name}{args.out_tag}"
log_file = open(results_dir / f"{out_stem}_diagnostics.txt", "w")
sys.stdout = Tee(sys.__stdout__, log_file)

metrics_summary: dict = {}


# ── Load core artifacts ──────────────────────────────────────────────────────

diff_path = PROCESSED_DATA_DIR / f"diffusion_matrix_{args.name}.npz"
if not diff_path.exists():
    raise FileNotFoundError(f"{diff_path} not found — run build_diffusion_targets.py --name {args.name} first.")
data = np.load(diff_path, allow_pickle=True)
S = data["S"].astype(np.float64)
codes = data["codes"]
versions = data["versions"]
target_node_ids = data["target_node_ids"]
n = S.shape[0]

edges_path = PROCESSED_DATA_DIR / f"graph_edges_{args.name}.parquet"
nodes_path = PROCESSED_DATA_DIR / f"graph_nodes_{args.name}.parquet"
if not edges_path.exists() or not nodes_path.exists():
    raise FileNotFoundError(f"{edges_path} / {nodes_path} not found — run build_diffusion_targets.py --name {args.name} first.")
edges_df = pd.read_parquet(edges_path)
nodes_df = pd.read_parquet(nodes_path)

if "split" not in edges_df.columns:
    raise ValueError(f"{edges_path.name} has no 'split' column — rebuild with the updated "
                      f"build_diffusion_targets.py.")

node_id_to_pos = {int(nid): i for i, nid in enumerate(target_node_ids)}

_section("Provenance")
print(f"  name             : {args.name}")
print(f"  combine_method   : {data['combine_method'].item() if 'combine_method' in data.files else '?'}")
if "diffusion_t" in data.files:
    print(f"  diffusion_t      : {data['diffusion_t'].item()}")
    print(f"  route_diffusion_t: {data['route_diffusion_t'].item()}")
    print(f"  repulsion        : {data['repulsion'].item()}")
    topk = int(data["target_topk"].item())
    print(f"  target_topk      : {'none' if topk < 0 else topk}")
    print(f"  split_seed       : {int(data['split_seed'].item())}")
else:
    print("  (older artifact, built before provenance fields were added — re-run build_diffusion_targets.py)")


# ── Section 1: source integrity ──────────────────────────────────────────────

def report_io_conservation() -> dict:
    """Fan-out context for the IO conservation fix (docs/conceptual_audit.md's
    Input-output relatedness section): reports how many divisions each raw IO code covers, from
    code_reference.csv (the audit trail build_input_output_relatedness.py
    already writes) — this m/n is exactly what used to multiply an
    aggregate cell's value before the equal-share allocation fix. The
    conservation ratio itself (expanded total / raw total, ~1.0 after the
    fix) is printed at BUILD time by that script's own _check_conservation
    calls; this section reports the fan-out distribution that made the old
    bug's magnitude possible, plus a sanity check on io_relatedness.parquet."""
    result: dict = {}
    ref_path = RAW_DATA_DIR / "io" / "code_reference.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        n_div = ref["divisions"].fillna("").apply(lambda s: len([d for d in s.split(",") if d]))
        print(f"  code_reference.csv: {len(ref)} codes, divisions-per-code "
              f"mean={n_div.mean():.2f} median={n_div.median():.0f} max={int(n_div.max())}")
        print(f"  (old bug: an aggregate cell's value was copied to up to "
              f"{int(n_div.max()) ** 2:,} division pairs instead of allocated — see "
              f"build_input_output_relatedness.py's _check_conservation printouts at build time)")
        result["io_max_divisions_per_code"] = int(n_div.max())
        result["io_mean_divisions_per_code"] = round(float(n_div.mean()), 2)
    else:
        print("  code_reference.csv not found — run build_input_output_relatedness.py first.")

    for version, fname in [("SNI07", "io_relatedness_SNI07.parquet"), ("SNI02", "io_relatedness_SNI02.parquet")]:
        io_path = PROCESSED_DATA_DIR / fname
        if io_path.exists():
            io_df = pd.read_parquet(io_path)
            sim = io_df["sim"].dropna()
            finite = bool(np.isfinite(sim).all())
            print(f"  {fname}: {len(io_df)} pairs, sim range "
                  f"[{sim.min():.4f}, {sim.max():.4f}], all finite={finite}")
            result[f"io_{version}_sim_all_finite"] = finite
        else:
            print(f"  {fname} not found — run build_input_output_relatedness.py first.")
    return result


def report_nace_sign_distribution(nace_embedding_file: str) -> dict:
    """% negative pairs at every NACE level, BEFORE vs AFTER the mean-
    centering load_nace_embeddings applies — makes the "centering
    manufactures negatives" claim (docs/conceptual_audit.md's NACE negative
    edges finding) numerically
    visible on every run. Re-implements the same centering as
    _center_and_renormalize in build_diffusion_targets.py (can't import that
    script directly — see module docstring)."""
    result: dict = {}
    path = PROCESSED_DATA_DIR / nace_embedding_file
    if not path.exists():
        print(f"  {nace_embedding_file} not found — skipping.")
        return result
    raw = np.load(path, allow_pickle=False)
    for level in ["class", "group", "division", "section"]:
        key = f"{level}_vecs"
        if key not in raw or len(raw[key]) == 0:
            continue
        vecs = raw[key].astype(np.float64)
        iu, ju = np.triu_indices(len(vecs), k=1)

        sims_raw = vecs @ vecs.T
        pct_neg_raw = 100 * float((sims_raw[iu, ju] < 0).mean())

        mean = vecs.mean(axis=0)
        centered = vecs - mean
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        centered = centered / norms
        sims_c = centered @ centered.T
        pct_neg_c = 100 * float((sims_c[iu, ju] < 0).mean())

        print(f"  {level:<10}: {len(vecs)} nodes, {len(iu):,} pairs — "
              f"% negative before centering={pct_neg_raw:.1f}%  after={pct_neg_c:.1f}%")
        result[f"nace_{level}_pct_neg_before"] = round(pct_neg_raw, 2)
        result[f"nace_{level}_pct_neg_after"] = round(pct_neg_c, 2)
    return result


_section("1. Source integrity")
metrics_summary.update({f"io_{k}": v for k, v in report_io_conservation().items()})
metrics_summary.update(report_nace_sign_distribution(args.nace_embedding_file))


# ── Section 2: source agreement ──────────────────────────────────────────────

def family_pairwise_agreement(edges_df: pd.DataFrame) -> pd.DataFrame:
    """For every pair of evidence-family edge types that share at least 10
    common (src, dst) pairs, report their raw-weight Spearman rank
    correlation and sign-agreement rate — do independent sources agree
    where they overlap, before any diffusion/fusion."""
    by_type: dict[str, dict[tuple[int, int], float]] = {}
    for etype, sub in edges_df.groupby("type"):
        by_type[etype] = {
            (min(int(r.src), int(r.dst)), max(int(r.src), int(r.dst))): float(r.weight)
            for r in sub.itertuples()
        }
    types = sorted(by_type)
    rows = []
    for i, ta in enumerate(types):
        for tb in types[i + 1:]:
            common = set(by_type[ta]) & set(by_type[tb])
            if len(common) < 10:
                continue
            a = [by_type[ta][k] for k in common]
            b = [by_type[tb][k] for k in common]
            rho, _ = spearmanr(a, b)
            sign_agree = float(np.mean([np.sign(x) == np.sign(y) for x, y in zip(a, b)]))
            rows.append({"family_a": ta, "family_b": tb, "n_shared": len(common),
                         "spearman_rho": round(float(rho), 4), "sign_agreement": round(sign_agree, 4)})
    out = pd.DataFrame(rows)
    if out.empty:
        print("  No family pairs share >=10 common (src, dst) pairs.")
    else:
        for r in out.itertuples():
            print(f"  {r.family_a:<16} vs {r.family_b:<16}: n_shared={r.n_shared:,}  "
                  f"rho={r.spearman_rho:.4f}  sign_agreement={r.sign_agreement:.4f}")
    return out


_section("2. Source agreement/conflict")
agreement_df = family_pairwise_agreement(edges_df)


# ── Section 3: fused-target feasibility ──────────────────────────────────────

def asymmetry_and_detailed_balance(S: np.ndarray) -> dict:
    """Frobenius asymmetry ||S - S.T|| / ||S|| and a detailed-balance
    residual treating (row-normalised) S as a Markov kernel: a reversible
    chain needs a stationary pi with pi_i P_ij == pi_j P_ji for all i, j.

    Row-normalizing any symmetric matrix - including the cosine-softmax
    neighbor distribution Q a trained encoder produces - always yields a
    nonsymmetric (P_ij != P_ji) but exactly reversible kernel; a nonzero
    Frobenius asymmetry is therefore expected and is not by itself evidence
    against representability by a cosine-similarity model. The residual is
    the diagnostic that matters: it is ~0 for a purely row-normalized
    symmetric matrix, and only rises when something breaks reversibility
    (row-wise top-k truncation here). Since Q is always exactly reversible
    regardless of embedding dimension, a nonzero residual is a real,
    dimension-independent mismatch (docs/conceptual_audit.md's central
    mathematical mismatch section)."""
    fro_S = np.linalg.norm(S)
    asymmetry = float(np.linalg.norm(S - S.T) / fro_S) if fro_S > 0 else float("nan")

    P, _ = row_normalize_target(S)
    pi = np.ones(S.shape[0]) / S.shape[0]
    for _ in range(2000):
        pi_next = pi @ P
        total = pi_next.sum()
        pi_next = pi_next / total if total > 0 else pi_next
        if np.linalg.norm(pi_next - pi) < 1e-12:
            pi = pi_next
            break
        pi = pi_next
    residual = float(np.max(np.abs(pi[:, None] * P - pi[None, :] * P.T)))

    print(f"  Frobenius asymmetry ||S - S.T|| / ||S||             : {asymmetry:.4f}  (0 = symmetric)")
    print(f"  Detailed-balance residual (max |pi_i P_ij - pi_j P_ji|): {residual:.2e}")
    return {"asymmetry": round(asymmetry, 4), "detailed_balance_residual": residual}


def spectral_reconstruction_ceiling(S: np.ndarray, rank_k: int) -> tuple[dict, np.ndarray]:
    """Symmetrize S, eigendecompose, report the best rank-k reconstruction
    error — 'is this even representable by a rank-k cosine Gram matrix'
    (docs/conceptual_audit.md's central mathematical mismatch section). A cosine-similarity model of embedding
    dimension d can only ever represent a rank-<=d Gram matrix; if most of
    S's Frobenius mass lives beyond the first d eigenvalues, no encoder of
    that dimension can fit it, regardless of training procedure."""
    rank_k = min(rank_k, S.shape[0])
    S_sym = (S + S.T) / 2.0
    eigvals = np.linalg.eigvalsh(S_sym)[::-1]
    total = float(np.sum(eigvals ** 2))
    kept = float(np.sum(eigvals[:rank_k] ** 2))
    frac_kept = kept / total if total > 0 else float("nan")
    verdict = "ceiling not binding" if frac_kept > 0.99 else "ceiling IS binding — representability constraint"
    print(f"  Eigenvalues: n={len(eigvals)}  top-5={np.round(eigvals[:5], 4)}  min={eigvals.min():.4f}")
    print(f"  Best rank-{rank_k} reconstruction: {frac_kept:.4f} of Frobenius mass captured ({verdict})")
    return {"rank_reconstruction_frac": round(frac_kept, 4)}, eigvals


def topk_truncation_fidelity(S_topk: np.ndarray, S_full: np.ndarray) -> dict:
    """How much of the untruncated target's own top-k mass survives
    truncation, row by row. This is NOT true per-family mass attribution —
    family identity is already destroyed by the time a single fused S
    exists; recovering that would
    require build_diffusion_targets.py to persist intermediate per-route
    matrices, out of scope here. This instead checks a narrower question:
    for each row, does S_topk keep S_full's own genuinely largest entries,
    or did truncation interact with upstream renormalisation/repulsion in
    a way that keeps a different set."""
    k = int((S_topk[0] > 0).sum()) or 1
    n = S_topk.shape[0]
    agreement = np.zeros(n)
    for i in range(n):
        full_top_idx = set(np.argpartition(-S_full[i], kth=min(k, n - 2) - 1)[:k])
        topk_idx = set(np.nonzero(S_topk[i])[0])
        agreement[i] = len(full_top_idx & topk_idx) / k
    print(f"  Row-wise agreement between S_topk's kept entries and S_full's own top-{k}: "
          f"mean={agreement.mean():.4f}  min={agreement.min():.4f}")
    return {"topk_agreement_with_full_ranking": round(float(agreement.mean()), 4)}


_section("3. Fused-target feasibility")
sparsity = sparsity_report(S)
print_sparsity_report(args.name, sparsity)
metrics_summary.update(asymmetry_and_detailed_balance(S))
spectral_metrics, eigvals = spectral_reconstruction_ceiling(S, args.rank_k)
metrics_summary.update(spectral_metrics)

if args.name_no_topk:
    no_topk_path = PROCESSED_DATA_DIR / f"diffusion_matrix_{args.name_no_topk}.npz"
    if no_topk_path.exists():
        S_full = np.load(no_topk_path, allow_pickle=True)["S"].astype(np.float64)
        if S_full.shape == S.shape:
            metrics_summary.update(topk_truncation_fidelity(S, S_full))
        else:
            print(f"  --name-no-topk artifact shape {S_full.shape} != {args.name} shape {S.shape} — skipping.")
    else:
        print(f"  {no_topk_path} not found — skipping top-k truncation fidelity.")
else:
    print("  --name-no-topk not given — skipping top-k truncation fidelity "
          "(needs a companion artifact built without --target-topk).")


# ── Section 4: model-independent ceilings ────────────────────────────────────

def fit_free_vectors(P: np.ndarray, active_mask: np.ndarray, dim: int,
                      epochs: int, lr: float, tau: float, seed: int) -> np.ndarray:
    """Free (n, dim) unit vectors optimized directly against P via the
    IDENTICAL forward-KL objective train_sentence_embedding.py uses (same
    loop, same temperature default) but with no text encoder at all — this
    measures whether S_target is representable by SOME rank-`dim` cosine
    geometry, independent of any sentence encoder's capacity or text-
    information limits (docs/conceptual_audit.md's central mathematical
    mismatch and recommended-architecture sections)."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = P.shape[0]
    P_t = torch.tensor(P, dtype=torch.float32, device=device)
    active_t = torch.tensor(active_mask, dtype=torch.bool, device=device)
    raw = torch.nn.Parameter(0.01 * torch.randn(n, dim, device=device))
    optimizer = torch.optim.AdamW([raw], lr=lr)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        E = F.normalize(raw, p=2, dim=-1)
        cos_sim = (E @ E.T).clone()
        cos_sim.fill_diagonal_(float("-inf"))
        Q = torch.softmax(cos_sim / tau, dim=-1)
        log_Q = torch.log(Q.clamp(min=1e-12))
        kl_rows = F.kl_div(log_Q, P_t, reduction="none").sum(dim=-1)
        loss = kl_rows[active_t].mean()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % max(epochs // 10, 1) == 0 or epoch == epochs:
            print(f"  free-vector fit  epoch {epoch:4d}/{epochs}  kl={loss.item():.5f}")

    return F.normalize(raw, p=2, dim=-1).detach().cpu().numpy()


def evaluate_ceiling_metrics(E: np.ndarray, edges_df: pd.DataFrame,
                              node_id_to_pos: dict[int, int]) -> pd.DataFrame:
    """Labor and nace_4d Spearman rho (raw source weight vs free-vector
    cosine similarity), computed separately on train vs test edges — a
    train/test gap here, even for
    UNCONSTRAINED free vectors, is direct evidence the target itself is
    source-destructive, independent of any encoder limitation. io/nace_3d/
    2d/sec are defined over hub nodes outside S_target's own node set, so
    they can't be evaluated this way without extending the fit to the full
    N-node graph — out of scope here."""
    rows = []
    for etype in ["labor", "nace_4d"]:
        sub = edges_df[edges_df["type"] == etype]
        for split in ["train", "test"]:
            split_sub = sub[sub["split"] == split]
            gt, sim = [], []
            for r in split_sub.itertuples():
                pi, pj = node_id_to_pos.get(int(r.src)), node_id_to_pos.get(int(r.dst))
                if pi is not None and pj is not None:
                    gt.append(float(r.weight))
                    sim.append(float(E[pi] @ E[pj]))
            if len(gt) >= 10:
                rho, _ = spearmanr(sim, gt)
                rows.append({"family": etype, "split": split, "n_pairs": len(gt), "rho": round(float(rho), 4)})
    out = pd.DataFrame(rows)
    for r in out.itertuples():
        print(f"  {r.family:<10} [{r.split:<5}]  n={r.n_pairs:,}  rho={r.rho:.4f}")
    if not out.empty:
        pivot = out.pivot(index="family", columns="split", values="rho")
        if "train" in pivot.columns and "test" in pivot.columns:
            for fam in pivot.index:
                gap = pivot.loc[fam, "train"] - pivot.loc[fam, "test"]
                print(f"  {fam:<10} train-test rho gap: {gap:.4f}")
    return out


_section("4. Model-independent ceilings (free-vector fit)")
P, active_mask = row_normalize_target(S)
E_free = fit_free_vectors(P, active_mask, args.free_vec_dim, args.free_vec_epochs,
                           args.free_vec_lr, args.free_vec_tau, args.seed)
ceiling_df = evaluate_ceiling_metrics(E_free, edges_df, node_id_to_pos)
for r in ceiling_df.itertuples():
    metrics_summary[f"ceiling_{r.family}_{r.split}_rho"] = r.rho


# ── Section 5: leakage-safe split report ─────────────────────────────────────

def _build_concept_groups_from_crosswalks(nodes_df: pd.DataFrame) -> dict[int, int]:
    """Independent re-derivation of the same concept groups build_diffusion_
    targets.py's _build_node_concept_groups computes at build time — reads
    the crosswalk CSVs directly (that script can't be imported, see module
    docstring), so this is a genuine self-consistency CHECK on the persisted
    split column, not just trusting it."""
    node_idx = {(row["node_type"], row["version"], str(row["code"])): int(nid)
                for nid, row in nodes_df.iterrows()}
    pairs = []
    for csv_name, old_ver, new_ver, as_int in [
        ("mapping_02_07.csv", "SNI02", "SNI07", False),
        ("mapping_92_02.csv", "SNI92", "SNI02", True),
    ]:
        path = metadata_dir / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        old_col = next((c for c in df.columns if old_ver[:5] in c.upper()), df.columns[0])
        new_col = next((c for c in df.columns if new_ver[:5] in c.upper()), df.columns[1])

        def to4(v):
            try:
                s5 = str(int(v)).zfill(5) if as_int else str(v).zfill(5)
            except (ValueError, TypeError):
                return None
            return s5[:4]

        for o, new_v in zip(df[old_col].map(to4), df[new_col].map(to4)):
            if o is None or new_v is None:
                continue
            r = node_idx.get(("4d", old_ver, o))
            c = node_idx.get(("4d", new_ver, new_v))
            if r is not None and c is not None:
                pairs.append((r, c))
    return union_find_groups(pairs)


def split_report(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> dict:
    """Per-family split-size distribution, plus a self-consistency check
    that no split key spans more than one split value across all edges
    currently in edges_df — a direct check that the grouping in
    build_diffusion_targets.py actually prevented leakage, not just an
    assumption that it did."""
    print(edges_df.groupby(["type", "split"]).size().unstack(fill_value=0).to_string())

    group = _build_concept_groups_from_crosswalks(nodes_df)
    key_to_splits: dict[tuple, set] = {}
    for r in edges_df.itertuples():
        key = split_key_for(int(r.src), int(r.dst), group)
        key_to_splits.setdefault(key, set()).add(r.split)
    violations = {k: v for k, v in key_to_splits.items() if len(v) > 1}
    print(f"\n  Split keys spanning >1 partition: {len(violations)} / {len(key_to_splits)}")
    if violations:
        print(f"  Example violating keys: {list(violations.items())[:5]}")
    return {"n_split_keys": len(key_to_splits), "n_leakage_violations": len(violations)}


_section("5. Leakage-safe split report")
metrics_summary.update(split_report(edges_df, nodes_df))


# ── Save ──────────────────────────────────────────────────────────────────────

with open(results_dir / f"{out_stem}_diagnostics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2, default=str)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(eigvals, marker=".", linewidth=0.5)
ax.set_xlabel("eigenvalue rank")
ax.set_ylabel("eigenvalue")
ax.set_title(f"Eigenvalue spectrum of symmetrized S ({out_stem})")
ax.axvline(min(args.rank_k, len(eigvals)), color="red", linestyle="--", label=f"rank-{args.rank_k} cutoff")
ax.legend()
fig.tight_layout()
fig.savefig(results_dir / f"{out_stem}_eigenvalue_spectrum.png", dpi=150)
plt.close(fig)

if not ceiling_df.empty:
    fig, ax = plt.subplots(figsize=(6, 5))
    pivot = ceiling_df.pivot(index="family", columns="split", values="rho")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Spearman rho (free-vector ceiling)")
    ax.set_title(f"Train vs test rho, free-vector ceiling ({out_stem})")
    fig.tight_layout()
    fig.savefig(results_dir / f"{out_stem}_ceiling_train_test_rho.png", dpi=150)
    plt.close(fig)

print(f"\nAll outputs written to {results_dir}/")
print("Done.")
