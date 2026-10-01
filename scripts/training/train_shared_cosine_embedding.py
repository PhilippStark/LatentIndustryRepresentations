"""train_shared_cosine_embedding.py — Phase 9 of the operator-preserving
regime (docs/operator_preserving_implementation_plan.md). The empty cell in
the 2x2 the earlier two runs (multilayer_v1 / operator_v1) fall into:

                    shared cosine only     per-layer decoder
one fused target    multilayer_v1          —
many per-layer      (this script)          operator_v1

`multilayer_v1` (train_sentence_embedding.py) fuses every layer into one
coupled supra-diffusion target BEFORE training — any conflict between
layers is already resolved (via the coupling parameter) before Z ever sees
a gradient. `operator_v1` (train_operator_embedding.py) keeps per-layer
targets but fits each with its own bilinear M_l — Phase 7 showed M_l can
absorb a layer's own structure without that fit ever reaching Z's own
plain-cosine geometry (the M_l-discard gap; see project history).

This script is the missing cell: many per-layer, per-SCALE targets (every
(layer, t) power kept separate, not weight-averaged — see
scripts/graph/build_layer_diffusion_targets.py --expanded), every one
scored against the SAME S_ij = cos(z_i, z_j) with at most a learnable
scalar temperature per target (operator_observation_models.CosineObservationModel)
— no M_l, no per-layer projection, nothing that can rotate the space. Any
conflict between layers now has to resolve inside Z itself, during
training, which is what every previous run's negative results were
pointing at.

Per docs/operator_preserving_implementation_plan.md Phase 9's own risk
note: diagnose_scale_redundancy.py should be run first against `--name`
to check whether a layer's own per-scale targets are highly mutually
redundant (if so, splitting that layer's target adds little). The
EXPANDED_DEFAULT_SCALES=(1,4) default already reflects that diagnostic's
2026-07-27 finding against multilayer_v1 (t=2/4/8 highly redundant with
each other for every layer; t=1 is not) — override --scales-<layer> only
if re-running the diagnostic against a different graph changes that
picture.

Usage
-----
  python scripts/graph/build_layer_diffusion_targets.py --name multilayer_v1 --expanded
  python scripts/training/train_shared_cosine_embedding.py --name multilayer_v1 --run-name shared_cosine_v1
  python scripts/training/train_shared_cosine_embedding.py --name multilayer_v1 --run-name shared_cosine_v3_textinit \
      --text-prior-init --embedding-dim 384

`--text-prior-init` (added 2026-07-27, confound check): shared_cosine_v1/v2
are a 64-d embedding trained from small random noise with no encoder at
all, compared against multilayer_v1's 384-d fine-tuned sentence-transformer.
Part of shared_cosine's semantic-metric drop vs. multilayer_v1 (e.g.
H_nace_rho_4d 0.47->0.31/0.26) could just be the absence of any text prior,
not a property of the shared-cosine training objective itself — this flag
isolates that: initialise Z from a frozen sentence-transformer encoding of
each node's own NACE description (`ni.nodes["description"]`, already
computed by NodeIndex — the same text multilayer_v1 fine-tunes on) instead
of `0.01 * randn`, at the matching dimension (384 for the default
all-MiniLM-L6-v2), then run the exact same joint shared-cosine training on
top. If semantic metrics hold up much better with this warm start, the
drop was mostly a missing-prior confound; if the joint objective still
degrades them from their encoder-only starting point, that's evidence
against the shared-cosine training dynamic itself, not just its lack of a
prior.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from latent_industry.common.paths import EMBEDDING_DIR
from latent_industry.graph.nodes import NodeIndex
from latent_industry.training.operator_data import DEFAULT_LAMBDAS, DEFAULT_LOSS_TYPES, load_operator_data_expanded
from latent_industry.training.operator_diagnostics import per_family_reference_metric, save_training_state
from latent_industry.training.operator_embedding import (
    build_hub_aggregation_matrix, export_embeddings, export_hub_coherence,
)
from latent_industry.training.operator_observation_models import make_observation_model
from latent_industry.training.operator_trainer import train_joint

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True,
                     help="Same --name used for build_layer_diffusion_targets.py --expanded.")
parser.add_argument("--run-name", default=None, help="Output dir under EMBEDDING_DIR. Defaults to --name.")
parser.add_argument("--embedding-dim", type=int, default=None,
                     help="d, the shared latent dimension. Defaults to 64, or (under "
                          "--text-prior-init without --text-prior-pca) to --text-prior-model's "
                          "own output dimension. Under --text-prior-init --text-prior-pca, must "
                          "be <= --text-prior-model's output dimension (defaults to 64).")
parser.add_argument("--text-prior-init", action="store_true",
                     help="Initialise Z from a frozen sentence-transformer encoding of each "
                          "node's own NACE description instead of small random noise — see "
                          "the module docstring. Requires --embedding-dim to match "
                          "--text-prior-model's output dim (auto-set if --embedding-dim is "
                          "left at its default).")
parser.add_argument("--text-prior-model", default="sentence-transformers/all-MiniLM-L6-v2",
                     help="Only used with --text-prior-init.")
parser.add_argument("--text-prior-pca", action="store_true",
                     help="With --text-prior-init: if --embedding-dim is smaller than "
                          "--text-prior-model's own output dimension, reduce the encoded "
                          "descriptions to --embedding-dim via PCA (fit on this graph's own N "
                          "node embeddings) instead of requiring an exact dimension match. Lets "
                          "Z keep a smaller --embedding-dim (e.g. 64) while still starting from "
                          "a (lossily compressed) semantic prior instead of random noise.")
parser.add_argument("--saturation-frac", type=float, default=0.0,
                     help="Item #1 (2026-07-27): a target's loss floor as a fraction of its own "
                          "null (trivial-baseline) loss — once its raw loss drops below this "
                          "fraction, it contributes exactly zero further gradient instead of "
                          "continuing to compete with unsatisfied targets. 0.0 (default) is a "
                          "no-op. Try e.g. 0.05-0.1.")
parser.add_argument("--dynamic-reweight", action="store_true",
                     help="Item #2 (2026-07-27): continuously re-derive each target's null-loss-"
                          "style normalisation from a running EMA of its own loss, instead of "
                          "freezing it at epoch 1 — keeps a target's contribution pinned near its "
                          "fair share throughout training instead of fading as it's solved. "
                          "Meant to be combined with --saturation-frac (which is what actually "
                          "stops an already-solved target, at exactly zero).")
parser.add_argument("--reweight-decay", type=float, default=0.9, help="EMA decay for --dynamic-reweight.")
parser.add_argument("--reweight-max-ratio", type=float, default=20.0,
                     help="Clamp on --dynamic-reweight's rescaling ratio (it -> infinity as a "
                          "target's running loss -> 0).")
parser.add_argument("--reweight-stall-ratio", type=float, default=5.0,
                     help="Item #4 (2026-07-27): once --dynamic-reweight's ratio would cross this, "
                          "the target's loss must have improved by --reweight-stall-min-improvement "
                          "since it last was below this threshold, or the ratio is capped here and "
                          "the target flagged as stalled (structurally broken) rather than hard — "
                          "guards against runaway boosts like shared_cosine_v4's "
                          "transition_SNI02_SNI07_t1 (~10-15x with no real improvement).")
parser.add_argument("--reweight-stall-min-improvement", type=float, default=0.1,
                     help="Required relative loss improvement to avoid the stall cap (0.1 = 10%%).")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--log-every", type=int, default=20)
parser.add_argument("--eval-every", type=int, default=0,
                     help="v7 §4.2: every N epochs (0 = off, default), compute one cheap "
                          "family-specific reference metric per family (operator_diagnostics."
                          "per_family_reference_metric — Spearman rho against labor/io/hierarchy "
                          "relatedness, or crosswalk MRR for transition) and record it against "
                          "that family's own training loss. Written to "
                          "{run_dir}/loss_metric_correlation.csv, with a final per-family "
                          "correlation printed: a healthy target has loss and metric moving in "
                          "OPPOSITE directions over training (negative correlation — loss falls "
                          "as the metric improves); a positive (or near-zero) correlation means "
                          "the loss is falling WITHOUT the metric improving (or while it gets "
                          "worse) — optimising a proxy that doesn't point at the object, exactly "
                          "how the io lift bug was found (io_t4's loss improving while I_io_rho "
                          "fell 0.598 -> 0.239, both moving the same direction over training).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--io-loss", default=DEFAULT_LOSS_TYPES["io"], choices=["kl", "mse"],
                     help="io family's likelihood — kl (default, categorical) or mse (Gaussian).")
for _lname, _default in DEFAULT_LAMBDAS.items():
    parser.add_argument(f"--lambda-{_lname}", type=float, default=_default, dest=f"lambda_{_lname}",
                         help=f"lambda_{_lname} — this FAMILY's (layer's) total weight in the joint "
                              f"loss, split evenly across however many scales it has as separate "
                              f"targets (see load_operator_data_expanded).")
args = parser.parse_args()

run_name = args.run_name or args.name
run_dir = EMBEDDING_DIR / run_name
run_dir.mkdir(parents=True, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\nGraph:  {args.name}\nOutput: {run_dir}\nSeed: {args.seed}")

text_prior_model = None
if args.text_prior_init:
    from sentence_transformers import SentenceTransformer
    print(f"\n--text-prior-init: loading {args.text_prior_model} …")
    text_prior_model = SentenceTransformer(args.text_prior_model, device=device)
    get_dim = getattr(text_prior_model, "get_embedding_dimension",
                       text_prior_model.get_sentence_embedding_dimension)
    text_prior_dim = get_dim()
    if args.embedding_dim is None:
        args.embedding_dim = 64 if args.text_prior_pca else text_prior_dim
    elif args.embedding_dim != text_prior_dim and not args.text_prior_pca:
        raise ValueError(
            f"--text-prior-init requires --embedding-dim to match {args.text_prior_model}'s "
            f"output dimension ({text_prior_dim}), got --embedding-dim={args.embedding_dim}. "
            f"Pass --text-prior-pca to instead reduce to --embedding-dim via PCA."
        )
    elif args.text_prior_pca and args.embedding_dim > text_prior_dim:
        raise ValueError(
            f"--text-prior-pca can only reduce dimensionality: --embedding-dim="
            f"{args.embedding_dim} exceeds {args.text_prior_model}'s output dimension "
            f"({text_prior_dim})."
        )
elif args.embedding_dim is None:
    args.embedding_dim = 64

loss_types_by_family = dict(DEFAULT_LOSS_TYPES)
loss_types_by_family["io"] = args.io_loss
family_weights = {lname: getattr(args, f"lambda_{lname}") for lname in DEFAULT_LAMBDAS}
print(f"loss_types_by_family={loss_types_by_family}\nfamily_weights={family_weights}\n"
      f"embedding_dim={args.embedding_dim}  lr={args.lr}  epochs={args.epochs}")

print("\nLoading expanded (per-layer, per-scale) diffusion targets …")
# v7 §1.4: hierarchy_t1 becomes the ordinal 4d-4d tier target (see
# graph.layer_diffusion.build_hierarchy_ordinal_tiers) instead of the
# categorical hub-inclusive one — a no-op if "hierarchy_t1" isn't among the
# loaded targets (e.g. --scales-hierarchy overridden to exclude t=1).
loss_type_overrides = {"hierarchy_t1": "ordinal"}
operator_data = load_operator_data_expanded(
    args.name, loss_types_by_family=loss_types_by_family, family_weights=family_weights,
    saturation_frac=args.saturation_frac, device=device, loss_type_overrides=loss_type_overrides,
)
if args.saturation_frac > 0:
    print(f"  saturation_frac={args.saturation_frac}  margins:")
    for target_id in sorted(operator_data.margins):
        print(f"    {target_id:<28}  margin={operator_data.margins[target_id]:.5f}")
print(f"  {len(operator_data.targets)} targets: {sorted(operator_data.targets)}")
print("  per-target weight (family_weight / n_scales_in_family / null_loss):")
for target_id in sorted(operator_data.weights):
    print(f"    {target_id:<16}  weight={operator_data.weights[target_id]:.5f}")

ni = NodeIndex()  # rebuilt deterministically from the same metadata files stage 1 used
if ni.N != next(iter(operator_data.targets.values())).shape[0]:
    raise ValueError(
        f"NodeIndex().N={ni.N} does not match the cached expanded diffusion targets' node count "
        f"({next(iter(operator_data.targets.values())).shape[0]}) — was "
        f"layer_diffusion_expanded_{args.name}.npz built from a different metadata snapshot?"
    )

# v7 §1.2: hub (3d/2d/section) rows become a function of their flat 4d-SNI07
# descendants instead of independently-trained free parameters — see
# operator_embedding.build_hub_aggregation_matrix.
hub_aggregation = build_hub_aggregation_matrix(ni).to(device)

if args.text_prior_init:
    print(f"\nEncoding {ni.N} node descriptions with {args.text_prior_model} for Z's initial value …")
    texts = ni.nodes["description"].tolist()
    with torch.no_grad():
        init_vecs = text_prior_model.encode(
            texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True,
        )
    if args.text_prior_pca and args.embedding_dim != text_prior_dim:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=args.embedding_dim, random_state=args.seed)
        init_vecs = pca.fit_transform(init_vecs)
        init_vecs = init_vecs / np.linalg.norm(init_vecs, axis=1, keepdims=True)
        print(f"  --text-prior-pca: {text_prior_dim}d -> {args.embedding_dim}d "
              f"({pca.explained_variance_ratio_.sum():.1%} of variance retained), "
              f"re-normalised to unit norm")
    Z = torch.nn.Parameter(torch.tensor(init_vecs, dtype=torch.float32, device=device))
    del text_prior_model
else:
    Z = torch.nn.Parameter(0.01 * torch.randn(ni.N, args.embedding_dim, device=device))
observation_models = {
    target_id: make_observation_model("cosine", args.embedding_dim).to(device)
    for target_id in operator_data.targets
}

loss_metric_rows: list[dict] = []
on_epoch = None
if args.eval_every > 0:
    print(f"\n--eval-every {args.eval_every}: tracking per-family loss <-> reference-metric "
          f"correlation (v7 §4.2) -> {run_dir / 'loss_metric_correlation.csv'}")

    def on_epoch(epoch: int, Z_eff: torch.Tensor, per_target_losses: dict[str, float]) -> None:
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            return
        family_losses: dict[str, list[float]] = {}
        for target_id, raw_loss in per_target_losses.items():
            family = operator_data.families.get(target_id, target_id)
            family_losses.setdefault(family, []).append(raw_loss)
        for family, losses in family_losses.items():
            metric = per_family_reference_metric(Z_eff, ni, family)
            loss_metric_rows.append({
                "epoch": epoch, "family": family,
                "mean_raw_loss": sum(losses) / len(losses), "reference_metric": metric,
            })

best_loss = train_joint(Z, observation_models, operator_data,
                         lr=args.lr, epochs=args.epochs, log_every=args.log_every,
                         dynamic_reweight=args.dynamic_reweight, reweight_decay=args.reweight_decay,
                         reweight_max_ratio=args.reweight_max_ratio,
                         reweight_stall_ratio=args.reweight_stall_ratio,
                         reweight_stall_min_improvement=args.reweight_stall_min_improvement,
                         hub_aggregation=hub_aggregation, on_epoch=on_epoch)
print(f"\nBest joint loss: {best_loss:.5f}")

if loss_metric_rows:
    loss_metric_df = pd.DataFrame(loss_metric_rows)
    loss_metric_path = run_dir / "loss_metric_correlation.csv"
    loss_metric_df.to_csv(loss_metric_path, index=False)
    print(f"\nPer-family loss <-> reference-metric trajectory -> {loss_metric_path}")
    print(f"  {'family':<14}{'corr(loss, metric)':>20}")
    for family, group in loss_metric_df.groupby("family"):
        valid = group.dropna(subset=["reference_metric"])
        if len(valid) >= 3 and valid["mean_raw_loss"].std() > 0 and valid["reference_metric"].std() > 0:
            corr = np.corrcoef(valid["mean_raw_loss"], valid["reference_metric"])[0, 1]
            # Healthy: loss down, metric up -> negative correlation. Positive
            # (or near-zero) means loss is falling without the metric
            # improving -- the io lift bug's signature (v7 §4.2).
            flag = "  <- target may not point at the object" if corr >= -0.1 else ""
            print(f"  {family:<14}{corr:>20.4f}{flag}")
        else:
            print(f"  {family:<14}{'n/a (too few points)':>20}")

print("\nExporting embeddings …")
export_embeddings(Z, ni, run_dir, hub_aggregation=hub_aggregation)

export_hub_coherence(Z, ni, hub_aggregation, run_dir / "hub_coherence.csv")

save_training_state(Z, observation_models, args.embedding_dim, run_dir / "operator_state.pt")
print(f"  operator_state.pt (Z + every per-target cosine temperature, diagnostic-only) "
      f"-> {run_dir / 'operator_state.pt'}")

training_provenance = {
    "run_name": run_name,
    "graph_name": args.name,
    "regime": "shared_cosine_expanded",
    "embedding_dim": args.embedding_dim,
    "seed": args.seed,
    "learning_rate": args.lr,
    "epochs": args.epochs,
    "loss_types_by_family": loss_types_by_family,
    "loss_type_overrides": loss_type_overrides,
    "per_target_loss_types": operator_data.loss_types,
    "family_weights": family_weights,
    "per_target_weights": operator_data.weights,
    "observation_model": "cosine",
    "hub_aggregation": True,
    "eval_every": args.eval_every,
    "text_prior_init": args.text_prior_init,
    "text_prior_model": args.text_prior_model if args.text_prior_init else None,
    "text_prior_pca": args.text_prior_pca if args.text_prior_init else None,
    "saturation_frac": args.saturation_frac,
    "per_target_margins": operator_data.margins,
    "dynamic_reweight": args.dynamic_reweight,
    "reweight_decay": args.reweight_decay if args.dynamic_reweight else None,
    "reweight_max_ratio": args.reweight_max_ratio if args.dynamic_reweight else None,
    "reweight_stall_ratio": args.reweight_stall_ratio if args.dynamic_reweight else None,
    "reweight_stall_min_improvement": args.reweight_stall_min_improvement if args.dynamic_reweight else None,
    "best_loss": best_loss,
}
(run_dir / "training_provenance.json").write_text(json.dumps(training_provenance, indent=2), encoding="utf-8")
print(f"Training provenance saved -> {run_dir / 'training_provenance.json'}")

print("\nDone.")
