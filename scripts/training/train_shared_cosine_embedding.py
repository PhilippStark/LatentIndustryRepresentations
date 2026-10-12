"""
train_shared_cosine_embedding.py
================================
Trains the shared industry embedding Z (one vector per graph node) so that
cosine similarities S_ij = cos(z_i, z_j) reproduce every multi-scale
diffusion target of every relational view at once
(scripts/graph/build_layer_diffusion_targets.py).

Model
-----
- One shared latent space: each target is scored against the same cosine
  similarity matrix, with only a learnable scalar temperature per target
  (operator_observation_models.CosineObservationModel). There are no
  per-view projections, so conflicts between views must be resolved in Z.
- Hierarchy hubs (3-digit, 2-digit and section codes) are aggregates of
  their 4-digit descendants (operator_embedding.build_hub_aggregation_matrix).
- Losses: KL for diffusion targets, cross-entropy for hierarchy/transition
  targets, an ordinal loss for hierarchy_t1. A view's weight --lambda-<view>
  is divided among its targets in proportion to the inverse of each target's
  null loss (operator_data.load_operator_data_expanded).
- Z is initialised from a sentence-transformer encoding of each node's SCB
  description (--text-prior-model), reduced to --embedding-dim by PCA and
  re-normalised; no loss term ties Z to this initialisation afterwards.
- --saturation-frac stops gradients from a target once its loss falls below
  that fraction of its null loss. Each target's weight is also rescaled by
  its epoch-1 loss divided by a running average of its current loss, so solved
  and unsolved targets keep comparable weight (capped by --reweight-max-ratio
  and --reweight-stall-ratio).

The defaults are the main run's settings.

Outputs (<LATENT_DATA_ROOT>/embedding/{run_name}/)
-------------------------------------------------
  embeddings_{sni92,sni02,sni07}.npz, embeddings_sni07_{section,2d,3d}.npz
  training_provenance.json, hub_coherence.csv,
  loss_metric_correlation.csv (with --eval-every > 0)

Usage (main run, shared_cosine_v16)
-----------------------------------
  python scripts/training/train_shared_cosine_embedding.py \
      --name layers_v1 --run-name shared_cosine_v16
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
from latent_industry.training.operator_data import (
    DEFAULT_LAMBDAS, DEFAULT_LOSS_TYPE_OVERRIDES, DEFAULT_LOSS_TYPES, load_operator_data_expanded,
)
from latent_industry.training.operator_diagnostics import per_family_reference_metric
from latent_industry.training.operator_embedding import (
    build_hub_aggregation_matrix, export_embeddings, export_hub_coherence,
)
from latent_industry.training.operator_observation_models import CosineObservationModel
from latent_industry.training.operator_trainer import train_joint

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True,
                     help="Same --name used for build_layer_diffusion_targets.py.")
parser.add_argument("--run-name", default=None, help="Output dir under EMBEDDING_DIR. Defaults to --name.")
parser.add_argument("--embedding-dim", type=int, default=64,
                     help="d, the shared latent dimension; at most --text-prior-model's output "
                          "dimension (the text prior is reduced to d by PCA).")
parser.add_argument("--text-prior-model", default="sentence-transformers/all-MiniLM-L6-v2",
                     help="Sentence-transformer whose encoding of each node's description "
                          "initialises Z.")
parser.add_argument("--saturation-frac", type=float, default=0.2,
                     help="Loss floor per target as a fraction of its null (trivial-baseline) "
                          "loss: below it the target contributes no further gradient. "
                          "0.0 = off.")
parser.add_argument("--reweight-decay", type=float, default=0.9,
                     help="EMA decay of the running loss used for reweighting.")
parser.add_argument("--reweight-max-ratio", type=float, default=20.0,
                     help="Upper bound on a target's reweighting ratio (epoch-1 loss / running loss).")
parser.add_argument("--reweight-stall-ratio", type=float, default=2.0,
                     help="Above this reweighting ratio, a target's loss must have improved "
                          "by --reweight-stall-min-improvement, otherwise the ratio is capped here "
                          "and the target is flagged as stalled.")
parser.add_argument("--reweight-stall-min-improvement", type=float, default=0.1,
                     help="Required relative loss improvement to avoid the stall cap (0.1 = 10%%).")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--log-every", type=int, default=100)
parser.add_argument("--eval-every", type=int, default=25,
                     help="Every N epochs (0 = off), record one reference metric per view "
                          "(operator_diagnostics.per_family_reference_metric: Spearman rho "
                          "against the source relatedness, or crosswalk MRR for transition) "
                          "next to that view's training loss, in "
                          "{run_dir}/loss_metric_correlation.csv. Loss and metric should move "
                          "in opposite directions (negative correlation).")
parser.add_argument("--seed", type=int, default=42)
for _lname, _default in DEFAULT_LAMBDAS.items():
    parser.add_argument(f"--lambda-{_lname}", type=float, default=_default, dest=f"lambda_{_lname}",
                         help=f"This view's total weight in the joint loss (default {_default}), "
                              f"divided among its targets in proportion to the inverse of each "
                              f"target's null loss (see load_operator_data_expanded).")
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

from sentence_transformers import SentenceTransformer
print(f"\nLoading text prior {args.text_prior_model} …")
text_prior_model = SentenceTransformer(args.text_prior_model, device=device)
get_dim = getattr(text_prior_model, "get_embedding_dimension",
                   text_prior_model.get_sentence_embedding_dimension)
text_prior_dim = get_dim()
if args.embedding_dim > text_prior_dim:
    raise ValueError(
        f"--embedding-dim={args.embedding_dim} exceeds {args.text_prior_model}'s output "
        f"dimension ({text_prior_dim}); the text prior can only be reduced by PCA."
    )

loss_types_by_family = dict(DEFAULT_LOSS_TYPES)
loss_type_overrides = dict(DEFAULT_LOSS_TYPE_OVERRIDES)
family_weights = {lname: getattr(args, f"lambda_{lname}") for lname in DEFAULT_LAMBDAS}
print(f"loss_types_by_family={loss_types_by_family}\nfamily_weights={family_weights}\n"
      f"embedding_dim={args.embedding_dim}  lr={args.lr}  epochs={args.epochs}")

print("\nLoading per-(view, scale) diffusion targets …")
operator_data = load_operator_data_expanded(
    args.name, loss_types_by_family=loss_types_by_family, family_weights=family_weights,
    saturation_frac=args.saturation_frac, device=device, loss_type_overrides=loss_type_overrides,
)
if args.saturation_frac > 0:
    print(f"  saturation_frac={args.saturation_frac}  margins:")
    for target_id in sorted(operator_data.margins):
        print(f"    {target_id:<28}  margin={operator_data.margins[target_id]:.5f}")
print(f"  {len(operator_data.targets)} targets: {sorted(operator_data.targets)}")
print("  per-target weight (family weight, divided in proportion to 1 / null loss):")
for target_id in sorted(operator_data.weights):
    print(f"    {target_id:<28}  weight={operator_data.weights[target_id]:.5f}")

ni = NodeIndex()  # rebuilt deterministically from the same metadata files stage 1 used
if ni.N != next(iter(operator_data.targets.values())).shape[0]:
    raise ValueError(
        f"NodeIndex().N={ni.N} does not match the cached diffusion targets' node count "
        f"({next(iter(operator_data.targets.values())).shape[0]}) — was "
        f"layer_diffusion_expanded_{args.name}.npz built from a different metadata snapshot?"
    )

# Hub (3d/2d/section) rows are a function of their 4d descendants
# (operator_embedding.build_hub_aggregation_matrix), not free parameters.
hub_aggregation = build_hub_aggregation_matrix(ni).to(device)

print(f"\nEncoding {ni.N} node descriptions with {args.text_prior_model} for Z's initial value …")
texts = ni.nodes["description"].tolist()
with torch.no_grad():
    init_vecs = text_prior_model.encode(
        texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True,
    )
if args.embedding_dim != text_prior_dim:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=args.embedding_dim, random_state=args.seed)
    init_vecs = pca.fit_transform(init_vecs)
    init_vecs = init_vecs / np.linalg.norm(init_vecs, axis=1, keepdims=True)
    print(f"  PCA: {text_prior_dim}d -> {args.embedding_dim}d "
          f"({pca.explained_variance_ratio_.sum():.1%} of variance retained), "
          f"re-normalised to unit norm")
Z = torch.nn.Parameter(torch.tensor(init_vecs, dtype=torch.float32, device=device))
del text_prior_model
observation_models = {target_id: CosineObservationModel().to(device) for target_id in operator_data.targets}

loss_metric_rows: list[dict] = []
on_epoch = None
if args.eval_every > 0:
    print(f"\n--eval-every {args.eval_every}: tracking per-family loss <-> reference-metric "
          f"correlation -> {run_dir / 'loss_metric_correlation.csv'}")

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

train_joint(Z, observation_models, operator_data,
            lr=args.lr, epochs=args.epochs, log_every=args.log_every,
            dynamic_reweight=True, reweight_decay=args.reweight_decay,
            reweight_max_ratio=args.reweight_max_ratio,
            reweight_stall_ratio=args.reweight_stall_ratio,
            reweight_stall_min_improvement=args.reweight_stall_min_improvement,
            hub_aggregation=hub_aggregation, on_epoch=on_epoch)

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
            # improving.
            flag = "  <- target may not point at the object" if corr >= -0.1 else ""
            print(f"  {family:<14}{corr:>20.4f}{flag}")
        else:
            print(f"  {family:<14}{'n/a (too few points)':>20}")

print("\nExporting embeddings …")
export_embeddings(Z, ni, run_dir, hub_aggregation=hub_aggregation)

export_hub_coherence(Z, ni, hub_aggregation, run_dir / "hub_coherence.csv")

training_provenance = {
    "run_name": run_name,
    "graph_name": args.name,
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
    "text_prior_model": args.text_prior_model,
    "text_prior_pca": args.embedding_dim != text_prior_dim,
    "saturation_frac": args.saturation_frac,
    "per_target_margins": operator_data.margins,
    "dynamic_reweight": True,
    "reweight_decay": args.reweight_decay,
    "reweight_max_ratio": args.reweight_max_ratio,
    "reweight_stall_ratio": args.reweight_stall_ratio,
    "reweight_stall_min_improvement": args.reweight_stall_min_improvement,
}
(run_dir / "training_provenance.json").write_text(json.dumps(training_provenance, indent=2), encoding="utf-8")
print(f"Training provenance saved -> {run_dir / 'training_provenance.json'}")

print("\nDone.")
