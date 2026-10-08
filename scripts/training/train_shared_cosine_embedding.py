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
  similarity matrix, with at most a learnable scalar temperature per target
  (operator_observation_models.CosineObservationModel). There are no
  per-view projections, so conflicts between views must be resolved in Z.
- Hierarchy hubs (SNI 2007 sections, 2- and 3-digit codes) are aggregates of
  their 4-digit descendants (operator_embedding.build_hub_aggregation_matrix).
- Losses: KL for diffusion targets, cross-entropy for hierarchy/transition
  targets, an ordinal loss for hierarchy_t1. A view's weight --lambda-<view>
  is split across its targets (operator_data.load_operator_data_expanded).
- --text-prior-init initialises Z from a sentence-transformer encoding of
  each node's NACE description (reduced by PCA with --text-prior-pca); no
  loss term ties Z to this initialisation afterwards.
- --saturation-frac stops gradients from a target once its loss falls below
  that fraction of its null loss; --dynamic-reweight rescales each target by
  a running average of its own loss so solved and unsolved targets keep
  comparable weight.

Outputs (<LATENT_DATA_ROOT>/embedding/{run_name}/)
-------------------------------------------------
  embeddings_{sni92,sni02,sni07}.npz, embeddings_sni07_{section,2d,3d}.npz
  operator_state.pt, training_provenance.json, hub_coherence.csv,
  loss_metric_correlation.csv (with --eval-every)

Usage (main run, shared_cosine_v16)
-----------------------------------
  python scripts/training/train_shared_cosine_embedding.py \
      --name layers_v1 --run-name shared_cosine_v16 \
      --text-prior-init --text-prior-pca --embedding-dim 64 \
      --lambda-hierarchy 0.10 --lambda-transition 0.10 --lambda-labor 0.30 \
      --lambda-io 0.25 --lambda-nace_text 0.0 --lambda-ownership 0.125 \
      --dynamic-reweight --reweight-stall-ratio 2.0 --saturation-frac 0.2 \
      --eval-every 25 --epochs 500 --seed 42 --log-every 100
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
                          "node's NACE description instead of small random noise. Requires "
                          "--embedding-dim to match --text-prior-model's output dim unless "
                          "--text-prior-pca is set.")
parser.add_argument("--text-prior-model", default="sentence-transformers/all-MiniLM-L6-v2",
                     help="Only used with --text-prior-init.")
parser.add_argument("--text-prior-pca", action="store_true",
                     help="With --text-prior-init: reduce the encoded descriptions to "
                          "--embedding-dim by PCA (fit on the graph's nodes).")
parser.add_argument("--saturation-frac", type=float, default=0.0,
                     help="Loss floor per target as a fraction of its null (trivial-baseline) "
                          "loss: below it the target contributes no further gradient. "
                          "0.0 = off.")
parser.add_argument("--dynamic-reweight", action="store_true",
                     help="Normalise each target by a running EMA of its own loss (instead of "
                          "its epoch-1 loss) so its share of the gradient stays balanced during "
                          "training. Used together with --saturation-frac.")
parser.add_argument("--reweight-decay", type=float, default=0.9, help="EMA decay for --dynamic-reweight.")
parser.add_argument("--reweight-max-ratio", type=float, default=20.0,
                     help="Upper bound on --dynamic-reweight's rescaling ratio.")
parser.add_argument("--reweight-stall-ratio", type=float, default=5.0,
                     help="Above this --dynamic-reweight ratio, a target's loss must have improved "
                          "by --reweight-stall-min-improvement, otherwise the ratio is capped here "
                          "and the target is flagged as stalled.")
parser.add_argument("--reweight-stall-min-improvement", type=float, default=0.1,
                     help="Required relative loss improvement to avoid the stall cap (0.1 = 10%%).")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--log-every", type=int, default=20)
parser.add_argument("--eval-every", type=int, default=0,
                     help="Every N epochs (0 = off), record one reference metric per view "
                          "(operator_diagnostics.per_family_reference_metric: Spearman rho "
                          "against the source relatedness, or crosswalk MRR for transition) "
                          "next to that view's training loss, in "
                          "{run_dir}/loss_metric_correlation.csv. Loss and metric should move "
                          "in opposite directions (negative correlation).")
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
# hierarchy_t1 is the ordinal 4d-4d tier target (see
# graph.layer_diffusion.build_hierarchy_ordinal_tiers); no-op if
# "hierarchy_t1" is not among the loaded targets.
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

# Hub (3d/2d/section) rows are a function of their 4d-SNI07 descendants
# (operator_embedding.build_hub_aggregation_matrix), not free parameters.
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
            # improving.
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
