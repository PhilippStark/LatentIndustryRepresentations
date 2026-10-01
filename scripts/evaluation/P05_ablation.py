"""P05_ablation.py — Result 5 (docs/Embedding validation.md): ablation /
source-contribution analysis. NOT YET IMPLEMENTED.

docs/Embedding validation.md frames this result as an open question with two
competing framings for the paper, not yet a technical spec:

  Option 1 (recommended there): representation learning is the primary
  contribution. Opportunity spaces are latent, multidimensional, and must be
  learned jointly from complementary sources rather than read off any single
  relatedness measure. Under this framing, an ablation would show what is
  lost — in source reconstruction (P02) and downstream regressions (P04) —
  when a source is dropped from training, not that the joint model "beats"
  any one source alone.

  Option 2: predictive performance is the primary contribution, with the
  embedding as an intermediate feature for a fixed downstream predictor.
  Under this framing an ablation is closer to a standard leave-one-source-out
  predictive comparison.

A more skeptical framing deprioritizes leave-one-source-out training and
per-source embeddings entirely, unless the paper commits to a causal/
marginal-contribution claim. That tension is unresolved — see
docs/shared_cosine_v14_architecture.md's "labor/io vs nace/hierarchy tension".

evaluate_route_weights.py / sweep_route_weights.py measured something
adjacent (per-relation contribution to the retired typed_gate/route_projection
diffusion-target construction) but are marked PENDING REPLACEMENT: they
target build_diffusion_targets.py's retired combination methods, not the
current multilayer supra-network diffusion (build_layer_transition_matrices.py
+ build_multilayer_diffusion.py) that P01-P04 evaluate against. Whatever this
script ends up measuring should target the current pipeline, not be a rename
of those two.

TODO before implementing
-------------------------
- [ ] Decide Option 1 vs Option 2 (or both) for the paper — see above.
- [ ] If Option 1: define what "ablation" means without retraining a full
      leave-one-source-out model per source (e.g. reuses P01/P02's
      source-specific evaluation distributions rather than new training
      runs) — or confirm a small leave-one-source-out training sweep is
      in scope after all.
- [ ] If Option 2: specify the fixed downstream predictor(s) and which P04
      outcome(s) it is evaluated against, and the train/held-out split.
- [ ] Decide whether "source contribution" is reported via P02's per-source
      reconstruction deltas (no retraining) or via genuine leave-one-out
      retraining (GPU, SLURM — see feedback memory on HPC job placement).
- [ ] Implement accordingly.

Usage
-----
  Not runnable yet.
"""

raise NotImplementedError(
    "P05_ablation.py is a scaffold — see the module docstring for the open "
    "design questions (Option 1 vs Option 2, retraining vs reusing P02 "
    "source-specific metrics) that need deciding before this can be written."
)
