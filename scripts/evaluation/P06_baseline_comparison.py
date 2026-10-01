"""P06_baseline_comparison.py — Result 6 (docs/Embedding validation.md):
compare the learned embedding against existing relatedness methods. NOT YET
IMPLEMENTED — and explicitly deferred in the source doc:

  "Result 6 — Compare against existing methods (leave out for now) ...
  This should actually come late. The paper isn't about beating Product
  Space. It's about learning opportunity spaces."

Candidate comparators named there:

  - Product Space (Hidalgo et al. 2007) — already the theoretical reference
    for the P04 density/relatedness regressions (see e.g. P04_entry.py,
    P04_growth.py docstrings), but no Product Space *proximity matrix* is
    built anywhere in this repo yet.
  - Tacchella (economic fitness/complexity) — not implemented anywhere.
  - text-only — the raw pre-fine-tuning sentence-transformer embedding.
    Already computed as the "baseline" comparison inside evaluate_embeddings.py
    (see --baseline / DEFAULT_BASELINE and section G/K's raw-vs-trained
    deltas, now split across P02_shared_geometry_reconstruction.py and
    P03_emergent_structure.py) — reusable here rather than reimplemented.
  - labor-only / IO-only — single-source diffusion targets/embeddings.
    Not currently built; would need either a single-relation diffusion
    target (build_multilayer_diffusion.py restricted to one layer) or the
    same leave-one-source-out question P05_ablation.py is blocked on.

TODO before implementing
-------------------------
- [ ] Confirm with the paper draft that Result 6 is still being left for a
      later pass (per the doc, it should be) before spending time here.
- [ ] Build a Product Space revealed-comparative-advantage proximity matrix
      from the same underlying trade/employment data, if/when this is
      picked back up.
- [ ] Decide whether Tacchella fitness/complexity is in scope at all.
- [ ] Reuse the existing raw-encoder baseline (already computed by
      P02/P03 via --baseline) for the "text-only" row instead of
      recomputing it.
- [ ] Decide whether labor-only/IO-only comparators reuse P05's
      leave-one-source-out machinery once that exists, rather than being
      built independently.
- [ ] Implement the comparison table/figure once the above are resolved.

Usage
-----
  Not runnable yet.
"""

raise NotImplementedError(
    "P06_baseline_comparison.py is a scaffold — docs/Embedding validation.md "
    "explicitly defers Result 6 ('leave out for now'). See the module "
    "docstring for what a real implementation would need."
)
