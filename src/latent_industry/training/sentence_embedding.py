"""sentence_embedding.py — KL-divergence sentence-embedding training against
a graph diffusion target, for scripts/training/train_sentence_embedding.py.

Trains a sentence transformer so that its cosine-similarity distribution
matches the graph diffusion distribution stored in
<LUNARC_DATA_ROOT>/processed/diffusion_matrix_{graph_name}.npz:

  p_i  = row i of the diffusion matrix  (row-normalised probability distribution)
  q_i  = softmax( cos(e_i, e_j) / tau,  j != i )
  L    = mean_{i active}  KL(p_i || q_i)   +  lam * MSE(E, E_initial)
                                            +  lam_hub * MSE(E_hub, E_hub_initial)

See the KL-vs-pairwise-regression rationale and the --lam-hub anchoring
rationale in the CLI script's docstring/help text — this module holds only
the mechanics, not the argument definitions.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

from latent_industry.common.paths import METADATA_DIR, PROCESSED_DATA_DIR


def load_diffusion_target(graph_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load diffusion_matrix_{graph_name}.npz and re-enforce row-normalisation
    (handles any numerical drift from the graph-build stage). Returns
    (P, active_mask, codes, versions) — P is (n, n) float32, active_mask is
    an (n,) bool of rows with any outgoing mass (isolated nodes are skipped
    in the KL loss so they don't contribute a degenerate uniform target)."""
    diff_path = PROCESSED_DATA_DIR / f"diffusion_matrix_{graph_name}.npz"
    if not diff_path.exists():
        raise FileNotFoundError(f"{diff_path} not found — run build_multilayer_diffusion.py first.")

    data = np.load(diff_path, allow_pickle=True)
    S = data["S"].astype(np.float32)
    codes = data["codes"]
    versions = data["versions"]

    np.fill_diagonal(S, 0.0)
    S = np.clip(S, 0.0, None)
    row_sums = S.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = S / row_sums

    active_mask = P.sum(axis=1) > 0
    return P, active_mask, codes, versions


def _load_level(pattern: str, drop_key: str) -> dict[str, str]:
    path = METADATA_DIR / pattern
    with open(path) as f:
        d = json.load(f)
    d.pop(drop_key, None)
    return d


def load_4d_mappings() -> dict[str, dict[str, str]]:
    """{'SNI07': {code: description}, 'SNI02': ..., 'SNI92': ...}."""
    return {
        "SNI07": _load_level("mapping_Sni_4digits_2007.json", "0000"),
        "SNI02": _load_level("mapping_Sni_4digits_2002.json", "0000"),
        "SNI92": _load_level("mapping_Sni_4digits_1992.json", "0000"),
    }


def load_target_texts(codes: np.ndarray, versions: np.ndarray,
                       ver_map: dict[str, dict[str, str]] | None = None) -> list[str]:
    """One description text per diffusion-matrix row, aligned to `codes`/`versions`."""
    ver_map = ver_map if ver_map is not None else load_4d_mappings()
    return [
        ver_map.get(str(ver), {}).get(str(code), f"{ver} {code}")
        for code, ver in zip(codes, versions)
    ]


def load_hub_texts() -> list[str]:
    """Hierarchy-hub (3d/2d/section, SNI07 only) description text. These are
    never diffusion-matrix targets (only 4d codes are), so they get no
    KL-loss gradient — loaded only so --lam-hub can anchor them against
    drift; never saved as embeddings."""
    sni07_3d = _load_level("mapping_Sni_3digits_2007.json", "000")
    sni07_2d = _load_level("mapping_Sni_2digits_2007.json", "00")
    sni07_sec = _load_level("mapping_Sni_section_2007.json", "Unknown")
    return list(sni07_3d.values()) + list(sni07_2d.values()) + list(sni07_sec.values())


def load_model(model_src: str, device: str):
    """Load a SentenceTransformer and return (model, preprocess_fn, dim).
    preprocess_fn is the tokenizer entry point — renamed from `tokenize` to
    `preprocess` in newer sentence-transformers releases; both are supported."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_src, device=device)
    # get_sentence_embedding_dimension was renamed to get_embedding_dimension in
    # newer sentence-transformers releases; support both (see build_nace_embeddings.py).
    get_dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)
    dim = get_dim()
    preprocess = model.preprocess if hasattr(model, "preprocess") else model.tokenize
    return model, preprocess, dim


def encode_differentiable(model, preprocess, device: str, texts: list[str]) -> torch.Tensor:
    """Tokenize all texts and return L2-normalised embeddings with gradient."""
    features = preprocess(texts)
    features = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()}
    out = model(features)
    emb = out["sentence_embedding"]
    return F.normalize(emb, p=2, dim=-1)


def encode_frozen(model, device: str, texts: list[str], hub_texts: list[str] | None,
                   lam_hub: float) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Initial (no-gradient) embeddings used as MSE regularisation anchors."""
    with torch.no_grad():
        E_init = model.encode(
            texts, batch_size=512, show_progress_bar=False,
            convert_to_tensor=True, normalize_embeddings=True,
        ).to(device)
        E_hub_init = model.encode(
            hub_texts, batch_size=512, show_progress_bar=False,
            convert_to_tensor=True, normalize_embeddings=True,
        ).to(device) if (lam_hub > 0 and hub_texts) else None
    return E_init.clone().detach(), (E_hub_init.clone().detach() if E_hub_init is not None else None)


def train_kl(model, preprocess, device: str, texts: list[str], hub_texts: list[str],
             P: np.ndarray, active_mask: np.ndarray, *, tau: float, lam: float, lam_hub: float,
             lr: float, epochs: int, log_every: int) -> float:
    """Run the full KL-divergence training loop in place on `model`'s weights.
    Returns the best (lowest) total loss observed. See module docstring for
    the loss formula."""
    P_t = torch.tensor(P, dtype=torch.float32, device=device)
    active_t = torch.tensor(active_mask, dtype=torch.bool, device=device)
    print(f"Active rows for KL loss: {int(active_mask.sum())} / {len(active_mask)}")

    E_init, E_hub_init = encode_frozen(model, device, texts, hub_texts, lam_hub)
    print(f"Initial embeddings: {tuple(E_init.shape)}"
          + (f"  +  hub anchor: {tuple(E_hub_init.shape)}" if E_hub_init is not None
             else "  (hub anchor disabled)"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\nTraining {epochs} epochs …")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        E = encode_differentiable(model, preprocess, device, texts)  # (n, dim), grad on

        # Cosine similarity: already unit-normalised, so it's just the dot product.
        cos_sim = E @ E.T  # (n, n)

        # Zero the diagonal before softmax so self-similarity doesn't inflate q_i[i].
        cos_sim = cos_sim.clone()
        cos_sim.fill_diagonal_(float("-inf"))

        # q_i = softmax(cos_sim_i / tau) — model's distribution over all other codes
        Q = torch.softmax(cos_sim / tau, dim=-1)  # (n, n)

        # KL(p_i || q_i) = sum_j p_i[j] * log(p_i[j] / q_i[j])
        # F.kl_div(log_input, target) computes target * (log(target) - log_input).
        # We want sum over j for each row i, then mean over active rows.
        log_Q = torch.log(Q.clamp(min=1e-12))  # numerical safety
        kl_rows = F.kl_div(log_Q, P_t, reduction="none").sum(dim=-1)  # (n,)
        kl_loss = kl_rows[active_t].mean()

        # MSE toward initial embeddings prevents the model from collapsing all codes
        # to a single point (which would trivially satisfy the KL loss for a flat p_i).
        mse_loss = F.mse_loss(E, E_init.detach())

        # Hierarchy-hub text gets no KL-loss gradient at all (see load_hub_texts'
        # docstring), so without its own anchor it just drifts as the shared
        # transformer's weights get pulled toward the 4d-code KL objective —
        # this term is what lam_hub controls.
        if E_hub_init is not None:
            E_hub = encode_differentiable(model, preprocess, device, hub_texts)
            hub_mse_loss = F.mse_loss(E_hub, E_hub_init.detach())
        else:
            hub_mse_loss = torch.tensor(0.0, device=device)

        loss = kl_loss + lam * mse_loss + lam_hub * hub_mse_loss
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(f"  epoch {epoch:4d}/{epochs}  "
                  f"loss={loss.item():.5f}  "
                  f"kl={kl_loss.item():.5f}  "
                  f"mse={mse_loss.item():.5f}  "
                  f"hub_mse={hub_mse_loss.item():.5f}")

    return best_loss


def embed_and_save(model, code_to_text: dict[str, str], out_path, label: str) -> None:
    """Encode every (code, text) pair with the final (non-differentiable)
    model and save to out_path/embeddings_{label}.npz."""
    texts_list = list(code_to_text.values())
    codes_list = list(code_to_text.keys())
    embs = model.encode(
        texts_list, batch_size=512, show_progress_bar=False,
        normalize_embeddings=True,
    )
    file_path = out_path / f"embeddings_{label}.npz"
    np.savez_compressed(file_path, embeddings=embs, codes=np.array(codes_list))
    print(f"  {label}: {embs.shape} → {file_path.name}")
