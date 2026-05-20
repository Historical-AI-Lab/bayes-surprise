"""mmr.py — Maximal Marginal Relevance selection.

Reduces a set of N candidates to k diverse, relevant items.
Used here to down-sample 20 belief candidates → 8, and 10
what-happens-next candidates → 7, before storing per-step data.

Reference: Carbonell & Goldstein, SIGIR 1998.
"""

import numpy as np

# Sentence-transformer model used for inter-candidate diversity.
# CPU-only, ~80 MB; change here to swap to a stronger model.
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_st_model = None  # lazy-loaded on first call to embed()


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised sentence embeddings, shape (N, D).

    The SentenceTransformer model is loaded once and cached for the
    process lifetime.  Runs on CPU so it never competes with the vLLM
    server for GPU memory.
    """
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(_EMBED_MODEL_NAME, device="cpu")
    vecs = _st_model.encode(texts, normalize_embeddings=True,
                             show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


def mmr_select(
    candidates: list[str],
    relevance: list[float],
    embeddings: np.ndarray,
    k: int,
    lambda_: float = 0.5,
) -> list[int]:
    """Select k indices via MMR.

    Parameters
    ----------
    candidates  : the text strings (unused except for length check).
    relevance   : log-probabilities (or any real-valued scores) —
                  higher is more relevant.  Min-max normalised into
                  [0, 1] inside this function before combining with
                  the diversity term.
    embeddings  : shape (N, D), L2-normalised, one row per candidate.
    k           : number of items to select.
    lambda_     : trade-off weight.  1.0 = pure relevance (no MMR);
                  0.0 = pure diversity; 0.5 = balanced default.

    Returns
    -------
    List of selected indices in selection order (best first).
    """
    n = len(candidates)
    if n == 0:
        return []
    k = min(k, n)

    rel = np.array(relevance, dtype=np.float64)

    # Min-max normalise relevance into [0, 1].
    rel_min, rel_max = rel.min(), rel.max()
    if rel_max > rel_min:
        rel_norm = (rel - rel_min) / (rel_max - rel_min)
    else:
        # All scores identical — relevance plays no role; just use diversity.
        rel_norm = np.zeros(n, dtype=np.float64)

    emb = embeddings  # (N, D), already L2-normalised

    selected = []
    remaining = list(range(n))

    # First pick: highest relevance.
    best = max(remaining, key=lambda i: rel_norm[i])
    selected.append(best)
    remaining.remove(best)

    while len(selected) < k and remaining:
        sel_emb = emb[selected]  # (|S|, D)

        best_score = -np.inf
        best_idx = remaining[0]
        for i in remaining:
            # Relevance term.
            r = lambda_ * rel_norm[i]
            # Diversity term: max cosine similarity to any already-selected.
            cos_to_selected = emb[i] @ sel_emb.T   # (|S|,) cosines in [-1,1]
            max_sim = float(cos_to_selected.max())
            d = (1.0 - lambda_) * max_sim
            score = r - d
            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected
