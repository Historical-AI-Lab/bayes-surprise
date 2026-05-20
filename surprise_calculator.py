"""surprise_calculator.py — stage 2 of the Bayesian-surprise pipeline.

Reads per-step belief files produced by belief_generator.py and computes:
  - Standard Bayesian surprise: D_KL(P_T ‖ P_{T-n}) over each lag n and
    each latent (conflict, mystery).  Posterior‖prior direction — expectation
    taken under the current/later belief, aligned with the time arrow.
  - Pooled Bayesian surprise: KL over the union of beliefs at T-n and T.
  - Embedding novelty baseline (cosine distance, no GPU).
  - What-happens-next improbability baselines.

Both sides of the KL use per-token-mean log-probabilities (normalize="per_token")
so that beliefs of different lengths are fairly compared.

Usage
-----
    python3 surprise_calculator.py <Book> [--model-family mixtral|gemma]
            [--backend vllm] [--tau 1.0] [--lags 3] [--no-pooled]
            [--baseline-b] [--max-steps N]

<Book> is the stem shared by all IO files (e.g. ShesNotSorryMaryKubica).

Inputs
------
    chunkedtexts/<Book>.json          chunk text + is_surprising gold labels
    beliefs/<Book>_beliefs.json       per-step belief objects from stage 1
    beliefs/<Book>_summaries.json     per-step summaries (for prompt reconstruction)
    characters/<Book>_characters.json optional character + premise data

Output
------
    surprise/<Book>_surprise.json     written incrementally, resumable
"""

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

from belief_generator import (
    build_context,
    _build_scoring_prompt,
    CONFLICT_ANCHOR,
    MYSTERY_ANCHOR,
    _load_characters,
)
from model_query import score_continuations, MODELS
from mmr import embed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATENTS = [
    ("conflict", CONFLICT_ANCHOR),
    ("mystery",  MYSTERY_ANCHOR),
]

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _softmax(log_probs: list[float], tau: float = 1.0) -> np.ndarray:
    """Softmax over per-token log-probs with temperature tau."""
    x = np.array(log_probs, dtype=np.float64) / tau
    x -= x.max()          # numerical stability
    e = np.exp(x)
    return e / e.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(P ‖ Q) = Σ P_i log(P_i / Q_i).  Posterior‖prior direction.

    Both P and Q must be positive; softmax output guarantees this.
    """
    return float(np.sum(p * (np.log(p) - np.log(q))))


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def _avg_logprobs(
    prompt: str,
    beliefs: list[str],
    model_id: str,
    backend: str,
) -> list[float]:
    """Return per-token avg log-prob for each belief continuation."""
    results = score_continuations(
        prompt, beliefs, model_id,
        normalize="per_token",
        backend=backend,
        use_prefix_cache=True,
    )
    return [r["avg_logprob"] for r in results]


# ---------------------------------------------------------------------------
# KL computation for one (latent, lag) pair
# ---------------------------------------------------------------------------

def _compute_kl_for_lag(
    beliefs_old: list[str],
    beliefs_new: list[str],
    prompt_old: str,
    prompt_new: str,
    model_id: str,
    backend: str,
    tau: float,
    pooled: bool,
    # pre-scored caches to avoid re-scoring the same (prompt, beliefs) pair
    cache_old_under_new: list[float] | None = None,  # beliefs_old under prompt_new
    cache_new_under_old: list[float] | None = None,  # beliefs_new under prompt_old (pooled)
    cache_new_under_new: list[float] | None = None,  # beliefs_new under prompt_new (pooled)
) -> tuple[float, float, dict]:
    """Return (kl_std, kl_pooled, score_cache) for one latent/lag combination.

    score_cache contains any newly computed scored arrays that callers may
    reuse for other lags sharing the same prompt.  Keys:
        "old_under_new"  — beliefs_old scored under prompt_new
        "new_under_old"  — beliefs_new scored under prompt_old (if pooled)
        "new_under_new"  — beliefs_new scored under prompt_new (if pooled)
    """
    score_cache = {}

    # --- beliefs_old under prompt_old (always the "prior" half) ---
    lp_old_under_old = _avg_logprobs(prompt_old, beliefs_old, model_id, backend)

    # --- beliefs_old under prompt_new (always the "posterior" half for std KL) ---
    if cache_old_under_new is not None:
        lp_old_under_new = cache_old_under_new
    else:
        lp_old_under_new = _avg_logprobs(prompt_new, beliefs_old, model_id, backend)
        score_cache["old_under_new"] = lp_old_under_new

    # Standard KL: D_KL(P_new ‖ P_old) over the 8 old beliefs.
    P_old = _softmax(lp_old_under_old, tau)
    P_new = _softmax(lp_old_under_new, tau)
    kl_std = kl_divergence(P_new, P_old)   # posterior ‖ prior

    kl_pool = None
    if pooled:
        # Pooled support = deduplicated union of beliefs_old and beliefs_new.
        seen = {}
        for b in beliefs_old:
            seen[b] = "old"
        for b in beliefs_new:
            if b not in seen:
                seen[b] = "new"
        support = list(seen.keys())
        idx_old = [i for i, b in enumerate(support) if seen[b] == "old"]
        idx_new = [i for i, b in enumerate(support) if seen[b] == "new"]

        lp_pool_under_old = [None] * len(support)
        lp_pool_under_new = [None] * len(support)

        # beliefs_old portion under prompt_old — already scored
        for j, i in enumerate(idx_old):
            lp_pool_under_old[i] = lp_old_under_old[j]

        # beliefs_old portion under prompt_new — already scored
        for j, i in enumerate(idx_old):
            lp_pool_under_new[i] = lp_old_under_new[j]

        if idx_new:
            extra_beliefs = [support[i] for i in idx_new]

            # beliefs_new under prompt_old
            if cache_new_under_old is not None:
                lp_new_under_old = cache_new_under_old
            else:
                lp_new_under_old = _avg_logprobs(prompt_old, extra_beliefs, model_id, backend)
                score_cache["new_under_old"] = lp_new_under_old
            for j, i in enumerate(idx_new):
                lp_pool_under_old[i] = lp_new_under_old[j]

            # beliefs_new under prompt_new
            if cache_new_under_new is not None:
                lp_new_under_new = cache_new_under_new
            else:
                lp_new_under_new = _avg_logprobs(prompt_new, extra_beliefs, model_id, backend)
                score_cache["new_under_new"] = lp_new_under_new
            for j, i in enumerate(idx_new):
                lp_pool_under_new[i] = lp_new_under_new[j]

        P_pool_old = _softmax(lp_pool_under_old, tau)
        P_pool_new = _softmax(lp_pool_under_new, tau)
        kl_pool = kl_divergence(P_pool_new, P_pool_old)  # posterior ‖ prior

    return kl_std, kl_pool, score_cache


# ---------------------------------------------------------------------------
# Per-step computation
# ---------------------------------------------------------------------------

def compute_surprise_for_step(
    T: int,
    chunks: list[dict],
    beliefs_steps: list[dict],
    summaries_steps: list[dict],
    characters: list[dict] | None,
    premise: str | None,
    model_id: str,
    backend: str,
    tau: float,
    n_lags: int,
    pooled: bool,
    baseline_b: bool,
    chunk_embeddings: dict,   # mutable cache: T -> (1, D) array
) -> dict:
    """Compute all surprise measures for step T.

    Returns a dict ready for JSON serialisation.  Also populates
    chunk_embeddings[T] as a side-effect.
    """
    record = {
        "step": T,
        "is_surprising": chunks[T].get("is_surprising"),
    }

    # ------------------------------------------------------------------ #
    # Embedding novelty (CPU, no model server needed)                     #
    # ------------------------------------------------------------------ #
    print(f"    embed ...", flush=True)
    if T not in chunk_embeddings:
        chunk_embeddings[T] = embed([chunks[T]["text"]])  # shape (1, D)
    e_T = chunk_embeddings[T][0]  # (D,)
    print(f"    embed done", flush=True)

    prior_indices = [t for t in range(max(0, T - 3), T) if t >= 0]
    if prior_indices:
        for t in prior_indices:
            if t not in chunk_embeddings:
                chunk_embeddings[t] = embed([chunks[t]["text"]])
        prior_vecs = np.stack([chunk_embeddings[t][0] for t in prior_indices])  # (k, D)
        mean_vec = prior_vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        record["novelty"] = float(1.0 - float(e_T @ mean_vec))
        sims = prior_vecs @ e_T
        record["novelty_maxsim"] = float(1.0 - float(sims.max()))
    else:
        record["novelty"] = None
        record["novelty_maxsim"] = None

    # ------------------------------------------------------------------ #
    # What-happens-next baselines                                         #
    # ------------------------------------------------------------------ #
    p_next = beliefs_steps[T].get("p_next_summary")
    record["p_next_summary"] = p_next
    record["improb_refined"] = float(-math.log(p_next)) if p_next is not None else None

    if baseline_b:
        if T + 1 < len(summaries_steps):
            next_summary = summaries_steps[T + 1]["summary"]
            ctx = build_context(chunks, summaries_steps, characters, premise, T)
            next_prompt = ctx + "\n\nWhat happens next:"
            print(f"    baseline_b vllm call ...", flush=True)
            lp = _avg_logprobs(next_prompt, [next_summary], model_id, backend)
            print(f"    baseline_b done", flush=True)
            record["improb_naive_logprob"] = lp[0]
        else:
            record["improb_naive_logprob"] = None

    # ------------------------------------------------------------------ #
    # Bayesian surprise (KL divergence)                                   #
    # ------------------------------------------------------------------ #
    kl_std_all    = {"conflict": {}, "mystery": {}}
    kl_pooled_all = {"conflict": {}, "mystery": {}} if pooled else None

    available_lags = [n for n in range(1, n_lags + 1) if T - n >= 0]

    if not available_lags:
        # No prior steps available yet.
        record["kl_std"] = kl_std_all
        if pooled:
            record["kl_pooled"] = kl_pooled_all
        record["kl_std_mean"]    = {"conflict": None, "mystery": None}
        record["kl_std_mean_all"] = None
        if pooled:
            record["kl_pooled_mean"]     = {"conflict": None, "mystery": None}
            record["kl_pooled_mean_all"] = None
        return record

    # Build prompt at T for each latent once; reuse across lags.
    ctx_T = build_context(chunks, summaries_steps, characters, premise, T)
    prompt_new_by_latent = {
        lat: _build_scoring_prompt(ctx_T, anchor)
        for lat, anchor in LATENTS
    }

    beliefs_new_by_latent = {
        lat: beliefs_steps[T].get(f"L_{lat}_beliefs", [])
        for lat, _ in LATENTS
    }

    for lat, anchor in LATENTS:
        prompt_new = prompt_new_by_latent[lat]
        beliefs_new = beliefs_new_by_latent[lat]

        # Cache: beliefs_new under prompt_new — reused across all lags.
        if pooled and beliefs_new:
            lp_new_under_new_cache = _avg_logprobs(
                prompt_new, beliefs_new, model_id, backend
            )
        else:
            lp_new_under_new_cache = None

        # Cache: beliefs_old under prompt_new — different per lag.
        # We compute inside the lag loop but pass it forward if lag shares prompt_new
        # (prompt_new is the same for all lags; only beliefs_old differs per lag).

        for n in available_lags:
            s = T - n  # prior step index
            beliefs_old = beliefs_steps[s].get(f"L_{lat}_beliefs", [])
            if not beliefs_old:
                continue

            ctx_s = build_context(chunks, summaries_steps, characters, premise, s)
            prompt_old = _build_scoring_prompt(ctx_s, anchor)

            # For pooled: beliefs_new under prompt_old — different per lag (prompt_old changes).
            if pooled and beliefs_new:
                lp_new_under_old_cache = _avg_logprobs(
                    prompt_old, beliefs_new, model_id, backend
                )
            else:
                lp_new_under_old_cache = None

            kl_s, kl_p, _ = _compute_kl_for_lag(
                beliefs_old=beliefs_old,
                beliefs_new=beliefs_new,
                prompt_old=prompt_old,
                prompt_new=prompt_new,
                model_id=model_id,
                backend=backend,
                tau=tau,
                pooled=pooled,
                cache_new_under_old=lp_new_under_old_cache,
                cache_new_under_new=lp_new_under_new_cache,
            )

            kl_std_all[lat][str(n)] = kl_s
            if pooled:
                kl_pooled_all[lat][str(n)] = kl_p

    # Aggregates: mean over available lags that were computed.
    def _mean_or_none(d: dict) -> float | None:
        vals = [v for v in d.values() if v is not None]
        return float(np.mean(vals)) if vals else None

    record["kl_std"] = kl_std_all
    record["kl_std_mean"] = {
        lat: _mean_or_none(kl_std_all[lat]) for lat, _ in LATENTS
    }
    lat_means_std = [v for v in record["kl_std_mean"].values() if v is not None]
    record["kl_std_mean_all"] = float(np.mean(lat_means_std)) if lat_means_std else None

    if pooled:
        record["kl_pooled"] = kl_pooled_all
        record["kl_pooled_mean"] = {
            lat: _mean_or_none(kl_pooled_all[lat]) for lat, _ in LATENTS
        }
        lat_means_pool = [v for v in record["kl_pooled_mean"].values() if v is not None]
        record["kl_pooled_mean_all"] = float(np.mean(lat_means_pool)) if lat_means_pool else None

    return record


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _make_surprise_metadata(
    model_family: str,
    model_id: str,
    tau: float,
    n_lags: int,
    pooled: bool,
    baseline_b: bool,
) -> dict:
    meta = {
        "model_family": model_family,
        "model_id": model_id,
        "phase": "surprise",
        "tau": tau,
        "lags": n_lags,
        "pooled": pooled,
        "baseline_b": baseline_b,
        "python": platform.python_version(),
    }
    try:
        import sentence_transformers as st
        meta["sentence_transformers_version"] = st.__version__
    except Exception:
        pass
    return meta


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_surprise(
    book_stem: str,
    model_family: str,
    backend: str,
    tau: float,
    n_lags: int,
    pooled: bool,
    baseline_b: bool,
    max_steps: int | None,
) -> None:
    repo = Path(__file__).parent

    chunks_path   = repo / "chunkedtexts" / f"{book_stem}.json"
    beliefs_path  = repo / "beliefs"      / f"{book_stem}_beliefs.json"
    summaries_path = repo / "beliefs"     / f"{book_stem}_summaries.json"
    out_dir        = repo / "surprise"
    out_path       = out_dir / f"{book_stem}_surprise.json"

    for p in (chunks_path, beliefs_path, summaries_path):
        if not p.exists():
            sys.exit(f"Required input not found: {p}")

    out_dir.mkdir(exist_ok=True)

    with open(chunks_path)   as f: chunks          = json.load(f)
    with open(beliefs_path)  as f: beliefs_data     = json.load(f)
    with open(summaries_path) as f: summaries_data  = json.load(f)

    # Index by step (they should already be in order, but sort to be safe).
    beliefs_steps  = sorted(beliefs_data["steps"],  key=lambda s: s["step"])
    summaries_steps = sorted(summaries_data["steps"], key=lambda s: s["step"])

    if len(beliefs_steps) != len(chunks):
        print(
            f"  warning: {len(beliefs_steps)} belief steps vs {len(chunks)} chunks",
            file=sys.stderr,
        )

    premise, characters = _load_characters(chunks_path)

    # Base model for scoring (instruct for generation, base for scoring per plan).
    model_id = MODELS[f"{model_family}-base"]

    n_steps = len(chunks)
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)

    # Resumability: load existing output if present.
    done_steps: set[int] = set()
    existing_steps: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            existing_data = json.load(f)
        existing_steps = existing_data.get("steps", [])
        done_steps = {s["step"] for s in existing_steps}
        print(f"  resuming: {len(done_steps)} steps already done")

    metadata = _make_surprise_metadata(
        model_family, model_id, tau, n_lags, pooled, baseline_b
    )

    chunk_embeddings: dict[int, np.ndarray] = {}  # T -> (1, D)
    steps_out = list(existing_steps)

    for T in range(n_steps):
        if T in done_steps:
            continue

        print(f"  step {T}/{n_steps - 1} ...", flush=True)

        record = compute_surprise_for_step(
            T=T,
            chunks=chunks,
            beliefs_steps=beliefs_steps,
            summaries_steps=summaries_steps,
            characters=characters,
            premise=premise,
            model_id=model_id,
            backend=backend,
            tau=tau,
            n_lags=n_lags,
            pooled=pooled,
            baseline_b=baseline_b,
            chunk_embeddings=chunk_embeddings,
        )
        steps_out.append(record)
        steps_out.sort(key=lambda s: s["step"])

        # Persist after every step (crash-resilient).
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata, "steps": steps_out}, f,
                      indent=1, ensure_ascii=False)

    print(f"  done → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    def _flag(name: str) -> bool:
        return name in args

    def _opt(name: str, default=None):
        try:
            return args[args.index(name) + 1]
        except (ValueError, IndexError):
            return default

    if not args or args[0].startswith("--"):
        sys.exit(
            "usage: python3 surprise_calculator.py <Book> "
            "[--model-family mixtral|gemma] [--backend vllm] "
            "[--tau 1.0] [--lags 3] [--no-pooled] [--baseline-b] "
            "[--max-steps N]"
        )

    book_stem    = args[0]
    model_family = _opt("--model-family", "mixtral")
    backend      = _opt("--backend", "vllm")
    tau          = float(_opt("--tau", "1.0"))
    n_lags       = int(_opt("--lags", "3"))
    pooled       = not _flag("--no-pooled")
    baseline_b   = _flag("--baseline-b")
    max_steps_s  = _opt("--max-steps")
    max_steps    = int(max_steps_s) if max_steps_s else None

    run_surprise(
        book_stem=book_stem,
        model_family=model_family,
        backend=backend,
        tau=tau,
        n_lags=n_lags,
        pooled=pooled,
        baseline_b=baseline_b,
        max_steps=max_steps,
    )
