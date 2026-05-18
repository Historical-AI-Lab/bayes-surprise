"""belief_generator.py — two-phase pipeline for the Bayesian-surprise project.

Reads a chunked-text JSON produced by chunker.py and generates, for each
~1000-word chunk, the Summary and Belief objects specified in the plan.

Two phases, run separately (the instruct model and base model cannot both
fit on an A100 40 GB):

    Phase 1 — generate (instruct model)
        Elicits candidate beliefs, what-happens-next guesses, per-step
        summaries, and optional character summaries.  Writes raw candidates
        plus sentence embeddings to work/<Book>_candidates.json.

    Phase 2 — score (base model)
        Loads candidates, scores continuation log-likelihoods, runs MMR
        selection (20→8 for conflict/mystery, 10→7 for what-next), and
        assembles the final Belief and Summary objects.  Also scores the
        what-happens-next improbability of summary_{t+1}.  Writes:
            beliefs/<Book>_beliefs.json
            beliefs/<Book>_summaries.json

Usage
-----
    # Generation (instruct server must be up at BAYES_VLLM_BASE_URL):
    python3 belief_generator.py chunkedtexts/ShesNotSorryMaryKubica.json \\
            --phase generate [--model-family mixtral|gemma] [--max-steps N]

    # Scoring (base server must be up at BAYES_VLLM_BASE_URL):
    python3 belief_generator.py chunkedtexts/ShesNotSorryMaryKubica.json \\
            --phase score [--model-family mixtral|gemma] [--lambda 0.5]

    # Dry-run: print the context prompt for step T, no model calls:
    python3 belief_generator.py chunkedtexts/ShesNotSorryMaryKubica.json \\
            --dry-run [--step T]

    # Log every context prompt to work/<Book>_contexts.json during generation:
    python3 belief_generator.py chunkedtexts/ShesNotSorryMaryKubica.json \\
            --phase generate --log-contexts

Characters (optional)
---------------------
If characters/<Book>_characters.json exists, characters are included in
context.  Expected format (a dict, not a list):

    {
      "premise": "One-sentence setting/premise always shown first.",
      "Canonical Name": {"aliases": ["Name", "Other Name"], "range": [0, 99]},
      ...  # up to 7 character entries
    }

Each character's description appears only when (a) the step T falls within its
inclusive chunk range and (b) at least one alias matches (word-boundary) the
current chunk or the last CONTEXT_SUMMARIES step-summaries.  If absent,
character context is silently omitted.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

from mmr import embed, mmr_select
from model_query import generate_answer, score_continuations, MODELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many raw candidates to generate per latent.
N_CONFLICT_CANDIDATES  = 20   # 4 calls × 5 answers
N_MYSTERY_CANDIDATES   = 20
N_NEXT_CANDIDATES      = 10   # 2 calls × 5 answers

# How many to keep after MMR selection.
K_BELIEFS = 8
K_NEXT    = 7

# How many recent step-summaries to show as context.
CONTEXT_SUMMARIES = 7

# Anchor phrases for conflict and mystery beliefs (the text that
# precedes the scorable continuation).
CONFLICT_ANCHOR = "In this story, the central conflict is"
MYSTERY_ANCHOR  = "By the end of this story, the reader will discover"

# Numbered-line regex: matches "1." / "1)" at the start (after stripping).
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s*")

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _alias_pattern(aliases: list[str]) -> re.Pattern:
    """Compile a word-boundary pattern matching any of the character's aliases."""
    # Longest first so shorter aliases don't mask longer ones in the alternation.
    alts = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
    return re.compile(r"\b(?:" + alts + r")\b", re.IGNORECASE)


def build_context(
    chunks: list[dict],
    summaries: list[dict],          # one dict per step so far
    characters: list[dict] | None,  # normalized character dicts, or None
    premise: str | None,            # stable setting/premise shown first always
    T: int,
    to_be_hidden: tuple = (),
) -> str:
    """Build the model input for step T.

    If premise is set it is always the first element in the prompt.

    At T==0 returns the premise (if any) followed by the chunk text.

    From T==1 onwards, also prepends:
      - One-sentence descriptions of up to 7 characters whose range covers T
        and who are mentioned (by any alias, word-boundary) in chunk T or in
        the last CONTEXT_SUMMARIES step-summaries (omitted if characters is
        None or the character's canonical name is in to_be_hidden).
      - One-sentence summaries of the last min(T, CONTEXT_SUMMARIES) steps.

    Then appends the ~1000-word chunk text.
    """
    chunk_text = chunks[T]["text"]

    parts = []
    if premise:
        parts.append(f"Premise:\n{premise}")

    if T == 0:
        parts.append(chunk_text)
        return "\n\n".join(parts)

    # --- character context ---
    char_lines = []
    if characters:
        window_start = max(0, T - CONTEXT_SUMMARIES)
        recent_texts = (
            [chunks[t]["text"] for t in range(window_start, T + 1)]
            + [s["summary"] for s in summaries[window_start:T]]
        )
        combined = " ".join(recent_texts)

        for char in characters:
            if char["name"] in to_be_hidden:
                continue
            r_start, r_end = char["range"]
            if not (r_start <= T <= r_end):
                continue
            if not _alias_pattern(char["aliases"]).search(combined):
                continue
            desc = _latest_char_desc(summaries, char["name"], T)
            if desc:
                char_lines.append(f"{char['name']}: {desc}")

    # --- step-summary context ---
    start = max(0, T - CONTEXT_SUMMARIES)
    summary_lines = [s["summary"] for s in summaries[start:T]]

    # --- assemble ---
    if char_lines:
        parts.append("Characters:\n" + "\n".join(char_lines))
    if summary_lines:
        parts.append("Recent events:\n" + "\n".join(summary_lines))
    parts.append(chunk_text)
    return "\n\n".join(parts)


def _latest_char_desc(summaries: list[dict], name: str, T: int) -> str | None:
    """Return the most recent character description for character name."""
    for s in reversed(summaries[:T]):
        descs = s.get("char_descriptions", {})
        if name in descs and descs[name]:
            return descs[name]
    return None


# ---------------------------------------------------------------------------
# Prompt/response helpers
# ---------------------------------------------------------------------------

def _parse_numbered_list(text: str, expected: int) -> list[str]:
    """Extract numbered items from a model reply.

    Tolerates "1." and "1)" delimiters, extra blank lines, and partial
    replies (returns fewer items rather than crashing).  Items that are
    empty after stripping are dropped.
    """
    lines  = text.splitlines()
    items  = []
    for line in lines:
        if _NUMBERED_RE.match(line):
            item = _NUMBERED_RE.sub("", line).strip()
            if item:
                items.append(item)
    if len(items) < expected:
        print(
            f"  warning: expected {expected} list items, got {len(items)}",
            file=sys.stderr,
        )
    return items


def _strip_anchor(text: str, anchor: str) -> str:
    """Return the substring of text that follows anchor (case-insensitive).

    The stored belief is the post-anchor continuation — this is exactly
    what gets scored during Phase 2.  E.g. if the model returns
    "In this story, the central conflict is the rivalry between X and Y."
    we store " the rivalry between X and Y."
    (with the leading space so the scorer sees it correctly).
    """
    lower = text.lower()
    a_lower = anchor.lower()
    idx = lower.find(a_lower)
    if idx == -1:
        # Anchor not found (model started mid-sentence) — keep as-is.
        return text
    return text[idx + len(anchor):]


def _elicit_beliefs(
    context: str,
    anchor: str,
    question_template: str,
    n_calls: int,
    per_call: int,
    model_id: str,
    max_new_tokens: int = 512,
) -> list[str]:
    """Call the instruct model n_calls times to collect candidate beliefs.

    question_template should be a format string with a {anchor} placeholder
    that will be filled with the anchor phrase.
    """
    raw_candidates: list[str] = []
    prompt = context + "\n\n" + question_template.format(anchor=anchor)
    for _ in range(n_calls):
        result = generate_answer(prompt, model_id, max_new_tokens=max_new_tokens,
                                 temperature=1.0, top_p=0.95)
        items  = _parse_numbered_list(result["text"], per_call)
        for item in items:
            raw_candidates.append(_strip_anchor(item, anchor))
    return raw_candidates


def _elicit_summary(context: str, model_id: str) -> str:
    """Elicit a one-sentence summary of the current chunk."""
    prompt = (
        context
        + "\n\nWrite a single sentence summarising what just happened "
          "in the passage above."
    )
    result = generate_answer(prompt, model_id, max_new_tokens=128)
    return result["text"].strip()


def _elicit_char_description(
    context: str,
    char_name: str,
    first_desc: str | None,
    latest_desc: str | None,
    model_id: str,
) -> str:
    """Elicit an updated one-sentence character description."""
    if first_desc and latest_desc and first_desc != latest_desc:
        prior = (
            f"Earlier description: {first_desc}\n"
            f"Most recent description: {latest_desc}\n\n"
        )
    elif first_desc or latest_desc:
        prior = f"Previous description: {first_desc or latest_desc}\n\n"
    else:
        prior = ""
    prompt = (
        context
        + f"\n\n{prior}Write a single sentence describing {char_name} "
          "based on everything you have read so far."
    )
    result = generate_answer(prompt, model_id, max_new_tokens=128)
    return result["text"].strip()


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _build_scoring_prompt(context: str, anchor: str) -> str:
    """Build the base-model scoring prompt: context + question + anchor.

    The scorer measures logP(continuation | prompt), where prompt ends
    with the anchor phrase and the continuation is the stored belief.
    """
    latent_question = {
        CONFLICT_ANCHOR: (
            "What is the central conflict in this story?\n"
            "Provide five distinct answers, each a short sentence on a "
            "numbered new line, beginning with the same phrase:\n\n"
            "1. " + CONFLICT_ANCHOR
        ),
        MYSTERY_ANCHOR: (
            "What does the reader expect to discover by the end of this story?\n"
            "Provide five distinct answers, each a short sentence on a "
            "numbered new line, beginning with the same phrase:\n\n"
            "1. " + MYSTERY_ANCHOR
        ),
    }.get(anchor)
    if latent_question:
        return context + "\n\n" + latent_question
    # What-happens-next scoring.
    return context + "\n\nWhat happens next:"


# ---------------------------------------------------------------------------
# Phase 1: generation
# ---------------------------------------------------------------------------

def run_phase_generate(
    chunks: list[dict],
    characters: list[dict] | None,
    premise: str | None,
    model_id: str,
    work_path: Path,
    max_steps: int | None = None,
    dry_run: bool = False,
    start_step: int = 0,
    log_contexts: bool = False,
) -> None:
    """Step through chunks, eliciting candidate beliefs and summaries.

    Writes/updates work_path after each step (resumable).

    If log_contexts is True, also writes work/<Book>_contexts.json — a list of
    {"step": T, "context": str} dicts, one per step processed in this run.
    Useful for manual inspection of what the model was shown at each step.
    Note: only steps processed in the current run are captured; skipped
    (already-done) steps from a resumed run are not re-logged.

    Intermediate format per step (list of step dicts):
    {
        "step": T,
        "summary": str,
        "char_descriptions": {canonical_name: str, ...},
        "conflict_candidates": [str, ...],    # up to 20, post-anchor
        "mystery_candidates":  [str, ...],
        "next_candidates":     [str, ...],    # up to 10
        "conflict_embeddings": [[float,...]], # all-MiniLM-L6-v2
        "mystery_embeddings":  [[float,...]],
        "next_embeddings":     [[float,...]],
    }
    """
    steps_done: dict[int, dict] = {}
    if work_path.exists():
        existing = json.loads(work_path.read_text())
        for s in existing:
            steps_done[s["step"]] = s
        print(f"  resuming: {len(steps_done)} steps already in {work_path}")

    summaries: list[dict] = []  # rebuilt from steps_done for context
    for t in range(len(chunks)):
        if t in steps_done:
            summaries.append(steps_done[t])
            continue

    # Rebuild summaries list in order for context building.
    summaries = [steps_done[t] for t in sorted(steps_done)]

    n_steps = min(len(chunks), max_steps) if max_steps else len(chunks)

    contexts_log: list[dict] = []  # populated only when log_contexts=True
    contexts_path = work_path.parent / (work_path.stem.replace("_candidates", "_contexts") + ".json")

    for T in range(start_step, n_steps):
        if T in steps_done:
            continue

        context = build_context(chunks, summaries, characters, premise, T)

        if log_contexts:
            contexts_log.append({"step": T, "context": context})

        if dry_run:
            print(f"\n{'='*70}\nStep {T} context prompt:\n{'='*70}")
            print(context)
            print(f"{'='*70}")
            continue

        print(f"  step {T}/{n_steps-1} — generating ...", end=" ", flush=True)

        step_data: dict = {"step": T, "char_descriptions": {}}

        # --- summary ---
        step_data["summary"] = _elicit_summary(context, model_id)

        # --- character descriptions (optional) ---
        if characters:
            chunk_text = chunks[T]["text"]
            for char in characters:
                r_start, r_end = char["range"]
                if not (r_start <= T <= r_end):
                    continue
                if not _alias_pattern(char["aliases"]).search(chunk_text):
                    continue
                first_desc  = _first_char_desc(summaries, char["name"])
                latest_desc = _latest_char_desc(summaries, char["name"], T)
                new_desc = _elicit_char_description(
                    context, char["name"], first_desc, latest_desc, model_id
                )
                step_data["char_descriptions"][char["name"]] = new_desc

        # --- conflict candidates ---
        q_conflict = (
            "What is the central conflict in this story?\n"
            "Provide five distinct answers, each a short sentence on a "
            "numbered new line, beginning with the same phrase:\n\n"
            "1. {anchor}"
        )
        conflict_cands = _elicit_beliefs(
            context, CONFLICT_ANCHOR, q_conflict,
            n_calls=4, per_call=5, model_id=model_id,
        )
        step_data["conflict_candidates"] = conflict_cands

        # --- mystery candidates ---
        q_mystery = (
            "What does the reader expect to discover by the end of this story?\n"
            "Provide five distinct answers, each a short sentence on a "
            "numbered new line, beginning with the same phrase:\n\n"
            "1. {anchor}"
        )
        mystery_cands = _elicit_beliefs(
            context, MYSTERY_ANCHOR, q_mystery,
            n_calls=4, per_call=5, model_id=model_id,
        )
        step_data["mystery_candidates"] = mystery_cands

        # --- what-happens-next candidates ---
        q_next = (
            'Predict what happens next in the form of a short present-tense '
            'sentence. E.g. "Red Riding Hood arrives at her grandmother\'s '
            'house." Provide five distinct predictions, each numbered and on '
            'a new line.\n\n1.'
        )
        next_cands: list[str] = []
        for _ in range(2):
            result = generate_answer(
                context + "\n\n" + q_next, model_id, max_new_tokens=256,
                temperature=1.0, top_p=0.95,
            )
            items = _parse_numbered_list(result["text"], 5)
            next_cands.extend(items)
        step_data["next_candidates"] = next_cands

        # --- embeddings (CPU, cached for offline MMR tuning) ---
        if conflict_cands:
            step_data["conflict_embeddings"] = embed(conflict_cands).tolist()
        else:
            step_data["conflict_embeddings"] = []

        if mystery_cands:
            step_data["mystery_embeddings"] = embed(mystery_cands).tolist()
        else:
            step_data["mystery_embeddings"] = []

        if next_cands:
            step_data["next_embeddings"] = embed(next_cands).tolist()
        else:
            step_data["next_embeddings"] = []

        steps_done[T] = step_data
        summaries.append(step_data)

        # Persist after every step (crash-resilient).
        ordered = [steps_done[t] for t in sorted(steps_done)]
        work_path.parent.mkdir(parents=True, exist_ok=True)
        with open(work_path, "w") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=1)

        if log_contexts:
            with open(contexts_path, "w") as f:
                json.dump(contexts_log, f, ensure_ascii=False, indent=1)

        print("done")

    if not dry_run:
        print(f"Generation complete.  Candidates written to {work_path}")
        if log_contexts:
            print(f"Context log written to {contexts_path}")


def _first_char_desc(summaries: list[dict], name: str) -> str | None:
    """Return the earliest recorded description for character name."""
    for s in summaries:
        descs = s.get("char_descriptions", {})
        if name in descs and descs[name]:
            return descs[name]
    return None


# ---------------------------------------------------------------------------
# Phase 2: scoring
# ---------------------------------------------------------------------------

def run_phase_score(
    chunks: list[dict],
    characters: list[dict] | None,
    premise: str | None,
    model_id: str,
    work_path: Path,
    beliefs_path: Path,
    summaries_path: Path,
    lambda_: float = 0.5,
    max_steps: int | None = None,
    run_metadata: dict | None = None,
) -> None:
    """Score candidates, run MMR, write final Belief and Summary objects.

    Reads the candidates produced by Phase 1 from work_path.
    """
    if not work_path.exists():
        sys.exit(f"Candidates file {work_path} not found; run --phase generate first.")

    candidates_data: list[dict] = json.loads(work_path.read_text())
    n_steps = min(len(candidates_data), max_steps) if max_steps else len(candidates_data)

    beliefs_list:  list[dict] = []
    summaries_list: list[dict] = []

    for idx in range(n_steps):
        step = candidates_data[idx]
        T    = step["step"]

        print(f"  step {T}/{n_steps-1} — scoring ...", end=" ", flush=True)

        # Rebuild summaries up to T for context (needed for scoring prompt).
        summaries_so_far = [
            {"summary": candidates_data[i]["summary"],
             "char_descriptions": candidates_data[i].get("char_descriptions", {})}
            for i in range(idx)
        ]
        context = build_context(chunks, summaries_so_far, characters, premise, T)

        # --- score conflict ---
        conflict_scoring_prompt = _build_scoring_prompt(context, CONFLICT_ANCHOR)
        conflict_cands    = step.get("conflict_candidates", [])
        conflict_emb      = np.array(step.get("conflict_embeddings", []),
                                     dtype=np.float32)
        conflict_beliefs, conflict_logits, conflict_logits_all = _score_and_select(
            conflict_scoring_prompt,
            conflict_cands,
            conflict_emb,
            model_id,
            k=K_BELIEFS,
            lambda_=lambda_,
        )

        # --- score mystery ---
        mystery_scoring_prompt = _build_scoring_prompt(context, MYSTERY_ANCHOR)
        mystery_cands     = step.get("mystery_candidates", [])
        mystery_emb       = np.array(step.get("mystery_embeddings", []),
                                     dtype=np.float32)
        mystery_beliefs, mystery_logits, mystery_logits_all = _score_and_select(
            mystery_scoring_prompt,
            mystery_cands,
            mystery_emb,
            model_id,
            k=K_BELIEFS,
            lambda_=lambda_,
        )

        # --- score what-happens-next ---
        next_scoring_prompt = _build_scoring_prompt(context, "")
        next_cands    = step.get("next_candidates", [])
        next_emb      = np.array(step.get("next_embeddings", []),
                                  dtype=np.float32)
        next_beliefs, next_logits, next_logits_all = _score_and_select(
            next_scoring_prompt,
            next_cands,
            next_emb,
            model_id,
            k=K_NEXT,
            lambda_=lambda_,
        )

        # Write all scored logits back into the candidates dict so that
        # inspect_generation.py can re-run MMR at any lambda offline.
        step["conflict_logits_all"] = conflict_logits_all
        step["mystery_logits_all"]  = mystery_logits_all
        step["next_logits_all"]     = next_logits_all

        # --- what-happens-next improbability (needs summary_{t+1}) ---
        p_next_summary = None
        if idx + 1 < len(candidates_data):
            summary_t1  = candidates_data[idx + 1]["summary"]
            # Score summary_{t+1} as a continuation after "What happens next:"
            scoring_res = score_continuations(
                next_scoring_prompt,
                [" " + b for b in next_beliefs] + [" " + summary_t1],
                model_id,
            )
            softmax_logits = [r["logprob"] for r in scoring_res]
            import math
            max_lp         = max(softmax_logits)
            exps           = [math.exp(lp - max_lp) for lp in softmax_logits]
            s              = sum(exps)
            probs          = [e / s for e in exps]
            p_next_summary = probs[-1]   # probability of the actual summary

        # --- assemble Belief object ---
        belief = {
            "step":              T,
            "L_conflict_beliefs": conflict_beliefs,
            "L_conflict_logits":  conflict_logits,
            "L_mystery_beliefs":  mystery_beliefs,
            "L_mystery_logits":   mystery_logits,
            "L_next_beliefs":     next_beliefs,
            "L_next_logits":      next_logits,
            "p_next_summary":     p_next_summary,
        }
        beliefs_list.append(belief)

        # --- assemble Summary object ---
        # Character descriptions: carry forward from previous step if none new.
        prev_char_descs = (
            summaries_list[-1].get("char_descriptions", {})
            if summaries_list else {}
        )
        char_descs = dict(prev_char_descs)
        char_descs.update(step.get("char_descriptions", {}))

        summary_obj = {
            "step":              T,
            "summary":           step["summary"],
            "char_descriptions": char_descs,
        }
        summaries_list.append(summary_obj)

        print("done")

    # --- write outputs ---
    beliefs_path.parent.mkdir(parents=True, exist_ok=True)

    beliefs_out = {"metadata": run_metadata or {}, "steps": beliefs_list}
    with open(beliefs_path, "w") as f:
        json.dump(beliefs_out, f, ensure_ascii=False, indent=1)

    summaries_out = {"metadata": run_metadata or {}, "steps": summaries_list}
    with open(summaries_path, "w") as f:
        json.dump(summaries_out, f, ensure_ascii=False, indent=1)

    # Write the full scored logits back to the candidates file so that
    # inspect_generation.py can re-run MMR at any lambda without re-scoring.
    # (Each step dict was mutated in-place above; we now persist that update.)
    with open(work_path, "w") as f:
        json.dump(candidates_data, f, ensure_ascii=False, indent=1)

    print(f"Beliefs written to   {beliefs_path}")
    print(f"Summaries written to {summaries_path}")
    print(f"Scored logits added to {work_path}")


def _score_and_select(
    scoring_prompt: str,
    candidates: list[str],
    embeddings: np.ndarray,
    model_id: str,
    k: int,
    lambda_: float,
) -> tuple[list[str], list[float], list[float]]:
    """Score candidates, run MMR, return (selected_texts, selected_logits, all_logits).

    all_logits is the full scored log-probability for every candidate (same
    order as candidates), not just the MMR-selected k.  It is saved back to
    the candidates file by run_phase_score so that inspect_generation.py can
    re-run MMR offline at any lambda value without re-querying the model.

    Continuations stored in Phase 1 already have the anchor stripped and
    begin with a leading space, so the scorer sees them correctly.
    """
    if not candidates:
        return [], [], []

    # Prefix each candidate with a space (boundary policy: caller supplies
    # the exact leading whitespace to be scored).
    conts = [" " + c if not c.startswith(" ") else c for c in candidates]

    scored = score_continuations(scoring_prompt, conts, model_id)
    logits = [r["logprob"] for r in scored]

    if embeddings.size == 0 or len(embeddings) != len(candidates):
        # Recompute embeddings if missing/mismatched.
        embeddings = embed(candidates)

    selected_idx = mmr_select(candidates, logits, embeddings, k, lambda_)

    selected_texts  = [candidates[i] for i in selected_idx]
    selected_logits = [logits[i]     for i in selected_idx]
    return selected_texts, selected_logits, logits


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_characters(
    chunks_path: Path,
) -> tuple[str | None, list[dict] | None]:
    """Load the optional character JSON.

    Returns (premise, characters) where characters is a list of normalized
    {"name": str, "aliases": [str], "range": [int, int]} dicts (up to 7),
    or (None, None) if no character file is found.

    Looks first in <repo_root>/characters/<stem>_characters.json, then falls
    back to the old <chunks_dir>/<stem>_characters.json location.
    """
    stem = chunks_path.stem + "_characters.json"
    primary   = chunks_path.parent.parent / "characters" / stem
    secondary = chunks_path.parent / stem
    char_path = primary if primary.exists() else secondary
    if not char_path.exists():
        return None, None
    with open(char_path) as f:
        data = json.load(f)
    print(f"  loaded character data from {char_path}")
    premise = data.pop("premise", None)
    characters = [
        {"name": name, "aliases": info["aliases"], "range": info["range"]}
        for name, info in data.items()
    ]
    if len(characters) > 7:
        print(f"  warning: {len(characters)} characters found; using first 7")
        characters = characters[:7]
    return premise, characters


def _make_run_metadata(model_family: str, phase: str, lambda_: float) -> dict:
    """Collect run provenance for the output JSON header."""
    import platform
    meta = {
        "model_family": model_family,
        "phase": phase,
        "lambda": lambda_,
        "python": platform.python_version(),
    }
    try:
        import sentence_transformers as st
        meta["sentence_transformers_version"] = st.__version__
    except Exception:
        pass
    return meta


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Minimal sys.argv argument parsing (no argparse; matches project style).
    # -----------------------------------------------------------------------
    args = sys.argv[1:]

    def _flag(name):
        return name in args

    def _opt(name, default=None):
        try:
            return args[args.index(name) + 1]
        except (ValueError, IndexError):
            return default

    if not args or args[0].startswith("--"):
        sys.exit(
            "usage: python3 belief_generator.py <chunkedtexts/Book.json> "
            "--phase generate|score [--model-family mixtral|gemma] "
            "[--max-steps N] [--lambda 0.5] [--step T] [--dry-run] "
            "[--log-contexts]"
        )

    chunks_path   = Path(args[0]).expanduser()
    phase         = _opt("--phase", "generate")
    model_family  = _opt("--model-family", "mixtral")
    max_steps_str = _opt("--max-steps")
    max_steps     = int(max_steps_str) if max_steps_str else None
    lambda_       = float(_opt("--lambda", "0.5"))
    step_str      = _opt("--step", "0")
    dry_run       = _flag("--dry-run")
    log_contexts  = _flag("--log-contexts")

    if not chunks_path.exists():
        sys.exit(f"Input file not found: {chunks_path}")

    with open(chunks_path) as f:
        chunks = json.load(f)

    premise, characters = _load_characters(chunks_path)
    if characters is None:
        print("  no character file found; running without character context")

    book_stem = chunks_path.stem
    work_path       = Path("work")     / f"{book_stem}_candidates.json"
    beliefs_path    = Path("beliefs")  / f"{book_stem}_beliefs.json"
    summaries_path  = Path("beliefs")  / f"{book_stem}_summaries.json"

    if dry_run:
        # Rebuild minimal summaries from candidates if they exist.
        summaries = []
        if work_path.exists():
            cands = json.loads(work_path.read_text())
            summaries = [
                {"summary": c["summary"],
                 "char_descriptions": c.get("char_descriptions", {})}
                for c in cands
            ]
        T = int(step_str)
        context = build_context(chunks, summaries, characters, premise, T)
        print(f"\n{'='*70}\nStep {T} context prompt:\n{'='*70}")
        print(context)
        print(f"{'='*70}")
        sys.exit(0)

    it_model = (
        MODELS["gemma-it"]   if model_family == "gemma"
        else MODELS["qwen-it"]   if model_family == "qwen"
        else MODELS["mixtral-it"]
    )
    base_model = (
        MODELS["gemma-base"]  if model_family == "gemma"
        else MODELS["qwen-base"]  if model_family == "qwen"
        else MODELS["mixtral-base"]
    )

    run_meta = _make_run_metadata(model_family, phase, lambda_)

    if phase == "generate":
        print(f"Phase 1: generating candidates for {book_stem} "
              f"({model_family} instruct)")
        run_phase_generate(
            chunks, characters, premise, it_model, work_path,
            max_steps=max_steps,
            start_step=0,
            log_contexts=log_contexts,
        )

    elif phase == "score":
        print(f"Phase 2: scoring + MMR for {book_stem} "
              f"({model_family} base, λ={lambda_})")
        run_phase_score(
            chunks, characters, premise, base_model,
            work_path, beliefs_path, summaries_path,
            lambda_=lambda_,
            max_steps=max_steps,
            run_metadata=run_meta,
        )

    else:
        sys.exit(f"Unknown phase {phase!r}; use 'generate' or 'score'.")
