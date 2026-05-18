# Bayesian Surprise in Novels

This project measures **Bayesian surprise** — KL divergence over a model's belief distribution — at each passage of a novel, and compares it against simpler surprise baselines: embedding novelty, improbability of the next summary, and refined improbability (what-happens-next predictions). Ground truth comes from reader reviews in `reviews.txt`, which contain marked spoilers identifying passages humans found surprising. The full experimental design is in `bayesian-surprise-plan.md`.

## Main parts

**Text extraction and chunking**
- `extractor.py` — reads an EPUB (directly as a zip, not via ebooklib) and writes a structured JSON to `rawtexts/` plus a flat `.txt` for eyeballing. Run as `python3 extractor.py <book.epub>`.
- `chunker.py` — splits the structured JSON into ~1000-word chunks, breaking at paragraph boundaries, and writes to `chunkedtexts/`. Only body sections (`kind == "body"`) are chunked.

**Belief generation pipeline** (the core of the project)
- `belief_generator.py` — two-phase pipeline. Phase 1 (`--phase generate`) uses an instruction-tuned model to elicit 20 candidate beliefs per latent (central conflict, central mystery) plus what-happens-next predictions, and writes raw candidates and embeddings to `work/`. Phase 2 (`--phase score`) scores continuation log-likelihoods with a base model, runs MMR selection (20→8 for beliefs, 10→7 for what-next), and writes final Belief and Summary objects to `beliefs/`. Use `--dry-run --step T` to inspect the context prompt for any step without model calls; use `--log-contexts` during generation to save every prompt to `work/<Book>_contexts.json`.
- `model_query.py` — low-level primitives: `generate_answer` (instruct model via vLLM chat-completions) and `score_continuations` (base model via vLLM prompt log-probabilities).
- `mmr.py` — Maximal Marginal Relevance implementation used to reduce oversampled candidates to a diverse final set.

**Character context**
- `characters/` — optional per-book JSON files (`<Book>_characters.json`) specifying a premise line and up to seven characters, each with an alias list and an inclusive chunk-index range. Used by `belief_generator.py` to include character descriptions in the context prompt when a character is mentioned.

**HPC scripts**
- `bayes_generate.slurm` / `bayes_score.slurm` — Slurm scripts for NCSA Delta (partition `gpuA100x4`, account `bdfx-delta-gpu`). Each starts a vLLM server on 4 × A100 40 GB GPUs, runs the corresponding phase, then shuts the server down.
- `slurm/` — additional server-only sbatch scripts (serve instruct / serve base) for interactive use.

**Inspection and tests**
- `tools/inspect_generation.py` — shows every candidate for a given step with its scored log-probability and MMR selection at a chosen lambda, enabling offline comparison of different lambda values. Requires Phase 2 to have run (which writes full scored logits back to the candidates file).
- `tests/` — pytest suite covering MMR correctness, response parsing, and character loading/range/alias logic.

## Workflow

```
# 1. Extract and chunk a book
python3 extractor.py ~/calibrebooks/Author/Title/book.epub
python3 chunker.py rawtexts/BookStem.json

# 2. (Optional) create characters/BookStem_characters.json by hand or with an LLM

# 3. Inspect prompts locally before spending allocation
python3 belief_generator.py chunkedtexts/BookStem.json --dry-run --step 0

# 4. Run on Delta (smoke test: add --max-steps 10 --log-contexts first)
sbatch bayes_generate.slurm
sbatch --dependency=afterok:<generate_jobid> bayes_score.slurm

# 5. Inspect MMR lambda tradeoff locally
python3 tools/inspect_generation.py BookStem --step 3 --lambda 0.5
```
