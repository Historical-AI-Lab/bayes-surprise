#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdfx-delta-gpu
#SBATCH --time=08:00:00
#SBATCH --job-name=bayes-score
#SBATCH --error=bayes-score-%j.err
#SBATCH --output=bayes-score-%j.out
#SBATCH --gpus-per-node=4
#SBATCH --mem=128g
#SBATCH --constraint="scratch"

# ===========================================================================
# Phase 2: score + MMR (base model)
#
# Starts a vLLM server serving Mixtral-8x7B (base), then runs
# belief_generator.py --phase score, then shuts the server down.
#
# Requires work/<BOOK>_candidates.json written by bayes_generate.slurm.
#
# Submit from the repo root after generation is complete:
#   sbatch bayes_score.slurm
#
# Or chain it automatically:
#   sbatch --dependency=afterok:<generate_jobid> bayes_score.slurm
#
# Prefix caching is critical here: scoring 20 continuations per latent
# per step reuses the same long prompt, so the KV cache is shared across
# almost all scoring calls at each step.
# ===========================================================================

# ---------------------------------------------------------------------------
# Configuration — edit these lines before submitting
# ---------------------------------------------------------------------------
BOOK="ShesNotSorryMaryKubica"          # must match the generate job
BAYES_ENV="$SLURM_SUBMIT_DIR/bayes-env/bin/activate"
LAMBDA="0.5"                           # MMR diversity/relevance tradeoff
# ---------------------------------------------------------------------------

echo "Job starting on $(hostname)"
date

module reset
module load cudatoolkit/25.3_12.8
module list

source "$BAYES_ENV"

export HF_HOME=/work/nvme/bdfx/.cache/huggingface
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"

MODEL_BASE="mistralai/Mixtral-8x7B-v0.1"
VLLM_PORT=8000

# --- start vLLM server (base model, 4-way tensor parallelism) --------------
# Mixtral-8x7B base is ~93 GB in bfloat16; 4 × A100 40 GB = 160 GB.
# Prefix caching (--enable-prefix-caching) is especially important here:
# the long prompt is shared across all 20 scoring calls per latent per step.
echo "Starting vLLM server for ${MODEL_BASE} ..."
vllm serve "$MODEL_BASE" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    > vllm-score-${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

# --- wait for server to be healthy (up to 10 minutes) ----------------------
MAX_WAIT=600
WAITED=0
until curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s; aborting."
        kill "$VLLM_PID" 2>/dev/null
        exit 1
    fi
    echo "  waiting for vLLM server ... (${WAITED}s elapsed)"
    sleep 15
    WAITED=$((WAITED + 15))
done
echo "vLLM server ready after ${WAITED}s."

# --- Phase 2: scoring + MMR ------------------------------------------------
export BAYES_VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

python -u belief_generator.py \
    "chunkedtexts/${BOOK}.json" \
    --phase score \
    --model-family mixtral \
    --lambda "$LAMBDA"

EXIT_CODE=$?

# --- shut down vLLM --------------------------------------------------------
echo "Stopping vLLM server (PID ${VLLM_PID}) ..."
kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null

echo "Scoring phase finished with exit code ${EXIT_CODE}."
date
exit $EXIT_CODE
