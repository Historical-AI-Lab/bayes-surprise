#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdfx-delta-gpu
#SBATCH --time=08:00:00
#SBATCH --job-name=bayes-generate
#SBATCH --error=bayes-generate-%j.err
#SBATCH --output=bayes-generate-%j.out
#SBATCH --gpus-per-node=4
#SBATCH --mem=128g
#SBATCH --constraint="scratch"

# ===========================================================================
# Phase 1: generate candidates (instruct model)
#
# Starts a vLLM server serving Mixtral-8x7B-Instruct, then runs
# belief_generator.py --phase generate, then shuts the server down.
#
# Submit from the repo root:
#   sbatch bayes_generate.slurm
#
# To chain scoring immediately after:
#   sbatch --dependency=afterok:<this_jobid> bayes_score.slurm
#
# Prerequisites — in BAYES_ENV you need:
#   pip install vllm openai transformers sentence-transformers numpy
# Both Mixtral models and all-MiniLM-L6-v2 must be pre-downloaded into
#   /work/nvme/bdfx/.cache/huggingface
# (sentence_transformers uses HF_HOME since version 3+; if you are on an
#  older version, also set SENTENCE_TRANSFORMERS_HOME accordingly)
# ===========================================================================

# ---------------------------------------------------------------------------
# Configuration — edit these two lines before submitting
# ---------------------------------------------------------------------------
BOOK="ShesNotSorryMaryKubica"          # stem of chunkedtexts/<BOOK>.json
BAYES_ENV="$SLURM_SUBMIT_DIR/bayes-env/bin/activate"
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

MODEL_IT="mistralai/Mixtral-8x7B-Instruct-v0.1"
VLLM_PORT=8000

# --- start vLLM server (instruct model, 4-way tensor parallelism) ----------
# Mixtral-8x7B-Instruct is ~93 GB in bfloat16; 4 × A100 40 GB = 160 GB.
# gpu-memory-utilization 0.90 leaves ~16 GB free for KV cache per GPU.
echo "Starting vLLM server for ${MODEL_IT} ..."
vllm serve "$MODEL_IT" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    > vllm-generate-${SLURM_JOB_ID}.log 2>&1 &
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

# --- Phase 1: generation ---------------------------------------------------
export BAYES_VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

python -u belief_generator.py \
    "chunkedtexts/${BOOK}.json" \
    --phase generate \
    --model-family mixtral

EXIT_CODE=$?

# --- shut down vLLM --------------------------------------------------------
echo "Stopping vLLM server (PID ${VLLM_PID}) ..."
kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null

echo "Generation phase finished with exit code ${EXIT_CODE}."
date
exit $EXIT_CODE
