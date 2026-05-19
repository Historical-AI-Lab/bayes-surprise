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
# vLLM runs inside an Apptainer container to work around Delta's glibc 2.28.
# Prefix caching is especially important here: the long prompt is shared
# across all 20 scoring calls per latent per step.
#
# Requires work/<BOOK>_candidates.json written by bayes_generate.slurm.
# Chain automatically: sbatch --dependency=afterok:<generate_jobid> bayes_score.slurm
# ===========================================================================

# ---------------------------------------------------------------------------
# Configuration — edit before submitting
# ---------------------------------------------------------------------------
BOOK="ShesNotSorryMaryKubica"
BAYES_ENV="$SLURM_SUBMIT_DIR/bayes-env/bin/activate"
LAMBDA="0.5"
SIF_IMAGE="/work/hdd/bdfx/zqiu1/models/vllm-openai_latest.sif"
# ---------------------------------------------------------------------------

echo "Job starting on $(hostname)"
date

module reset
module load cudatoolkit/25.3_12.8
module list

source "$BAYES_ENV"

export HF_HOME=/work/nvme/bdfx/.cache/huggingface
export HF_HUB_OFFLINE=1

unset CC CXX FC CFLAGS CXXFLAGS LDFLAGS

cd "$SLURM_SUBMIT_DIR"

MODEL_BASE="mistralai/Mixtral-8x7B-v0.1"
VLLM_PORT=8000

# --- start vLLM server via Apptainer (base model, 4-way TP) ----------------
echo "Starting vLLM server for ${MODEL_BASE} ..."
apptainer exec --nv \
    --bind "${HF_HOME}:${HF_HOME}" \
    --bind /tmp:/tmp \
    --bind /dev/shm:/dev/shm \
    "$SIF_IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_BASE" \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.90 \
        --dtype bfloat16 \
        --enforce-eager \
        --enable-prefix-caching \
        --host 127.0.0.1 \
        --port "$VLLM_PORT" \
    > vllm-score-${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

# --- wait for server to be healthy (up to 15 minutes) ----------------------
MAX_WAIT=900
WAITED=0
until curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s; aborting."
        echo "Check vllm-score-${SLURM_JOB_ID}.log for details."
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
