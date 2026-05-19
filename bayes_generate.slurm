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
# vLLM runs inside an Apptainer container to work around Delta's glibc 2.28.
# The Python client (belief_generator.py) runs in bayes-env as normal.
#
# Prerequisites in BAYES_ENV (NOT vllm — that lives in the container):
#   pip install openai transformers sentence-transformers numpy
#
# Both Mixtral models and sentence-transformers/all-MiniLM-L6-v2 must be
# pre-downloaded into /work/nvme/bdfx/.cache/huggingface.
#
# Submit from the repo root:
#   sbatch bayes_generate.slurm
# Chain scoring after:
#   sbatch --dependency=afterok:<this_jobid> bayes_score.slurm
# ===========================================================================

# ---------------------------------------------------------------------------
# Configuration — edit before submitting
# ---------------------------------------------------------------------------
BOOK="ShesNotSorryMaryKubica"
BAYES_ENV="$SLURM_SUBMIT_DIR/bayes-env/bin/activate"
# Apptainer image built from docker://vllm/vllm-openai:latest
# Use Ziliang's existing image or build your own:
#   apptainer build vllm_openai.sif docker://vllm/vllm-openai:latest
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

# Clear host compiler flags that can conflict inside the container.
unset CC CXX FC CFLAGS CXXFLAGS LDFLAGS

cd "$SLURM_SUBMIT_DIR"

MODEL_IT="mistralai/Mixtral-8x7B-Instruct-v0.1"
VLLM_PORT=8000

# --- start vLLM server via Apptainer (instruct model, 4-way TP) ------------
# Mixtral-8x7B-Instruct is ~93 GB in bfloat16; 4 × A100 40 GB = 160 GB.
echo "Starting vLLM server for ${MODEL_IT} ..."
apptainer exec --nv \
    --bind "${HF_HOME}:${HF_HOME}" \
    --bind /tmp:/tmp \
    --bind /dev/shm:/dev/shm \
    "$SIF_IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_IT" \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.90 \
        --dtype bfloat16 \
        --enforce-eager \
        --enable-prefix-caching \
        --host 127.0.0.1 \
        --port "$VLLM_PORT" \
    > vllm-generate-${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

# --- wait for server to be healthy (up to 15 minutes) ----------------------
MAX_WAIT=900
WAITED=0
until curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s; aborting."
        echo "Check vllm-generate-${SLURM_JOB_ID}.log for details."
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
