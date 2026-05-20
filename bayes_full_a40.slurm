#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdfx-delta-gpu
#SBATCH --time=7:00:00
#SBATCH --job-name=bayes-full-a40
#SBATCH --error=bayes-full-a40-%j.err
#SBATCH --output=bayes-full-a40-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --constraint="scratch"

# ===========================================================================
# Full run: Qwen2.5-7B on a single A40, all chunks, both phases.
#
# Phase 1 (generate) is crash-resumable: it skips steps already written to
# work/<BOOK>_candidates.json, so a resubmit picks up where it left off.
# Phase 2 (score) writes everything at the end and is NOT step-resumable;
# if it is interrupted, Phase 2 must be re-run from the start.
#
# Submit from the repo root:
#   sbatch bayes_full_a40.slurm
# ===========================================================================

BOOK="ShesNotSorryMaryKubica"
BAYES_ENV="$SLURM_SUBMIT_DIR/bayes-env/bin/activate"
SIF_IMAGE="/work/hdd/bdfx/zqiu1/models/vllm-openai_latest.sif"

echo "Job starting on $(hostname)"
date

module reset
module load cudatoolkit/25.3_12.8
module list

source "$BAYES_ENV"

export HF_HOME=/work/nvme/bdfx/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT=/tmp/vllm-cache-${SLURM_JOB_ID}

unset CC CXX FC CFLAGS CXXFLAGS LDFLAGS

cd "$SLURM_SUBMIT_DIR"

VLLM_PORT=8000
VLLM_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export BAYES_VLLM_BASE_URL="$VLLM_URL"

# ---------------------------------------------------------------------------
# Helper: start vLLM via Apptainer, wait for health, print PID
# ---------------------------------------------------------------------------
start_vllm() {
    local model="$1"
    local logfile="$2"
    echo "Starting vLLM server for ${model} ..." >&2
    apptainer exec --nv \
        --bind "${HF_HOME}:${HF_HOME}" \
        --bind /tmp:/tmp \
        --bind /dev/shm:/dev/shm \
        "$SIF_IMAGE" \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$model" \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.90 \
            --dtype bfloat16 \
            --enable-prefix-caching \
            --host 127.0.0.1 \
            --port "$VLLM_PORT" \
        > "$logfile" 2>&1 &
    local pid=$!
    local waited=0
    until curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; do
        if [ $waited -ge 300 ]; then
            echo "ERROR: vLLM did not start within 300s. Check ${logfile}." >&2
            kill "$pid" 2>/dev/null
            exit 1
        fi
        echo "  waiting for vLLM ... (${waited}s)" >&2
        sleep 10
        waited=$((waited + 10))
    done
    echo "vLLM ready after ${waited}s." >&2
    echo "$pid"
}

# ---------------------------------------------------------------------------
# Phase 1: generate (instruct model)
# ---------------------------------------------------------------------------
VLLM_PID=$(start_vllm "Qwen/Qwen2.5-7B-Instruct" "vllm-full-a40-generate-${SLURM_JOB_ID}.log")

python -u belief_generator.py \
    "chunkedtexts/${BOOK}.json" \
    --phase generate \
    --model-family qwen

GEN_EXIT=$?
echo "Generation exited with code ${GEN_EXIT}."

kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
echo "Instruct server stopped."

if [ $GEN_EXIT -ne 0 ]; then
    echo "Generation failed; skipping scoring."
    exit $GEN_EXIT
fi

# ---------------------------------------------------------------------------
# Phase 2: score (base model)
# ---------------------------------------------------------------------------
VLLM_PID=$(start_vllm "Qwen/Qwen2.5-7B" "vllm-full-a40-score-${SLURM_JOB_ID}.log")

python -u belief_generator.py \
    "chunkedtexts/${BOOK}.json" \
    --phase score \
    --model-family qwen

SCORE_EXIT=$?
echo "Scoring exited with code ${SCORE_EXIT}."

kill "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
echo "Base server stopped."

echo "Full run complete."
date
exit $SCORE_EXIT
