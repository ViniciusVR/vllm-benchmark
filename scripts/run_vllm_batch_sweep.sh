#!/bin/bash
#
# run_vllm_batch_sweep.sh
#
# vLLM batch-size sweep.
#
# Experiment:
#   - Prompt length: 512 tokens
#   - Output length: 512 generated tokens
#   - Batch sizes: 1, 2, 4, 8, 16, 32
#   - Concurrency: 1
#
# Run from project root:
#   sbatch scripts/run_vllm_batch_sweep.sh
#

#SBATCH --job-name=vllm_batch_sweep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --output=scripts/logs/vllm_batch_sweep_%j.out
#SBATCH --error=scripts/logs/vllm_batch_sweep_%j.err
#SBATCH --time=08:00:00

set -euo pipefail

echo "============================================================"
echo "Starting vLLM batch sweep"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "============================================================"

PROJECT_ROOT="$(pwd)"
cd "${PROJECT_ROOT}"

mkdir -p scripts/logs results

# Load cluster modules.
module purge
module load python/python-3.11.4-gcc-12.2.0 || module load python
module load cuda || true

source "${PROJECT_ROOT}/venv/bin/activate"

# Limit CPU threads to avoid cluster thread-limit errors.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face cache and authentication.
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

# Load Hugging Face token from project folder.
if [ -f "${PROJECT_ROOT}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${PROJECT_ROOT}/.hf_token")"
    echo "HF_TOKEN loaded from project .hf_token file"
else
    echo "WARNING: .hf_token not found. Gated models may fail."
fi

echo "============================================================"
echo "Loaded modules and GPU info"
echo "============================================================"
module list || true
nvidia-smi || true
which python
python --version

MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"

# Use a job-specific port to reduce collision risk if multiple jobs run.
PORT=$((8000 + SLURM_JOB_ID % 1000))
SERVER_URL="http://127.0.0.1:${PORT}"

echo "============================================================"
echo "Starting vLLM OpenAI-compatible server"
echo "============================================================"
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"

# Start vLLM server in the background.
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    > "scripts/logs/vllm_server_${SLURM_JOB_ID}.out" \
    2> "scripts/logs/vllm_server_${SLURM_JOB_ID}.err" &

SERVER_PID=$!

# Stop the background server when the job exits.
cleanup() {
    echo "Stopping vLLM server with PID ${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to become ready..."

# Poll the server until the OpenAI-compatible endpoint is available.
python - <<PY
import time
import requests
import sys

url = "${SERVER_URL}/v1/models"

for attempt in range(180):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            print("vLLM server is ready.")
            sys.exit(0)
    except Exception:
        pass

    time.sleep(5)

print("ERROR: vLLM server did not become ready in time.")
sys.exit(1)
PY

echo "============================================================"
echo "Running vLLM benchmark client"
echo "============================================================"

OUTPUT_CSV="results/vllm_batch_sweep.csv"
PROMPT_FILE="prompts/prompts_512.jsonl"

# Run sweep.
python python/benchmark_vllm_openai.py \
    --model-name "${MODEL_NAME}" \
    --server-url "${SERVER_URL}" \
    --prompt-files "${PROMPT_FILE}" \
    --output-csv "${OUTPUT_CSV}" \
    --sweep-name "batch_sweep" \
    --batch-sizes 1 2 4 8 16 32 \
    --concurrency-levels 1 \
    --max-new-tokens 512 \
    --output-length-mode fixed

echo "============================================================"
echo "vLLM batch sweep complete"
echo "Results: ${OUTPUT_CSV}"
echo "End time: $(date)"
echo "============================================================"
