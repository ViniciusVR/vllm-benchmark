#!/bin/bash
#
# run_sample_hf_outputs.sbatch
#
# Runs a small qualitative Hugging Face generate() sample and saves a CSV with:
#   - full text prompt
#   - generated output
#   - token counts
#   - latency
#   - throughput metrics
#
# Run from project root:
#   sbatch scripts/run_sample_hf_outputs.sbatch
#

#SBATCH --job-name=sample_hf_outputs
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --output=scripts/logs/sample_hf_outputs_%j.out
#SBATCH --error=scripts/logs/sample_hf_outputs_%j.err
#SBATCH --time=01:00:00

set -euo pipefail

echo "============================================================"
echo "Starting qualitative HF output sampling with metrics"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "============================================================"

PROJECT_ROOT="$(pwd)"
cd "${PROJECT_ROOT}"

mkdir -p scripts/logs
mkdir -p results

module purge
module load python/python-3.11.4-gcc-12.2.0 || module load python
module load cuda || true

echo "============================================================"
echo "Loaded modules"
echo "============================================================"
module list || true

echo "============================================================"
echo "GPU info"
echo "============================================================"
nvidia-smi || true

source "${PROJECT_ROOT}/venv/bin/activate"

# Limit CPU-side threading to avoid OpenBLAS/MKL process-limit errors.
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

if [ -f "${PROJECT_ROOT}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${PROJECT_ROOT}/.hf_token")"
    echo "HF_TOKEN loaded from project .hf_token file"
else
    echo "WARNING: .hf_token not found. Gated models may fail."
fi

# ---------------------------------------------------------------------
# Sample generation configuration.
#
# Use batch size 1 for report examples because per-prompt latency and
# throughput are easiest to interpret.
# ---------------------------------------------------------------------
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
PROMPT_FILE="prompts/prompts_512.jsonl"

OUTPUT_CSV="results/sample_outputs_512_with_metrics.csv"

NUM_SAMPLES=10
MAX_NEW_TOKENS=256
BATCH_SIZE=1
SEED=42

echo "============================================================"
echo "Sample generation config"
echo "============================================================"
echo "Model: ${MODEL_NAME}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Output CSV: ${OUTPUT_CSV}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Seed: ${SEED}"

echo "============================================================"
echo "Python environment check"
echo "============================================================"
which python
python --version

python - <<'PY'
import torch
import transformers

print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "============================================================"
echo "Running sample_hf_outputs_with_metrics.py"
echo "============================================================"

python python/sample_hf_outputs_with_metrics.py \
    --model-name "${MODEL_NAME}" \
    --prompt-file "${PROMPT_FILE}" \
    --output-csv "${OUTPUT_CSV}" \
    --num-samples "${NUM_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --dtype float16

echo "============================================================"
echo "Qualitative HF output sampling complete"
echo "CSV results: ${OUTPUT_CSV}"
echo "End time: $(date)"
echo "============================================================"
