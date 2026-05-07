#!/bin/bash
#
# run_hf_concurrency_sweep.sbatch
#
# Hugging Face generate() simulated concurrency sweep.
#
# Experiment:
#   - Prompt length: 512 tokens
#   - Output length: 512 generated tokens
#   - Batch size: 1
#   - Simulated concurrency: 1, 2, 4, 8, 16, 32
#
# Important:
# Hugging Face generate() is not a serving system. This sweep uses Python
# threads to create request pressure, but it does not reproduce vLLM's dynamic
# batching or request scheduler.
#
# Run from project root:
#   sbatch scripts/run_hf_concurrency_sweep.sbatch
#

#SBATCH --job-name=hf_concurrency_sweep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --output=scripts/logs/hf_concurrency_sweep_%j.out
#SBATCH --error=scripts/logs/hf_concurrency_sweep_%j.err
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT="$(pwd)"
cd "${PROJECT_ROOT}"

mkdir -p scripts/logs results

module purge
module load python/python-3.11.4-gcc-12.2.0 || module load python
module load cuda || true

source "${PROJECT_ROOT}/venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

echo "============================================================"
echo "Starting HF concurrency sweep"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "============================================================"

nvidia-smi || true
which python
python --version

MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
PROMPT_FILE="prompts/prompts_512.jsonl"
OUTPUT_CSV="results/hf_concurrency_sweep.csv"

python python/benchmark_hf_generate.py \
    --model-name "${MODEL_NAME}" \
    --prompt-files "${PROMPT_FILE}" \
    --output-csv "${OUTPUT_CSV}" \
    --sweep-name "concurrency_sweep" \
    --batch-sizes 1 \
    --concurrency-levels 1 2 4 8 16 32 \
    --max-new-tokens 512 \
    --output-length-mode fixed \
    --dtype float16

echo "============================================================"
echo "HF concurrency sweep complete"
echo "Results: ${OUTPUT_CSV}"
echo "End time: $(date)"
echo "============================================================"
