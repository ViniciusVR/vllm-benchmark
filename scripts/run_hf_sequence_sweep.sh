#!/bin/bash
#
# run_hf_sequence_sweep.sbatch
#
# Hugging Face generate() sequence-length sweep.
#
# Experiment:
#   - Prompt lengths: 128, 256, 512, 1024, 2048
#   - Output length: equal to prompt target length
#   - Batch size: 1
#   - Concurrency: 1
#
# This isolates the cost of longer contexts and longer outputs without adding
# batching or simulated concurrent request pressure.
#
# Run from project root:
#   sbatch scripts/run_hf_sequence_sweep.sbatch
#

#SBATCH --job-name=hf_sequence_sweep
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --output=scripts/logs/hf_sequence_sweep_%j.out
#SBATCH --error=scripts/logs/hf_sequence_sweep_%j.err
#SBATCH --time=08:00:00

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
echo "Starting HF sequence sweep"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "============================================================"

nvidia-smi || true
which python
python --version

MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
OUTPUT_CSV="results/hf_sequence_sweep.csv"

python python/benchmark_hf_generate.py \
    --model-name "${MODEL_NAME}" \
    --prompt-files \
        prompts/prompts_128.jsonl \
        prompts/prompts_256.jsonl \
        prompts/prompts_512.jsonl \
        prompts/prompts_1024.jsonl \
        prompts/prompts_2048.jsonl \
    --output-csv "${OUTPUT_CSV}" \
    --sweep-name "sequence_sweep" \
    --batch-sizes 1 \
    --concurrency-levels 1 \
    --output-length-mode prompt_target \
    --dtype float16

echo "============================================================"
echo "HF sequence sweep complete"
echo "Results: ${OUTPUT_CSV}"
echo "End time: $(date)"
echo "============================================================"
