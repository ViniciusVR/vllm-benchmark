#!/bin/bash
#
# setup_vllm_env.sbatch
#
# This job creates a Python virtual environment and installs all dependencies
# needed for the project, including:
#   - PyTorch
#   - Hugging Face Transformers
#   - vLLM
#   - pandas/tqdm/aiohttp/openai for benchmarking and CSV logging
#
# Run with:
#   sbatch scripts/setup_vllm_env.sbatch
#

#SBATCH --job-name=setup_vllm_env
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --output=scripts/logs/setup_vllm_env_%j.out
#SBATCH --error=scripts/logs/setup_vllm_env_%j.err
#SBATCH --time=02:00:00

set -euo pipefail

echo "============================================================"
echo "Starting vLLM/Hugging Face environment setup"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "============================================================"

# ---------------------------------------------------------------------
# Move to the project root.
# This assumes the sbatch command is submitted from the project root.
# Example:
#   sbatch scripts/setup_vllm_env.sbatch
# ---------------------------------------------------------------------
PROJECT_ROOT="$(pwd)"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------
# Create directories used by this project.
# logs/ stores Slurm output.
# prompts/ stores generated JSONL prompts.
# results/ stores benchmark CSV files.
# ---------------------------------------------------------------------
mkdir -p scripts/logs
mkdir -p prompts
mkdir -p results

# ---------------------------------------------------------------------
# Load cluster modules.
#
# Important:
# Your cluster may use a different module name for Python or CUDA.
# If this fails, run:
#   module avail python
#   module avail cuda
#
# Then replace the module names below with the correct ones.
# ---------------------------------------------------------------------
module purge

module load python/python-3.11.4-gcc-12.2.0 || module load python
module load cuda || true

echo "Loaded modules:"
module list || true

# ---------------------------------------------------------------------
# Create the virtual environment.
#
# The environment is stored inside the project folder so that all later
# SBATCH scripts can activate the same environment.
# ---------------------------------------------------------------------
VENV_DIR="${PROJECT_ROOT}/venv"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
else
    echo "Virtual environment already exists at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------
# Upgrade core Python packaging tools.
# ---------------------------------------------------------------------
python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------
# Install PyTorch.
#
# This uses the CUDA 12.1 PyTorch wheels, which are commonly compatible
# with H100 systems. If your cluster recommends a different CUDA version,
# adjust the index URL.
# ---------------------------------------------------------------------
python -m pip install --upgrade \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------
# Install project dependencies.
#
# transformers/accelerate: Hugging Face model loading and generate()
# vllm: high-throughput LLM serving engine
# pandas: CSV/data handling
# tqdm: progress bars
# aiohttp/openai: async HTTP and OpenAI-compatible vLLM API client support
# sentencepiece/protobuf: needed by many tokenizers
# ---------------------------------------------------------------------

python -m pip install --upgrade \
    accelerate \
    datasets \
    pandas \
    matplotlib \
    tqdm \
    aiohttp \
    requests \
    openai \
    sentencepiece \
    protobuf \
    huggingface_hub
    
python -m pip install \
    "transformers==4.46.3" \
    "tokenizers<0.21"
    
python -m pip install "vllm==0.6.4.post1"
# ---------------------------------------------------------------------
# Print environment verification information.
# This helps debug cluster setup issues from the Slurm log file.
# ---------------------------------------------------------------------
echo "============================================================"
echo "Environment verification"
echo "============================================================"

which python
python --version

python - <<'PY'
import torch
import transformers

print("Torch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
PY

python - <<'PY'
try:
    import vllm
    print("vLLM import: OK")
    print("vLLM version:", vllm.__version__)
except Exception as e:
    print("vLLM import failed:", repr(e))
    raise
PY

echo "============================================================"
echo "Setup complete"
echo "End time: $(date)"
echo "============================================================"