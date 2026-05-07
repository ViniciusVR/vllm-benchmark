#!/bin/bash
#
# setup_env.sh
#
# This job creates a Python virtual environment and installs all dependencies
# needed for the project.
#
# Run with:
#   sbatch scripts/setup_env.sh
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

# Move to the project root.
PROJECT_ROOT="$(pwd)"
cd "${PROJECT_ROOT}"

# Create project output folders.
mkdir -p scripts/logs
mkdir -p prompts
mkdir -p results

# Load cluster modules.
module purge

module load python/python-3.11.4-gcc-12.2.0 || module load python
module load cuda || true

echo "Loaded modules:"
module list || true

# Create or open the project virtual environment.
VENV_DIR="${PROJECT_ROOT}/venv"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
else
    echo "Virtual environment already exists at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# Upgrade Python packaging tools.
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA wheels.
python -m pip install --upgrade \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install general project dependencies.
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
    
# Install the Transformers stack version compatible with this vLLM version.
python -m pip install \
    "transformers==4.46.3" \
    "tokenizers<0.21"
    
# Install the pinned vLLM version used in the experiments.
python -m pip install "vllm==0.6.4.post1"

echo "============================================================"
echo "Environment verification"
echo "============================================================"

which python
python --version

# Verify PyTorch, Transformers, and CUDA visibility.
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

# Verify vLLM import.
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