#!/bin/bash
#
# show_results.sh
#
# Creates grouped bar charts comparing Hugging Face generate() and vLLM
# using requests/sec as the plotted metric.
#
# Run from project root:
#   sbatch scripts/show_results.sh
#

#SBATCH --job-name=plot_requests_per_sec
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=scripts/logs/plot_requests_per_sec_%j.out
#SBATCH --error=scripts/logs/plot_requests_per_sec_%j.err
#SBATCH --time=00:15:00

set -euo pipefail

echo "============================================================"
echo "Creating requests/sec figures"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "============================================================"

PROJECT_ROOT="$(pwd)"
cd "${PROJECT_ROOT}"

mkdir -p scripts/logs
mkdir -p figures

# Load Python and activate the project environment.
module purge
module load python/python-3.11.4-gcc-12.2.0 || module load python

source "${PROJECT_ROOT}/venv/bin/activate"

echo "============================================================"
echo "Python environment"
echo "============================================================"
which python
python --version

echo "============================================================"
echo "Running plotting script"
echo "============================================================"

# Generate requests/sec plots for all sweeps.
python python/plot_sweep_results.py \
    --results-dir results \
    --output-dir figures \
    --metric requests_per_sec

echo "============================================================"
echo "Plotting complete"
echo "Figures saved in: figures/"
echo "End time: $(date)"
echo "============================================================"