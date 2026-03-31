#!/usr/bin/env bash
#SBATCH --job-name=svg-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=a100_short        # A100 partition, 3-day limit
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ──────────────────────────────────────────────────────────────
# SVG Training Job — BigPurple A100
# ──────────────────────────────────────────────────────────────
set -euo pipefail

LAB_ENV="/gpfs/data/schultebrauckslab/Users/yadavj02/conda_envs/svg-gen"
PROJECT="/gpfs/data/schultebrauckslab/Users/yadavj02/svg-competition"
PY="${LAB_ENV}/bin/python"

export PATH="${LAB_ENV}/bin:$PATH"

cd "${PROJECT}"
mkdir -p logs

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
nvidia-smi
${PY} -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Train
echo "=== Starting training ==="
${PY} train.py

echo "=== Job finished: $(date) ==="
