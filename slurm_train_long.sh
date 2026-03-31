#!/usr/bin/env bash
#SBATCH --job-name=svg-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=a100_long         # 28-day limit — enough to finish
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ──────────────────────────────────────────────────────────────
# SVG Training Job — Resume from checkpoint on a100_long
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
echo ""

# List existing checkpoints
echo "=== Existing checkpoints ==="
ls -d checkpoints/checkpoint-* 2>/dev/null || echo "No checkpoints found (starting fresh)"
echo ""

# Train (will auto-resume from latest checkpoint if found)
echo "=== Starting training ==="
${PY} train.py

echo "=== Job finished: $(date) ==="
