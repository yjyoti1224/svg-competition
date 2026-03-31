#!/usr/bin/env bash
#SBATCH --job-name=svg-infer
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --partition=a100_dev          # A100 dev partition, 4-hour limit
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ──────────────────────────────────────────────────────────────
# SVG Inference Job — generate submission.csv
# ──────────────────────────────────────────────────────────────
set -euo pipefail

LAB_ENV="/gpfs/data/schultebrauckslab/Users/yadavj02/conda_envs/svg-gen"
PROJECT="/gpfs/data/schultebrauckslab/Users/yadavj02/svg-competition"
PY="${LAB_ENV}/bin/python"

export PATH="${LAB_ENV}/bin:$PATH"

cd "${PROJECT}"
mkdir -p logs output

echo "=== Inference started: $(date) ==="
nvidia-smi

${PY} generate.py

echo "=== Inference done: $(date) ==="
echo "Submission file: output/submission.csv"
