#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Deploy SVG competition pipeline to BigPurple HPC
# Run from your Mac:  bash deploy_to_hpc.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

HPC="bigpurple"   # Uses SSH config alias
REMOTE_DIR="/gpfs/data/schultebrauckslab/Users/yadavj02/svg-competition"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Step 1: Upload project to BigPurple ==="
echo "Local:  ${LOCAL_DIR}"
echo "Remote: ${HPC}:${REMOTE_DIR}"

# Create remote directory
ssh "${HPC}" "mkdir -p ${REMOTE_DIR}/{data,logs,output,checkpoints}"

# Upload all project files
scp "${LOCAL_DIR}/config.py" \
    "${LOCAL_DIR}/utils.py" \
    "${LOCAL_DIR}/prepare_data.py" \
    "${LOCAL_DIR}/train.py" \
    "${LOCAL_DIR}/generate.py" \
    "${LOCAL_DIR}/evaluate_local.py" \
    "${LOCAL_DIR}/setup_env.sh" \
    "${LOCAL_DIR}/slurm_train.sh" \
    "${LOCAL_DIR}/slurm_infer.sh" \
    "${LOCAL_DIR}/submit.sh" \
    "${HPC}:${REMOTE_DIR}/"

# Upload competition data
echo ""
echo "=== Step 2: Upload competition data ==="
scp "${LOCAL_DIR}/data/train.csv" \
    "${LOCAL_DIR}/data/test.csv" \
    "${LOCAL_DIR}/data/sample_submission.csv" \
    "${HPC}:${REMOTE_DIR}/data/"

# Upload Kaggle credentials
echo ""
echo "=== Step 3: Upload Kaggle credentials ==="
ssh "${HPC}" "mkdir -p ~/.kaggle && chmod 700 ~/.kaggle"
scp ~/.kaggle/kaggle.json "${HPC}:~/.kaggle/kaggle.json"
ssh "${HPC}" "chmod 600 ~/.kaggle/kaggle.json"

echo ""
echo "=== Step 4: Set up conda environment on HPC ==="
echo "This runs the setup script on BigPurple (may take 5-10 min)..."
ssh "${HPC}" "cd ${REMOTE_DIR} && bash setup_env.sh" 2>&1 | tail -20

echo ""
echo "============================================="
echo " Deployment complete!"
echo "============================================="
echo ""
echo "Now SSH into BigPurple and run:"
echo "  ssh bigpurple"
echo "  cd ${REMOTE_DIR}"
echo "  sbatch slurm_train.sh"
echo ""
echo "Monitor with:"
echo "  squeue -u yadavj02"
echo "  tail -f logs/train_*.out"
