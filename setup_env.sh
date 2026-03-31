#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Environment setup for SVG pipeline — BigPurple (NYU HPC)
# Run ONCE:  bash setup_env.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

ENV_NAME="svg-gen"

echo "==> Loading BigPurple modules"
module purge
module load anaconda3/gpu
module load cuda/12.1

echo "==> Creating conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python=3.11 -y
source activate "${ENV_NAME}"

echo "==> Installing PyTorch (CUDA 12.x)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing core training packages"
pip install \
    transformers \
    accelerate \
    peft \
    trl \
    bitsandbytes \
    datasets \
    huggingface_hub \
    sentencepiece \
    protobuf

echo "==> Installing Unsloth (fast LoRA training)"
pip install unsloth

echo "==> Installing inference / eval utilities"
pip install \
    cairosvg \
    lxml \
    pandas \
    numpy \
    tqdm \
    kaggle \
    scikit-image \
    Pillow

echo "==> Installing flash-attention for A100"
pip install flash-attn --no-build-isolation

echo ""
echo "============================================="
echo " Environment '${ENV_NAME}' ready on BigPurple!"
echo "============================================="
