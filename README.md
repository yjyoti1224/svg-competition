# SVG Generation from Text Prompts

NYU DL Spring 2026 Kaggle Competition — Team J_and_J

## Task

Generate SVG code from natural language text descriptions. Scored by CLIP similarity between rendered SVG and text prompt.

Competition: `dl-spring-2026-svg-generation-from-text-prompts-extended-deadline`

## Model

- **Base model:** Qwen/Qwen3.5-2B (1.5B parameters, within 2B limit)
- **Fine-tuning:** QLoRA (4-bit NF4 quantization) via Unsloth
- **LoRA config:** r=16, alpha=32, dropout=0.0
- **Trainable parameters:** 20.2M (1.32% of total)

## Repository Structure

```
config.py          — Central configuration (paths, hyperparameters)
utils.py           — SVG extraction, validation, post-processing
prepare_data.py    — Data loading, filtering, tokenization
train.py           — QLoRA fine-tuning with Unsloth + SFTTrainer
generate.py        — Batched inference with retry logic
svg_pipeline.ipynb — End-to-end notebook (executed with outputs)
setup_env.sh       — Environment setup script for HPC
environment.yml    — Conda environment specification
```

## Model Weights

Fine-tuned model weights (merged, ~4.3GB):
**[Google Drive link]** *(see submission for link)*

## Reproducing Results

### 1. Environment Setup

```bash
# Option A: Conda (recommended for HPC)
conda env create -f environment.yml
conda activate svg-gen

# Option B: pip
pip install -r requirements.txt
```

### 2. Download Competition Data

```bash
# Requires Kaggle API key at ~/.kaggle/kaggle.json
python prepare_data.py
```

This downloads train/test CSVs, filters training data against competition constraints, and saves processed datasets.

### 3. Training

```bash
python train.py
```

- Trains for 5 epochs on A100 GPU (~18h total)
- Saves LoRA adapter to `checkpoints/final/`
- Saves merged model to `checkpoints/merged/`
- Supports checkpoint resumption automatically

### 4. Inference

```bash
python generate.py --use-merged --checkpoint checkpoints/merged/
```

- Batched generation (batch_size=8) at temperature 0.6
- Failed SVGs retried individually at descending temperatures (0.5, 0.4, 0.3, 0.2)
- Post-processing enforces competition constraints (dimensions, allowed tags, path limits)
- Outputs `submission.csv`

### 5. Configuration

All hyperparameters are centralized in `config.py`. Key settings:

| Parameter | Value |
|-----------|-------|
| Max sequence length | 2048 |
| Effective batch size | 16 (4 x 4 grad accum) |
| Learning rate | 2e-4 (cosine schedule) |
| Max new tokens (inference) | 1536 |
| SVG dimensions | 256x256 |

## Results

| Submission | Score |
|------------|-------|
| 3-epoch model | 14.92 |
| 5-epoch merged | **15.13** |

## Hardware

- Training & inference: NVIDIA A100 80GB (NYU BigPurple HPC)
- SLURM job scheduler with `a100_short` (12h) and `a100_long` (24h) partitions
