"""
Central configuration for SVG generation pipeline.
NYU DL Spring 2026 — SVG Generation from Text Prompts
Adjust paths and hyperparameters here — everything else imports from this file.
"""
import os

# ──────────────────────────── Paths ────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Competition files (downloaded via kaggle CLI)
KAGGLE_COMP = "dl-spring-2026-svg-generation"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# ──────────────────────────── Model ────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-2B"  # ≤2B as per midterm rules

LOAD_IN_4BIT = True           # QLoRA 4-bit quantization
MAX_SEQ_LENGTH = 2048          # 2B sweet spot

# ──────────────────────────── LoRA ─────────────────────────────
LORA_R = 16
LORA_ALPHA = 32                # alpha/r = 2.0 scaling
LORA_DROPOUT = 0.0             # Unsloth is fastest with dropout=0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]

# ──────────────────────────── Training ─────────────────────────
SEED = 42
NUM_TRAIN_EPOCHS = 5           # More epochs for better convergence
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4    # Effective batch = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 25
EVAL_STEPS = 200
SAVE_STEPS = 400
MAX_GRAD_NORM = 1.0
EVAL_FRACTION = 0.02               # 2% held out for eval

# ──────────────────────────── SVG Constraints (from competition) ───
MAX_SVG_CHARS = 8_000
MAX_PATH_ELEMENTS = 256
ALLOWED_TAGS = {
    "svg", "g", "path", "rect", "circle", "ellipse", "line",
    "polyline", "polygon", "defs", "use", "symbol", "clipPath",
    "mask", "linearGradient", "radialGradient", "stop", "text",
    "tspan", "title", "desc", "style", "pattern", "marker", "filter",
}

# ──────────────────────────── Generation ───────────────────────
SYSTEM_PROMPT = (
    "Generate valid SVG code. Use root <svg> with "
    'xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256". '
    "Output only SVG, nothing else."
)
GEN_MAX_NEW_TOKENS = 1536      # Longer SVGs = more visual detail
GEN_TEMPERATURE = 0.6
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.05

# ──────────────────────────── Fallback ─────────────────────────
FALLBACK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
    'viewBox="0 0 256 256">'
    '<rect width="256" height="256" fill="#f0f0f0"/>'
    '<circle cx="128" cy="128" r="48" fill="#666"/>'
    '</svg>'
)

# ──────────────────────────── Optional: External augmentation ──
# Set USE_EXTERNAL_DATA = True to augment competition data with HF datasets
USE_EXTERNAL_DATA = False
EXTERNAL_SOURCES = [
    {
        "id": "xingxm/SVGX-SFT-1M",
        "split": "train",
        "prompt_fields": ["prompt", "instruction", "input", "query"],
        "svg_fields": ["completion", "output", "svg", "response"],
        "max_samples": 10_000,
    },
    {
        "id": "thesantatitan/deepseek-svg-dataset",
        "split": "train",
        "prompt_fields": ["prompt", "instruction", "input"],
        "svg_fields": ["completion", "output", "svg"],
        "max_samples": 5_000,
    },
]
