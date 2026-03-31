"""
Local evaluation: score predicted SVGs against ground-truth on a held-out set.
Approximates the competition metric (Visual + Structural + Compactness).

Usage:
    python evaluate_local.py

Uses the eval split from prepare_data.py to give you a rough score estimate.
This runs on the TRAINING data (since test labels are hidden), so it's only
an approximation — but useful for comparing experiments.
"""
import io
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, TRAIN_CSV, SEED

# Optional: pip install cairosvg scikit-image Pillow
try:
    import cairosvg
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    HAS_RENDER = True
except ImportError:
    HAS_RENDER = False
    print("WARNING: cairosvg/scikit-image not installed. Skipping visual metrics.")
    print("  pip install cairosvg scikit-image Pillow")


def render_svg_to_gray(svg_text: str, size: int = 256) -> np.ndarray | None:
    """Render SVG to 256x256 grayscale numpy array."""
    if not HAS_RENDER:
        return None
    try:
        png_bytes = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=size, output_height=size,
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("L")
        return np.array(img, dtype=np.float64) / 255.0
    except Exception:
        return None


def visual_similarity(pred_svg: str, gt_svg: str) -> float:
    """SSIM between rendered grayscale images (approximation of the V metric)."""
    pred_img = render_svg_to_gray(pred_svg)
    gt_img = render_svg_to_gray(gt_svg)
    if pred_img is None or gt_img is None:
        return 0.0
    try:
        score = ssim(pred_img, gt_img, data_range=1.0)
        return max(0.0, score)
    except Exception:
        return 0.0


def tag_sequence(svg_text: str) -> list[str]:
    """Extract tag sequence from SVG for structural comparison."""
    try:
        root = ET.fromstring(svg_text)
        tags = []
        for elem in root.iter():
            tag = elem.tag
            if "}" in tag:
                tag = tag.split("}")[-1]
            tags.append(tag.lower())
        return tags
    except ET.ParseError:
        return []


def tree_edit_distance_approx(seq1: list[str], seq2: list[str]) -> int:
    """Simple Levenshtein distance on tag sequences as TED approximation."""
    m, n = len(seq1), len(seq2)
    if m == 0:
        return n
    if n == 0:
        return m
    # Use memory-efficient single-row DP
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = curr
    return prev[n]


def structural_similarity_score(pred_svg: str, gt_svg: str) -> float:
    """exp(-TED/25) where TED is tree-edit distance on tag sequences."""
    pred_tags = tag_sequence(pred_svg)
    gt_tags = tag_sequence(gt_svg)
    if not pred_tags or not gt_tags:
        return 0.0
    ted = tree_edit_distance_approx(pred_tags, gt_tags)
    return np.exp(-ted / 25.0)


def compactness_score(pred_svg: str, gt_svg: str) -> float:
    """Penalize SVGs much longer or shorter than reference."""
    pred_len = len(pred_svg)
    gt_len = len(gt_svg)
    if gt_len == 0:
        return 0.0
    ratio = pred_len / gt_len
    # Simple penalty: 1.0 if ratio=1, decays for deviation
    return np.exp(-abs(np.log(max(ratio, 0.01))) ** 2)


def score_single(pred_svg: str, gt_svg: str) -> dict:
    """Score a single prediction against ground truth."""
    v = visual_similarity(pred_svg, gt_svg) if HAS_RENDER else 0.0
    s = structural_similarity_score(pred_svg, gt_svg)
    c = compactness_score(pred_svg, gt_svg)

    # Geometric mean with weights (85%, 12%, 3%)
    if HAS_RENDER:
        combined = (v ** 0.85) * (s ** 0.12) * (c ** 0.03)
    else:
        # Without visual, just use structural + compactness
        combined = (s ** 0.80) * (c ** 0.20)

    return {"visual": v, "structural": s, "compactness": c, "combined": combined}


def main():
    """Run local evaluation on a sample of training data."""
    print("Loading training data for local eval ...")
    df = pd.read_csv(TRAIN_CSV).dropna(subset=["prompt", "svg"])

    # Take a small random sample for evaluation
    n_eval = min(100, len(df))
    sample = df.sample(n=n_eval, random_state=SEED)

    print(f"Evaluating {n_eval} samples ...")
    print("(This tests your post-processing pipeline on ground-truth SVGs)\n")

    scores = []
    for _, row in tqdm(sample.iterrows(), total=n_eval):
        # Score GT against itself (should be ~1.0) as sanity check
        result = score_single(row["svg"], row["svg"])
        scores.append(result)

    scores_df = pd.DataFrame(scores)
    print("\n=== Self-score (GT vs GT — should be ~1.0) ===")
    print(f"  Visual:     {scores_df['visual'].mean():.4f}")
    print(f"  Structural: {scores_df['structural'].mean():.4f}")
    print(f"  Compactness:{scores_df['compactness'].mean():.4f}")
    print(f"  Combined:   {scores_df['combined'].mean():.4f}")

    print("\n=== To evaluate your model's predictions: ===")
    print("  Modify this script to load your submission.csv and compare")
    print("  against train.csv ground truths for the eval split.")


if __name__ == "__main__":
    main()
