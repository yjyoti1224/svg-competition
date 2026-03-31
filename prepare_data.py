"""
Step 2: Download competition data + prepare training dataset.

Usage:
    python prepare_data.py

What it does:
1. Downloads competition data from Kaggle (train.csv, test.csv)
2. Filters training SVGs against competition constraints
3. Formats data for SFT (chat template)
4. Optionally augments with external HuggingFace datasets
5. Saves processed datasets to disk
"""
import os
import sys
import random
import zipfile

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASE_DIR, DATA_DIR, KAGGLE_COMP, TRAIN_CSV, TEST_CSV,
    MODEL_NAME, MAX_SEQ_LENGTH, SEED, EVAL_FRACTION,
    SYSTEM_PROMPT, USE_EXTERNAL_DATA, EXTERNAL_SOURCES,
)
from utils import check_constraints, pick_first_field

random.seed(SEED)
np.random.seed(SEED)


def download_kaggle_data():
    """Download competition data using Kaggle CLI."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV):
        print(f"Data already exists at {DATA_DIR}, skipping download.")
        return

    print(f"Downloading competition data: {KAGGLE_COMP} ...")
    ret = os.system(f"kaggle competitions download -c {KAGGLE_COMP} -p {DATA_DIR}")
    if ret != 0:
        print("ERROR: kaggle download failed. Make sure:")
        print("  1. kaggle CLI is installed: pip install kaggle")
        print("  2. API key is at ~/.kaggle/kaggle.json")
        print("  3. You have accepted the competition rules on kaggle.com")
        sys.exit(1)

    # Unzip if needed
    zip_path = os.path.join(DATA_DIR, f"{KAGGLE_COMP}.zip")
    if os.path.exists(zip_path):
        print("Extracting zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        os.remove(zip_path)

    print(f"Data ready at {DATA_DIR}")


def load_and_filter_train():
    """Load train.csv and filter against competition constraints."""
    print(f"Loading {TRAIN_CSV} ...")
    df = pd.read_csv(TRAIN_CSV)
    print(f"  Raw rows: {len(df)}")

    # Drop rows with missing prompt or svg
    df = df.dropna(subset=["prompt", "svg"])
    df = df[df["prompt"].str.strip().astype(bool)]
    df = df[df["svg"].str.strip().astype(bool)]
    print(f"  After dropping empty: {len(df)}")

    # Validate SVGs against competition constraints
    valid_mask = []
    for svg in df["svg"]:
        ok, _ = check_constraints(svg)
        valid_mask.append(ok)
    df = df[valid_mask]
    print(f"  After constraint filtering: {len(df)}")

    return df


def format_chat(prompt: str, svg: str) -> str:
    """Format a prompt-SVG pair into chat template for SFT."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{svg}<|im_end|>"
    )


def prepare_competition_dataset(df: pd.DataFrame) -> Dataset:
    """Convert filtered DataFrame to HuggingFace Dataset with chat-formatted text."""
    texts = []
    for _, row in df.iterrows():
        texts.append(format_chat(row["prompt"], row["svg"]))

    return Dataset.from_dict({"text": texts})


def load_external_source(source_cfg: dict) -> Dataset:
    """Load and normalize an external HuggingFace dataset."""
    from datasets import load_dataset as hf_load

    ds_id = source_cfg["id"]
    max_samples = source_cfg.get("max_samples", 10_000)
    print(f"Loading external: {ds_id} ...")

    try:
        ds = hf_load(ds_id, split=source_cfg["split"])
    except Exception as e:
        print(f"  SKIP {ds_id}: {e}")
        return None

    if len(ds) > max_samples:
        ds = ds.shuffle(seed=SEED).select(range(max_samples))

    texts = []
    for ex in ds:
        prompt = pick_first_field(ex, source_cfg["prompt_fields"])
        svg = pick_first_field(ex, source_cfg["svg_fields"])
        if not prompt or not svg:
            continue
        ok, _ = check_constraints(svg)
        if not ok:
            continue
        texts.append(format_chat(prompt, svg))

    print(f"  {ds_id}: {len(texts)} usable samples")
    if not texts:
        return None
    return Dataset.from_dict({"text": texts})


def filter_by_token_length(dataset: Dataset, tokenizer, max_len: int) -> Dataset:
    """Remove samples that exceed max sequence length."""
    def _fits(example):
        ids = tokenizer(example["text"], truncation=False)["input_ids"]
        return len(ids) <= max_len

    before = len(dataset)
    dataset = dataset.filter(_fits, desc="Filtering by token length")
    print(f"  Token-length filter: {before} -> {len(dataset)}")
    return dataset


def main():
    # Step 1: Download data
    download_kaggle_data()

    # Step 2: Load and filter competition training data
    train_df = load_and_filter_train()
    comp_dataset = prepare_competition_dataset(train_df)
    print(f"Competition dataset: {len(comp_dataset)} samples")

    # Step 3: Optionally augment with external data
    all_datasets = [comp_dataset]
    if USE_EXTERNAL_DATA:
        for src in EXTERNAL_SOURCES:
            ext_ds = load_external_source(src)
            if ext_ds is not None:
                all_datasets.append(ext_ds)

    if len(all_datasets) > 1:
        full_dataset = concatenate_datasets(all_datasets).shuffle(seed=SEED)
        print(f"Combined dataset: {len(full_dataset)} samples")
    else:
        full_dataset = comp_dataset.shuffle(seed=SEED)

    # Step 4: Filter by token length
    print("Loading tokenizer for length filtering ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_dataset = filter_by_token_length(full_dataset, tokenizer, MAX_SEQ_LENGTH)

    # Step 5: Split into train/eval
    splits = full_dataset.train_test_split(test_size=EVAL_FRACTION, seed=SEED)
    train_ds = splits["train"]
    eval_ds = splits["test"]

    print(f"\nFinal splits:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Eval:  {len(eval_ds)}")

    # Step 6: Save to disk
    save_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(save_dir, exist_ok=True)
    train_ds.save_to_disk(os.path.join(save_dir, "train"))
    eval_ds.save_to_disk(os.path.join(save_dir, "eval"))
    print(f"\nSaved processed data to {save_dir}")

    # Show a sample
    print("\n--- Sample training text (first 500 chars) ---")
    print(train_ds[0]["text"][:500])


if __name__ == "__main__":
    main()
