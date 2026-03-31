"""
Step 3: QLoRA fine-tuning with Unsloth on A100.

Usage:
    python train.py
"""
import os
import sys
import random

# Import unsloth FIRST (required by unsloth for optimizations)
from unsloth import FastLanguageModel

import numpy as np
import torch
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, CHECKPOINT_DIR, MODEL_NAME, MAX_SEQ_LENGTH, LOAD_IN_4BIT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    SEED, NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, WARMUP_RATIO,
    WEIGHT_DECAY, LOGGING_STEPS, EVAL_STEPS, SAVE_STEPS, MAX_GRAD_NORM,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def main():
    # ── 1. Load processed data ──────────────────────────────────
    processed_dir = os.path.join(DATA_DIR, "processed")
    train_ds = load_from_disk(os.path.join(processed_dir, "train"))
    eval_ds = load_from_disk(os.path.join(processed_dir, "eval"))
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── 2. Load model with Unsloth ──────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # ── 3. Apply LoRA adapters ──────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    model.print_trainable_parameters()

    # ── 4. Training config ──────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Calculate warmup steps from ratio
    total_steps = (len(train_ds) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * NUM_TRAIN_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    sft_config = SFTConfig(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        seed=SEED,
        dataloader_num_workers=4,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
    )

    # ── 5. SFT Trainer ─────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
    )

    # ── 6. Train! (with checkpoint resume support) ──────────────
    # Find last checkpoint to resume from if one exists
    resume_from = None
    if os.path.isdir(CHECKPOINT_DIR):
        checkpoints = [
            os.path.join(CHECKPOINT_DIR, d)
            for d in os.listdir(CHECKPOINT_DIR)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            resume_from = max(checkpoints, key=os.path.getmtime)
            print(f"Resuming from checkpoint: {resume_from}")

    print("Starting training ...")
    result = trainer.train(resume_from_checkpoint=resume_from)
    print(f"Training complete. Metrics: {result.metrics}")

    # ── 7. Save final adapter ───────────────────────────────────
    final_dir = os.path.join(CHECKPOINT_DIR, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final adapter to {final_dir}")

    # Save merged model for faster inference
    print("Merging adapter into base model ...")
    merged_dir = os.path.join(CHECKPOINT_DIR, "merged")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"Saved merged model to {merged_dir}")


if __name__ == "__main__":
    main()
