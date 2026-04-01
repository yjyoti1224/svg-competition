"""
Step 4: Generate SVGs for test prompts and create submission.csv.

Usage:
    python generate.py [--checkpoint PATH] [--use-merged]

Loads the fine-tuned model and generates SVGs for each test prompt.
Uses batched inference for speed. Applies post-processing and constraint validation.
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CHECKPOINT_DIR, OUTPUT_DIR, TEST_CSV, SUBMISSION_PATH,
    MODEL_NAME, MAX_SEQ_LENGTH, SYSTEM_PROMPT, FALLBACK_SVG,
    GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE, GEN_TOP_P, GEN_REPETITION_PENALTY,
)
from utils import postprocess_svg, check_constraints
import xml.etree.ElementTree as ET


def svg_has_content(svg_text: str) -> bool:
    """Check if SVG has actual visual content (not empty/self-closing)."""
    try:
        root = ET.fromstring(svg_text)
        return len(list(root)) > 0
    except ET.ParseError:
        return False


# Retry temperatures: descend from 0.6 to 0.2 for increasingly deterministic output
RETRY_TEMPS = [0.5, 0.4, 0.3, 0.2]


def load_model(checkpoint_path: str, use_merged: bool):
    """Load the fine-tuned model for inference."""
    if use_merged:
        print(f"Loading merged model from {checkpoint_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print(f"Loading base model + LoRA adapter from {checkpoint_path} ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()

    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on {model.device}")
    return model, tokenizer


def build_prompt(user_prompt: str) -> str:
    """Build the chat-formatted prompt for inference."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str]) -> list[str]:
    """Generate SVGs for a batch of prompts."""
    chats = [build_prompt(p) for p in prompts]
    inputs = tokenizer(
        chats, return_tensors="pt", padding=True, truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        repetition_penalty=GEN_REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the generated part for each sample
    results = []
    input_len = inputs["input_ids"].shape[1]
    for i in range(len(prompts)):
        gen_ids = output_ids[i][input_len:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append(decoded)
    return results


@torch.no_grad()
def generate_single(model, tokenizer, prompt_text: str, temperature: float = None) -> str:
    """Generate SVG for a single prompt. Supports custom temperature for retries."""
    temp = temperature if temperature is not None else GEN_TEMPERATURE
    chat = build_prompt(prompt_text)
    inputs = tokenizer(chat, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LENGTH).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temp,
        top_p=GEN_TOP_P,
        repetition_penalty=GEN_REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(CHECKPOINT_DIR, "final"),
                        help="Path to adapter or merged checkpoint")
    parser.add_argument("--use-merged", action="store_true",
                        help="Load a merged (non-adapter) checkpoint")
    parser.add_argument("--test-csv", type=str, default=TEST_CSV)
    parser.add_argument("--output", type=str, default=SUBMISSION_PATH)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference")
    args = parser.parse_args()

    # Try merged dir first if it exists
    merged_dir = os.path.join(CHECKPOINT_DIR, "merged")
    if os.path.exists(merged_dir) and not args.use_merged:
        args.checkpoint = merged_dir
        args.use_merged = True
        print(f"Found merged model, using {merged_dir}")

    # Load model
    model, tokenizer = load_model(args.checkpoint, args.use_merged)

    # Load test prompts
    test_df = pd.read_csv(args.test_csv)
    print(f"Test prompts: {len(test_df)}")

    # --- Batched inference + individual retry for failures ---
    all_prompts = test_df["prompt"].tolist()
    all_ids = test_df["id"].tolist()
    n_prompts = len(all_prompts)

    t0 = time.time()

    # Step 1: Batched generation (fast)
    print(f"\n=== Batched generation (temp={GEN_TEMPERATURE}) ===")
    raw_outputs = []
    for i in tqdm(range(0, n_prompts, args.batch_size), desc="Batch"):
        batch_prompts = all_prompts[i:i + args.batch_size]
        chats = [build_prompt(p) for p in batch_prompts]
        inputs = tokenizer(
            chats, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                repetition_penalty=GEN_REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for j in range(len(batch_prompts)):
            gen_ids = output_ids[j][input_len:]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
            raw_outputs.append(decoded)

    # Step 2: Post-process and identify failures
    results = [None] * n_prompts
    failed_indices = []

    for idx, raw in enumerate(raw_outputs):
        svg = postprocess_svg(raw)
        is_fallback = (svg == FALLBACK_SVG)
        valid, _ = check_constraints(svg)
        has_content = svg_has_content(svg) if valid and not is_fallback else False

        if valid and not is_fallback and has_content:
            results[idx] = svg
        else:
            failed_indices.append(idx)

    elapsed_batch = time.time() - t0
    print(f"\nBatch done: {n_prompts - len(failed_indices)}/{n_prompts} succeeded, "
          f"{len(failed_indices)} need retry, {elapsed_batch/60:.1f}min elapsed")

    # Step 3: Retry failures individually at descending temperatures
    if failed_indices:
        print(f"\n=== Retrying {len(failed_indices)} failed prompts ===")
        still_failed = []

        for idx in tqdm(failed_indices, desc="Retry"):
            prompt_text = all_prompts[idx]
            success = False

            for temp in RETRY_TEMPS:
                raw = generate_single(model, tokenizer, prompt_text, temperature=temp)
                svg = postprocess_svg(raw)
                is_fallback = (svg == FALLBACK_SVG)
                valid, _ = check_constraints(svg)
                has_content = svg_has_content(svg) if valid and not is_fallback else False

                if valid and not is_fallback and has_content:
                    results[idx] = svg
                    success = True
                    break

            if not success:
                results[idx] = FALLBACK_SVG
                still_failed.append(idx)

        print(f"  Retry recovered {len(failed_indices) - len(still_failed)}/{len(failed_indices)}")
        print(f"  Still failed: {len(still_failed)}")

    # Count fallbacks
    fallback_count = sum(1 for s in results if s == FALLBACK_SVG)
    final_results = [{"id": all_ids[idx], "svg": results[idx]} for idx in range(n_prompts)]

    print(f"\nFinal stats: {fallback_count} fallbacks out of {n_prompts}")

    elapsed = time.time() - t0

    # Save submission
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sub_df = pd.DataFrame(final_results)
    sub_df.to_csv(args.output, index=False)

    print(f"\n{'='*50}")
    print(f"Submission saved: {args.output}")
    print(f"Total rows: {len(sub_df)}")
    print(f"Fallbacks: {fallback_count}")
    print(f"Runtime: {elapsed:.1f}s ({elapsed/len(test_df):.2f}s per prompt)")
    print(f"{'='*50}")

    # Sanity check
    print("\nSample outputs:")
    for i in range(min(3, len(final_results))):
        print(f"  [{final_results[i]['id'][:12]}...] {final_results[i]['svg'][:120]}...")


if __name__ == "__main__":
    main()
