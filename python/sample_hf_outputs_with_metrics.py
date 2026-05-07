#!/usr/bin/env python3
"""
sample_hf_outputs_with_metrics.py

Qualitative Hugging Face generate() sampling with per-prompt text and throughput
metrics.

This script samples a small number of prompts, generates model outputs, and
writes a CSV containing:
  - full prompt text
  - generated output text
  - prompt/output token counts
  - latency
  - tokens/sec
  - generated tokens/sec
  - requests/sec

This is meant for report examples, not the full benchmark sweep.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(path: str):
    """Load JSONL prompt records."""
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def choose_prompts(rows, num_samples: int, seed: int):
    """Choose a reproducible random sample of prompts."""
    rng = random.Random(seed)

    if len(rows) <= num_samples:
        return rows

    return rng.sample(rows, num_samples)


def write_csv(path: str, rows, fieldnames):
    """Write result rows to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-csv", default="results/sample_outputs_with_metrics.csv")

    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )

    args = parser.parse_args()

    # Cluster-safe CPU threading limits.
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "4")
    os.environ.setdefault("OMP_NUM_THREADS", cpus)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", cpus)
    os.environ.setdefault("MKL_NUM_THREADS", cpus)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", cpus)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Loading model for sample generation with metrics")
    print("=" * 60)
    print("Model:", args.model_name)
    print("Prompt file:", args.prompt_file)
    print("Device:", device)
    print("Batch size:", args.batch_size)
    print("Max new tokens:", args.max_new_tokens)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )

    model.eval()

    all_rows = load_prompts(args.prompt_file)
    selected = choose_prompts(all_rows, args.num_samples, args.seed)

    result_rows = []

    global_start = time.perf_counter()

    # Process in small batches. For report examples, batch_size=1 is easiest
    # because latency and throughput are truly per prompt. If batch_size > 1,
    # the same batch latency is assigned to each prompt in the batch, and an
    # approximate per-prompt latency is also reported.
    for batch_start in range(0, len(selected), args.batch_size):
        batch = selected[batch_start:batch_start + args.batch_size]
        prompts = [row["PROMPT"] for row in batch]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()

        batch_time_start = time.perf_counter()

        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        batch_time_end = time.perf_counter()
        batch_latency = batch_time_end - batch_time_start

        batch_prompt_tokens = int(sum(input_lengths))
        batch_generated_tokens = 0
        generated_texts = []
        generated_lengths = []

        for local_idx, row in enumerate(batch):
            input_len = int(input_lengths[local_idx])
            output_len = int(outputs[local_idx].shape[0])
            generated_len = max(0, output_len - input_len)

            generated_ids = outputs[local_idx][input_len:]
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            batch_generated_tokens += generated_len
            generated_lengths.append(generated_len)
            generated_texts.append(generated_text)

        batch_total_tokens = batch_prompt_tokens + batch_generated_tokens

        batch_tokens_per_sec = (
            batch_total_tokens / batch_latency if batch_latency > 0 else 0.0
        )
        batch_generated_tokens_per_sec = (
            batch_generated_tokens / batch_latency if batch_latency > 0 else 0.0
        )
        batch_requests_per_sec = (
            len(batch) / batch_latency if batch_latency > 0 else 0.0
        )

        for local_idx, row in enumerate(batch):
            prompt_tokens = int(input_lengths[local_idx])
            generated_tokens = generated_lengths[local_idx]
            total_tokens = prompt_tokens + generated_tokens

            # With batch_size=1, this is exact. With larger batches, individual
            # latency is not directly observable from one generate() call.
            approx_prompt_latency = batch_latency / max(1, len(batch))

            prompt_tokens_per_sec = (
                total_tokens / approx_prompt_latency
                if approx_prompt_latency > 0 else 0.0
            )
            prompt_generated_tokens_per_sec = (
                generated_tokens / approx_prompt_latency
                if approx_prompt_latency > 0 else 0.0
            )
            prompt_requests_per_sec = (
                1.0 / approx_prompt_latency
                if approx_prompt_latency > 0 else 0.0
            )

            result_rows.append({
                "backend": "huggingface_generate",
                "model_name": args.model_name,
                "prompt_file": Path(args.prompt_file).name,
                "sample_index": len(result_rows),
                "prompt_id": row.get("ID"),
                "target_tokens": row.get("TARGET_TOKENS"),
                "actual_tokens_file": row.get("ACTUAL_TOKENS"),
                "batch_size": len(batch),
                "batch_index": batch_start // args.batch_size,
                "position_in_batch": local_idx,
                "max_new_tokens": args.max_new_tokens,
                "prompt_tokens_model": prompt_tokens,
                "generated_tokens": generated_tokens,
                "total_tokens": total_tokens,
                "batch_prompt_tokens": batch_prompt_tokens,
                "batch_generated_tokens": batch_generated_tokens,
                "batch_total_tokens": batch_total_tokens,
                "batch_latency_sec": batch_latency,
                "approx_prompt_latency_sec": approx_prompt_latency,
                "batch_tokens_per_sec": batch_tokens_per_sec,
                "batch_generated_tokens_per_sec": batch_generated_tokens_per_sec,
                "batch_requests_per_sec": batch_requests_per_sec,
                "prompt_tokens_per_sec": prompt_tokens_per_sec,
                "prompt_generated_tokens_per_sec": prompt_generated_tokens_per_sec,
                "prompt_requests_per_sec": prompt_requests_per_sec,
                "prompt": row["PROMPT"],
                "generated_output": generated_texts[local_idx],
            })

        print(
            f"Finished batch {batch_start // args.batch_size}: "
            f"latency={batch_latency:.3f}s, "
            f"batch_tokens/sec={batch_tokens_per_sec:.2f}, "
            f"requests/sec={batch_requests_per_sec:.2f}",
            flush=True,
        )

    global_end = time.perf_counter()

    fieldnames = [
        "backend",
        "model_name",
        "prompt_file",
        "sample_index",
        "prompt_id",
        "target_tokens",
        "actual_tokens_file",
        "batch_size",
        "batch_index",
        "position_in_batch",
        "max_new_tokens",
        "prompt_tokens_model",
        "generated_tokens",
        "total_tokens",
        "batch_prompt_tokens",
        "batch_generated_tokens",
        "batch_total_tokens",
        "batch_latency_sec",
        "approx_prompt_latency_sec",
        "batch_tokens_per_sec",
        "batch_generated_tokens_per_sec",
        "batch_requests_per_sec",
        "prompt_tokens_per_sec",
        "prompt_generated_tokens_per_sec",
        "prompt_requests_per_sec",
        "prompt",
        "generated_output",
    ]

    write_csv(args.output_csv, result_rows, fieldnames)

    print("=" * 60)
    print("Sample generation complete")
    print("CSV saved to:", args.output_csv)
    print("Total wall time:", global_end - global_start)
    print("=" * 60)


if __name__ == "__main__":
    main()
