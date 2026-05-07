#!/usr/bin/env python3
"""
benchmark_hf_generate.py

Hugging Face generate() benchmark for LLM inference.

This script is designed for three experiment types:
  1. Batch sweep:
       fixed prompt length, fixed output length, varying batch size
  2. Sequence sweep:
       varying prompt length, output length equal to prompt target length
  3. Concurrency sweep:
       fixed prompt length, fixed batch size, varying simulated concurrency

Important note:
Hugging Face generate() is not a production serving engine. The concurrency
mode in this script uses Python threads to create simulated request pressure.
This is useful as a baseline, but it does not provide the same scheduler,
dynamic batching, or KV-cache management behavior as vLLM.

Input JSONL format:
Each line should contain:
  - ID
  - PROMPT
  - TARGET_TOKENS
  - ACTUAL_TOKENS

Output CSV:
Each row summarizes one benchmark condition.
"""

import argparse
import csv
import gc
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MAX_CONTEXT_TOKENS = 4096
TOKEN_SAFETY_MARGIN = 32


def safe_max_new_tokens_from_encoded(encoded: Dict[str, torch.Tensor], requested_max_new_tokens: int) -> int:
    """
    Reduce max_new_tokens so the longest prompt in the batch stays inside
    the context window.
    """
    if requested_max_new_tokens <= 0:
        raise ValueError(f"requested_max_new_tokens must be positive, got {requested_max_new_tokens}")

    prompt_lengths = encoded["attention_mask"].sum(dim=1)
    max_prompt_tokens = int(prompt_lengths.max().item()) if prompt_lengths.numel() > 0 else 0

    available = MAX_CONTEXT_TOKENS - max_prompt_tokens - TOKEN_SAFETY_MARGIN
    if available <= 0:
        raise ValueError(
            "Prompt is too long for the configured context window: "
            f"max_prompt_tokens={max_prompt_tokens}, "
            f"max_context={MAX_CONTEXT_TOKENS}, "
            f"safety_margin={TOKEN_SAFETY_MARGIN}"
        )

    return min(requested_max_new_tokens, available)


def load_prompt_file(path: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Load prompts from a JSONL file.

    The script accepts uppercase keys because the provided prompt files use:
      ID, PROMPT, TARGET_TOKENS, ACTUAL_TOKENS

    limit is useful for quick testing.
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

            if limit is not None and len(rows) >= limit:
                break

    return rows


def chunked(items: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    """
    Split a list of prompt records into batches.

    For Hugging Face generate(), batch size is manually controlled by grouping
    prompts before tokenization and generation.
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def safe_token_count(tokenizer, text: str) -> int:
    """
    Count tokens using the actual benchmark model tokenizer.

    This is more accurate than trusting the precomputed ACTUAL_TOKENS field,
    because that field may have been created with a lightweight tokenizer.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


@torch.inference_mode()
def run_one_batch(
    model,
    tokenizer,
    batch: List[Dict],
    max_new_tokens: int,
    device: str,
) -> Dict:
    """
    Run one Hugging Face generate() call on a batch of prompts.

    Important details:
    - Padding is enabled so prompts in the same batch have equal length.
    - Left padding is preferred for decoder-only causal language models.
    - do_sample=False keeps generation deterministic across runs.
    """
    prompts = [row["PROMPT"] for row in batch]

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    max_new_tokens = safe_max_new_tokens_from_encoded(
        encoded=encoded,
        requested_max_new_tokens=max_new_tokens,
    )

    # Count input tokens using the attention mask so padding tokens are not
    # incorrectly counted as real prompt tokens.
    prompt_tokens = int(encoded["attention_mask"].sum().item())

    start = time.perf_counter()

    outputs = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    end = time.perf_counter()

    # Generated token count is computed per sequence to avoid counting padding.
    input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    generated_tokens = 0

    for seq_idx, input_len in enumerate(input_lengths):
        generated_tokens += max(0, int(outputs[seq_idx].shape[0]) - int(input_len))

    latency = end - start

    return {
        "num_requests": len(batch),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": prompt_tokens + generated_tokens,
        "latency_sec": latency,
    }


def run_condition(
    model,
    tokenizer,
    prompts: List[Dict],
    model_name: str,
    prompt_file: str,
    sweep_name: str,
    batch_size: int,
    concurrency: int,
    max_new_tokens: int,
    prompt_target_tokens: int,
    device: str,
) -> Dict:
    """
    Run one benchmark condition.

    For batch sweep:
      - concurrency is usually 1
      - batch size changes

    For sequence sweep:
      - batch size and concurrency are fixed
      - prompt file and max_new_tokens change

    For concurrency sweep:
      - batch size is usually 1
      - concurrency changes

    If an OOM or other runtime error occurs, the function returns a CSV row
    marked as failed instead of crashing the entire sweep.
    """
    condition_start = time.perf_counter()
    batches = list(chunked(prompts, batch_size))

    try:
        results = []

        # Simulated concurrency:
        # Multiple Python threads submit generate() calls to the same model.
        # This is not equivalent to a serving scheduler, but it creates a
        # useful baseline for comparison against vLLM request concurrency.
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    run_one_batch,
                    model,
                    tokenizer,
                    batch,
                    max_new_tokens,
                    device,
                )
                for batch in batches
            ]

            for future in as_completed(futures):
                results.append(future.result())

        condition_end = time.perf_counter()
        total_time = condition_end - condition_start

        total_requests = sum(r["num_requests"] for r in results)
        prompt_tokens = sum(r["prompt_tokens"] for r in results)
        generated_tokens = sum(r["generated_tokens"] for r in results)
        total_tokens = prompt_tokens + generated_tokens
        avg_batch_latency = sum(r["latency_sec"] for r in results) / max(1, len(results))

        return {
            "status": "ok",
            "error_message": "",
            "backend": "huggingface_generate",
            "sweep_name": sweep_name,
            "model_name": model_name,
            "prompt_file": Path(prompt_file).name,
            "prompt_target_tokens": prompt_target_tokens,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "max_new_tokens": max_new_tokens,
            "num_prompts": len(prompts),
            "num_batches": len(batches),
            "num_requests": total_requests,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": total_tokens,
            "total_time_sec": total_time,
            "avg_batch_latency_sec": avg_batch_latency,
            "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0.0,
            "generated_tokens_per_sec": generated_tokens / total_time if total_time > 0 else 0.0,
            "requests_per_sec": total_requests / total_time if total_time > 0 else 0.0,
        }

    except RuntimeError as e:
        # CUDA OOM can happen for large batches or long sequences.
        # Clear cache so the next condition can continue.
        error_text = repr(e)
        traceback.print_exc()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        condition_end = time.perf_counter()

        return {
            "status": "failed",
            "error_message": error_text[:500],
            "backend": "huggingface_generate",
            "sweep_name": sweep_name,
            "model_name": model_name,
            "prompt_file": Path(prompt_file).name,
            "prompt_target_tokens": prompt_target_tokens,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "max_new_tokens": max_new_tokens,
            "num_prompts": len(prompts),
            "num_batches": len(batches),
            "num_requests": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "total_tokens": 0,
            "total_time_sec": condition_end - condition_start,
            "avg_batch_latency_sec": 0.0,
            "tokens_per_sec": 0.0,
            "generated_tokens_per_sec": 0.0,
            "requests_per_sec": 0.0,
        }


def infer_prompt_target_from_rows(rows: List[Dict], fallback: int = -1) -> int:
    """
    Infer target token length from the prompt file rows.
    """
    if not rows:
        return fallback

    value = rows[0].get("TARGET_TOKENS", fallback)

    try:
        return int(value)
    except Exception:
        return fallback


def write_row(output_csv: str, row: Dict, fieldnames: List[str]) -> None:
    """
    Append one benchmark result row to CSV.

    The file is flushed every row so partial results are preserved if the Slurm
    job reaches its time limit or a later condition fails.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompt-files", nargs="+", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sweep-name", required=True)

    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1])

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Fixed output length unless --output-length-mode prompt_target is used.",
    )

    parser.add_argument(
        "--output-length-mode",
        choices=["fixed", "prompt_target"],
        default="fixed",
        help="Use fixed max_new_tokens or set max_new_tokens equal to the prompt target length.",
    )

    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Optional limit for quick debugging. Default uses all prompts.",
    )

    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )

    args = parser.parse_args()

    # Cluster-friendly CPU thread limits. These protect against OpenBLAS/MKL
    # creating too many threads in constrained Slurm environments.
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
    print("Loading model and tokenizer")
    print("=" * 60)
    print("Model:", args.model_name)
    print("Device:", device)
    print("Dtype:", args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Decoder-only models should use left padding for batched generation.
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )

    model.eval()

    fieldnames = [
        "status",
        "error_message",
        "backend",
        "sweep_name",
        "model_name",
        "prompt_file",
        "prompt_target_tokens",
        "batch_size",
        "concurrency",
        "max_new_tokens",
        "num_prompts",
        "num_batches",
        "num_requests",
        "prompt_tokens",
        "generated_tokens",
        "total_tokens",
        "total_time_sec",
        "avg_batch_latency_sec",
        "tokens_per_sec",
        "generated_tokens_per_sec",
        "requests_per_sec",
    ]

    for prompt_file in args.prompt_files:
        prompts = load_prompt_file(prompt_file, limit=args.limit_prompts)
        prompt_target = infer_prompt_target_from_rows(prompts)

        if args.output_length_mode == "prompt_target":
            condition_max_new_tokens = prompt_target
        else:
            condition_max_new_tokens = args.max_new_tokens

        print("=" * 60)
        print("Prompt file:", prompt_file)
        print("Loaded prompts:", len(prompts))
        print("Prompt target:", prompt_target)
        print("Max new tokens:", condition_max_new_tokens)
        print("=" * 60)

        for batch_size in args.batch_sizes:
            for concurrency in args.concurrency_levels:
                print(
                    f"Running condition: sweep={args.sweep_name}, "
                    f"prompt_target={prompt_target}, batch={batch_size}, "
                    f"concurrency={concurrency}, max_new_tokens={condition_max_new_tokens}",
                    flush=True,
                )

                row = run_condition(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    model_name=args.model_name,
                    prompt_file=prompt_file,
                    sweep_name=args.sweep_name,
                    batch_size=batch_size,
                    concurrency=concurrency,
                    max_new_tokens=condition_max_new_tokens,
                    prompt_target_tokens=prompt_target,
                    device=device,
                )

                write_row(args.output_csv, row, fieldnames)

                print(
                    f"Finished condition with status={row['status']}, "
                    f"tokens/sec={row['tokens_per_sec']:.2f}, "
                    f"requests/sec={row['requests_per_sec']:.2f}",
                    flush=True,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                gc.collect()

    print("=" * 60)
    print("Benchmark complete")
    print("Results saved to:", args.output_csv)
    print("=" * 60)


if __name__ == "__main__":
    main()
