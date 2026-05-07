#!/usr/bin/env python3
"""
benchmark_vllm_openai.py

vLLM benchmark client using the OpenAI-compatible /v1/completions endpoint.

This script is designed to match the Hugging Face generate() sweeps:
  1. Batch sweep:
       prompt length 512, output length 512, batch sizes 1/2/4/8/16/32
  2. Sequence sweep:
       prompt lengths 128/256/512/1024/2048, output length equals prompt target
  3. Concurrency sweep:
       prompt length 512, output length 512, concurrency 1/2/4/8/16/32

Important:
  - vLLM performs its own dynamic batching internally.
  - batch_size in this script is the number of prompts sent in one HTTP
    completion request as a list.
  - concurrency is the number of HTTP completion requests in flight.

Input JSONL fields:
  ID, PROMPT, TARGET_TOKENS, ACTUAL_TOKENS

Output:
  One CSV row per benchmark condition, similar to the Hugging Face summary CSV.
"""

import argparse
import asyncio
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from transformers import AutoTokenizer



# Global context settings used to keep requests within the model window.
MAX_CONTEXT_TOKENS = 4096
TOKEN_SAFETY_MARGIN = 32



# Prompt loading, batching, and request-size helpers.
def safe_max_new_tokens_for_prompts(tokenizer, prompts: List[str], requested_max_new_tokens: int) -> int:
    """
    Reduce max_new_tokens so each prompt stays inside the context window.

    Uses the longest prompt in the request batch because OpenAI-compatible
    completions accepts one max_tokens value for the whole prompt list.
    """
    if requested_max_new_tokens <= 0:
        raise ValueError(f"requested_max_new_tokens must be positive, got {requested_max_new_tokens}")

    prompt_token_counts = [
        len(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    ]
    max_prompt_tokens = max(prompt_token_counts) if prompt_token_counts else 0

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
    """Load prompt records from JSONL."""
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

            if limit is not None and len(rows) >= limit:
                break

    return rows


def chunked(items: List[Dict], batch_size: int):
    """Split records into client-side request batches."""
    for i in range(0, len(items), batch_size):
        yield i // batch_size, items[i:i + batch_size]


def infer_prompt_target(rows: List[Dict], fallback: int = -1) -> int:
    """Infer target prompt length from the first JSONL row."""
    if not rows:
        return fallback

    try:
        return int(rows[0].get("TARGET_TOKENS", fallback))
    except Exception:
        return fallback



# OpenAI-compatible HTTP request helper.
async def post_completion(
    session: aiohttp.ClientSession,
    url: str,
    model_name: str,
    prompt_batch: List[str],
    max_new_tokens: int,
    temperature: float,
) -> Dict:
    """
    Send one completion request to vLLM.

    prompt_batch may contain one prompt or multiple prompts. vLLM's completions
    endpoint accepts a list of prompts and returns one choice per prompt.
    """
    payload = {
        "model": model_name,
        "prompt": prompt_batch,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }

    start = time.perf_counter()

    async with session.post(url, json=payload) as response:
        text = await response.text()
        end = time.perf_counter()

        if response.status != 200:
            raise RuntimeError(
                f"vLLM request failed with status {response.status}: {text[:1000]}"
            )

        data = json.loads(text)

    return {
        "latency_sec": end - start,
        "data": data,
    }



# One full vLLM benchmark condition and its CSV summary row.
async def run_condition(
    tokenizer,
    server_url: str,
    model_name: str,
    prompt_file: str,
    prompts: List[Dict],
    sweep_name: str,
    batch_size: int,
    concurrency: int,
    max_new_tokens: int,
    prompt_target_tokens: int,
    temperature: float,
) -> Dict:
    """
    Run one vLLM benchmark condition.

    The condition is measured with wall-clock time around all HTTP requests.
    """
    completions_url = f"{server_url.rstrip('/')}/v1/completions"

    request_batches = list(chunked(prompts, batch_size))
    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request(batch_index: int, batch_rows: List[Dict]):
        async with semaphore:
            prompt_texts = [row["PROMPT"] for row in batch_rows]
            safe_max_new_tokens = safe_max_new_tokens_for_prompts(
                tokenizer=tokenizer,
                prompts=prompt_texts,
                requested_max_new_tokens=max_new_tokens,
            )
            result = await post_completion(
                session=session,
                url=completions_url,
                model_name=model_name,
                prompt_batch=prompt_texts,
                max_new_tokens=safe_max_new_tokens,
                temperature=temperature,
            )
            return batch_index, batch_rows, result

    condition_start = time.perf_counter()

    try:
        timeout = aiohttp.ClientTimeout(total=None)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                limited_request(batch_index, batch_rows)
                for batch_index, batch_rows in request_batches
            ]

            responses = await asyncio.gather(*tasks)

        condition_end = time.perf_counter()
        total_time = condition_end - condition_start

        total_prompt_tokens = 0
        total_generated_tokens = 0
        total_prompt_requests = 0
        http_latencies = []

        for batch_index, batch_rows, response in responses:
            http_latencies.append(response["latency_sec"])
            data = response["data"]

            choices = data.get("choices", [])

            # Choices should match prompts. Use index if present, otherwise order.
            choices_by_index = {}
            for fallback_index, choice in enumerate(choices):
                choice_index = choice.get("index", fallback_index)
                choices_by_index[choice_index] = choice

            for local_idx, row in enumerate(batch_rows):
                prompt = row["PROMPT"]
                choice = choices_by_index.get(local_idx, {})
                generated_text = choice.get("text", "")

                prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                generated_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))

                total_prompt_tokens += prompt_tokens
                total_generated_tokens += generated_tokens
                total_prompt_requests += 1

        total_tokens = total_prompt_tokens + total_generated_tokens
        avg_http_latency = sum(http_latencies) / max(1, len(http_latencies))

        return {
            "status": "ok",
            "error_message": "",
            "backend": "vllm_openai_server",
            "sweep_name": sweep_name,
            "model_name": model_name,
            "prompt_file": Path(prompt_file).name,
            "prompt_target_tokens": prompt_target_tokens,
            "batch_size_requested": batch_size,
            "concurrency": concurrency,
            "max_new_tokens": max_new_tokens,
            "num_prompts": len(prompts),
            "num_http_requests": len(request_batches),
            "num_requests": total_prompt_requests,
            "prompt_tokens": total_prompt_tokens,
            "generated_tokens": total_generated_tokens,
            "total_tokens": total_tokens,
            "total_time_sec": total_time,
            "avg_http_latency_sec": avg_http_latency,
            "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0.0,
            "generated_tokens_per_sec": (
                total_generated_tokens / total_time if total_time > 0 else 0.0
            ),
            "requests_per_sec": (
                total_prompt_requests / total_time if total_time > 0 else 0.0
            ),
            "http_requests_per_sec": (
                len(request_batches) / total_time if total_time > 0 else 0.0
            ),
        }

    except Exception as e:
        condition_end = time.perf_counter()

        return {
            "status": "failed",
            "error_message": repr(e)[:1000],
            "backend": "vllm_openai_server",
            "sweep_name": sweep_name,
            "model_name": model_name,
            "prompt_file": Path(prompt_file).name,
            "prompt_target_tokens": prompt_target_tokens,
            "batch_size_requested": batch_size,
            "concurrency": concurrency,
            "max_new_tokens": max_new_tokens,
            "num_prompts": len(prompts),
            "num_http_requests": len(request_batches),
            "num_requests": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "total_tokens": 0,
            "total_time_sec": condition_end - condition_start,
            "avg_http_latency_sec": 0.0,
            "tokens_per_sec": 0.0,
            "generated_tokens_per_sec": 0.0,
            "requests_per_sec": 0.0,
            "http_requests_per_sec": 0.0,
        }



# CSV output helper.
def write_row(output_csv: str, row: Dict, fieldnames: List[str]) -> None:
    """Append one condition row to CSV."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
        f.flush()



# Main async benchmark loop.
async def main_async(args):
    """Main async benchmark loop."""
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "4")
    os.environ.setdefault("OMP_NUM_THREADS", cpus)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", cpus)
    os.environ.setdefault("MKL_NUM_THREADS", cpus)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", cpus)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("=" * 60)
    print("Loading tokenizer")
    print("=" * 60)
    print("Model:", args.model_name)
    print("Server URL:", args.server_url)
    print("Output CSV:", args.output_csv)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    fieldnames = [
        "status",
        "error_message",
        "backend",
        "sweep_name",
        "model_name",
        "prompt_file",
        "prompt_target_tokens",
        "batch_size_requested",
        "concurrency",
        "max_new_tokens",
        "num_prompts",
        "num_http_requests",
        "num_requests",
        "prompt_tokens",
        "generated_tokens",
        "total_tokens",
        "total_time_sec",
        "avg_http_latency_sec",
        "tokens_per_sec",
        "generated_tokens_per_sec",
        "requests_per_sec",
        "http_requests_per_sec",
    ]

    for prompt_file in args.prompt_files:
        prompts = load_prompt_file(prompt_file, limit=args.limit_prompts)
        prompt_target = infer_prompt_target(prompts)

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
                    f"Running vLLM condition: sweep={args.sweep_name}, "
                    f"prompt_target={prompt_target}, batch={batch_size}, "
                    f"concurrency={concurrency}, max_new_tokens={condition_max_new_tokens}",
                    flush=True,
                )

                row = await run_condition(
                    tokenizer=tokenizer,
                    server_url=args.server_url,
                    model_name=args.model_name,
                    prompt_file=prompt_file,
                    prompts=prompts,
                    sweep_name=args.sweep_name,
                    batch_size=batch_size,
                    concurrency=concurrency,
                    max_new_tokens=condition_max_new_tokens,
                    prompt_target_tokens=prompt_target,
                    temperature=args.temperature,
                )

                write_row(args.output_csv, row, fieldnames)

                print(
                    f"Finished condition status={row['status']}, "
                    f"tokens/sec={row['tokens_per_sec']:.2f}, "
                    f"requests/sec={row['requests_per_sec']:.2f}, "
                    f"http req/sec={row['http_requests_per_sec']:.2f}",
                    flush=True,
                )

    print("=" * 60)
    print("vLLM benchmark complete")
    print("Results saved to:", args.output_csv)
    print("=" * 60)



# Command-line argument parsing.
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt-files", nargs="+", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sweep-name", required=True)

    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1])

    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--output-length-mode",
        choices=["fixed", "prompt_target"],
        default="fixed",
    )

    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
