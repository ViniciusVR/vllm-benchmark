#!/usr/bin/env python3
"""
plot_sweep_results.py

Create grouped bar charts comparing HF and vLLM results for each benchmark
sweep.

Expected input files inside --results-dir:
  hf_batch_sweep.csv
  vllm_batch_sweep.csv
  hf_sequence_sweep.csv
  vllm_sequence_sweep.csv
  hf_concurrency_sweep.csv
  vllm_concurrency_sweep.csv

Example:
  python python/plot_sweep_results.py \
      --results-dir results \
      --output-dir figures \
      --metric requests_per_sec
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BACKEND_LABELS = {
    "huggingface_generate": "HF",
    "vllm_openai_server": "vLLM",
}


def read_csv_checked(path: Path) -> pd.DataFrame:
    """Read one CSV file and fail clearly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing expected result file: {path}")
    return pd.read_csv(path)


def normalize_backend_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create clean backend labels for plotting."""
    df = df.copy()
    if "backend" not in df.columns:
        raise ValueError("Expected column 'backend' was not found.")
    df["backend_label"] = df["backend"].map(BACKEND_LABELS).fillna(df["backend"])
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names across HF and vLLM result files.

    Important fix:
    The combined dataframe can contain both `batch_size` and
    `batch_size_requested`. HF batch results may only fill `batch_size`, while
    vLLM fills `batch_size_requested`. Therefore, we fill missing
    `batch_size_requested` values from `batch_size` row-by-row.
    """
    df = df.copy()

    if "batch_size_requested" not in df.columns:
        df["batch_size_requested"] = pd.NA

    if "batch_size" in df.columns:
        df["batch_size_requested"] = df["batch_size_requested"].fillna(df["batch_size"])

    numeric_columns = [
        "batch_size_requested",
        "batch_size",
        "concurrency",
        "prompt_target_tokens",
        "requests_per_sec",
        "generated_tokens_per_sec",
        "tokens_per_sec",
        "total_time_sec",
        "avg_batch_latency_sec",
        "avg_http_latency_sec",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_pair(results_dir: Path, sweep: str) -> pd.DataFrame:
    """Load and combine HF and vLLM CSVs for one sweep."""
    hf_path = results_dir / f"hf_{sweep}_sweep.csv"
    vllm_path = results_dir / f"vllm_{sweep}_sweep.csv"

    hf = read_csv_checked(hf_path)
    vllm = read_csv_checked(vllm_path)

    combined = pd.concat([hf, vllm], ignore_index=True)
    combined = normalize_backend_label(combined)
    combined = normalize_columns(combined)

    # Keep successful rows only so failed/OOM rows do not appear as zeros.
    if "status" in combined.columns:
        combined = combined[combined["status"].astype(str).str.strip().str.lower() == "ok"].copy()

    print(f"\n{sweep.title()} sweep backend counts:")
    print(combined["backend_label"].value_counts(dropna=False).to_string())

    expected = {"HF", "vLLM"}
    present = set(combined["backend_label"].dropna().unique())
    missing = expected - present
    if missing:
        print(
            f"WARNING: {sweep} sweep is missing backend(s): "
            f"{', '.join(sorted(missing))}."
        )

    return combined


def prepare_sweep_data(df: pd.DataFrame, sweep: str, metric: str):
    """Select the x-axis variable for each sweep and aggregate values."""
    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found. Available columns: {list(df.columns)}"
        )

    if sweep == "batch":
        x_col = "batch_size_requested"
        x_label = "Batch Size"
    elif sweep == "sequence":
        x_col = "prompt_target_tokens"
        x_label = "Prompt / Output Target Tokens"
    elif sweep == "concurrency":
        x_col = "concurrency"
        x_label = "Concurrency Level"
    else:
        raise ValueError(f"Unknown sweep: {sweep}")

    if x_col not in df.columns:
        raise ValueError(f"Expected x-axis column '{x_col}' not found.")

    df = df.dropna(subset=[x_col, metric]).copy()

    plot_df = (
        df.groupby([x_col, "backend_label"], as_index=False)[metric]
        .mean()
        .sort_values(x_col)
    )

    print(f"\nData used for {sweep} plot:")
    print(plot_df.to_string(index=False))

    return plot_df, x_col, x_label


def pretty_metric_name(metric: str) -> str:
    """Convert CSV metric names into title-case labels."""
    names = {
        "requests_per_sec": "Requests Per Second",
        "generated_tokens_per_sec": "Generated Tokens Per Second",
        "tokens_per_sec": "Total Tokens Per Second",
        "total_time_sec": "Total Runtime (Seconds)",
        "avg_batch_latency_sec": "Average Batch Latency (Seconds)",
        "avg_http_latency_sec": "Average HTTP Latency (Seconds)",
    }
    return names.get(metric, metric.replace("_", " ").title())


def pretty_sweep_title(sweep: str) -> str:
    """Return title-case sweep names."""
    names = {
        "batch": "Batch Size Sweep",
        "sequence": "Sequence Length Sweep",
        "concurrency": "Concurrency Sweep",
    }
    return names[sweep]


def plot_grouped_bar(
    plot_df: pd.DataFrame,
    x_col: str,
    x_label: str,
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    """Create one grouped bar chart with one bar per backend."""
    pivot = plot_df.pivot(index=x_col, columns="backend_label", values=metric)

    preferred_order = ["HF", "vLLM"]
    ordered_columns = [c for c in preferred_order if c in pivot.columns]
    ordered_columns += [c for c in pivot.columns if c not in ordered_columns]
    pivot = pivot[ordered_columns]

    # Clean integer x-axis labels.
    try:
        if all(float(x).is_integer() for x in pivot.index):
            pivot.index = pivot.index.astype(int)
    except Exception:
        pass

    ax = pivot.plot(kind="bar", figsize=(8, 4.8), width=0.8)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(pretty_metric_name(metric))
    ax.legend(title="Backend")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="x", labelrotation=0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument(
        "--metric",
        default="requests_per_sec",
        help=(
            "Metric to plot. Examples: requests_per_sec, "
            "generated_tokens_per_sec, tokens_per_sec, total_time_sec"
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    for sweep in ["batch", "sequence", "concurrency"]:
        df = load_pair(results_dir, sweep)
        plot_df, x_col, x_label = prepare_sweep_data(df, sweep, args.metric)

        output_path = output_dir / f"{sweep}_sweep_{args.metric}.png"

        plot_grouped_bar(
            plot_df=plot_df,
            x_col=x_col,
            x_label=x_label,
            metric=args.metric,
            title=f"{pretty_sweep_title(sweep)}: {pretty_metric_name(args.metric)}",
            output_path=output_path,
        )

        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
