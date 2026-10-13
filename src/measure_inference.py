"""Benchmark model size and inference latency for every trained model.

The benchmark reports both:

* single-query latency, which approximates a live planner call; and
* batched latency/throughput, which shows the model's processing capacity
  without timing Python loop overhead for every row.

Inputs are deterministic by default so results from different models and runs
are comparable.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from gen_training_data import FEAT_DIM
from model_loader import BASE, CardModel, available_models

RESULTS = BASE / "results"
N_QUERIES = 1000
MAX_SINGLE_QUERY_SAMPLES = 100
RANDOM_SEED = 42


def _input_dim(model: CardModel) -> int:
    """Return and validate the feature dimension expected by a model."""
    configured = model.meta.get("input_dim", FEAT_DIM)
    try:
        input_dim = int(configured)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid input_dim in metadata for {model.name}: {configured!r}"
        ) from exc
    if input_dim < 1:
        raise ValueError(f"input_dim must be positive for {model.name}: {input_dim}")
    return input_dim


def bench_one(
    name: str,
    n_queries: int = N_QUERIES,
    seed: int = RANDOM_SEED,
) -> dict[str, float | str]:
    """Benchmark one model and return CSV-compatible measurements."""
    if n_queries < 1:
        raise ValueError(f"n_queries must be positive, got {n_queries}")

    model = CardModel(name)
    input_dim = _input_dim(model)
    rng = np.random.default_rng(seed)
    queries = rng.random((n_queries, input_dim), dtype=np.float32)

    warmup_count = min(100, n_queries)
    model.raw(queries[:warmup_count])

    batch_start = time.perf_counter()
    model.raw(queries)
    batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0

    single_count = min(MAX_SINGLE_QUERY_SAMPLES, n_queries)
    single_times = np.empty(single_count, dtype=np.float64)
    for i, query in enumerate(queries[:single_count]):
        start = time.perf_counter()
        model.raw(query.reshape(1, -1))
        single_times[i] = (time.perf_counter() - start) * 1000.0

    size_kb = Path(model.path).stat().st_size / 1024.0
    result: dict[str, float | str] = {
        "model": name,
        "size_kb": size_kb,
        "input_dim": input_dim,
        "single_mean_ms": float(single_times.mean()),
        "single_median_ms": float(np.median(single_times)),
        "single_p95_ms": float(np.percentile(single_times, 95)),
        "single_max_ms": float(single_times.max()),
        "batch_mean_ms": float(batch_elapsed_ms / n_queries),
        "throughput_qps": float(n_queries / (batch_elapsed_ms / 1000.0)),
    }

    print(f"=== Evaluating {name.upper()} ===")
    print(f"Model storage size       : {size_kb:.2f} KB")
    print(
        f"Single-query mean/median: {result['single_mean_ms']:.4f} / "
        f"{result['single_median_ms']:.4f} ms"
    )
    print(
        f"Single-query p95/max    : {result['single_p95_ms']:.4f} / "
        f"{result['single_max_ms']:.4f} ms"
    )
    print(f"Single-query samples    : {single_count}")
    print(
        f"Batch mean / throughput  : {result['batch_mean_ms']:.4f} ms / "
        f"{result['throughput_qps']:.1f} queries/s\n"
    )
    return result


def run_benchmark(
    n_queries: int = N_QUERIES,
    seed: int = RANDOM_SEED,
    output: Path = RESULTS / "inference_benchmark.csv",
) -> list[dict[str, float | str]]:
    """Run the benchmark for all available models and save the results."""
    names = available_models()
    if not names:
        raise FileNotFoundError(
            "No trained models found in results/. Run the train_*.py scripts first."
        )

    print(
        f"Benchmarking: {', '.join(names)} over {n_queries} deterministic "
        f"feature vectors (seed={seed}).\n"
    )
    results = [bench_one(name, n_queries, seed) for name in names]

    print("=" * 112)
    print("=== Summary (sorted by single-query median latency) ===")
    print(
        f"{'Model':<8} | {'Size (KB)':>10} | {'median (ms)':>12} | "
        f"{'p95 (ms)':>10} | {'batch (ms)':>10} | {'QPS':>10}"
    )
    print("-" * 112)
    for result in sorted(results, key=lambda item: float(item["single_median_ms"])):
        print(
            f"{str(result['model']).upper():<8} | "
            f"{float(result['size_kb']):>10.2f} | "
            f"{float(result['single_median_ms']):>12.4f} | "
            f"{float(result['single_p95_ms']):>10.4f} | "
            f"{float(result['batch_mean_ms']):>10.4f} | "
            f"{float(result['throughput_qps']):>10.1f}"
        )
    print("=" * 112 + "\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved -> {output}")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--queries",
        type=int,
        default=N_QUERIES,
        help=f"number of benchmark vectors per model (default: {N_QUERIES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"random seed for reproducible inputs (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=RESULTS / "inference_benchmark.csv",
        help="CSV output path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(args.queries, args.seed, args.output)
