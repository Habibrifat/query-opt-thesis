# Gemini Version: 1.0
"""Benchmark model storage size and inference latency for every trained model.

================================================================================
SYSTEM OVERHEAD EVALUATION & RESEARCH LITERATURE REFERENCES (2015 - 2025)
================================================================================
Why this 3rd evaluation script is needed alongside `evaluate_tpch.py` and
`evaluate_join_order_quality.py`:

  In a real relational database optimizer (such as PostgreSQL), a cardinality
  estimator is invoked dozens or hundreds of times during dynamic-programming
  join enumeration (once per candidate sub-plan). Therefore, a learned
  cardinality model must be evaluated not only on:
    1. Estimation Accuracy (`evaluate_tpch.py` -> Q-Error, Symmetric Accuracy %)
    2. Plan-Choice Quality (`evaluate_join_order_quality.py` -> Regret, Pair OK)
  but also on:
    3. System Overhead (`measure_inference.py` -> Model Size KB, Latency ms, QPS)
       to prove that the ML inference time (~0.01 - 2.0 ms) is negligible
       compared to the ~2,000+ ms saved by choosing a better join order!

--- PRIMARY OVERHEAD METRICS (Quote these first in your thesis) ---

  1. single_median_ms (Typical Single-Subplan Planner Latency — Lower is better)
     * Meaning : Median wall-clock time (ms) to predict 1 sub-plan feature
                 vector (1x23), simulating a live PostgreSQL planner call.
     * Target  : Must be << query execution time (typically < 1-5 ms) so
                 optimization overhead never outweighs execution savings.
     * Papers  : - Wang et al. (PVLDB 2021): "Are We Ready For Learned
                   Cardinality Estimation?", PVLDB 14(9), pp. 1640-1654.
                   URL: https://www.vldb.org/pvldb/vol14/p1640-wang.pdf
                 - Kipf et al. (CIDR 2019): "Learned Cardinalities: Estimating
                   Correlated Joins with Deep Learning" (MSCN).
                   URL: https://www.cidrdb.org/cidr2019/papers/p101-kipf-cidr19.pdf

  2. size_kb (Disk / Memory Footprint in Kilobytes — Lower is better)
     * Meaning : Serialized model artifact size on disk (`results/<model>.*`).
                 Shows how lightweight the model is compared to database catalog
                 histograms and multi-megabyte data-driven deep models.
     * Papers  : - Wang et al. (PVLDB 2021); Dutt et al. (PVLDB 2019):
                   "Selectivity Estimation for Range Predicates using
                   Lightweight Models", PVLDB 12(9), pp. 1044-1057.

--- SECONDARY OVERHEAD METRICS (Tail latency & batched capacity) ---

  3. single_p95_ms (95th-Percentile Tail Inference Latency — Lower is better)
     * Meaning : 95% of single-query planner calls finish within this time (ms).
     * Papers  : - Marcus et al. (PVLDB 2019 - Neo); Negi et al. (PVLDB 2023):
                   URL: https://www.vldb.org/pvldb/vol16/p1520-negi.pdf

  4. batch_mean_ms & throughput_qps (Vectorized Batch Latency & Queries/Sec)
     * Meaning : Amortized per-query latency (ms) and throughput (queries/sec)
                 when scoring all candidate sub-plans of a join graph in a
                 single vectorized batch call (eliminating Python loop overhead).
     * Papers  : - Chronis et al. (arXiv 2024 - CardBench):
                   URL: https://arxiv.org/html/2408.16170v1
                 - Zhu et al. (PVLDB 2024/2025 - PRICE):
                   URL: https://www.vldb.org/pvldb/vol18/p637-zhu.pdf

USAGE EXAMPLES:
  python src/measure_inference.py
  python src/measure_inference.py --models rf mlp
  python src/measure_inference.py -n 2000 --seed 42
================================================================================
"""
import argparse
import csv
import time
import warnings
from pathlib import Path

# Suppress NumPy 2.5 / Python 3.13 DeprecationWarning inside joblib/sklearn pickle loader
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from gen_training_data import FEAT_DIM
from model_loader import BASE, CardModel, available_models

RESULTS = BASE / "results"
N_QUERIES = 1000
MAX_SINGLE_QUERY_SAMPLES = 100
RANDOM_SEED = 42


def _input_dim(model: CardModel) -> int:
    """Return and validate the feature dimension (23-D) expected by a model."""
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
    """Benchmark one model's storage size, single-query latency, and batch QPS.

    Methodology:
      1. Warm-up pass (up to 100 queries) so JIT/cache initialization does not
         pollute timing measurements.
      2. Vectorized batch pass (`n_queries` at once) using `time.perf_counter()`
         to measure vectorized throughput (`batch_mean_ms`, `throughput_qps`).
      3. Single-query loop (`MAX_SINGLE_QUERY_SAMPLES` individual `1 x 23` calls)
         to measure live query-planner latency (`single_median_ms`, `single_p95_ms`).
    """
    if n_queries < 1:
        raise ValueError(f"n_queries must be positive, got {n_queries}")

    model = CardModel(name)
    input_dim = _input_dim(model)
    rng = np.random.default_rng(seed)
    queries = rng.random((n_queries, input_dim), dtype=np.float32)

    # 1. Warm-up pass to initialize internal buffers / thread pools
    warmup_count = min(100, n_queries)
    model.raw(queries[:warmup_count])

    # 2. Vectorized batch inference timing
    batch_start = time.perf_counter()
    model.raw(queries)
    batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0

    # 3. Single-query inference timing (simulating live optimizer sub-plan calls)
    single_count = min(MAX_SINGLE_QUERY_SAMPLES, n_queries)
    single_times = np.empty(single_count, dtype=np.float64)
    for i, query in enumerate(queries[:single_count]):
        start = time.perf_counter()
        model.raw(query.reshape(1, -1))
        single_times[i] = (time.perf_counter() - start) * 1000.0

    size_kb = Path(model.path).stat().st_size / 1024.0
    result: dict[str, float | str] = {
        "model": name,
        "size_kb": round(size_kb, 2),
        "input_dim": input_dim,
        "single_mean_ms": round(float(single_times.mean()), 4),
        "single_median_ms": round(float(np.median(single_times)), 4),
        "single_p95_ms": round(float(np.percentile(single_times, 95)), 4),
        "single_max_ms": round(float(single_times.max()), 4),
        "batch_mean_ms": round(float(batch_elapsed_ms / n_queries), 4),
        "throughput_qps": round(float(n_queries / (batch_elapsed_ms / 1000.0)), 1),
    }

    print(f"=== Evaluating {name.upper()} ===")
    print(f"Model storage size       : {size_kb:.2f} KB")
    print(
        f"Single-query mean/median: {float(result['single_mean_ms']):.4f} / "
        f"{float(result['single_median_ms']):.4f} ms"
    )
    print(
        f"Single-query p95/max    : {float(result['single_p95_ms']):.4f} / "
        f"{float(result['single_max_ms']):.4f} ms"
    )
    print(f"Single-query samples    : {single_count}")
    print(
        f"Batch mean / throughput  : {float(result['batch_mean_ms']):.4f} ms / "
        f"{float(result['throughput_qps']):.1f} queries/s\n"
    )
    return result


def run_benchmark(
    n_queries: int = N_QUERIES,
    seed: int = RANDOM_SEED,
    output: Path = RESULTS / "inference_benchmark.csv",
    models: list[str] | None = None,
) -> list[dict[str, float | str]]:
    """Run the inference overhead benchmark for selected models and save CSV."""
    names = models or available_models()
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
    print("=== Inference Overhead Summary (sorted by single-query median latency) ===")
    print("PRIMARY   : median (ms) [live planner latency, lower=better] | Size (KB) [memory footprint, lower=better]")
    print("SECONDARY : p95 (ms) [tail latency], batch (ms) [vectorized latency], QPS [batch queries/sec, higher=better]")
    print("-" * 112)
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
    """Parse command-line arguments for the inference latency benchmark."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="subset of models to benchmark: mlp lgbm rf xgb (default: all available)",
    )
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
    run_benchmark(args.queries, args.seed, args.output, args.models)
