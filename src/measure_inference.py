"""Benchmark: model storage size + inference latency, for ALL 4 models.

FIXED vs previous version: only benchmarked MLP + LightGBM. RF and XGBoost
were missing entirely, so the "is this lightweight enough to run inside a
live planner" comparison was incomplete. Now loops over every model that
has a saved artifact (via model_loader.available_models()).
"""
import os
import time
import csv

import numpy as np

from model_loader import CardModel, BASE, available_models

RESULTS = BASE / "results"
FEAT_DIM = 81
N_QUERIES = 1000


# def bench_one(name):
#     model = CardModel(name)
#     size_kb = os.path.getsize(model.path) / 1024
    
#     print(f"=== Evaluating {name.upper()} ===")
#     print(f"Model Storage Size: {size_kb:.2f} KB")

#     queries = np.random.rand(N_QUERIES, FEAT_DIM).astype(np.float32)
#     _ = model.raw(queries[:10])  # warm-up

#     t0 = time.perf_counter()
#     for q in queries:
#         _ = model.raw(q.reshape(1, -1))
#     ms = (time.perf_counter() - t0) / N_QUERIES * 1000
    
#     print(f"Average Inference Latency: {ms:.4f} ms per query\n")
#     return name, size_kb, ms


# def run_benchmark():
#     names = available_models()
#     if not names:
#         print("No trained models found in results/ — run the train_*.py scripts first.")
#         return
        
#     print(f"Benchmarking: {', '.join(names)}\n")
    
#     results = []
#     for name in names:
#         results.append(bench_one(name))

#     # --- Summary Table ---
#     print("=" * 55)
#     print("=== Summary (sorted by inference latency) ===")
#     print(f"{'Model':<12} | {'Size (KB)':>12} | {'Latency (ms)':>14}")
#     print("-" * 55)
    
#     # Sort results by latency (index 2) to easily see the fastest model
#     for name, size_kb, ms in sorted(results, key=lambda x: x[2]):
#         print(f"{name.upper():<12} | {size_kb:>12.2f} | {ms:>14.4f}")
        
#     print("=" * 55 + "\n")

def bench_one(name):
    model = CardModel(name)
    size_kb = os.path.getsize(model.path) / 1024

    queries = np.random.rand(N_QUERIES, FEAT_DIM).astype(np.float32)
    _ = model.raw(queries[:100])            # more warm-up

    times = np.empty(N_QUERIES, dtype=np.float64)
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        _ = model.raw(q.reshape(1, -1))
        times[i] = (time.perf_counter() - t0) * 1000.0

    print(f"=== Evaluating {name.upper()} ===")
    print(f"Model Storage Size       : {size_kb:.2f} KB")
    print(f"Latency mean / median    : {times.mean():.4f} / {np.median(times):.4f} ms")
    print(f"Latency std / p95 / max  : {times.std():.4f} / "
          f"{np.percentile(times, 95):.4f} / {times.max():.4f} ms\n")

    return {
        "model": name, "size_kb": size_kb,
        "mean_ms":  float(times.mean()),
        "median_ms":float(np.median(times)),
        "p95_ms":   float(np.percentile(times, 95)),
        "max_ms":   float(times.max()),
    }


def run_benchmark():
    names = available_models()
    if not names:
        print("No trained models found in results/.")
        return

    print(f"Benchmarking: {', '.join(names)} over {N_QUERIES} random feature vectors.\n")
    results = [bench_one(n) for n in names]

    print("=" * 78)
    print("=== Summary (sorted by median latency) ===")
    print(f"{'Model':<8} | {'Size (KB)':>10} | {'median (ms)':>12} | "
          f"{'mean (ms)':>10} | {'p95 (ms)':>10} | {'max (ms)':>10}")
    print("-" * 78)
    for r in sorted(results, key=lambda x: x["median_ms"]):
        print(f"{r['model'].upper():<8} | {r['size_kb']:>10.2f} | "
              f"{r['median_ms']:>12.4f} | {r['mean_ms']:>10.4f} | "
              f"{r['p95_ms']:>10.4f} | {r['max_ms']:>10.4f}")
    print("=" * 78 + "\n")

    out = RESULTS / "inference_benchmark.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"Saved -> {out}")

if __name__ == "__main__":
    run_benchmark()