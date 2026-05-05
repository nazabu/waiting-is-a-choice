#!/usr/bin/env python3
"""
Rough roofline scaffolding for WIC microbench: compares a naive flop model vs order-of-magnitude
hardware roofs. Replace `mem-gbps` and `peak-*` defaults with Numbers from NVIDIA briefs +
measured cuda-memcpy / Nsight SOL once you finalize hardware.
"""

from __future__ import annotations

import argparse
import csv


def read_mean_us(csv_path: str, row_name: str) -> float | None:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip().strip('"')
            if name == row_name:
                return float(row["mean_us"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/microbench.csv")
    parser.add_argument("--row", default="fp16_matmul_64")
    parser.add_argument("--mem-gbps", type=float, default=900.0)
    parser.add_argument("--peak-fp16-tflops", type=float, default=600.0)
    args = parser.parse_args()

    try:
        mean_us = read_mean_us(args.csv, args.row)
    except FileNotFoundError:
        mean_us = None

    flops = 2 * 64 * 64 * 64  # must match naive kernel sizes in cuda/main_cuda_bench.cpp
    ridge = args.peak_fp16_tflops / args.mem_gbps  # TFLOP/s per (GB/s) ≈ flop/byte OOM ridge

    print("Waiting is a Choice — rough roof check")
    print(f"  DRAM roof (GB/s, OOM default): {args.mem_gbps}")
    print(f"  FP16 tensor roof (TFLOP/s, OOM): {args.peak_fp16_tflops}")
    print(f"  Ridge intensity (OOM): {ridge:.3f} (TFLOPs·s^-1)/(GB/s) proxy")
    if mean_us is None:
        print(f"  Missing CSV `{args.csv}` — run scripts/run_microbench.sh first.")
        return

    achieved = flops / mean_us / 1000.0  # GFLOP/s naive (µs timings)
    print(f"  Row `{args.row}` mean latency: {mean_us:.3f} µs")
    print(f"  Naive GEMM-ish throughput: {achieved:.3f} GFLOP/s (model only)")
    roof_gfps = args.peak_fp16_tflops * 1000.0  # TFLOPs/s → GFLOPs/s
    pct = min(100.0, 100.0 * achieved / roof_gfps) if roof_gfps else 0.0
    print(f"  Vs OOM FP16 tensor roof (~{roof_gfps:.0f} GFLOP/s from {args.peak_fp16_tflops} TFLOP/s): ~{pct:.2f}%")


if __name__ == "__main__":
    main()
