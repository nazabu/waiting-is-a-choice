#!/usr/bin/env python3
"""
Measured roofline helper: combines microbench CSV latency with peaks taken from Nsight Compute
(or product brief). Use this for publication-grade “% of SOL” claims instead of canned defaults.

Export Nsight metrics to CSV (example):
  ncu --csv -f --metrics gpu__time_duration.sum,dram__bytes_read.sum,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
  -k regex:k_gemm_fp16_wmma ./build/wic_cuda_bench --skip-correctness --iters 8 \
  > results/ncu_fp16_gemm.csv

Then pass --peak-tflops and --peak-mem-gbps from the same capture or separate memcpy/SOL sweep.
"""

from __future__ import annotations

import argparse
import csv


def read_mean_us_from_bench(csv_path: str, row_name: str) -> float | None:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip().strip('"')
            if name == row_name:
                return float(row["mean_us"])
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Roofline using measured bench latency + measured/hw peaks")
    p.add_argument("--bench-csv", default="results/microbench.csv")
    p.add_argument("--row", default="fp16_matmul_wmma_64")
    p.add_argument("--flops", type=float, default=2 * 64**3, help="Model FLOPs for that row")
    p.add_argument("--peak-tflops", type=float, required=True, help="Tensor/SM peak TFLOP/s (from ncu/SOL)")
    p.add_argument("--peak-mem-gbps", type=float, required=True, help="Achievable DRAM GB/s for this workload")
    args = p.parse_args()

    mean_us = read_mean_us_from_bench(args.bench_csv, args.row)
    if mean_us is None:
        raise SystemExit(f"Row `{args.row}` not found in {args.bench_csv}")

    achieved_gflops = args.flops / mean_us / 1e3  # flops / µs / 1000 → GFLOP/s
    roof_gflops = args.peak_tflops * 1e3
    pct_sol = 100.0 * min(1.0, achieved_gflops / roof_gflops) if roof_gflops else 0.0

    bytes_moved = 2 * (64 * 64) * 2  # FP16 A+B read rough model (two 64² matrices half)
    achieved_bw = bytes_moved / mean_us / 1e3  # GB/s (bytes / µs / 1e3)
    pct_bw = 100.0 * min(1.0, achieved_bw / args.peak_mem_gbps) if args.peak_mem_gbps else 0.0

    ridge = args.peak_tflops / args.peak_mem_gbps

    print("WIC measured roofline (publication-style)")
    print(f"  Row: {args.row}  mean: {mean_us:.3f} µs")
    print(f"  Modeled FLOPs: {args.flops:g}")
    print(f"  Achieved: {achieved_gflops:.1f} GFLOP/s")
    print(f"  Peak tensor (input): {args.peak_tflops} TFLOP/s  →  ~{pct_sol:.1f}% of that roof")
    print(f"  Rough DRAM bytes (A+B FP16): {bytes_moved} B → {achieved_bw:.1f} GB/s vs peak {args.peak_mem_gbps} → ~{pct_bw:.1f}%")
    print(f"  Ridge intensity proxy (TFLOP/s per GB/s): {ridge:.3f}")


if __name__ == "__main__":
    main()
