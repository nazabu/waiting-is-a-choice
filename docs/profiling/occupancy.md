# Occupancy and register budget (warp-resource balance)

Heterogeneous kernels (`k_fused_pipeline`, `k_kv_tile_pipeline`, `k_cluster_tma_dsmem_kv`, WMMA GEMMs) carry two roles on the same SM. Use this checklist to document **measured** resource use and justify `__launch_bounds__` choices for publication.

## Build-time register dump

```bash
cmake --build build --target wic_cuda_kernels VERBOSE=1 2>&1 | tee build.log
# or add to CUDA flags: -Xptxas -v
```

Record **registers / thread** and **spills** for each hot `__global__`.

## Nsight Compute table (fill after capture)

> **Status (2026-06-02):** Columns marked `—` are pending a dedicated headless `ncu --set full`
> session with sufficient counter permissions. Desktop GPU processes (display, compositor) were
> present during headline latency capture and may restrict hardware counter access. Run on a
> headless or exclusive-compute-mode machine and populate this table before strengthening
> occupancy claims. See `docs/bench_quality_gate.md` §D for the capture command.

| Kernel                   | Block | `__launch_bounds__` | Reg/thread (ncu) | Theoretical max blocks/SM | Achieved active warps | Notes                    |
| ------------------------ | ----- | ------------------- | ---------------- | ------------------------- | --------------------- | ------------------------ |
| `k_fused_pipeline`       | 256   | `(256, 2)`          | —                | —                         | —                     | draft vs verify halves   |
| `k_kv_tile_pipeline`     | 256   | `(256, 2)`          | —                | —                         | —                     | prefetch + compute warps |
| `k_cluster_tma_dsmem_kv` | 128   | `(128, 2)`          | —                | —                         | —                     | cluster 2×1×1            |
| `k_gemm_fp16_wmma_64_nn` | 512   | `(512, 2)`          | —                | —                         | —                     | 16 WMMA tiles            |
| `k_gemm_bf16_wmma_64_nn` | 512   | `(512, 2)`          | —                | —                         | —                     | BF16 WMMA                |


Example `ncu` query (adjust metrics to toolkit):

```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    -k regex:k_fused ./build/wic_cuda_bench --bench-scope minimal --skip-correctness --iters 8
```

**“50% occupancy” definition:** State explicitly (e.g. active warps / 48 or / 64 for your SKU, or achieved warps vs hardware max concurrent warps per SM).

## Tuning playbook

1. If one role forces high register pressure, shrink unroll factors in that branch or move the slow path to a second launch.
2. Increase `minBlocksPerMultiprocessor` in `__launch_bounds__` only when register count allows **and** sustained wave count improves in `ncu`.

