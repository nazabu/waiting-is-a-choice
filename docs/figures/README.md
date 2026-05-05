# Figure sources

## Nsight Systems (timeline overlap)

1. Build with NVTX enabled (header-only `nvtx3` ranges are compiled when CUDA NVTX headers are present).
2. Capture a short trace (avoids huge `.nsys-rep` files):
  ```bash
   scripts/capture_nsys.sh -- --bench-scope minimal --skip-correctness --iters 20 --warmup 3
  ```
3. Open `docs/figures/nsys_wic_demo.nsys-rep` in Nsight Systems.
4. Export PNG: select **CUDA HW** + **NVTX** tracks; zoom to `WIC:fused_pipeline_mixed_float` vs `WIC:two_kernels_host_sync`.
5. Save as `docs/figures/overlap_nvtx_minimal.png` and reference the exact command line in the figure caption.

The committed `overlap_nvtx_minimal.png` is generated from the NVTX push/pop trace export (`nsys stats --report nvtx_pushpop_trace`) to keep labels clean and deterministic across runs.

Screenshots illustrating **compute/memory overlap** between prefetch warps and validator warps belong here with versioned filenames.

## Nsight Compute

Follow `[docs/profiling/nsight.md](../profiling/nsight.md)` for `.ncu-rep` recipes; export charts from NCU GUI as needed.