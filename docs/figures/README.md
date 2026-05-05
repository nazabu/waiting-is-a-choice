# Figure sources

## Nsight Systems (timeline overlap)

1. Build with NVTX enabled (CMake finds `libnvToolsExt` under the CUDA toolkit).
2. Capture a short trace (avoids huge `.nsys-rep` files):
   ```bash
   scripts/capture_nsys.sh -- --bench-scope minimal --skip-correctness --iters 20 --warmup 3
   ```
3. Open `docs/figures/nsys_wic_demo*.nsys-rep` in Nsight Systems.
4. Export PNG: select **CUDA HW** + **NVTX** tracks; zoom to `WIC:fused_pipeline_mixed_float` vs back-to-back `WIC:two_kernels_host_sync` child kernels.
5. Save as e.g. `overlap_nvtx_minimal_2026.png` and reference the exact command line in the figure caption.

Screenshots illustrating **compute/memory overlap** between prefetch warps and validator warps belong here with versioned filenames.

## Nsight Compute

Follow [`docs/profiling/nsight.md`](../profiling/nsight.md) for `.ncu-rep` recipes; export charts from NCU GUI as needed.
