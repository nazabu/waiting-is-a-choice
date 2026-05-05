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

### Manual crop checklist (step-by-step)

1. Keep only tracks needed for the paper claim:
   - `NVTX`
   - `CUDA HW` kernels
   - `CUDA API` (optional, only if launch/sync gaps are visible)
2. Hide unrelated CPU threads, memory engines, and empty tracks.
3. Set one contiguous time window that includes:
   - `WIC:serial_two_phase_stream`
   - `WIC:two_kernels_host_sync`
   - `WIC:fused_pipeline_mixed_float`
4. Start slightly before the serial range and end slightly after the fused range.
5. Ensure text labels are readable at publication size (target >=2200 px width).
6. Export with light background for print clarity.
7. Save to `docs/figures/overlap_nvtx_minimal.png` (same filename keeps paper include paths stable).

The committed `overlap_nvtx_minimal.png` is a paper-color, two-panel figure:
- Panel A: NVTX timeline from `nsys stats --report nvtx_pushpop_trace`.
- Panel B: bar chart from `results/final_bench.csv` (means with stddev error bars).

If you want a pure Nsight GUI export instead, replace this file with your manually curated screenshot and keep the same filename.

Screenshots illustrating **compute/memory overlap** between prefetch warps and validator warps belong here with versioned filenames.

## Suggested paper figure pack

- `architecture_overview.png` (method section)
- `benchmark_protocol.png` (experimental protocol section)
- `overlap_nvtx_minimal.png` (results section, two-panel)
- `occupancy_registers.png` (discussion section)

## Nsight Compute

Follow `[docs/profiling/nsight.md](../profiling/nsight.md)` for `.ncu-rep` recipes; export charts from NCU GUI as needed.