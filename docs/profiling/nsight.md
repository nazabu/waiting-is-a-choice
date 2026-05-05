# Nsight profiling playbook (Waiting is a Choice)

This project’s microbench emits three GPU paths that correspond to manuscript figures:

- **Serial-but-pipelined on one stream**: `run_serial_two_phase` (still two kernels chained without host sync between them—the stream carries the dependence).
- **Two-kernel with host fences**: models cross-launch stalls from CPU orchestration.
- **Single-kernel fused / warp-specialized**: `run_single_kernel_fused` demonstrating shared-memory handoff inside one grid launch.

## Nsight Systems (timeline / overlap narrative)

Capture a timeline (GPU rows + CUDA API):

```bash
scripts/capture_nsys.sh
# inspect docs/figures/nsys_wic_demo.nsys-rep in Nsight Systems GUI
```

`wic_cuda_bench` pushes **NVTX** ranges (`WIC:serial_two_phase_stream`, `WIC:two_kernels_host_sync`, `WIC:fused_pipeline_mixed_float`, `WIC:kv_tile_pipeline_prefetch`, `WIC:cluster_tma_dsmem_`*, `WIC:fp16_wmma_gemm_64`, etc.) when CUDA `nvtx3` headers are present at build time. Use the NVTX row in the timeline to show **Draft/Verify analog** spacing: compare `two_kernels_host_sync` (gaps between kernels) vs `fused_pipeline_mixed_float` (single continuous kernel).

Recommended manual crop procedure (for paper figure quality):

1. Keep only `NVTX`, `CUDA HW`, and (optionally) `CUDA API` rows.
2. Hide unrelated thread/process rows to reduce clutter.
3. Select one window containing all three benchmark regions:
  - `WIC:serial_two_phase_stream`
  - `WIC:two_kernels_host_sync`
  - `WIC:fused_pipeline_mixed_float`
4. Export high-resolution PNG (>=2200 px width) with readable labels.
5. Use filename `docs/figures/overlap_nvtx_minimal.png` so LaTeX paths remain unchanged.

For short traces (recommended for screenshot prep):

```bash
./build/wic_cuda_bench --bench-scope minimal --skip-correctness --iters 10 --warmup 2
scripts/capture_nsys.sh -- --bench-scope minimal --skip-correctness --iters 10 --warmup 2
```

For a narrower manual slice:

```bash
nsys profile --trace cuda,nvtx -o results/nsys_mini ./build/wic_cuda_bench --skip-correctness --bench-scope minimal
```

Interpretation cues:

1. Prefer **overlap** between memory traffic (HtoD/engine) and kernel waves when KV tile prefetch + compute kernels are active.
2. Compare **CUDA API gaps** (`cudaDeviceSynchronize`, extra launches) vs the single-kernel path.

Place curated screenshots under [docs/figures/](../figures/) and cite the exact command line beside each figure caption.

## Nsight Compute (kernel-level)

Recommended sections for `results/*.ncu-rep`:

- Achieved occupancy, register spills, warp stall breakdown (often `smsp__warp_issue_stalled_`*).
- Memory workload analysis for the KV prefetch kernel vs fused pipeline.

Example CLI (adapt metric sets to your toolkit version):

```bash
ncu --set full -k regex:k_fused -o results/fused.ncu-rep ./build/wic_cuda_bench --skip-correctness --iters 8
ncu --set roofline --kernel-name-base demangled ./build/wic_cuda_bench --skip-correctness
```

For **`k_cluster_tma_dsmem_kv`** (hardware TMA + cluster shared), use Nsight Compute sections that surface **TMA bulk** traffic and **barrier / mbarrier** behavior in addition to the usual L1TEX/L2 story. Helpful starting points (names vary by toolkit build):

- **Memory workload / TMA**: metrics such as `l1tex__t_sectors_pipe_tma`* or toolkit-specific “Tensor Memory Accelerator” counters when present.
- **Stalls**: `smsp__warp_issue_stalled_mbarrier` (or similarly named stall buckets) alongside `smsp__warp_issue_stalled_lg_throttle` / `..._long_scoreboard` for global dependence.
- Capture: `ncu --set full -k regex:k_cluster_tma_dsmem -o results/cluster_tma.ncu-rep ./build/wic_cuda_bench --skip-correctness --iters 8`

Refine the metric set after the first on-device capture; figures stay user-generated.

## Occupancy / register worksheet

See [occupancy.md](occupancy.md) for the kernel table template and `__launch_bounds__` tuning notes.

## Measured roofline

After exporting `ncu` peaks, combine with `results/microbench.csv` via [scripts/roofline_measured.py](../../scripts/roofline_measured.py) (requires `--peak-tflops` and `--peak-mem-gbps` from your capture).