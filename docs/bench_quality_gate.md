# Benchmark Quality Gate (Submission Readiness)

Use this checklist before claiming benchmark readiness in paper text.

## Scope

This gate certifies **kernel-level** benchmark quality for this repo.  
It does **not** certify end-to-end serving quality.

---

## A) Environment Control (must pass all)

- [ ] GPU is dedicated (no other heavy jobs running).
- [ ] Driver/CUDA versions recorded in notes.
- [ ] Power mode/clocks policy recorded (or explicitly left default and documented).
- [ ] Same binary used for all headline runs (`cmake --build build -j` once before sweeps).
- [ ] Benchmark command lines captured verbatim in artifact notes.

Suggested capture:

```bash
scripts/env_info.sh > results/env_info.txt
```

---

## B) Headline Latency Protocol (must pass all)

Run without Nsight Compute attached for headline numbers.

- [ ] Use minimal scope for headline comparison.
- [ ] Use fixed warmup/measure settings.
- [ ] Run repeated trials (>=10) and store raw CSV per trial.
- [ ] Report median + p95 + mean/stddev across trials.

Example trial command:

```bash
OUT=results/final_bench_trial01.csv scripts/run_microbench.sh \
  --bench-scope minimal --skip-fused-correctness --iters 30 --warmup 5
```

Pass condition:

- [ ] Fused path consistently faster than two-kernel host-sync in most trials.
- [ ] Trial-to-trial variance is acceptable and documented.

---

## C) Profiling Separation (must pass all)

Keep profiling metrics separate from latency claims.

- [ ] Nsight Systems used for timeline/overlap narrative.
- [ ] Nsight Compute used for per-kernel metrics only.
- [ ] No headline latency numbers are taken from `ncu --set full` runs.
- [ ] Report text explains replay/pass overhead under NCU.

---

## D) Occupancy/Register Evidence (must pass all)

- [ ] `docs/profiling/occupancy.md` has measured values for target kernels.
- [ ] If counters unavailable, permission status is explicitly documented.
- [ ] `__launch_bounds__` choices are justified with measured data (not only intuition).

Recommended kernels:

- `k_fused_pipeline`
- `k_kv_tile_pipeline`
- `k_cluster_tma_dsmem_kv`
- `k_gemm_fp16_wmma_64_nn`
- `k_gemm_bf16_wmma_64_nn`

---

## E) Scale Robustness (must pass all)

Run at multiple sizes (example: small/medium/large).

- [ ] At least 3 `--n` settings tested (e.g., 256, 1024, 2048).
- [ ] Speedup trend vs size reported (not just one point).
- [ ] Any regressions or inversions are explained.

Example:

```bash
OUT=results/final_bench_n256.csv  scripts/run_microbench.sh --bench-scope minimal --skip-fused-correctness --iters 30 --warmup 5 --n 256
OUT=results/final_bench_n1024.csv scripts/run_microbench.sh --bench-scope minimal --skip-fused-correctness --iters 30 --warmup 5 --n 1024
OUT=results/final_bench_n2048.csv scripts/run_microbench.sh --bench-scope minimal --skip-fused-correctness --iters 30 --warmup 5 --n 2048
```

---

## F) Memory Claim Hygiene (must pass all)

- [ ] Paper/README clearly separate implemented memory mechanisms vs out-of-scope serving memory manager behavior.
- [ ] Timeline figure supports overlap claim visually.
- [ ] Memory claim wording avoids implying full KV-cache lifecycle optimization unless measured.

---

## G) Figure Readiness (must pass all)

- [ ] `docs/figures/overlap_nvtx_minimal.png` is readable at paper scale.
- [ ] Caption includes exact command context.
- [ ] Panel subtitles are explicit and descriptive.
- [ ] Figure file names and paper include paths are stable.

Optional recommended pack:

- `architecture_overview.png`
- `benchmark_protocol.png`
- `overlap_nvtx_minimal.png`
- `occupancy_registers.png`

---

## H) Artifact Completeness (must pass all)

- [ ] `results/final_bench.csv` present and matches quoted headline values.
- [ ] Supporting trial CSVs retained.
- [ ] Nsight reports/screenshots retained under `docs/figures/` (or documented storage location).
- [ ] Paper text and README numbers agree exactly with final CSV.

---

## Exit Criteria

Mark benchmark package as **submission-ready** only when:

1. All sections A-H pass, and  
2. Headline claims are reproducible from checked-in artifacts/commands, and  
3. Scope disclaimers remain accurate (kernel-level vs full serving).
