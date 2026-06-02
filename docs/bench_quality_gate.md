# Benchmark Quality Gate (Submission Readiness)

Use this checklist before claiming benchmark readiness in paper text.

## Scope

This gate certifies **kernel-level** benchmark quality for this repo.  
It does **not** certify end-to-end serving quality.

**Current status (2026-06-02):** Sections B, C, E, F, G, H pass. Sections A and D are
partially complete and annotated below.

---

## A) Environment Control

- [x] Driver/CUDA versions recorded — see `results/env_info.txt` (NVIDIA-SMI 595.58.03, CUDA 13.2, RTX 5070 Ti).
- [x] Toolchain versions recorded — g++ 13.3.0, cmake 4.3.3, nvcc built 2025-08-20.
- [x] Same binary used for all headline runs — single `cmake --build build -j` before trial sweep.
- [x] Benchmark command lines captured verbatim — see `results/benchmark_summary.md` and paper §Experimental Protocol.
- [ ] **GPU dedicated (no other heavy jobs)** — ⚠️ desktop/display processes (Xorg, gnome-shell, Cursor) were present on the GPU during headline capture. This inflates variance. For a tighter latency claim, rerun with display off or on a headless machine.
- [ ] **Power mode/clocks policy** — left at default (P5 idle state at capture time). For publication-grade numbers, lock clocks with `nvidia-smi -lgc <freq>` and document the setting.

Capture command used:

```bash
scripts/env_info.sh > results/env_info.txt
```

---

## B) Headline Latency Protocol ✓

- [x] Minimal scope used for headline comparison.
- [x] Fixed warmup (5) and measure (30 iterations) settings.
- [x] 10 repeated trials stored as `results/trials/final_bench_trial01.csv` … `trial10.csv`.
- [x] Median, p95, mean, and stddev reported across trials in `results/benchmark_summary.md` and paper Table 1.
- [x] Fused path faster than two-kernel host-sync in 9 of 10 trials (median 1.56×).
- [x] Trial-to-trial variance documented honestly — high p95 explained in paper §Results and §Limitations.

---

## C) Profiling Separation ✓

- [x] Nsight Systems used for timeline/overlap narrative only (`docs/figures/overlap_nvtx_minimal.png`).
- [x] Nsight Compute reserved for per-kernel resource analysis (`results/fused.ncu-rep`).
- [x] No headline latency numbers taken from `ncu --set full` runs.
- [x] Paper §Experimental Protocol explains that profiler replay can perturb timing.

---

## D) Occupancy/Register Evidence

- [ ] **`docs/profiling/occupancy.md` measured values** — ⚠️ table rows are present but Reg/thread, max blocks/SM, and achieved active warps columns are empty. Nsight Compute counter capture requires elevated permissions (`--set full` may be limited by system policy). Populate after a dedicated headless `ncu` session.
- [x] Counter-unavailability is explicitly documented — paper §Resource Status states "measured Nsight Compute register and occupancy entries are not yet committed."
- [ ] **`__launch_bounds__` choices justified with measured data** — current annotations are based on design intent (`(256, 2)` for heterogeneous kernels); measured register counts pending NCU capture.

Capture command (run with sufficient permissions):

```bash
ncu --set full -k regex:k_fused -o results/fused_full.ncu-rep \
    ./build/wic_cuda_bench --bench-scope minimal --skip-correctness --iters 8
```

---

## E) Scale Robustness ✓

- [x] Three `--n` settings tested: 256, 1024, 2048 (see `results/final_bench_n256/1024/2048.csv`).
- [x] Speedup trend vs size reported in paper Table 2.
- [x] Non-monotonic behavior at n=256 explained in paper §Size sweep: single-run per size, fusion overhead dominates at small n; n=1024 shows favorable fused result.

Note: size-sweep runs are single-trial each. Repeated size sweeps remain future work.

---

## F) Memory Claim Hygiene ✓

- [x] Paper and README clearly separate implemented kernel memory mechanisms from out-of-scope serving memory manager behavior (see paper §Memory scope clarification and README §Memory Scope).
- [x] Timeline figure (`overlap_nvtx_minimal.png`) supports overlap claim visually.
- [x] Memory claim wording avoids implying full KV-cache lifecycle optimization.

---

## G) Figure Readiness ✓

- [x] `docs/figures/overlap_nvtx_minimal.png` committed and readable at paper scale.
- [x] Caption includes exact capture command (`scripts/capture_nsys.sh -- --bench-scope minimal --skip-correctness --iters 20 --warmup 3`).
- [x] `docs/figures/architecture_overview.png` committed.
- [x] `docs/figures/benchmark_protocol.png` committed.
- [x] Figure filenames and paper `\IfFileExists` include paths are stable.
- [ ] `occupancy_registers.png` — optional; pending NCU capture (Section D above).

---

## H) Artifact Completeness ✓

- [x] `results/final_bench.csv` present; headline values match paper Table 1 and README exactly.
- [x] Supporting trial CSVs retained (`results/trials/final_bench_trial01–10.csv`).
- [x] Nsight timeline PNG committed under `docs/figures/`; `.nsys-rep` binary is local-only (large; see note in `docs/figures/README.md`).
- [x] Paper text and README numbers agree with final CSV.
- [x] `results/env_info.txt` committed (captured 2026-06-02).

---

## Exit Criteria

Mark benchmark package as **submission-ready** only when:

1. All sections A-H pass, and
2. Headline claims are reproducible from checked-in artifacts/commands, and
3. Scope disclaimers remain accurate (kernel-level vs full serving).

**Current verdict:** Sections B, C, E, F, G, H pass. Section A passes for toolchain/command
documentation but has an open note on GPU dedication and clock locking. Section D is the main
remaining gap (occupancy table unpopulated). The paper already states this limitation
explicitly. The artifact is suitable for submission as a **mechanism study with honest variance
reporting**; upgrade to a full latency claim requires completing Section A environment controls
and Section D occupancy capture.
