# Waiting is a Choice

Microbenchmarks for **asynchronous heterogeneous speculative decoding** on NVIDIA Blackwell-class GPUs (default `sm_120`, e.g., RTX 5070 Ti family). Everything in this repo lives under `waiting-is-a-choice/`; do not sprinkle artifacts into the parent `research/` tree.

This is **not** an integrated vLLM/Triton stack. It isolates warp-specialization, cluster barriers, prefetch/compute overlap, and precision stubs so manuscript claims remain tied to reproducible kernels and CSV timelines.

---

## Warp partitioning logic (deep dive)

The GPU code maps two conceptual warp groups onto the same streaming multiprocessor (SM):

| Role | Canonical responsibility | Canonical resources |
| --- | --- | --- |
| **Messenger / Draft** | Speculative low-precision arithmetic and staging into shared/cluster-visible buffers | Registers + shared memory for ephemeral tiles; occupies lower warp IDs inside `k_fused_pipeline`. |
| **Validator / Target** | Verifies speculative states (higher-precision or redundant math) consuming the staged payloads | Occupies upper warp IDs; guarded by `_syncthreads()` / cooperative cluster scopes. |

Concrete implementations in-tree:

1. **`k_fused_pipeline` (`cuda/baselines.cu`)** splits the resident warps roughly in half (`is_draft_side`) and uses **static shared memory** (`__shared__ float sm[2048]`) as the lane between draft and verify phases. This models the *zero-external-launch* path that proves “waiting” is orchestration—not an SM limitation.
2. **`k_cluster_messenger_validator` (`cuda/dsmem_pipeline.cu`)** launches **two clustered thread blocks**, each exposing dynamic shared scratch. Explicit `cuda::launch::cluster_dims` synchronization (`cluster.sync()`) substitutes for distributed-shared-memory choreography when cross-block DSMEM primitives are gated by toolchain maturity. Messenger block 0 writes the global handshake buffer; validator block 1 consumes it—the same causal structure as DSMEM payloads.
3. **`k_kv_tile_pipeline` (`cuda/tma_prefetch.cu`)** reserves **warp groups** for prefetch and compute iterations over KV-shaped tiles (`tile_a`/`tile_b` double buffer). Prefetch lanes issue wide global reads; compute lanes perform dependent math concurrently on the trailing tile—a stand-in pattern for documenting **Nsight-visible memory/compute overlap** until CUTLASS-style TMA descriptors land in-repo.

Occupancy takeaway: heterogeneous kernels pay in **register pressure** (`__launch_bounds__` hints annotate hot paths). When draft + validators share residency, spilled registers bounce through local DRAM and erase wins—benchmark with Nsight Compute (see [`docs/profiling/nsight.md`](docs/profiling/nsight.md)).

---

## Repository map

```
cmake/               # WIC_CUDA_ARCH defaulting to 120 for RTX Blackwell configs
cpu/                 # Greedy LM-head reference + OpenMP producer/consumer ring
cuda/                # Benchmark kernels + NVIDIA bench driver (`wic_cuda_bench`)
scripts/             # Build/env/microbench/nsys/roofline helpers
docs/profiling/      # Exact Nsight command recipes
paper/               # LaTeX skeleton for the manuscript
results/             # CSV outputs (.gitignored large traces remain ignored)
Dockerfile           # CUDA 12.8+ reproducible toolchain
```

---

## Build & run

```bash
scripts/build.sh
./build/wic_cpu_demo --bind --threads 4
GIT_COMMIT="$(git rev-parse HEAD)" scripts/run_microbench.sh
scripts/roofline.py --csv results/microbench.csv
```

Environment notes:

| Variable | Meaning |
| --- | --- |
| `OMP_PROC_BIND` / `OMP_PLACES` | Set implicitly via `./wic_cpu_demo --bind` (`spread`, `cores`) to mimic deterministic socket pinning analogous to warp residency. |
| `WIC_CUDA_ARCH` | Passed through `cmake` (`scripts/build.sh` defaults to **120**). Override for non-Blackwell GPUs, e.g. `WIC_CUDA_ARCH=89`. |
| `GIT_COMMIT` | Injected into CSV metadata by `scripts/run_microbench.sh`. |

---

## CPU demo & greedy parity

`wic_cpu_demo` verifies that the specialized OpenMP drafting/verification pathway matches greedy sampling on deterministic synthetic weights (`cpu/greedy_reference.cpp`). `scripts/compare_cpu_parity.sh` wraps the parity check while allowing `V`, `H`, and `STEP`/`SEED` environment overrides:

```bash
V=768 H=128 STEPS=32 SEED=2 scripts/compare_cpu_parity.sh --draft 2 --verify 2
```

---

## Comparators versus HuggingFace / vLLM

Full framework stacks multiplex CPU scheduling, graph capture, quantization, KV paging, NCCL collectives, and more—these contributions are orthogonal to the kernels above. Whenever you cite end-to-end speedups *vs.* HuggingFace `generate()` or vLLM, treat them as **system-level anecdotes** assembled outside this repo unless you wire explicit benchmarking harnesses. The authoritative artifacts here are **`wic_cuda_bench` timings + CSV** and Nsight timelines.

---

## Roofline methodology

[`scripts/roofline.py`](scripts/roofline.py) prints a naive arithmetic intensity scaffold. Replace `--mem-gbps` and `--peak-fp16-tflops` with **measured** peaks from NVIDIA product briefings plus `memcpy`/Nsight-derived bandwidth after you characterize your exact board SKU.

---

## Cascaded speculation (future work outline)

Supporting **multiple draft models** concurrently implies:

1. Expanding DSMEM / cluster ring epochs with **explicit sequence numbers** to prevent validators from overtaking stale draft buffers.
2. Duplicating prefetch engines (conceptually multiple `k_kv_tile_pipeline` instances) multiplexed onto distinct global KV segments.
3. Revisiting warp budgets: cascading increases **register footprints** unless draft models shrink to quantized experts.

Discuss trade-offs openly in manuscripts; the codebase currently instantiates single-tier draft/target roles to keep profiler diff clean.

---

## Container workflow

```bash
docker build -t wic:devel .
docker run --gpus all -it wic:devel ./build/wic_cuda_bench --skip-correctness --iters 30
```

The image pins **CUDA 12.8.0-devel** (`nvidia/cuda:12.8.0-devel-ubuntu24.04`). Match host drivers accordingly.

---

## License

Specify your preferred OSS license externally (artifact omitted here).
