#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace wic {

void run_serial_two_phase(const float* d_x, float* d_tmp, float* d_y, int n, cudaStream_t stream);

void run_two_kernels_with_sync(const float* d_x, float* d_tmp, float* d_y, int n);

void run_single_kernel_fused(const float* d_x, float* d_y, int n, int iters);

float max_abs_diff_fused_vs_two_kernel(const float* d_x, float* d_tmp, float* d_y, int n);

cudaError_t launch_dsmem_cluster_demo(const float* d_in, float* d_out, int n, int repeats,
                                      cudaDeviceProp& prop);

cudaError_t launch_tma_style_prefetch_demo(const float* d_kv, float* d_out, std::size_t total_elems,
                                           std::size_t tile_elems, int batches, cudaDeviceProp& prop);

cudaError_t launch_cluster_tma_dsmem_kv_demo(const float* d_kv, float* d_tile_out, std::size_t tile_elems,
                                             int num_tiles, cudaDeviceProp& prop);

cudaError_t compare_cluster_tma_kv_to_cpu_ref(const float* d_kv, const float* d_partial, int tile_elems,
                                              int num_tiles, float* max_abs_host);

void run_precision_matmul_fp16(const half* d_A, const half* d_B, float* d_C, int M, int N, int K);

/** BF16 GEMM — WMMA tile path for {64,64,64}; otherwise naive fallback. Requires SM ≥ 8.x for WMMA. */
void run_precision_matmul_bf16_wmma(const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, float* d_C, int M, int N,
                                    int K);

void run_precision_matvec_fp8_e4m3_lut(const std::uint8_t* d_x, const std::uint8_t* d_W, float* d_y,
                                       int vocab, int hidden);

void run_precision_fp4_packed_microbench(const std::uint8_t* d_packed, float* d_sum,
                                         std::size_t num_nibbles);

void run_precision_fp4_draft_stub();

cudaError_t run_stochastic_verify_demo(int n, int iters, float corrupt_fraction);

}  // namespace wic
