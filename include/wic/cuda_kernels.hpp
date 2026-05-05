#pragma once

#include <cuda_runtime.h>

namespace wic {

void run_serial_two_phase(const float* d_x, float* d_tmp, float* d_y, int n, cudaStream_t stream);

void run_two_kernels_with_sync(const float* d_x, float* d_tmp, float* d_y, int n);

void run_single_kernel_fused(const float* d_x, float* d_y, int n, int iters);

cudaError_t launch_dsmem_cluster_demo(const float* d_in, float* d_out, int n, int repeats,
                                      cudaDeviceProp& prop);

cudaError_t launch_tma_style_prefetch_demo(const float* d_kv, float* d_out, std::size_t total_elems,
                                           std::size_t tile_elems, int batches, cudaDeviceProp& prop);

void run_precision_matmul_fp16(const half* d_A, const half* d_B, float* d_C, int M, int N, int K);

}  // namespace wic
