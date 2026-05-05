#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "common.cuh"
#include "wic/cuda_kernels.hpp"

namespace wic {

__global__ void k_draft_burst(const float* __restrict__ x, float* __restrict__ y, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i] * scale);
}

__global__ void k_verify_burst(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * x[i] + 1.f;
}

// Single-kernel heterogeneous-style: lower half warps "draft", upper half "verify" on shared handshake.
// Occupancy targets: tune (maxThreads,minBlocksPerSM) with ncu + docs/profiling/occupancy.md — raise minBlocks
// only when register count allows sustained multi-block waves on your SKU.
__launch_bounds__(256, 2) __global__ void k_fused_pipeline(const float* __restrict__ gx, float* __restrict__ gy,
                                                           int n, int iters) {
    __shared__ float sm[2048];
    const int tid = threadIdx.x;
    const int warps = blockDim.x >> 5;
    const int wid = tid >> 5;
    const bool is_draft_side = (wid < (warps >> 1));

    for (int it = 0; it < iters; ++it) {
        if (is_draft_side) {
            for (int idx = tid; idx < n && idx < 2048; idx += blockDim.x) {
                sm[idx] = tanhf(gx[idx] * 0.25f);
            }
        }
        __syncthreads();
        if (!is_draft_side) {
            for (int idx = tid; idx < n && idx < 2048; idx += blockDim.x) {
                const float v = sm[idx];
                gy[idx] = v * v + 1.f;
            }
        }
        __syncthreads();
    }
}

void run_serial_two_phase(const float* d_x, float* d_tmp, float* d_y, int n, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    k_draft_burst<<<blocks, threads, 0, stream>>>(d_x, d_tmp, n, 0.25f);
    WIC_CUDA_OK(cudaGetLastError());
    k_verify_burst<<<blocks, threads, 0, stream>>>(d_tmp, d_y, n);
    WIC_CUDA_OK(cudaGetLastError());
}

void run_two_kernels_with_sync(const float* d_x, float* d_tmp, float* d_y, int n) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    k_draft_burst<<<blocks, threads>>>(d_x, d_tmp, n, 0.25f);
    WIC_CUDA_OK(cudaDeviceSynchronize());
    k_verify_burst<<<blocks, threads>>>(d_tmp, d_y, n);
    WIC_CUDA_OK(cudaDeviceSynchronize());
}

void run_single_kernel_fused(const float* d_x, float* d_y, int n, int iters) {
    const int threads = 256;
    const int blocks = 1;
    k_fused_pipeline<<<blocks, threads>>>(d_x, d_y, n, iters);
    WIC_CUDA_OK(cudaGetLastError());
}

float max_abs_diff_fused_vs_two_kernel(const float* d_x, float* d_tmp, float* d_y, int n) {
    run_two_kernels_with_sync(d_x, d_tmp, d_y, n);

    float* d_fused = nullptr;
    WIC_CUDA_OK(cudaMalloc(&d_fused, static_cast<std::size_t>(n) * sizeof(float)));

    run_single_kernel_fused(d_x, d_fused, n, 1);

    std::vector<float> href(static_cast<std::size_t>(n));
    std::vector<float> hf(static_cast<std::size_t>(n));
    WIC_CUDA_OK(
        cudaMemcpy(href.data(), d_y, static_cast<std::size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    WIC_CUDA_OK(cudaMemcpy(hf.data(), d_fused, static_cast<std::size_t>(n) * sizeof(float),
                           cudaMemcpyDeviceToHost));

    float m = 0.f;
    for (int i = 0; i < n; ++i) {
        m = fmaxf(m, fabsf(href[static_cast<std::size_t>(i)] - hf[static_cast<std::size_t>(i)]));
    }
    WIC_CUDA_OK(cudaFree(d_fused));
    return m;
}

}  // namespace wic
