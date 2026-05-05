#include <cuda_fp16.h>

#include <cstdint>
#include <cmath>

#include <cuda_runtime.h>

#include "common.cuh"
#include "wic/cuda_kernels.hpp"

namespace wic {

__global__ void k_matmul_fp16_naive(const half* __restrict__ A, const half* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = acc;
}

void run_precision_matmul_fp16(const half* d_A, const half* d_B, float* d_C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    k_matmul_fp16_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    WIC_CUDA_OK(cudaGetLastError());
}

// Device-side FP8 E4M3 -> float (microbench stand-in; not full IEEE edge-case complete).
__device__ float fp8_e4m3_to_float(std::uint8_t v) {
    const unsigned sign = v >> 7;
    const unsigned exp = (v >> 3) & 0x0Fu;
    const unsigned mant = v & 0x07u;
    if (exp == 0u) {
        if (mant == 0u) return sign ? -0.f : 0.f;
        return ldexpf((sign ? -1.f : 1.f) * (mant / 8.f), 1 - 7);
    }
    if (exp == 0x0Fu)
        if (mant == 0u) return sign ? -1e10f : 1e10f;
        return 0.f;
    const int e = static_cast<int>(exp) - 7;
    const float m = 1.0f + mant / 8.0f;
    const float f = ldexpf(m, e);
    return sign ? -f : f;
}

__global__ void k_matvec_fp8_e4m3_lut(const std::uint8_t* __restrict__ x,
                                      const std::uint8_t* __restrict__ W, float* __restrict__ y, int vocab,
                                      int hidden) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= vocab) return;
    float acc = 0.f;
    for (int h = 0; h < hidden; ++h) {
        const float xv = fp8_e4m3_to_float(x[h]);
        const float wv = fp8_e4m3_to_float(W[row * hidden + h]);
        acc += xv * wv;
    }
    y[row] = acc;
}

void run_precision_matvec_fp8_e4m3_lut(const std::uint8_t* d_x, const std::uint8_t* d_W, float* d_y,
                                       int vocab, int hidden) {
    const int threads = 256;
    const int blocks = (vocab + threads - 1) / threads;
    k_matvec_fp8_e4m3_lut<<<blocks, threads>>>(d_x, d_W, d_y, vocab, hidden);
    WIC_CUDA_OK(cudaGetLastError());
}

/// Nibble-packed FP4 stand-in (two nibbles per byte): validates decode + memory traffic pattern only.
__global__ void k_fp4_packed_lookup_accum(const std::uint8_t* __restrict__ packed_weights,
                                           float* __restrict__ accum, std::size_t num_nibbles) {
    float local = 0.f;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nibbles;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::uint8_t b = packed_weights[i >> 1];
        const int n = (i & 1u) ? (b >> 4) : (b & 0x0F);
        local += 0.0625f * static_cast<float>(n) - 0.5f;
    }
    local += __shfl_down_sync(0xffffffffu, local, 16);
    local += __shfl_down_sync(0xffffffffu, local, 8);
    local += __shfl_down_sync(0xffffffffu, local, 4);
    local += __shfl_down_sync(0xffffffffu, local, 2);
    local += __shfl_down_sync(0xffffffffu, local, 1);
    if ((threadIdx.x & 31) == 0) atomicAdd(accum, local);
}

void run_precision_fp4_packed_microbench(const std::uint8_t* d_packed, float* d_sum,
                                           std::size_t num_nibbles) {
    const int threads = 256;
    const int blocks = 8;
    k_fp4_packed_lookup_accum<<<blocks, threads>>>(d_packed, d_sum, num_nibbles);
    WIC_CUDA_OK(cudaGetLastError());
}

void run_precision_fp4_draft_stub() {
    // Reserved for native FP4 Tensor Core path on sm_120; see `run_precision_fp4_packed_microbench`.
}

}  // namespace wic
