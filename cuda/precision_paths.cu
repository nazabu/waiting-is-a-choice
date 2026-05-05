#include <cuda_fp16.h>

#include <cuda_runtime.h>

#include "common.cuh"
#include "wic/cuda_kernels.hpp"

namespace wic {

// Baseline FP16 matmul-like accumulation (not full WMMA) — validates toolchain + gives a precision path
// to extend toward WMMA FP8/FP4 on sm_120.
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

}  // namespace wic
