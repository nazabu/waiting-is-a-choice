#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

#include <mma.h>

#include <cuda_runtime.h>

#include "common.cuh"
#include "wic/cuda_kernels.hpp"

namespace wic {

namespace {

constexpr int WMMA_SZ = 16;

__global__ void k_matmul_fp16_naive(const half* __restrict__ A, const half* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int kk = 0; kk < K; ++kk) {
        acc += __half2float(A[row * K + kk]) * __half2float(B[kk * N + col]);
    }
    C[row * N + col] = acc;
}

// 64x64x64 WMMA GEMM — 16 warps, each owns one 16×16 output tile.
__global__ void __launch_bounds__(512, 2) k_gemm_fp16_wmma_64_nn(const half* __restrict__ A,
                                                                  const half* __restrict__ B,
                                                                  float* __restrict__ C) {
    using namespace nvcuda::wmma;
    constexpr int M = 64, N = 64, K = 64;
    const int warpId = static_cast<int>(threadIdx.x >> 5);
    const int tiles_m = M / WMMA_SZ;
    const int tiles_n = N / WMMA_SZ;
    const int nw = tiles_m * tiles_n;
    if (warpId >= nw) return;

    const int tile_row = (warpId / tiles_n) * WMMA_SZ;
    const int tile_col = (warpId % tiles_n) * WMMA_SZ;

    fragment<accumulator, WMMA_SZ, WMMA_SZ, WMMA_SZ, float> c_frag{};
    fill_fragment(c_frag, 0.0f);

    for (int kk = 0; kk < K; kk += WMMA_SZ) {
        fragment<matrix_a, WMMA_SZ, WMMA_SZ, WMMA_SZ, half, row_major> a_frag{};
        fragment<matrix_b, WMMA_SZ, WMMA_SZ, WMMA_SZ, half, col_major> b_frag{};
        load_matrix_sync(a_frag, A + tile_row * K + kk, K);
        load_matrix_sync(b_frag, B + kk * N + tile_col, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    store_matrix_sync(C + tile_row * N + tile_col, c_frag, N, mem_row_major);
}

__global__ void __launch_bounds__(512, 2) k_gemm_bf16_wmma_64_nn(const __nv_bfloat16* __restrict__ A,
                                                                 const __nv_bfloat16* __restrict__ B,
                                                                 float* __restrict__ C) {
    using namespace nvcuda::wmma;
    constexpr int M = 64, N = 64, K = 64;
    const int warpId = static_cast<int>(threadIdx.x >> 5);
    const int tiles_m = M / WMMA_SZ;
    const int tiles_n = N / WMMA_SZ;
    const int nw = tiles_m * tiles_n;
    if (warpId >= nw) return;

    const int tile_row = (warpId / tiles_n) * WMMA_SZ;
    const int tile_col = (warpId % tiles_n) * WMMA_SZ;

    fragment<accumulator, WMMA_SZ, WMMA_SZ, WMMA_SZ, float> c_frag{};
    fill_fragment(c_frag, 0.0f);

    for (int kk = 0; kk < K; kk += WMMA_SZ) {
        fragment<matrix_a, WMMA_SZ, WMMA_SZ, WMMA_SZ, __nv_bfloat16, row_major> a_frag{};
        fragment<matrix_b, WMMA_SZ, WMMA_SZ, WMMA_SZ, __nv_bfloat16, col_major> b_frag{};
        load_matrix_sync(a_frag, A + tile_row * K + kk, K);
        load_matrix_sync(b_frag, B + kk * N + tile_col, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    store_matrix_sync(C + tile_row * N + tile_col, c_frag, N, mem_row_major);
}

__global__ void k_matmul_bf16_naive(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int kk = 0; kk < K; ++kk) {
        acc += __bfloat162float(A[row * K + kk]) * __bfloat162float(B[kk * N + col]);
    }
    C[row * N + col] = acc;
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 890)
__device__ float fp8_e4m3_to_float_lut(std::uint8_t v) {
    const unsigned sign = v >> 7;
    const unsigned exp = (v >> 3) & 0x0Fu;
    const unsigned mant = v & 0x07u;
    if (exp == 0u) {
        if (mant == 0u) return sign ? -0.f : 0.f;
        return ldexpf((sign ? -1.f : 1.f) * (mant / 8.f), 1 - 7);
    }
    if (exp == 0x0Fu) {
        if (mant == 0u) return sign ? -1e10f : 1e10f;
        return 0.f;
    }
    const int e = static_cast<int>(exp) - 7;
    const float m = 1.0f + mant / 8.0f;
    const float f = ldexpf(m, e);
    return sign ? -f : f;
}
#endif

__device__ float fp8_e4m3_to_float_hw(std::uint8_t storage) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    const half h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(storage), __NV_E4M3);
    return __half2float(h);
#else
    return fp8_e4m3_to_float_lut(storage);
#endif
}

__global__ void __launch_bounds__(256, 4) k_matvec_fp8_e4m3_lut(const std::uint8_t* __restrict__ x,
                                                                const std::uint8_t* __restrict__ W,
                                                                float* __restrict__ y, int vocab, int hidden) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= vocab) return;
    float acc = 0.f;
    for (int h = 0; h < hidden; ++h) {
        const float xv = fp8_e4m3_to_float_hw(x[h]);
        const float wv = fp8_e4m3_to_float_hw(W[row * hidden + h]);
        acc += xv * wv;
    }
    y[row] = acc;
}

__global__ void k_fp4_packed_lookup_accum(const std::uint8_t* __restrict__ packed_weights, float* __restrict__ accum,
                                          std::size_t num_nibbles) {
    float local = 0.f;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nibbles;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::uint8_t b = packed_weights[i >> 1];
        const int n = static_cast<int>((i & 1u) ? (b >> 4) : (b & 0x0F));
        local += 0.0625f * static_cast<float>(n) - 0.5f;
    }
    local += __shfl_down_sync(0xffffffffu, local, 16);
    local += __shfl_down_sync(0xffffffffu, local, 8);
    local += __shfl_down_sync(0xffffffffu, local, 4);
    local += __shfl_down_sync(0xffffffffu, local, 2);
    local += __shfl_down_sync(0xffffffffu, local, 1);
    if ((threadIdx.x & 31) == 0) atomicAdd(accum, local);
}

void fprintf_fp4_stub_once(const char* msg) {
    static int guard = 0;
    if (__atomic_fetch_add(&guard, 1, __ATOMIC_RELAXED) == 0) std::fprintf(stderr, "%s", msg);
}

}  // namespace

void run_precision_matmul_fp16(const half* d_A, const half* d_B, float* d_C, int M, int N, int K) {
    if (M == 64 && N == 64 && K == 64) {
        k_gemm_fp16_wmma_64_nn<<<1, 512>>>(d_A, d_B, d_C);
        WIC_CUDA_OK(cudaGetLastError());
        return;
    }
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    k_matmul_fp16_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    WIC_CUDA_OK(cudaGetLastError());
}

void run_precision_matmul_bf16_wmma(const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, float* d_C, int M, int N,
                                    int K) {
    cudaDeviceProp prop{};
    WIC_CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    if (M == 64 && N == 64 && K == 64 && prop.major >= 8) {
        k_gemm_bf16_wmma_64_nn<<<1, 512>>>(d_A, d_B, d_C);
        WIC_CUDA_OK(cudaGetLastError());
        return;
    }
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    k_matmul_bf16_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    WIC_CUDA_OK(cudaGetLastError());
}

void run_precision_matvec_fp8_e4m3_lut(const std::uint8_t* d_x, const std::uint8_t* d_W, float* d_y, int vocab,
                                       int hidden) {
    const int threads = 256;
    const int blocks = (vocab + threads - 1) / threads;
    k_matvec_fp8_e4m3_lut<<<blocks, threads>>>(d_x, d_W, d_y, vocab, hidden);
    WIC_CUDA_OK(cudaGetLastError());
}

void run_precision_fp4_packed_microbench(const std::uint8_t* d_packed, float* d_sum, std::size_t num_nibbles) {
    const int threads = 256;
    const int blocks = 8;
    k_fp4_packed_lookup_accum<<<blocks, threads>>>(d_packed, d_sum, num_nibbles);
    WIC_CUDA_OK(cudaGetLastError());
}

void run_precision_fp4_draft_stub() {
    fprintf_fp4_stub_once("[WIC] FP4 TCGEN05 tensor-core path is not implemented in-tree; keep "
                          "k_fp4_packed_lookup_accum for traffic + pair with CUTLASS tcgen05 on sm_100+.\n");
}

}  // namespace wic
