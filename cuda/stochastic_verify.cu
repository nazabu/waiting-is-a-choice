/*
 * Host-orchestrated stochastic verification: inject draft errors with a configurable rate,
 * then flag mismatches on device to prove the verifier path observes disagreements cleanly.
 */

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "common.cuh"

namespace wic {

namespace {

__global__ void k_flag_squared_error(const float* draft, const float* ref, unsigned char* flags, float tol_sq,
                                     int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float e = draft[i] - ref[i];
    flags[i] = (e * e > tol_sq) ? static_cast<unsigned char>(1) : static_cast<unsigned char>(0);
}

}  // namespace

cudaError_t run_stochastic_verify_demo(int n, int iters, float corrupt_fraction) {
    if (n < 1 || iters < 1 || corrupt_fraction < 0.f || corrupt_fraction > 1.f) return cudaErrorInvalidValue;

    float *d_draft = nullptr, *d_ref = nullptr;
    unsigned char* d_flags = nullptr;
    const cudaError_t a0 = cudaMalloc(&d_draft, static_cast<size_t>(n) * sizeof(float));
    if (a0 != cudaSuccess) return a0;
    const cudaError_t a1 = cudaMalloc(&d_ref, static_cast<size_t>(n) * sizeof(float));
    if (a1 != cudaSuccess) {
        cudaFree(d_draft);
        return a1;
    }
    const cudaError_t a2 = cudaMalloc(&d_flags, static_cast<size_t>(n) * sizeof(unsigned char));
    if (a2 != cudaSuccess) {
        cudaFree(d_draft);
        cudaFree(d_ref);
        return a2;
    }

    std::vector<float> href(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) href[static_cast<size_t>(i)] = std::sin(0.03f * static_cast<float>(i));

    std::mt19937 rng(static_cast<unsigned>(92411u ^ static_cast<unsigned>(n)));
    std::uniform_real_distribution<float> uni01(0.f, 1.f);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const float tol_sq = 1.0f;  // corruption adds +10 so |e|>>1

    for (int it = 0; it < iters; ++it) {
        WIC_CUDA_OK(cudaMemcpy(d_ref, href.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float> draft = href;
        std::vector<unsigned char> expect_corrupt(static_cast<size_t>(n), 0);
        for (int i = 0; i < n; ++i) {
            if (uni01(rng) < corrupt_fraction) {
                draft[static_cast<size_t>(i)] = href[static_cast<size_t>(i)] + 10.f;
                expect_corrupt[static_cast<size_t>(i)] = 1;
            }
        }
        WIC_CUDA_OK(cudaMemcpy(d_draft, draft.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));
        k_flag_squared_error<<<blocks, threads>>>(d_draft, d_ref, d_flags, tol_sq, n);
        const cudaError_t lr = cudaGetLastError();
        if (lr != cudaSuccess) {
            cudaFree(d_draft);
            cudaFree(d_ref);
            cudaFree(d_flags);
            return lr;
        }
        WIC_CUDA_OK(cudaDeviceSynchronize());

        std::vector<unsigned char> flags(static_cast<size_t>(n));
        WIC_CUDA_OK(cudaMemcpy(flags.data(), d_flags, static_cast<size_t>(n) * sizeof(unsigned char),
                               cudaMemcpyDeviceToHost));

        for (int i = 0; i < n; ++i) {
            const size_t z = static_cast<size_t>(i);
            if (expect_corrupt[z] && flags[z] != 1) {
                std::fprintf(stderr, "stochastic_verify: missed corruption at index %d (iter %d)\n", i, it);
                cudaFree(d_draft);
                cudaFree(d_ref);
                cudaFree(d_flags);
                return cudaErrorUnknown;
            }
            if (!expect_corrupt[z] && flags[z] != 0) {
                std::fprintf(stderr, "stochastic_verify: false positive at index %d (iter %d)\n", i, it);
                cudaFree(d_draft);
                cudaFree(d_ref);
                cudaFree(d_flags);
                return cudaErrorUnknown;
            }
        }
    }

    cudaFree(d_draft);
    cudaFree(d_ref);
    cudaFree(d_flags);
    return cudaSuccess;
}

}  // namespace wic
