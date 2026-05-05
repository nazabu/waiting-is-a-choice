#include <cooperative_groups.h>

#include <vector>

#include <cuda_runtime.h>

#include "common.cuh"
#include "wic/cuda_kernels.hpp"

namespace cg = cooperative_groups;

namespace wic {

// Hopper+ thread-block cluster: two blocks share cluster.sync() barriers; block 0 acts as messenger
// (draft transform), block 1 as validator (secondary transform), with a global staging buffer.
__global__ void __launch_bounds__(128, 2) __cluster_dims__(2, 1, 1) k_cluster_messenger_validator(
    float* __restrict__ g_payload, int elems, int repeats) {
    const cg::cluster_group cluster = cg::this_cluster();
    const unsigned bid = cluster.block_rank();

    extern __shared__ float smem[];

    const int stride = static_cast<int>(blockDim.x);
    for (int r = 0; r < repeats; ++r) {
        for (int i = static_cast<int>(threadIdx.x); i < elems; i += stride) {
            smem[i] = sinf(static_cast<float>(i + r * 17));
        }
        cg::this_thread_block().sync();
        cluster.sync();

        if (bid == 0u) {
            for (int i = static_cast<int>(threadIdx.x); i < elems; i += stride) {
                g_payload[i] = smem[i] * 0.25f;
            }
        }

        cluster.sync();

        if (bid == 1u) {
            for (int i = static_cast<int>(threadIdx.x); i < elems; i += stride) {
                const float v = g_payload[i];
                smem[i] = v * v + 1.f;
            }
        }

        cg::this_thread_block().sync();
        cluster.sync();
    }
}

cudaError_t launch_dsmem_cluster_demo(const float* /*d_in*/, float* d_out, int n, int repeats,
                                      cudaDeviceProp& prop) {
    if (prop.major < 9) return cudaErrorNotSupported;

    const int elems_cap = 1024;
    const int elems = n > elems_cap ? elems_cap : (n < 1 ? 1 : n);
    float* d_payload = nullptr;
    const cudaError_t a0 = cudaMalloc(&d_payload, static_cast<size_t>(elems_cap) * sizeof(float));
    if (a0 != cudaSuccess) return a0;

    constexpr int threads = 128;
    const size_t smem_bytes = static_cast<size_t>(elems_cap) * sizeof(float);

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(2, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.dynamicSmemBytes = static_cast<unsigned>(smem_bytes);

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    const cudaError_t st =
        cudaLaunchKernelEx(&cfg, k_cluster_messenger_validator, d_payload, elems, repeats);
    if (st != cudaSuccess) {
        cudaFree(d_payload);
        return st;
    }
    WIC_CUDA_OK(cudaDeviceSynchronize());

    if (d_out != nullptr) {
        std::vector<float> host(static_cast<size_t>(elems_cap));
        WIC_CUDA_OK(cudaMemcpy(host.data(), d_payload, static_cast<size_t>(elems) * sizeof(float),
                                cudaMemcpyDeviceToHost));
        float sum = 0.f;
        for (int i = 0; i < elems; ++i) sum += host[static_cast<size_t>(i)];
        WIC_CUDA_OK(cudaMemcpy(d_out, &sum, sizeof(float), cudaMemcpyHostToDevice));
    }

    cudaFree(d_payload);
    return cudaSuccess;
}

}  // namespace wic
