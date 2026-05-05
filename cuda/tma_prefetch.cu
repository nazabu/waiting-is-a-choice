#include <cuda_runtime.h>

#include "common.cuh"
#include "wic/cuda_kernels.hpp"

namespace wic {

// Software-pipelined KV-style reader: low warp IDs prefetch the *next* tile from global into shared
// while high warp IDs compute on the *current* tile (stand-in for TMA bulk async on Blackwell).
__launch_bounds__(256, 2) __global__ void k_kv_tile_pipeline(const float* __restrict__ kv_base,
                                                             float* __restrict__ out_partial,
                                                             std::size_t tile_elems, int num_tiles) {
    __shared__ float tile_a[1024];
    __shared__ float tile_b[1024];
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int prefetch_warps = 4;
    const int compute_warps = 4;

    const int tile_stride = static_cast<int>(tile_elems > 1024 ? 1024 : tile_elems);

    int cur = 0;
    int nxt = 1;
    float* cur_buf = tile_a;
    float* nxt_buf = tile_b;

    auto load_tile = [&](float* dst, int tile_id) {
        const std::size_t off = static_cast<std::size_t>(tile_id) * tile_elems;
        for (int i = lane + wid * 32; i < tile_stride; i += 32 * prefetch_warps) {
            dst[i] = kv_base[off + static_cast<std::size_t>(i)];
        }
    };

    auto compute_tile = [&](const float* src, int tile_id) {
        float acc = 0.f;
        const int base = 32 * (wid - prefetch_warps);
        if (wid >= prefetch_warps && wid < prefetch_warps + compute_warps) {
            for (int i = lane + base; i < tile_stride; i += 32 * compute_warps) {
                acc += src[i] * src[i] + __sinf(src[i]);
            }
        }
        __syncwarp();
        for (int m = 16; m > 0; m >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, m);
        }
        if (lane == 0 && wid == prefetch_warps) {
            atomicAdd(out_partial + tile_id, acc);
        }
    };

    if (wid < prefetch_warps && num_tiles > 0) load_tile(cur_buf, 0);
    __syncthreads();

    for (int t = 0; t < num_tiles; ++t) {
        if (t + 1 < num_tiles && wid < prefetch_warps) load_tile(nxt_buf, t + 1);
        if (wid >= prefetch_warps) compute_tile(cur_buf, t);
        __syncthreads();
        float* tmp = cur_buf;
        cur_buf = nxt_buf;
        nxt_buf = tmp;
        (void)cur;
        cur = 1 - cur;
        nxt = 1 - nxt;
        __syncthreads();
    }
}

cudaError_t launch_tma_style_prefetch_demo(const float* d_kv, float* d_out, std::size_t total_elems,
                                           std::size_t tile_elems, int batches,
                                           cudaDeviceProp& prop) {
    (void)batches;
    (void)prop;
    const std::size_t max_tile = 1024;
    if (tile_elems == 0 || tile_elems > max_tile || total_elems < tile_elems) return cudaErrorInvalidValue;

    const int num_tiles = static_cast<int>(total_elems / tile_elems);
    if (num_tiles < 1) return cudaErrorInvalidValue;

    cudaMemset(d_out, 0, static_cast<size_t>(num_tiles) * sizeof(float));

    k_kv_tile_pipeline<<<1, dim3(256), 0>>>(d_kv, d_out, tile_elems, num_tiles);
    return cudaGetLastError();
}

}  // namespace wic
