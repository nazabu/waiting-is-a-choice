/*
 * Slice (C): Hopper+/Blackwell — `cp.async.bulk.tensor` (rank-3 map) into shared::cta, combined with
 * `mbarrier.arrive.expect_tx` + `mbarrier_try_wait_parity`; validator reads messenger SMEM via map_shared_rank.
 * A volatile producer counter in rank-0 SMEM is updated after each tile (visible through DSMEM mapping).
 */

#include <cooperative_groups.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#include <vector>

#include <cuda/ptx>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.cuh"
#include "wic/cuda_kernels.hpp"

namespace cg = cooperative_groups;

namespace wic {

namespace {

using cuda::ptx::cp_async_bulk_tensor;
using cuda::ptx::mbarrier_arrive_expect_tx;
using cuda::ptx::mbarrier_init;
using cuda::ptx::mbarrier_try_wait_parity;
using cuda::ptx::scope_cta;
using cuda::ptx::sem_release;
using cuda::ptx::space_global;
using cuda::ptx::space_shared;

constexpr unsigned kBarrierOff = 0u;
constexpr unsigned kProdOff = 16u;
constexpr unsigned kConsOff = 20u;  // uint32 consumer epoch (anti-lapping with producer)
constexpr unsigned kTileBaseOff = 128u;

__host__ __device__ __forceinline__ unsigned round_up_align(unsigned v, unsigned a) {
    return (v + a - 1u) & ~(a - 1u);
}

__device__ float warp_reduce_sum(float v) {
    for (int m = 16; m > 0; m >>= 1) v += __shfl_down_sync(0xffffffffu, v, m);
    return v;
}

__device__ float block_reduce_sum(float v) {
    __shared__ float sw[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sw[wid] = v;
    __syncthreads();
    const int nw = static_cast<int>((blockDim.x + 31u) >> 5);
    float b = (static_cast<int>(threadIdx.x) < nw) ? sw[threadIdx.x] : 0.f;
    b = warp_reduce_sum(b);
    return b;
}

__launch_bounds__(128, 2)
    __global__ void __cluster_dims__(2, 1, 1)
        k_cluster_tma_dsmem_kv(const __grid_constant__ CUtensorMap tensor_map, float* __restrict__ g_out,
                               int tile_elems, int num_tiles) {
    cg::cluster_group cluster = cg::this_cluster();
    const unsigned br = cluster.block_rank();

    extern __shared__ __align__(128) unsigned char smem_raw[];

    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_raw + kBarrierOff);
    volatile uint32_t* prod_local = reinterpret_cast<volatile uint32_t*>(smem_raw + kProdOff);
    volatile uint32_t* cons_local = reinterpret_cast<volatile uint32_t*>(smem_raw + kConsOff);
    unsigned char* tile_base = smem_raw + kTileBaseOff;

    const unsigned xfer_bytes_u = static_cast<unsigned>(tile_elems) * sizeof(float);
    float* tile_this = reinterpret_cast<float*>(tile_base);

    if (br == 0u) {
        if (threadIdx.x == 0) {
            *prod_local = 0u;
            *cons_local = 0u;
            mbarrier_init(mbar, 1u);
        }
        __syncthreads();
    }
    cluster.sync();

    for (int t = 0; t < num_tiles; ++t) {
        if (br == 0u) {
            if (threadIdx.x == 0 && t > 0) {
                while (*cons_local < static_cast<uint32_t>(t)) {
                    __threadfence_cluster();
                    __nanosleep(128);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                (void)mbarrier_arrive_expect_tx(sem_release, scope_cta, space_shared, mbar, xfer_bytes_u);
                const int32_t coords[3] = {t * tile_elems, 0, 0};
                cp_async_bulk_tensor(space_shared, space_global, tile_this, &tensor_map, coords, mbar);
            }
            __syncthreads();

            const uint32_t phase_parity = static_cast<uint32_t>(t & 1);
            for (;;) {
                if (mbarrier_try_wait_parity(mbar, phase_parity)) break;
                __nanosleep(256);
            }
            __threadfence_cluster();
            if (threadIdx.x == 0) *prod_local = static_cast<uint32_t>(t + 1);
            __threadfence_cluster();
            __syncthreads();
        }

        cluster.sync();

        if (br == 1u) {
            float* peer_tile = cluster.map_shared_rank(tile_this, 0);
            float acc = 0.f;
            for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
                const float x = peer_tile[i];
                acc += x * x + __sinf(x);
            }
            acc = block_reduce_sum(acc);
            if (threadIdx.x == 0) atomicAdd(g_out + t, acc);
            if (threadIdx.x == 0) {
                unsigned int* peer_cons =
                    reinterpret_cast<unsigned int*>(cluster.map_shared_rank(smem_raw + kConsOff, 0));
                atomicAdd(peer_cons, 1u);
            }
            __threadfence_cluster();
            __syncthreads();
        }

        cluster.sync();
    }
}

inline bool drv_ok(CUresult r, const char* ctx) {
    if (r == CUDA_SUCCESS) return true;
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    std::fprintf(stderr, "[cluster_tma_dsmem] %s : %s\n", ctx, s ? s : "driver error");
    return false;
}

bool encode_kv_tensor_rank3(float* kv_ptr, int tile_elems, int num_tiles, CUtensorMap* tm) {
    std::memset(tm, 0, sizeof(CUtensorMap));
    const std::size_t esz = sizeof(float);
    const int ntot = tile_elems * num_tiles;
    cuuint64_t gdim[3] = {(cuuint64_t)ntot, 1ull, 1ull};
    const cuuint64_t stride0 = (cuuint64_t)gdim[0] * esz;
    cuuint64_t gstride[2] = {stride0, stride0};
    cuuint32_t box[3] = {(cuuint32_t)tile_elems, 1u, 1u};
    cuuint32_t elem_stride[3] = {1u, 1u, 1u};
    CUresult rc = cuTensorMapEncodeTiled(tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3u, kv_ptr, gdim, gstride, box,
                                         elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                         CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return drv_ok(rc, "cuTensorMapEncodeTiled");
}

float cpu_kv_tile_reduce_ref(const float* kv, std::size_t tile_elems, int tile_idx) {
    float acc = 0.f;
    const std::size_t base = static_cast<std::size_t>(tile_idx) * tile_elems;
    for (std::size_t i = 0; i < tile_elems; ++i) {
        const float x = kv[base + i];
        acc += x * x + std::sin(static_cast<float>(x));
    }
    return acc;
}

}  // namespace

cudaError_t launch_cluster_tma_dsmem_kv_demo(const float* d_kv, float* d_tile_out,
                                               std::size_t tile_elems, int num_tiles,
                                               cudaDeviceProp& prop) {
    if (prop.major < 9) return cudaErrorNotSupported;
    if (tile_elems == 0 || tile_elems > 1024u || num_tiles < 1) return cudaErrorInvalidValue;
    if (((tile_elems * sizeof(float)) & 15u) != 0u) return cudaErrorInvalidValue;

    (void)cuInit(0);

    const int te = static_cast<int>(tile_elems);
    float* kv_nc = const_cast<float*>(d_kv);

    alignas(64) CUtensorMap tm_host{};
    if (!encode_kv_tensor_rank3(kv_nc, te, num_tiles, &tm_host)) return cudaErrorNotSupported;

    const unsigned tile_bytes = round_up_align(static_cast<unsigned>(te) * sizeof(float), 128u);
    const size_t dyn = static_cast<size_t>(kTileBaseOff) + tile_bytes + 512u;

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(2, 1, 1);
    cfg.blockDim = dim3(128, 1, 1);
    cfg.dynamicSmemBytes = static_cast<unsigned>(dyn);
    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    WIC_CUDA_OK(cudaMemset(d_tile_out, 0, static_cast<size_t>(num_tiles) * sizeof(float)));

    CUtensorMap tm_arg = tm_host;
    cudaError_t err = cudaLaunchKernelEx(&cfg, k_cluster_tma_dsmem_kv, tm_arg, d_tile_out, te, num_tiles);
    return err != cudaSuccess ? err : cudaGetLastError();
}

cudaError_t compare_cluster_tma_kv_to_cpu_ref(const float* d_kv, const float* d_partial, int tile_elems,
                                              int num_tiles, float* max_abs_host) {
    std::vector<float> h_kv(static_cast<size_t>(tile_elems) * static_cast<size_t>(num_tiles));
    std::vector<float> h_part(static_cast<size_t>(num_tiles));
    WIC_CUDA_OK(cudaMemcpy(h_kv.data(), d_kv, h_kv.size() * sizeof(float), cudaMemcpyDeviceToHost));
    WIC_CUDA_OK(cudaMemcpy(h_part.data(), d_partial, h_part.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float m = 0.f;
    for (int t = 0; t < num_tiles; ++t) {
        const float gold = cpu_kv_tile_reduce_ref(h_kv.data(), static_cast<size_t>(tile_elems), t);
        m = fmaxf(m, fabsf(h_part[static_cast<size_t>(t)] - gold));
    }
    *max_abs_host = m;
    return cudaSuccess;
}

}  // namespace wic
