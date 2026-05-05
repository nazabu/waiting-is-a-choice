#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "wic/config.hpp"
#include "wic/cuda_kernels.hpp"
#include "wic/results.hpp"

namespace {

inline void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(e) << '\n';
        std::abort();
    }
}

double mean_us(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

template <typename F>
double bench_mean_us(int warmup, int measure, F&& f) {
    for (int i = 0; i < warmup; ++i) {
        f();
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(measure));
    for (int i = 0; i < measure; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        samples.push_back(wic::us_since(t0));
    }
    return mean_us(samples);
}

}  // namespace

int main(int argc, char** argv) {
    wic::BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--warmup" && i + 1 < argc) cfg.warmup_iters = std::atoi(argv[++i]);
        if (a == "--iters" && i + 1 < argc) cfg.measure_iters = std::atoi(argv[++i]);
        if (a == "--n" && i + 1 < argc) cfg.dim = static_cast<std::size_t>(std::atoll(argv[++i]));
    }

    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

    const int n = static_cast<int>(std::min<std::size_t>(cfg.dim, 2048u));
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    float *d_x = nullptr, *d_tmp = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_x, bytes), "cudaMalloc d_x");
    check_cuda(cudaMalloc(&d_tmp, bytes), "cudaMalloc d_tmp");
    check_cuda(cudaMalloc(&d_y, bytes), "cudaMalloc d_y");
    std::vector<float> h(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) h[static_cast<std::size_t>(i)] = std::sin(0.01f * static_cast<float>(i));
    check_cuda(cudaMemcpy(d_x, h.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");

    cudaStream_t stream{};
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    const double us_serial = bench_mean_us(cfg.warmup_iters, cfg.measure_iters, [&]() {
        wic::run_serial_two_phase(d_x, d_tmp, d_y, n, stream);
    });
    const double us_two_sync = bench_mean_us(cfg.warmup_iters, cfg.measure_iters, [&]() {
        wic::run_two_kernels_with_sync(d_x, d_tmp, d_y, n);
    });
    const int fused_iters = 8;
    const double us_fused = bench_mean_us(cfg.warmup_iters, cfg.measure_iters, [&]() {
        wic::run_single_kernel_fused(d_x, d_y, n, fused_iters);
    });

    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    const std::size_t tile = std::min<std::size_t>(cfg.seq_tile, 1024u);
    const std::size_t tiles = std::max<std::size_t>(cfg.batch, 4u);
    const std::size_t kv_elems = tile * tiles;

    float* d_kv = nullptr;
    float* d_tile_out = nullptr;
    check_cuda(cudaMalloc(&d_kv, kv_elems * sizeof(float)), "cudaMalloc d_kv");
    check_cuda(cudaMalloc(&d_tile_out, tiles * sizeof(float)), "cudaMalloc d_tile_out");
    std::vector<float> hkv(kv_elems, 0.5f);
    check_cuda(cudaMemcpy(d_kv, hkv.data(), kv_elems * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy d_kv");

    const double us_kv = bench_mean_us(cfg.warmup_iters, cfg.measure_iters, [&]() {
        const cudaError_t st = wic::launch_tma_style_prefetch_demo(d_kv, d_tile_out, kv_elems, tile, 1, prop);
        if (st != cudaSuccess) {
            std::cerr << "launch_tma_style_prefetch_demo: " << cudaGetErrorString(st) << '\n';
            std::abort();
        }
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize kv");
    });

    cudaFree(d_kv);
    cudaFree(d_tile_out);

    const int M = 64;
    const int N = 64;
    const int K = 64;
    half *d_A = nullptr, *d_B = nullptr;
    float* d_C = nullptr;
    check_cuda(cudaMalloc(&d_A, static_cast<std::size_t>(M * K) * sizeof(half)), "cudaMalloc A");
    check_cuda(cudaMalloc(&d_B, static_cast<std::size_t>(K * N) * sizeof(half)), "cudaMalloc B");
    check_cuda(cudaMalloc(&d_C, static_cast<std::size_t>(M * N) * sizeof(float)), "cudaMalloc C");
    const double us_fp16 = bench_mean_us(cfg.warmup_iters, cfg.measure_iters, [&]() {
        wic::run_precision_matmul_fp16(d_A, d_B, d_C, M, N, K);
    });
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaError_t cluster_st = cudaSuccess;
    {
        float* d_sum = nullptr;
        check_cuda(cudaMalloc(&d_sum, sizeof(float)), "cudaMalloc d_sum cluster");
        cluster_st = wic::launch_dsmem_cluster_demo(d_x, d_sum, n, 2, prop);
        cudaFree(d_sum);
    }

    cudaFree(d_x);
    cudaFree(d_tmp);
    cudaFree(d_y);

    std::cout << "GPU: " << prop.name << " (SM " << prop.major << '.' << prop.minor << ")\n";
    std::cout << "n=" << n << "  warmup=" << cfg.warmup_iters << "  measure=" << cfg.measure_iters << '\n';
    std::cout << "serial_two_phase (one stream): " << us_serial << " us (mean)\n";
    std::cout << "two_kernels_host_sync:         " << us_two_sync << " us (mean)\n";
    std::cout << "fused_single_kernel iters=" << fused_iters << ": " << us_fused << " us (mean)\n";
    std::cout << "kv_tile_pipeline:              " << us_kv << " us (mean)\n";
    std::cout << "fp16_matmul " << M << 'x' << N << 'x' << K << ": " << us_fp16 << " us (mean)\n";
    if (cluster_st == cudaSuccess) {
        std::cout << "cluster_dsmem_demo:            OK (Hopper+)\n";
    } else if (cluster_st == cudaErrorNotSupported) {
        std::cout << "cluster_dsmem_demo:            skipped (requires cluster / SM 9.x+)\n";
    } else {
        std::cerr << "cluster_dsmem_demo: CUDA error " << cudaGetErrorString(cluster_st) << '\n';
        return 2;
    }

    return 0;
}
