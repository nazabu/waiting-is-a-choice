#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double stddev_sample(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean(v);
    double s2 = 0.0;
    for (double x : v) {
        const double d = x - m;
        s2 += d * d;
    }
    return std::sqrt(s2 / static_cast<double>(v.size() - 1));
}

template <typename F>
void bench_samples(int warmup, int measure, F&& f, std::vector<double>* out_us) {
    for (int i = 0; i < warmup; ++i) {
        f();
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    out_us->clear();
    out_us->reserve(static_cast<std::size_t>(measure));
    for (int i = 0; i < measure; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        out_us->push_back(wic::us_since(t0));
    }
}

std::string git_sha() {
    const char* g = std::getenv("GIT_COMMIT");
    if (g && g[0]) return g;
    return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
    wic::BenchConfig cfg;
    const char* csv_path = nullptr;
    bool skip_correctness = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--warmup" && i + 1 < argc) cfg.warmup_iters = std::atoi(argv[++i]);
        if (a == "--iters" && i + 1 < argc) cfg.measure_iters = std::atoi(argv[++i]);
        if (a == "--n" && i + 1 < argc) cfg.dim = static_cast<std::size_t>(std::atoll(argv[++i]));
        if (a == "--csv" && i + 1 < argc) csv_path = argv[++i];
        if (a == "--skip-correctness") skip_correctness = true;
    }

    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

    const int cuda_rt = CUDART_VERSION;
    std::string cuda_ver =
        std::to_string(cuda_rt / 1000) + "." + std::to_string((cuda_rt % 1000) / 10);

    const int n = static_cast<int>(std::min<std::size_t>(cfg.dim, 2048u));
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    float *d_x = nullptr, *d_tmp = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_x, bytes), "cudaMalloc d_x");
    check_cuda(cudaMalloc(&d_tmp, bytes), "cudaMalloc d_tmp");
    check_cuda(cudaMalloc(&d_y, bytes), "cudaMalloc d_y");
    std::vector<float> h(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) h[static_cast<std::size_t>(i)] = std::sin(0.01f * static_cast<float>(i));
    check_cuda(cudaMemcpy(d_x, h.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");

    if (!skip_correctness) {
        const float err = wic::max_abs_diff_fused_vs_two_kernel(d_x, d_tmp, d_y, n);
        std::cout << "correctness max|fused - two_kernel| (1 fused iter): " << err << '\n';
        if (!(err < 5e-2f)) {
            std::cerr << "Correctness check failed (max abs error " << err << ").\n";
            return 4;
        }
        check_cuda(cudaMemcpy(d_x, h.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy reset d_x");
    }

    cudaStream_t stream{};
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    std::vector<double> samples;

    bench_samples(cfg.warmup_iters, cfg.measure_iters,
                  [&]() { wic::run_serial_two_phase(d_x, d_tmp, d_y, n, stream); }, &samples);
    const double us_serial_m = mean(samples);
    const double us_serial_s = stddev_sample(samples);

    bench_samples(cfg.warmup_iters, cfg.measure_iters,
                  [&]() { wic::run_two_kernels_with_sync(d_x, d_tmp, d_y, n); }, &samples);
    const double us_two_sync_m = mean(samples);
    const double us_two_sync_s = stddev_sample(samples);

    const int fused_iters = 8;
    bench_samples(cfg.warmup_iters, cfg.measure_iters, [&]() {
        wic::run_single_kernel_fused(d_x, d_y, n, fused_iters);
    }, &samples);
    const double us_fused_m = mean(samples);
    const double us_fused_s = stddev_sample(samples);

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

    bench_samples(cfg.warmup_iters, cfg.measure_iters, [&]() {
        const cudaError_t st = wic::launch_tma_style_prefetch_demo(d_kv, d_tile_out, kv_elems, tile, 1, prop);
        if (st != cudaSuccess) {
            std::cerr << "launch_tma_style_prefetch_demo: " << cudaGetErrorString(st) << '\n';
            std::abort();
        }
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize kv");
    }, &samples);
    const double us_kv_m = mean(samples);
    const double us_kv_s = stddev_sample(samples);

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
    bench_samples(cfg.warmup_iters, cfg.measure_iters,
                  [&]() { wic::run_precision_matmul_fp16(d_A, d_B, d_C, M, N, K); }, &samples);
    const double us_fp16_m = mean(samples);
    const double us_fp16_s = stddev_sample(samples);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    const int V = 128;
    const int Hh = 64;
    std::uint8_t *d_x8 = nullptr, *d_W8 = nullptr;
    float* d_y8 = nullptr;
    check_cuda(cudaMalloc(&d_x8, static_cast<std::size_t>(Hh)), "cudaMalloc x8");
    check_cuda(cudaMalloc(&d_W8, static_cast<std::size_t>(V * Hh)), "cudaMalloc W8");
    check_cuda(cudaMalloc(&d_y8, static_cast<std::size_t>(V) * sizeof(float)), "cudaMalloc y8");
    {
        std::vector<std::uint8_t> hx(static_cast<std::size_t>(Hh), 0x3b);
        std::vector<std::uint8_t> hW(static_cast<std::size_t>(V * Hh), 0x3c);
        check_cuda(cudaMemcpy(d_x8, hx.data(), hx.size(), cudaMemcpyHostToDevice), "cudaMemcpy x8");
        check_cuda(cudaMemcpy(d_W8, hW.data(), hW.size(), cudaMemcpyHostToDevice), "cudaMemcpy W8");
    }
    bench_samples(cfg.warmup_iters, cfg.measure_iters,
                  [&]() { wic::run_precision_matvec_fp8_e4m3_lut(d_x8, d_W8, d_y8, V, Hh); }, &samples);
    const double us_fp8_m = mean(samples);
    const double us_fp8_s = stddev_sample(samples);
    cudaFree(d_x8);
    cudaFree(d_W8);
    cudaFree(d_y8);

    std::uint8_t* d_p4 = nullptr;
    float* d_s4 = nullptr;
    const std::size_t num_nibbles = 1u << 16;
    check_cuda(cudaMalloc(&d_p4, num_nibbles / 2), "cudaMalloc fp4");
    check_cuda(cudaMalloc(&d_s4, sizeof(float)), "cudaMalloc fp4 sum");
    check_cuda(cudaMemset(d_p4, 0xab, num_nibbles / 2), "memset p4");
    bench_samples(cfg.warmup_iters, cfg.measure_iters, [&]() {
        check_cuda(cudaMemset(d_s4, 0, sizeof(float)), "memset s4");
        wic::run_precision_fp4_packed_microbench(d_p4, d_s4, num_nibbles);
    }, &samples);
    const double us_fp4_m = mean(samples);
    const double us_fp4_s = stddev_sample(samples);
    cudaFree(d_p4);
    cudaFree(d_s4);

    wic::run_precision_fp4_draft_stub();

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
    std::cout << "serial_two_phase (one stream): " << us_serial_m << " us (mean), " << us_serial_s
              << " us (stddev)\n";
    std::cout << "two_kernels_host_sync:         " << us_two_sync_m << " us (mean), " << us_two_sync_s
              << " us (stddev)\n";
    std::cout << "fused_single_kernel iters=" << fused_iters << ": " << us_fused_m << " us (mean), "
              << us_fused_s << " us (stddev)\n";
    std::cout << "kv_tile_pipeline:              " << us_kv_m << " us (mean), " << us_kv_s
              << " us (stddev)\n";
    std::cout << "fp16_matmul " << M << 'x' << N << 'x' << K << ": " << us_fp16_m << " us (mean), "
              << us_fp16_s << " us (stddev)\n";
    std::cout << "fp8_e4m3_lut matvec " << V << 'x' << Hh << ": " << us_fp8_m << " us (mean), " << us_fp8_s
              << " us (stddev)\n";
    std::cout << "fp4_packed_microbench:         " << us_fp4_m << " us (mean), " << us_fp4_s
              << " us (stddev)\n";

    if (cluster_st == cudaSuccess) {
        std::cout << "cluster_dsmem_demo:            OK (Hopper+ cluster launch)\n";
    } else if (cluster_st == cudaErrorNotSupported) {
        std::cout << "cluster_dsmem_demo:            skipped (requires SM 9.x+ cluster hardware)\n";
    } else {
        std::cerr << "cluster_dsmem_demo: CUDA error " << cudaGetErrorString(cluster_st) << '\n';
        return 2;
    }

    if (csv_path) {
        std::ofstream out(csv_path);
        wic::write_csv_header(out);
        const std::string gpu = prop.name;
        const std::string sha = git_sha();
        auto row = [&](const char* name, double m, double s, double gops) {
            wic::BenchResult r;
            r.name = name;
            r.mean_us = m;
            r.stddev_us = s;
            r.throughput_gops = gops;
            wic::write_csv_row(out, r, gpu, cuda_ver, sha);
        };
        row("serial_two_phase_stream", us_serial_m, us_serial_s, 0.0);
        row("two_kernels_sync", us_two_sync_m, us_two_sync_s, 0.0);
        row("fused_pipeline", us_fused_m, us_fused_s, 0.0);
        row("kv_tile_pipeline", us_kv_m, us_kv_s, 0.0);
        row("fp16_matmul_64", us_fp16_m, us_fp16_s,
            (2.0 * M * N * K) / (us_fp16_m * 1e-3));  // naive FLOP model
        row("fp8_matvec_lut", us_fp8_m, us_fp8_s,
            (2.0 * static_cast<double>(V) * Hh) / (us_fp8_m * 1e-3));
        row("fp4_packed_microbench", us_fp4_m, us_fp4_s, static_cast<double>(num_nibbles) / (us_fp4_m * 1e-3));
        std::cout << "Wrote CSV: " << csv_path << '\n';
    }

    return 0;
}
