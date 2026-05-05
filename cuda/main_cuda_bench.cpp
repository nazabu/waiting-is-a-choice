#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "wic/config.hpp"
#include "wic/cuda_kernels.hpp"
#include "wic/nvtx_scope.hpp"
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

enum class BenchScope { Full, Core, Minimal };

BenchScope parse_bench_scope(const std::string& s) {
    if (s == "minimal") return BenchScope::Minimal;
    if (s == "core") return BenchScope::Core;
    return BenchScope::Full;
}

}  // namespace

int main(int argc, char** argv) {
    wic::BenchConfig cfg;
    const char* csv_path = nullptr;
    bool skip_correctness = false;
    bool skip_fused_correctness = false;
    std::string bench_scope_str = "full";
    int stochastic_iters = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--warmup" && i + 1 < argc) cfg.warmup_iters = std::atoi(argv[++i]);
        if (a == "--iters" && i + 1 < argc) cfg.measure_iters = std::atoi(argv[++i]);
        if (a == "--n" && i + 1 < argc) cfg.dim = static_cast<std::size_t>(std::atoll(argv[++i]));
        if (a == "--csv" && i + 1 < argc) csv_path = argv[++i];
        if (a == "--skip-correctness") skip_correctness = true;
        if (a == "--skip-fused-correctness") skip_fused_correctness = true;
        if (a == "--bench-scope" && i + 1 < argc) bench_scope_str = argv[++i];
        if (a == "--stochastic-verify-iters" && i + 1 < argc)
            stochastic_iters = std::max(0, std::atoi(argv[++i]));
    }

    const BenchScope bench_scope = parse_bench_scope(bench_scope_str);
    const bool do_kv_cluster = bench_scope != BenchScope::Minimal;
    const bool do_heavy_prec = bench_scope == BenchScope::Full;

    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

    const int cuda_rt = CUDART_VERSION;
    std::string cuda_ver =
        std::to_string(cuda_rt / 1000) + "." + std::to_string((cuda_rt % 1000) / 10);

    const int n = static_cast<int>(std::min<std::size_t>(cfg.dim, 2048u));
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    if (stochastic_iters > 0) {
        wic::NvtxRange rz("WIC:stochastic_verify");
        cudaError_t sv =
            wic::run_stochastic_verify_demo(std::max(64, std::min(n, 4096)), stochastic_iters, 0.20f);
        if (sv != cudaSuccess) {
            std::cerr << "run_stochastic_verify_demo failed\n";
            return 9;
        }
        std::cout << "stochastic_verify (fraction=0.2 iters=" << stochastic_iters << "): OK\n";
    }

    float *d_x = nullptr, *d_tmp = nullptr, *d_y = nullptr;
    check_cuda(cudaMalloc(&d_x, bytes), "cudaMalloc d_x");
    check_cuda(cudaMalloc(&d_tmp, bytes), "cudaMalloc d_tmp");
    check_cuda(cudaMalloc(&d_y, bytes), "cudaMalloc d_y");
    std::vector<float> h(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) h[static_cast<size_t>(i)] = std::sin(0.01f * static_cast<float>(i));
    check_cuda(cudaMemcpy(d_x, h.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");

    if (!skip_correctness && !skip_fused_correctness) {
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

    {
        wic::NvtxRange rz("WIC:serial_two_phase_stream");
        bench_samples(cfg.warmup_iters, cfg.measure_iters,
                      [&]() { wic::run_serial_two_phase(d_x, d_tmp, d_y, n, stream); }, &samples);
    }
    const double us_serial_m = mean(samples);
    const double us_serial_s = stddev_sample(samples);

    {
        wic::NvtxRange rz("WIC:two_kernels_host_sync");
        bench_samples(cfg.warmup_iters, cfg.measure_iters,
                      [&]() { wic::run_two_kernels_with_sync(d_x, d_tmp, d_y, n); }, &samples);
    }
    const double us_two_sync_m = mean(samples);
    const double us_two_sync_s = stddev_sample(samples);

    const int fused_iters = 8;
    {
        wic::NvtxRange rz("WIC:fused_pipeline_mixed_float");
        bench_samples(cfg.warmup_iters, cfg.measure_iters, [&]() {
            wic::run_single_kernel_fused(d_x, d_y, n, fused_iters);
        }, &samples);
    }
    const double us_fused_m = mean(samples);
    const double us_fused_s = stddev_sample(samples);

    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    double us_kv_m = 0., us_kv_s = 0.;
    cudaError_t cluster_tma_st = cudaErrorNotSupported;
    double us_cluster_tma_m = 0., us_cluster_tma_s = 0.;

    const std::size_t tile = std::min<std::size_t>(cfg.seq_tile, 1024u);
    const std::size_t tiles = std::max<std::size_t>(cfg.batch, 4u);
    const std::size_t kv_elems = tile * tiles;

    float *d_kv = nullptr, *d_tile_out = nullptr;
    if (do_kv_cluster) {
        check_cuda(cudaMalloc(&d_kv, kv_elems * sizeof(float)), "cudaMalloc d_kv");
        check_cuda(cudaMalloc(&d_tile_out, tiles * sizeof(float)), "cudaMalloc d_tile_out");
        std::vector<float> hkv(kv_elems, 0.5f);
        check_cuda(cudaMemcpy(d_kv, hkv.data(), kv_elems * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy d_kv");

        if (prop.major >= 9 && (tile * sizeof(float)) % 16u == 0u) {
            wic::NvtxRange rz("WIC:cluster_tma_dsmem_probe");
            cluster_tma_st = wic::launch_cluster_tma_dsmem_kv_demo(d_kv, d_tile_out, tile,
                                                                   static_cast<int>(tiles), prop);
            check_cuda(cudaDeviceSynchronize(), "cluster_tma_dsmem probe sync");
            if (cluster_tma_st == cudaSuccess && !skip_correctness) {
                float kv_err = 0.f;
                check_cuda(wic::compare_cluster_tma_kv_to_cpu_ref(d_kv, d_tile_out, static_cast<int>(tile),
                                                                  static_cast<int>(tiles), &kv_err),
                           "compare_cluster_tma_kv_to_cpu_ref");
                std::cout << "correctness cluster_tma_dsmem vs CPU ref (KV tiles): max abs " << kv_err << '\n';
                if (!(kv_err < 5e-2f)) {
                    std::cerr << "cluster_tma_dsmem correctness failed (max abs " << kv_err << ").\n";
                    return 5;
                }
            } else if (cluster_tma_st != cudaSuccess && cluster_tma_st != cudaErrorNotSupported) {
                std::cerr << "launch_cluster_tma_dsmem_kv_demo: " << cudaGetErrorString(cluster_tma_st) << '\n';
                return 5;
            }
            check_cuda(cudaMemset(d_tile_out, 0, tiles * sizeof(float)), "cudaMemset d_tile_out reset");
        }

        {
            wic::NvtxRange rz("WIC:kv_tile_pipeline_prefetch");
            bench_samples(cfg.warmup_iters, cfg.measure_iters, [&]() {
                const cudaError_t st =
                    wic::launch_tma_style_prefetch_demo(d_kv, d_tile_out, kv_elems, tile, 1, prop);
                if (st != cudaSuccess) {
                    std::cerr << "launch_tma_style_prefetch_demo: " << cudaGetErrorString(st) << '\n';
                    std::abort();
                }
                check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize kv");
            }, &samples);
        }
        us_kv_m = mean(samples);
        us_kv_s = stddev_sample(samples);

        if (cluster_tma_st == cudaSuccess) {
            wic::NvtxRange rz("WIC:cluster_tma_dsmem_kv_timing");
            bench_samples(cfg.warmup_iters, cfg.measure_iters, [&]() {
                check_cuda(wic::launch_cluster_tma_dsmem_kv_demo(d_kv, d_tile_out, tile,
                                                                 static_cast<int>(tiles), prop),
                           "launch_cluster_tma_dsmem_kv_demo bench launch");
                check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize cluster_tma_dsmem");
            }, &samples);
            us_cluster_tma_m = mean(samples);
            us_cluster_tma_s = stddev_sample(samples);
        }
    }

    const int M = 64;
    const int N = 64;
    const int K = 64;

    half *d_A = nullptr, *d_B = nullptr;
    float* d_C = nullptr;
    double us_fp16_m = 0., us_fp16_s = 0.;
    double us_bf16_m = 0., us_bf16_s = 0.;
    double us_fp8_m = 0., us_fp8_s = 0.;
    double us_fp4_m = 0., us_fp4_s = 0.;
    const std::size_t num_nibbles = do_heavy_prec ? (1u << 16) : 0u;
    std::uint8_t* d_p4 = nullptr;
    float* d_s4 = nullptr;
    cudaError_t cluster_st = cudaSuccess;

    if (do_heavy_prec) {
        check_cuda(cudaMalloc(&d_A, static_cast<std::size_t>(M * K) * sizeof(half)), "cudaMalloc A");
        check_cuda(cudaMalloc(&d_B, static_cast<std::size_t>(K * N) * sizeof(half)), "cudaMalloc B");
        check_cuda(cudaMalloc(&d_C, static_cast<std::size_t>(M * N) * sizeof(float)), "cudaMalloc C");
        {
            wic::NvtxRange rz("WIC:fp16_wmma_gemm_64");
            bench_samples(cfg.warmup_iters, cfg.measure_iters,
                          [&]() { wic::run_precision_matmul_fp16(d_A, d_B, d_C, M, N, K); }, &samples);
        }
        us_fp16_m = mean(samples);
        us_fp16_s = stddev_sample(samples);

        __nv_bfloat16 *d_Ab = nullptr, *d_Bb = nullptr;
        check_cuda(cudaMalloc(&d_Ab, static_cast<std::size_t>(M * K) * sizeof(__nv_bfloat16)), "cudaMalloc Ab");
        check_cuda(cudaMalloc(&d_Bb, static_cast<std::size_t>(K * N) * sizeof(__nv_bfloat16)), "cudaMalloc Bb");
        {
            std::vector<__nv_bfloat16> hb(static_cast<std::size_t>(M * K), __float2bfloat16(0.03125f)),
                hbB(static_cast<std::size_t>(K * N), __float2bfloat16(0.03125f));
            check_cuda(cudaMemcpy(d_Ab, hb.data(), hb.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
                       "Memcpy Ab");
            check_cuda(cudaMemcpy(d_Bb, hbB.data(), hbB.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice),
                       "Memcpy Bb");
        }
        {
            wic::NvtxRange rz("WIC:bf16_wmma_gemm_64");
            bench_samples(cfg.warmup_iters, cfg.measure_iters, [&]() {
                wic::run_precision_matmul_bf16_wmma(d_Ab, d_Bb, d_C, M, N, K);
            }, &samples);
        }
        us_bf16_m = mean(samples);
        us_bf16_s = stddev_sample(samples);
        cudaFree(d_Ab);
        cudaFree(d_Bb);

        cudaFree(d_A);
        cudaFree(d_B);
        d_A = nullptr;
        d_B = nullptr;

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
        {
            wic::NvtxRange rz("WIC:fp8_e4m3_matvec");
            bench_samples(cfg.warmup_iters, cfg.measure_iters, [&]() {
                wic::run_precision_matvec_fp8_e4m3_lut(d_x8, d_W8, d_y8, V, Hh);
            }, &samples);
        }
        us_fp8_m = mean(samples);
        us_fp8_s = stddev_sample(samples);
        cudaFree(d_x8);
        cudaFree(d_W8);
        cudaFree(d_y8);

        check_cuda(cudaMalloc(&d_p4, num_nibbles / 2), "cudaMalloc fp4");
        check_cuda(cudaMalloc(&d_s4, sizeof(float)), "cudaMalloc fp4 sum");
        check_cuda(cudaMemset(d_p4, 0xab, num_nibbles / 2), "memset p4");
        {
            wic::NvtxRange rz("WIC:fp4_packed_microbench");
            bench_samples(cfg.warmup_iters, cfg.measure_iters, [&]() {
                check_cuda(cudaMemset(d_s4, 0, sizeof(float)), "memset s4");
                wic::run_precision_fp4_packed_microbench(d_p4, d_s4, num_nibbles);
            }, &samples);
        }
        us_fp4_m = mean(samples);
        us_fp4_s = stddev_sample(samples);

        cudaFree(d_p4);
        cudaFree(d_s4);

        {
            wic::NvtxRange rz("WIC:fp4_tensorcore_stub_notice");
            wic::run_precision_fp4_draft_stub();
        }

        cluster_st = cudaSuccess;
        {
            wic::NvtxRange rz("WIC:cluster_dsmem_global_demo");
            float* d_sum = nullptr;
            check_cuda(cudaMalloc(&d_sum, sizeof(float)), "cudaMalloc d_sum cluster");
            cluster_st = wic::launch_dsmem_cluster_demo(d_x, d_sum, n, 2, prop);
            cudaFree(d_sum);
        }
        cudaFree(d_C);
        d_C = nullptr;
    }

    if (do_kv_cluster) {
        cudaFree(d_kv);
        cudaFree(d_tile_out);
    }
    cudaFree(d_x);
    cudaFree(d_tmp);
    cudaFree(d_y);

    std::cout << "GPU: " << prop.name << " (SM " << prop.major << '.' << prop.minor << ")\n";
    std::cout << "bench_scope=" << bench_scope_str << "  n=" << n << "  warmup=" << cfg.warmup_iters
              << "  measure=" << cfg.measure_iters << '\n';
    std::cout << "serial_two_phase (one stream): " << us_serial_m << " us (mean), " << us_serial_s
              << " us (stddev)\n";
    std::cout << "two_kernels_host_sync:         " << us_two_sync_m << " us (mean), " << us_two_sync_s
              << " us (stddev)\n";
    std::cout << "fused_single_kernel iters=" << fused_iters << ": " << us_fused_m << " us (mean), " << us_fused_s
              << " us (stddev)\n";
    if (do_kv_cluster) {
        std::cout << "kv_tile_pipeline:              " << us_kv_m << " us (mean), " << us_kv_s
                  << " us (stddev)\n";
        if (cluster_tma_st == cudaSuccess) {
            std::cout << "cluster_tma_dsmem_kv:           " << us_cluster_tma_m << " us (mean), " << us_cluster_tma_s
                      << " us (stddev)\n";
        } else if (cluster_tma_st == cudaErrorNotSupported) {
            std::cout << "cluster_tma_dsmem_kv:           skipped (SM < 9 or tensor encode / stride gating)\n";
        }
    }

    if (do_heavy_prec) {
        std::cout << "fp16_tc_gemm " << M << 'x' << N << 'x' << K << ": " << us_fp16_m << " us (mean), " << us_fp16_s
                  << " us (stddev)\n";
        std::cout << "bf16_tc_gemm " << M << 'x' << N << 'x' << K << ": " << us_bf16_m << " us (mean), " << us_bf16_s
                  << " us (stddev)\n";
        std::cout << "fp8_e4m3_lut matvec 128x64:       " << us_fp8_m << " us (mean), " << us_fp8_s
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
    }

    if (csv_path) {
        std::ofstream out(csv_path);
        wic::write_csv_header(out);
        const std::string gpu = prop.name;
        auto row = [&](const char* name, double m, double s, double gops) {
            wic::BenchResult r;
            r.name = name;
            r.mean_us = m;
            r.stddev_us = s;
            r.throughput_gops = gops;
            wic::write_csv_row(out, r, gpu, cuda_ver);
        };
        row("serial_two_phase_stream", us_serial_m, us_serial_s, 0.0);
        row("two_kernels_sync", us_two_sync_m, us_two_sync_s, 0.0);
        row("fused_pipeline", us_fused_m, us_fused_s, 0.0);
        if (do_kv_cluster) {
            row("kv_tile_pipeline", us_kv_m, us_kv_s, 0.0);
            if (cluster_tma_st == cudaSuccess) {
                row("cluster_tma_dsmem_kv", us_cluster_tma_m, us_cluster_tma_s, 0.0);
            }
        }
        if (do_heavy_prec) {
            row("fp16_matmul_wmma_64", us_fp16_m, us_fp16_s, (2.0 * M * N * K) / (us_fp16_m * 1e-3));
            row("bf16_matmul_wmma_64", us_bf16_m, us_bf16_s, (2.0 * M * N * K) / (us_bf16_m * 1e-3));
            constexpr int V = 128, Hh = 64;
            row("fp8_matvec_lut", us_fp8_m, us_fp8_s, (2.0 * static_cast<double>(V) * Hh) / (us_fp8_m * 1e-3));
            row("fp4_packed_microbench", us_fp4_m, us_fp4_s, static_cast<double>(num_nibbles) / (us_fp4_m * 1e-3));
        }
        std::cout << "Wrote CSV: " << csv_path << '\n';
    }

    return 0;
}
