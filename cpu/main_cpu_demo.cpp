#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "wic/cpu_pipeline.hpp"

int main(int argc, char** argv) {
    wic::CpuPipelineConfig cfg;
    cfg.vocab = 512;
    cfg.hidden = 128;
    cfg.gen_len = 64;
    cfg.draft_threads = 1;
    cfg.verify_threads = 1;
    bool spread_bind = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--vocab" && i + 1 < argc) cfg.vocab = static_cast<std::size_t>(std::atoi(argv[++i]));
        if (a == "--hidden" && i + 1 < argc) cfg.hidden = static_cast<std::size_t>(std::atoi(argv[++i]));
        if (a == "--steps" && i + 1 < argc) cfg.gen_len = static_cast<std::size_t>(std::atoi(argv[++i]));
        if (a == "--draft" && i + 1 < argc) cfg.draft_threads = std::atoi(argv[++i]);
        if (a == "--verify" && i + 1 < argc) cfg.verify_threads = std::atoi(argv[++i]);
        if (a == "--threads" && i + 1 < argc) {
            const int t = std::atoi(argv[++i]);
            cfg.draft_threads = std::max(1, t / 2);
            cfg.verify_threads = std::max(1, t - cfg.draft_threads);
        }
        if (a == "--seed" && i + 1 < argc) cfg.seed = static_cast<unsigned>(std::atoi(argv[++i]));
        if (a == "--bind") spread_bind = true;
        if (a == "--help" || a == "-h") {
            std::cout
                << "wic_cpu_demo [--vocab N] [--hidden H] [--steps S] [--seed R]\n"
                << "  [--draft D] [--verify V] | [--threads T] split across draft/verify\n"
                << "  [--bind] set OMP_PROC_BIND=spread OMP_PLACES=cores for thread pinning mimic\n";
            return 0;
        }
    }

#ifdef _OPENMP
    if (spread_bind) {
        setenv("OMP_PROC_BIND", "spread", 1);
        setenv("OMP_PLACES", "cores", 1);
    }
    const int team = cfg.draft_threads + cfg.verify_threads;
    omp_set_num_threads(team > 2 ? team : 2);
#else
    (void)spread_bind;
#endif

    const auto greedy = wic::greedy_reference(cfg);
    const auto pipe = wic::pipeline_openmp_specialized(cfg);

    if (pipe.empty()) {
        std::cerr << "CPU pipeline FAILED (verification mismatch).\n";
        return 2;
    }
    if (!wic::tokens_equal(greedy, pipe)) {
        std::cerr << "Mismatch vs greedy reference.\n";
        return 3;
    }
    std::cout << "CPU demo OK (" << greedy.size() << " tokens)."
#ifdef _OPENMP
              << " OpenMP threads=" << (cfg.draft_threads + cfg.verify_threads)
#endif
              << '\n';
    std::cout << "First 16 tokens:";
    for (std::size_t i = 0; i < std::min<std::size_t>(16, greedy.size()); ++i)
        std::cout << ' ' << greedy[i];
    std::cout << '\n';
    return 0;
}
