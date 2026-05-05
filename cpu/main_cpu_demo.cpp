#include <cstdlib>
#include <iostream>

#include "wic/cpu_pipeline.hpp"

int main(int argc, char** argv) {
    wic::CpuPipelineConfig cfg;
    cfg.vocab = 512;
    cfg.hidden = 128;
    cfg.gen_len = 64;

    int threads_total = 2;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--vocab" && i + 1 < argc) cfg.vocab = static_cast<std::size_t>(std::atoi(argv[++i]));
        if (a == "--hidden" && i + 1 < argc) cfg.hidden = static_cast<std::size_t>(std::atoi(argv[++i]));
        if (a == "--steps" && i + 1 < argc) cfg.gen_len = static_cast<std::size_t>(std::atoi(argv[++i]));
        if (a == "--threads" && i + 1 < argc) threads_total = std::atoi(argv[++i]);
        if (a == "--seed" && i + 1 < argc) cfg.seed = static_cast<unsigned>(std::atoi(argv[++i]));
    }
    if (threads_total < 2) threads_total = 2;
    cfg.draft_threads = 1;
    cfg.verify_threads = 1;

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
    std::cout << "CPU demo OK (" << greedy.size() << " tokens).\n";
    std::cout << "First 16 tokens:";
    for (std::size_t i = 0; i < std::min<std::size_t>(16, greedy.size()); ++i)
        std::cout << ' ' << greedy[i];
    std::cout << '\n';
    return 0;
}
