#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wic {

struct CpuPipelineConfig {
    int draft_threads = 2;
    int verify_threads = 2;
    std::size_t vocab = 1024;
    std::size_t hidden = 256;
    std::size_t gen_len = 32;
    unsigned seed = 1;
};

// Reference greedy generation (single-threaded), returns token ids.
std::vector<int> greedy_reference(const CpuPipelineConfig& cfg);

// OpenMP draft/verify pipeline with double-buffered handoff; same interface as greedy for comparison.
std::vector<int> pipeline_openmp_specialized(const CpuPipelineConfig& cfg);

bool tokens_equal(const std::vector<int>& a, const std::vector<int>& b);

}  // namespace wic
