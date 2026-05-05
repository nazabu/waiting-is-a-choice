#pragma once

#include <cstddef>
#include <cstdint>

namespace wic {

struct BenchConfig {
    int warmup_iters = 5;
    int measure_iters = 50;
    std::size_t batch = 64;
    std::size_t dim = 256;
    std::size_t kv_heads = 8;
    std::size_t head_dim = 64;
    std::size_t seq_tile = 32;
};

}  // namespace wic
