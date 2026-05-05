#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "wic/cpu_pipeline.hpp"

namespace wic {

namespace {

void xoshiro_step(std::uint64_t& s0, std::uint64_t& s1, std::uint64_t& s2, std::uint64_t& s3) {
    const std::uint64_t r = s0 * 5;
    const std::uint64_t t = s1 << 17;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = (s3 << 45) | (s3 >> 19);
    (void)r;
}

float rndf(std::uint64_t& s0, std::uint64_t& s1, std::uint64_t& s2, std::uint64_t& s3) {
    xoshiro_step(s0, s1, s2, s3);
    const std::uint32_t u = static_cast<std::uint32_t>(s0 >> 32);
    return (u & 0xffffff) / 16777216.0f - 0.5f;
}

void fill_weights(std::vector<float>& w_lm, std::vector<float>& emb, std::uint64_t seed) {
    std::uint64_t s0 = seed ^ 0x9E3779B97F4A7C15ULL;
    std::uint64_t s1 = seed ^ 0xBF58476D1CE4E5B9ULL;
    std::uint64_t s2 = seed ^ 0x94D049BB133111EBULL;
    std::uint64_t s3 = seed ^ 0xC6BC279692B5C323ULL;
    for (auto& x : w_lm) x = rndf(s0, s1, s2, s3);
    for (auto& x : emb) x = rndf(s0, s1, s2, s3);
}

int argmax_matvec(const std::vector<float>& w_lm, const float* h, std::size_t vocab,
                  std::size_t hidden) {
    int best = 0;
    float best_val = -1e30f;
    for (std::size_t v = 0; v < vocab; ++v) {
        float acc = 0.f;
        const float* row = &w_lm[v * hidden];
        for (std::size_t i = 0; i < hidden; ++i) acc += row[i] * h[i];
        if (acc > best_val) {
            best_val = acc;
            best = static_cast<int>(v);
        }
    }
    return best;
}

void embedding_lookup(const std::vector<float>& emb, int tok, std::size_t hidden, float* out) {
    const float* e = &emb[static_cast<std::size_t>(tok) * hidden];
    for (std::size_t i = 0; i < hidden; ++i) out[i] = e[i];
}

}  // namespace

std::vector<int> greedy_reference(const CpuPipelineConfig& cfg) {
    const std::size_t V = cfg.vocab;
    const std::size_t H = cfg.hidden;
    std::vector<float> w_lm(V * H);
    std::vector<float> emb(V * H);
    fill_weights(w_lm, emb, cfg.seed);

    std::vector<float> h(H);
    {
        std::uint64_t s0 = cfg.seed + 1, s1 = cfg.seed + 2, s2 = cfg.seed + 3, s3 = cfg.seed + 4;
        for (std::size_t i = 0; i < H; ++i) h[i] = rndf(s0, s1, s2, s3);
    }

    std::vector<int> out;
    out.reserve(cfg.gen_len);
    for (std::size_t t = 0; t < cfg.gen_len; ++t) {
        const int tok = argmax_matvec(w_lm, h.data(), V, H);
        out.push_back(tok);
        embedding_lookup(emb, tok, H, h.data());
    }
    return out;
}

bool tokens_equal(const std::vector<int>& a, const std::vector<int>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

}  // namespace wic
