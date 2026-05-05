#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include <omp.h>

#include "wic/cpu_pipeline.hpp"

namespace wic {

namespace {

void xoshiro_step(std::uint64_t& s0, std::uint64_t& s1, std::uint64_t& s2, std::uint64_t& s3) {
    std::uint64_t t = s1 << 17;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = (s3 << 45) | (s3 >> 19);
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

struct alignas(64) Slot {
    std::vector<float> h{};
    int token = -1;
};

constexpr int kRingCap = 16;

}  // namespace

std::vector<int> pipeline_openmp_specialized(const CpuPipelineConfig& cfg) {
    const std::size_t V = cfg.vocab;
    const std::size_t H = cfg.hidden;
    std::vector<float> w_lm(V * H);
    std::vector<float> emb(V * H);
    fill_weights(w_lm, emb, cfg.seed);

    std::vector<float> h_init(H);
    {
        std::uint64_t s0 = cfg.seed + 1, s1 = cfg.seed + 2, s2 = cfg.seed + 3, s3 = cfg.seed + 4;
        for (std::size_t i = 0; i < H; ++i) h_init[i] = rndf(s0, s1, s2, s3);
    }

    Slot ring[kRingCap];
    for (auto& s : ring) s.h.resize(H);

    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};
    std::atomic<bool> failed{false};

    std::vector<int> verified;
    verified.resize(cfg.gen_len);

    omp_set_dynamic(0);
    omp_set_max_active_levels(32);
#pragma omp parallel num_threads(2) default(shared)
#pragma omp sections
    {
#pragma omp section
        {
            // Drafting unit (producer): compute token + embed; handshake without global orchestration stall.
            std::vector<float> h_local(h_init);
            for (std::size_t t = 0; t < cfg.gen_len; ++t) {
                while (produced.load(std::memory_order_acquire) -
                           consumed.load(std::memory_order_acquire) >=
                       kRingCap - 2) {
                    std::this_thread::yield();
                }
                const int slot = produced.load(std::memory_order_acquire) % kRingCap;
                Slot& dst = ring[slot];
                dst.h.assign(h_local.begin(), h_local.end());
                const int tok = argmax_matvec(w_lm, dst.h.data(), V, H);
                dst.token = tok;
                embedding_lookup(emb, tok, H, h_local.data());
                produced.fetch_add(1, std::memory_order_release);
            }
        }
#pragma omp section
        {
            // Verification unit (consumer): recompute greedy argmax(state) — must agree with draft.
            for (std::size_t t = 0; t < cfg.gen_len; ++t) {
                while (produced.load(std::memory_order_acquire) <= static_cast<int>(t)) {
                    std::this_thread::yield();
                }
                const int slot = static_cast<int>(t) % kRingCap;
                const Slot& src = ring[slot];
                const int ref_tok = argmax_matvec(w_lm, src.h.data(), V, H);
                if (ref_tok != src.token) {
                    failed.store(true, std::memory_order_release);
                    break;
                }
                verified[t] = src.token;
                consumed.fetch_add(1, std::memory_order_release);
            }
        }
    }

    if (failed.load(std::memory_order_acquire)) return {};
    return verified;
}

}  // namespace wic
