#pragma once

// Nsight Systems ranges; prefer header-only NVTX3 when available.
#if __has_include(<nvtx3/nvtx3.hpp>)
#include <nvtx3/nvtx3.hpp>

namespace wic {

struct NvtxRange {
    explicit NvtxRange(const char* name) : range_(name) {}
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;

private:
    nvtx3::scoped_range range_;
};

}  // namespace wic

#else

namespace wic {

struct NvtxRange {
    explicit NvtxRange(const char*) {}
};

}  // namespace wic

#endif
