#pragma once

// Nsight Systems ranges; no-op when built with WIC_HAS_NVTX=0 (missing libnvToolsExt).
#if WIC_HAS_NVTX
#include <nvtx3/nvToolsExt.h>

namespace wic {

struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace wic

#else

namespace wic {

struct NvtxRange {
    explicit NvtxRange(const char*) {}
};

}  // namespace wic

#endif
