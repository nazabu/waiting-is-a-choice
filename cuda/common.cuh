#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace wic {

inline void cuda_ok(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
#if __GNUC__ || defined(__clang__)
        __builtin_trap();
#else
#endif
        (void)fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), file, line);
        std::abort();
    }
}

#define WIC_CUDA_OK(x) wic::cuda_ok((x), __FILE__, __LINE__)

}  // namespace wic
