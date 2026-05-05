# Allow override: cmake -DWIC_CUDA_ARCH=89 ..
if(NOT DEFINED WIC_CUDA_ARCH)
  set(WIC_CUDA_ARCH 120 CACHE STRING "CUDA real architecture (native SASS), e.g. 120 for RTX 5070 Ti")
endif()
set(CMAKE_CUDA_ARCHITECTURES ${WIC_CUDA_ARCH})
