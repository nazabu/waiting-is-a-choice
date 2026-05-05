cmake_minimum_required(VERSION 3.24)
project(waiting-is-a-choice VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(cmake/WICCudaArch.cmake)

find_package(OpenMP REQUIRED)

add_library(wic_cpu STATIC
  cpu/greedy_reference.cpp
  cpu/pipeline_openmp.cpp
)
target_include_directories(wic_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(wic_cpu PUBLIC OpenMP::OpenMP_CXX)

add_executable(wic_cpu_demo cpu/main_cpu_demo.cpp)
target_link_libraries(wic_cpu_demo PRIVATE wic_cpu)

# --- CUDA benchmarks ---
enable_language(CUDA)

add_library(wic_cuda_kernels STATIC
  cuda/baselines.cu
  cuda/dsmem_pipeline.cu
  cuda/tma_prefetch.cu
  cuda/precision_paths.cu
)
target_include_directories(wic_cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_property(TARGET wic_cuda_kernels PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET wic_cuda_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(wic_cuda_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda -lineinfo>)

add_executable(wic_cuda_bench cuda/main_cuda_bench.cpp)
target_link_libraries(wic_cuda_bench PRIVATE wic_cuda_kernels CUDA::cudart)
target_include_directories(wic_cuda_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_property(TARGET wic_cuda_bench PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Optional symlink-friendly install
install(TARGETS wic_cpu_demo wic_cuda_bench RUNTIME DESTINATION bin)
