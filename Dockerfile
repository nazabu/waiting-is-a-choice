# CUDA 12.8 development image for WIC microbench (targets sm_120 by default).
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    ninja-build \
    git \
    g++ \
    libomp-dev \
    python3 \
    python3-pip \
    ca-certificates \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/waiting-is-a-choice
COPY . .

ARG WIC_CUDA_ARCH=120
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DWIC_CUDA_ARCH=${WIC_CUDA_ARCH} \
    && cmake --build build --parallel "$(nproc)"

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
CMD ["bash", "-lc", "scripts/env_info.sh && ./build/wic_cuda_bench --skip-correctness --iters 5"]
