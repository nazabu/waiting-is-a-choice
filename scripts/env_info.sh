#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "### Host"
uname -a || true
echo

echo "### Git"
git rev-parse HEAD 2>/dev/null || echo "no-git"
git status -sb 2>/dev/null || true
echo

echo "### Toolchain"
command -v g++ >/dev/null && g++ --version | head -1 || echo "g++ missing"
command -v cmake >/dev/null && cmake --version | head -1 || echo "cmake missing"
command -v nvcc >/dev/null && nvcc --version | head -n 3 || echo "nvcc missing"
echo

echo "### NVIDIA"
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi unavailable"
