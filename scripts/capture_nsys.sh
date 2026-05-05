#!/usr/bin/env bash
# Example Nsight Systems capture for compute/memory overlap stories.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/wic_cuda_bench"
OUT="${OUT:-$ROOT/docs/figures/nsys_wic_demo}"
mkdir -p "$(dirname "$OUT")"

command -v nsys >/dev/null || { echo "Install Nsight Systems (nsys)." >&2; exit 2; }

nsys profile --trace cuda,nvtx,osrt -o "$OUT" "$BIN" --skip-correctness --iters 20 --warmup 3 "$@"
echo "Wrote: ${OUT}.nsys-rep (add to docs/figures/ after review)"
