#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/wic_cuda_bench"
OUT="${OUT:-$ROOT/results/microbench.csv}"
mkdir -p "$(dirname "$OUT")"

if [[ ! -x "$BIN" ]]; then
  echo "Build first: scripts/build.sh" >&2
  exit 1
fi

exec "$BIN" "${@}" --csv "$OUT"
