#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/wic_cpu_demo"
if [[ ! -x "$BIN" ]]; then
  echo "Build first: scripts/build.sh" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$BIN" --bind "${@}"
