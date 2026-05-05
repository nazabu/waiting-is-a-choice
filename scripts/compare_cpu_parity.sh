#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/wic_cpu_demo"
if [[ ! -x "$BIN" ]]; then
  echo "Build first: scripts/build.sh" >&2
  exit 1
fi

exec "$BIN" --vocab "${V:-512}" --hidden "${H:-128}" --steps "${STEPS:-64}" --seed "${SEED:-1}" --bind "${@}"
