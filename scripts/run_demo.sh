#!/usr/bin/env bash
set -euo pipefail

TMPDIR="${TMPDIR:-/tmp}"

python -m hyperquant generate-live-data \
  --scenario online \
  --output "$TMPDIR/online.npy" \
  --n-vectors 4096 \
  --dim 128

python -m hyperquant vector-benchmark \
  --input "$TMPDIR/online.npy" \
  --bits 3 \
  --group-size 128 \
  --iterations 3 \
  --warmup 1

python -m hyperquant dense-baseline-benchmark \
  --input "$TMPDIR/online.npy" \
  --bits 3 \
  --group-size 128 \
  --iterations 3 \
  --warmup 1

python -m hyperquant resident-plan \
  --input "$TMPDIR/online.npy" \
  --page-size 64 \
  --group-size 128 \
  --hot-pages 8 \
  --active-window-tokens 256 \
  --concurrent-sessions 8 \
  --runtime-value-bytes 2
