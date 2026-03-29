# Evaluation guide

This repository is strongest when evaluated on real traces, not only the synthetic examples.

## 1. Start with route fit

For a structured long-context trace:

```bash
python -m hyperquant context-benchmark --input trace.npy --with-guarantee --fail-closed
python -m hyperquant context-audit-input --input trace.npy --bundle bundle.npz
```

For vector-heavy traces:

```bash
python -m hyperquant vector-benchmark --input trace.npy
python -m hyperquant dense-baseline-benchmark --input trace.npy
```

## 2. Model resident memory

Use realistic values for concurrency, active window size, and runtime value bytes.

```bash
python -m hyperquant resident-plan   --input trace.npy   --concurrent-sessions 8   --active-window-tokens 256   --runtime-value-bytes 2
```

## 3. Build and verify a resident artifact

```bash
python -m hyperquant build-resident-store --input trace.npy --output ./resident_store
python -m hyperquant verify-resident-store --store ./resident_store
python -m hyperquant read-resident-slice --store ./resident_store --start 0 --end 256 --output /tmp/window.npy
```

## 4. Benchmark route trade-offs

```bash
python -m hyperquant route-benchmark --json-output docs/evidence/route-benchmark.json
python -m hyperquant resident-benchmark --json-output docs/evidence/resident-benchmark.json
```

## 5. Decide with operating criteria

A good pilot decision is tied to workload-specific outcomes:

- resident/session under the target budget,
- session density increase,
- reconstruction error limits,
- artifact verification success,
- read-slice latency for the replay or serving path.
