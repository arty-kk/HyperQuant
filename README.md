# HyperQuant

HyperQuant is an open-source, memory-first compression stack for teams that are hitting a real resident-memory limit in vector, long-context, replay, or session-state workloads.

It is not positioned as a universal “better quantizer.” The product goal is narrower and more useful:

> reduce resident memory in a way that can be measured, reproduced, and rolled out safely.

## What problem it solves

Most infrastructure teams do not need another synthetic compression demo. They need a clear answer to one of these questions:

- Why does resident state cap session density per node?
- How many sessions fit into the current RAM budget if only a hot window stays resident?
- Can long-context artifacts be tiered out of RAM without hiding quality loss?
- Can route selection stay explicit instead of silently over-claiming wins?

## Routes

HyperQuant ships four explicit routes.

- `conservative_codebook`: trained codebook route for arbitrary numeric tensors.
- `vector_codec`: training-free rotated-scalar route for vector-like data, with a residual rescue side-channel for the hardest coefficients.
- `context_codec`: page-aware route for structured long-context data, with an explicit fail-closed contract.
- `resident_tier`: resident-memory planning and tiered artifact store for bounded hot-window serving.

These routes are deliberately separate. The repository is easier to trust when it says where each path fits.

## Why test it

The main reason to evaluate HyperQuant is not headline ratio. It is operating leverage:

- fewer resident bytes per session,
- higher session density under a fixed RAM budget,
- reproducible route-level trade-offs,
- offline tiered artifacts that can be reopened, verified, and read lazily by slice.

The flagship path is `resident_tier`. The other routes exist to make that path measurable and explainable.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

Generate a live-like dataset and project resident memory:

```bash
python -m hyperquant generate-live-data   --scenario online   --output /tmp/online.npy   --n-vectors 16384   --dim 128

python -m hyperquant resident-plan   --input /tmp/online.npy   --page-size 64   --group-size 128   --residual-topk 1   --hot-pages 8   --active-window-tokens 256   --concurrent-sessions 8   --runtime-value-bytes 2
```

Build, verify, and read from a tiered resident store:

```bash
python -m hyperquant build-resident-store   --input /tmp/online.npy   --output /tmp/resident_store   --page-size 64   --group-size 128   --residual-topk 1   --hot-pages 8

python -m hyperquant verify-resident-store   --store /tmp/resident_store

python -m hyperquant read-resident-slice   --store /tmp/resident_store   --start 0   --end 256   --output /tmp/window.npy
```

Compare the vector route against the dense rotation baseline:

```bash
python -m hyperquant vector-benchmark   --input /tmp/online.npy   --bits 3   --group-size 128   --residual-topk 1   --iterations 5   --warmup 1

python -m hyperquant dense-baseline-benchmark   --input /tmp/online.npy   --bits 3   --group-size 128   --residual-topk 1   --iterations 5   --warmup 1
```

Evaluate the structured context route with a fail-closed contract:

```bash
python -m hyperquant context-benchmark   --input /tmp/online.npy   --with-guarantee   --fail-closed
```

Regenerate the checked-in evidence pack:

```bash
python scripts/build_proof_pack.py
```

## What to test first

### Resident memory is the bottleneck
Start with:

- `resident-plan`
- `build-resident-store`
- `verify-resident-store`
- `read-resident-slice`
- `resident-benchmark`

### Vector compression throughput is the bottleneck
Start with:

- `vector-benchmark`
- `dense-baseline-benchmark`
- `route-benchmark`

### Structured long-context state is the bottleneck
Start with:

- `context-benchmark --with-guarantee --fail-closed`
- `route-benchmark`
- `context-audit-input`

## Evidence

The repository ships checked-in evidence under `docs/evidence/` and a script that regenerates it from source.

Start with:

- `docs/evidence/route-benchmark.md`
- `docs/evidence/resident-benchmark.md`
- `docs/evidence/capacity-example.md`

## API surface

- `GET /healthz`
- `GET /metrics`
- `POST /v1/codebook/compress`
- `POST /v1/codebook/decompress`
- `POST /v1/vector/compress`
- `POST /v1/vector/decompress`
- `POST /v1/context/compress`
- `POST /v1/context/decompress`
- `POST /v1/resident/plan`

## What this repository proves

- the four routes work end to end;
- resident-memory planning is reproducible from source;
- the resident store can be built, reopened, verified, and read lazily by slice;
- route trade-offs can be measured with the included harness;
- the context route can run with a fail-closed contract.

## What it does not claim

- universal superiority across every model or runtime;
- end-task quality on your production traces without a pilot;
- vendor-runtime integration such as auth, admission control, or multi-node cache coherence;
- a drop-in replacement for serving engines or vector indexes.

That boundary is deliberate. It is more credible than pretending to be a universal winner.

## Read next

- `docs/problem-solution-profit.md`
- `docs/architecture.md`
- `docs/evaluation.md`
- `docs/guarantees.md`
- `docs/pilot-playbook.md`

## License

Licensed under the Apache License, Version 2.0.

Copyright © 2026 Сацук Артём Венедиктович (Satsuk Artem).

Synchatica is the project brand. See `LICENSE` and `NOTICE`.
