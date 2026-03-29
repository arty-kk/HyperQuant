# Architecture

HyperQuant is organized as a small route portfolio around one product goal: resident-memory reduction.

## Route map

### `conservative_codebook`
A trained codebook route for arbitrary numeric tensors. This is the conservative fallback when the structured routes are not a fit.

### `vector_codec`
A training-free rotated-scalar route for vector-like data. It trades some reconstruction error for strong ratio and throughput, with a residual rescue side-channel for the hardest coefficients.

### `context_codec`
A page-aware route for structured long-context data. It combines page references, low-rank pages, int8 fallback, and fp16 fallback, then applies an explicit contour check and optional fail-closed guarantee.

### `resident_tier`
The product-facing route. It uses the other routes to model and build a tiered artifact where only a bounded hot window remains resident.

## Design principles

- no universal route claim;
- route selection is explicit and inspectable;
- fail-closed behavior is available where over-claiming would be dangerous;
- benchmarks and evidence are reproducible from source;
- resident-memory planning is treated as a first-class outcome.

## Operational flow

1. Generate or load a representative trace.
2. Run `resident-plan` to estimate resident/session and budgeted capacity.
3. Build a resident store with `build-resident-store`.
4. Verify the artifact with `verify-resident-store`.
5. Read realistic slices with `read-resident-slice`.
6. Compare route trade-offs with `route-benchmark` and `resident-benchmark`.
