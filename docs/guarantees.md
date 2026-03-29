# Guarantees and fail-closed behavior

Only the `context_codec` route ships with an explicit fail-closed contract.

## Why

The context route is intentionally narrow. It is designed for workloads where page structure is materially present. When that structure is not present, the correct behavior is to say so.

## What is checked

The route records:

- contour classification,
- contour support decision,
- contour reasons,
- optional guarantee profile results.

The checked guarantee profile can enforce thresholds on:

- compression ratio,
- cosine similarity,
- RMS error,
- maximum absolute error.

## Contours

The public contour values are:

- `context_structured`
- `generic_conservative`
- `reject`

If the input does not fit the structured contour and fail-closed mode is enabled, the route rejects instead of quietly passing through.

## What it does not promise

The context route does not promise:

- a fixed ratio on arbitrary tensors,
- applicability to every long-context workload,
- universal quality neutrality without trace-level validation.

That limitation is deliberate and improves trust.
