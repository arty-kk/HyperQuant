# Deployment notes

## Local API shell

The FastAPI shell starts with a trained codebook bundle loaded because the codebook route needs one. The vector, context, and resident routes can be evaluated without separate training, but the API process still expects a bundle path at startup.

```bash
hyperquant serve --bundle bundle.npz --host 0.0.0.0 --port 8080
```

## Recommended validation order

1. Run `pytest -q`.
2. Build the native FWHT helper if you want the structured vector fast path.
3. Validate `resident-plan` on a representative trace.
4. Build a resident store and run `verify-resident-store`.
5. Validate `read-resident-slice` against the same trace family.
6. Expose the API only after the route behavior is understood.

## Current scope

This OSS build is intentionally focused on local and pilot workflows. It does not ship opinionated auth, admission control, object-store integration, or multi-node cache coordination.
