# Evidence Pack

This directory contains the checked-in benchmark artifacts for the public release.

Files:

- `route-benchmark.{md,json}` — route benchmark on live-like synthetic workloads
- `resident-benchmark.{md,json}` — resident-memory benchmark and slice-read timings
- `capacity-example.{md,json}` — projected session density under a fixed RAM budget
- `SHA256SUMS.txt` — hashes for the files in this directory

Regenerate from source:

```bash
python scripts/build_proof_pack.py
```
