# Contributing

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Validation before opening a pull request

```bash
python -m compileall hyperquant
pytest -q
python scripts/build_proof_pack.py --skip-tests --iterations 1 --warmup 0 --slice-iterations 1
```

## Contribution rules

- keep claims tied to reproducible evidence;
- update docs when public behavior changes;
- do not check in opaque binary artifacts without a clear reason;
- keep benchmarks reproducible from source;
- prefer explicit route naming and honest boundaries over hype.

## Licensing

By submitting a contribution, you agree that it will be licensed under Apache-2.0 unless you explicitly state otherwise in writing.
