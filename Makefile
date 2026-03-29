PYTHON ?= python

.PHONY: install test serve codebook-benchmark context-benchmark vector-benchmark dense-baseline-benchmark route-benchmark resident-benchmark native proof-pack

install:
	$(PYTHON) -m pip install -e .[dev]

native:
	$(PYTHON) -c "from hyperquant.native_core import build_native_fwht; print(build_native_fwht(force=True))"

test:
	pytest -q

serve:
	hyperquant serve --bundle bundle.npz --host 0.0.0.0 --port 8080

codebook-benchmark:
	hyperquant codebook-benchmark --bundle bundle.npz --input demo.npy

context-benchmark:
	hyperquant context-benchmark --input context_demo.npy --with-guarantee --fail-closed

vector-benchmark:
	hyperquant vector-benchmark --input demo.npy

dense-baseline-benchmark:
	hyperquant dense-baseline-benchmark --input demo.npy

route-benchmark:
	hyperquant route-benchmark --markdown-output docs/evidence/route-benchmark.md --json-output docs/evidence/route-benchmark.json

resident-benchmark:
	hyperquant resident-benchmark --markdown-output docs/evidence/resident-benchmark.md --json-output docs/evidence/resident-benchmark.json

proof-pack:
	$(PYTHON) scripts/build_proof_pack.py
