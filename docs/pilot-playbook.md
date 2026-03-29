# Pilot playbook

## Goal

Answer one decision question with evidence:

- does resident/session fall enough to change capacity?
- does the context route actually fit the trace family?
- is the resident store operationally useful for replay or serving?

## Step 1: prepare a trace

Export or synthesize a representative array and keep the shape, dtype, and workload label.

## Step 2: run the route checks

```bash
python -m hyperquant vector-benchmark --input trace.npy
python -m hyperquant context-benchmark --input trace.npy --with-guarantee --fail-closed
python -m hyperquant resident-plan --input trace.npy ...
```

## Step 3: build the artifact

```bash
python -m hyperquant build-resident-store --input trace.npy --output ./pilot_store ...
python -m hyperquant verify-resident-store --store ./pilot_store
python -m hyperquant read-resident-slice --store ./pilot_store --start 0 --end 256 --output /tmp/window.npy
```

## Step 4: capture the decision

Record:

- baseline resident/session,
- projected resident/session,
- capacity delta under the same RAM budget,
- route recommendation,
- guarantee outcome,
- slice-read latency.

A pilot is successful when it makes a real deployment decision easier, not when it produces the prettiest ratio.
