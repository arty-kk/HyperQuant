# Capacity example

Generated: `2026-03-29`

This document shows illustrative node-capacity planning on live-like synthetic workloads.
For each workload, the RAM budget equals the amount needed to keep the baseline fully resident for 8 sessions.

## online_vector_stream

- shape: `(16384, 128)`
- budget: `33,554,432 B`
- chosen route: `resident_tier`
- baseline resident/session: `4,194,304 B`
- projected resident/session: `159,943 B`
- max sessions within budget (baseline): `8`
- max sessions within budget (projected): `209`
- capacity gain vs baseline: `26.12x`

## structured_long_context

- shape: `(16384, 128)`
- budget: `33,554,432 B`
- chosen route: `resident_tier`
- baseline resident/session: `4,194,304 B`
- projected resident/session: `159,901 B`
- max sessions within budget (baseline): `8`
- max sessions within budget (projected): `209`
- capacity gain vs baseline: `26.12x`

## mixed_long_context

- shape: `(32768, 128)`
- budget: `67,108,864 B`
- chosen route: `resident_tier`
- baseline resident/session: `8,388,608 B`
- projected resident/session: `252,008 B`
- max sessions within budget (baseline): `8`
- max sessions within budget (projected): `266`
- capacity gain vs baseline: `33.25x`
