# Problem, solution, payoff

## Problem

Teams usually do not buy a compression component because a benchmark chart looked clever. They test one because an operating constraint is already painful:

- resident state caps session density,
- replay artifacts outgrow RAM budgets,
- long-context traffic is expensive because too much state stays hot,
- route selection is opaque and hard to defend.

## Solution

HyperQuant answers that problem with four explicit routes:

- `conservative_codebook` for arbitrary numeric tensors,
- `vector_codec` for fast training-free vector compression,
- `context_codec` for structured long-context pages with a fail-closed contract,
- `resident_tier` for resident-memory planning and tiered storage.

The important point is not the number of routes. It is that the repository makes the trade-off explicit instead of hiding it behind one overbroad promise.

## Payoff

A useful pilot should produce at least one concrete operating result:

- lower resident bytes per session,
- more sessions in the same RAM budget,
- a reproducible benchmark for route choice,
- a tiered artifact that can be verified and sliced safely,
- a decision about whether the context route fits the workload at all.

That is the category HyperQuant is built for.
