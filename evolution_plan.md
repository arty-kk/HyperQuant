# Evolution Plan

## 0. Baseline (from audit)
- Architecture map:
  - Product is a four-route compression stack (`conservative_codebook`, `vector_codec`, `context_codec`, `resident_tier`) with explicit route boundaries and a resident-memory-first objective (`docs/architecture.md:3-18`).
  - Primary entry points are CLI subcommands (`hyperquant/cli.py:525-695`, `hyperquant/__main__.py:15-18`) and FastAPI endpoints (`hyperquant/api/app.py:168-400`, `README.md:122-132`).
  - Core domain logic lives in route codecs/planner: vector (`hyperquant/vector_codec.py:261-310`, `hyperquant/vector_codec.py#RotatedScalarCodec`), context (`hyperquant/context_codec.py:291-316`, `hyperquant/context_codec.py#ContextCodec`), resident planning/store (`hyperquant/resident_tier.py:821-860`, `hyperquant/resident_tier.py#ResidentPlanner`; `hyperquant/resident_tier.py:667-819`, `hyperquant/resident_tier.py#ResidentTierStore`).
  - Observability is via Prometheus counters/gauges/histograms (`hyperquant/telemetry.py:25-190`) exposed at `/metrics` (`hyperquant/api/app.py:186-188`).
  - Validation and constraints are centralized in numeric/shape validators (`hyperquant/validation.py:29-72`) plus Pydantic request models (`hyperquant/api/models.py:22-183`).
- Critical flows:
  1. Train codebook bundle (`hyperquant/cli.py:143-157`).
  2. Codebook compress/decompress (CLI and API) (`hyperquant/cli.py:161-179`, `hyperquant/api/app.py:190-233`).
  3. Vector compress/decompress benchmark loop (`hyperquant/cli.py:241-289`, `hyperquant/api/app.py:234-275`).
  4. Context compress with contour + optional fail-closed guarantee (`hyperquant/context_codec.py:580-620`, `docs/guarantees.md:3-34`, `hyperquant/api/app.py:323-378`).
  5. Context decompress from envelope (`hyperquant/context_codec.py:622-667`, `hyperquant/api/app.py:380-400`).
  6. Resident plan estimation (`hyperquant/resident_tier.py:850-985`, `hyperquant/api/app.py:277-321`).
  7. Build/open/verify/read resident store (`hyperquant/resident_tier.py:988-1062`, `hyperquant/resident_tier.py:760-819`, `hyperquant/cli.py:431-460`, `hyperquant/cli.py:653-669`).
  8. Route/resident benchmark evidence generation (`hyperquant/cli.py:310-330`, `hyperquant/cli.py:442-460`, `Makefile:29-36`).
- Current pain points:
  - **P0: Internal server errors are surfaced as HTTP 400 (client error), not server error, across endpoints.** Evidence: broad `except Exception` branches rethrow `HTTPException(status_code=400, ...)` in all handlers (`hyperquant/api/app.py:206-208,228-230,248-250,271-273,317-319,371-373,396-398`).
  - **P1: Request size enforcement depends on `Content-Length` and can be bypassed when header is absent/invalid, causing avoidable large-body processing.** Evidence: middleware checks only declared header (`hyperquant/api/app.py:152-166`), while decoding limits happen later after body parse (`hyperquant/api/app.py:195,220,239,262,282,343,385`; `hyperquant/utils.py:44-51`).
  - **P1: Resident planner swallows all context-route exceptions and silently downgrades candidate quality, including unexpected bugs.** Evidence: `except Exception` converts any error to `context_error` (`hyperquant/resident_tier.py:883-887,930-937`).
  - **P1: Same domain defaults are duplicated across dataclass config, API models, and CLI parser defaults, increasing drift risk.** Evidence: context defaults repeated in `ContextCodecConfig` (`hyperquant/context_codec.py:41-53`), API request model (`hyperquant/api/models.py:50-63`), CLI args (`hyperquant/cli.py:477-488`); resident/vector defaults similarly duplicated (`hyperquant/resident_tier.py:42-61`, `hyperquant/api/models.py:127-145`, `hyperquant/cli.py:508-521`).
  - **P2: `context-decompress-file` exposes many context-tuning flags that do not affect decompression behavior, creating UX friction and misleading contracts.** Evidence: parser adds full context args (`hyperquant/cli.py:611-615,477-488`), but decompression logic depends on envelope payload and does not consult config thresholds (`hyperquant/context_codec.py:622-667`).
  - **P2: Test data builders for context-like/random arrays are duplicated across suites, increasing maintenance effort and inconsistency risk.** Evidence: repeated helpers in `tests/test_context_codec.py:31-72` and `tests/test_api.py:36-73`.
- Constraints:
  - Repo-native validation commands are `pytest -q` and Makefile targets (`Makefile:11-36`); CI uses compile, native FWHT smoke, tests, package build, and proof-pack smoke (`.github/workflows/ci.yml:19-38`).
  - Project explicitly limits scope to local/pilot workflows without auth/admission/multi-node concerns (`docs/deployment.md:20-22`).

## 1. North Star
- UX outcomes:
  - API error semantics are predictable: malformed requests remain 4xx, unhandled server failures become 5xx, and monitoring dashboards can separate client misuse from server regressions.
  - CLI decompression flow has only necessary flags, reducing operator friction (proxy: number of required/visible flags for `context-decompress-file`).
  - Large-body rejection is deterministic regardless of `Content-Length` presence.
- Domain outcomes:
  - Route/planner decisions rely on explicit, domain-expected failure classes only; unexpected defects fail fast instead of being masked.
  - Single source of truth for route defaults and tunable invariants to avoid behavior skew across CLI/API/runtime.
- Engineering outcomes:
  - Lower regression risk via targeted tests for API error codes and body-size enforcement.
  - Faster safe changes by removing duplicated fixture/config logic.

## 2. Roadmap (incremental)

### Phase 1 (Stabilize Core) - up to 10 highest-impact tasks (prioritize P0/P1)
1. **Goal:** Correct API error contract and stop misclassifying server failures.
   - Scope (what we touch / what we don’t): Touch FastAPI exception handling in `hyperquant/api/app.py`; do not change codec math/algorithms.
   - Deliverables: Replace broad `Exception -> 400` mapping with `Exception -> 500` + stable error payload; keep `ValueError` as 400 and contour/guarantee as 422.
   - Dependencies: none.
   - Risk & Rollback strategy: Low-risk behavior change; rollback by reverting only handler/status mapping.
   - Validation: `pytest -q`.

2. **Goal:** Enforce request body limits independent of header trust.
   - Scope: API ingress path/middleware only.
   - Deliverables: Add streaming body-size guard (ASGI receive wrapper) that rejects >`max_http_body_bytes` even when `Content-Length` missing/wrong.
   - Dependencies: Task 1 recommended first to lock status semantics.
   - Risk & Rollback strategy: Medium risk to request handling path; rollback by reverting middleware wrapper and keeping legacy header check.
   - Validation: `pytest -q` (with new tests from Phase 1 task 4).

3. **Goal:** Prevent silent planner downgrade on internal defects.
   - Scope: `ResidentPlanner.plan` error handling around context candidate.
   - Deliverables: Catch only known/expected context qualification errors and propagate unexpected exceptions.
   - Dependencies: none.
   - Risk & Rollback strategy: Medium (can surface latent defects); rollback by restoring broad catch.
   - Validation: `pytest -q`.

4. **Goal:** Add regression tests for the corrected contracts.
   - Scope: API and resident planner tests only.
   - Deliverables: Tests for (a) internal error -> 500, (b) oversized body rejection without reliable `Content-Length`, (c) planner propagates unexpected internal exceptions.
   - Dependencies: Tasks 1-3.
   - Risk & Rollback strategy: Low; tests-only.
   - Validation: `pytest -q`.

### Phase 2 (UX & Domain Consolidation) - up to 10 tasks
1. **Goal:** Make route defaults single-source to prevent drift.
   - Scope: Config defaults in route dataclasses + API models + CLI parser wiring.
   - Deliverables: Introduce centralized default constants module and reference it from `ContextCodecConfig`, `ResidentTierConfig`, API model defaults, and CLI argument defaults.
   - Dependencies: Phase 1 complete.
   - Risk & Rollback strategy: Medium (touches multiple surfaces); rollback via targeted revert of default-source integration.
   - Validation: `pytest -q`; `python -m compileall hyperquant`.

2. **Goal:** Remove misleading CLI decompression knobs.
   - Scope: `context-decompress-file` parser and command behavior/docs.
   - Deliverables: Decompress command accepts only input/output (and truly used controls, if any), aligned with envelope-driven decode path.
   - Dependencies: Task 1 (default consolidation) optional.
   - Risk & Rollback strategy: Low; rollback by restoring prior parser wiring.
   - Validation: `pytest -q`.

3. **Goal:** Reduce duplicated test-fixture logic for synthetic arrays.
   - Scope: test helpers only.
   - Deliverables: Shared fixture/helper module reused by API/context tests.
   - Dependencies: none.
   - Risk & Rollback strategy: Low.
   - Validation: `pytest -q`.

### Phase 3 (Scale & Maintainability)- up to 10 tasks (only if it truly blocks progress)
1. **Goal:** Align CI checks with high-risk contracts to prevent regressions from shipping.
   - Scope: tests/CI definitions only.
   - Deliverables: Ensure new contract tests are part of default suite; keep CI matrix unchanged.
   - Dependencies: Phases 1-2.
   - Risk & Rollback strategy: Low.
   - Validation: `pytest -q` and existing CI job commands from `.github/workflows/ci.yml:24-38`.

## 3. Task Specs (atomic, single-strategy)

### ID: EVO-001
- Priority: P0
- Theme: Reliability
- Problem: API maps unhandled server exceptions to 400, violating client/server fault boundaries.
- Evidence: `hyperquant/api/app.py:206-208,228-230,248-250,271-273,317-319,371-373,396-398` (`create_app` endpoint handlers).
- Root Cause: Copy-pasted exception blocks classify all non-`ValueError` failures as bad request.
- Impact: Clients receive misleading feedback; observability/alerting cannot distinguish malformed requests from server defects.
- Fix (single solution): Replace each broad exception mapping with `HTTPException(status_code=500, detail="internal server error")` (or equivalent shared helper) while preserving existing 400/422 branches.
- Steps:
  1. Introduce one internal helper in `create_app` to map unexpected exceptions.
  2. Apply helper uniformly to all endpoint `except Exception` blocks.
  3. Keep metrics reason `internal_error` unchanged.
- Acceptance Criteria (verifiable): Any injected runtime exception in endpoint worker path returns HTTP 500; malformed payload still returns 400; contour/guarantee rejection remains 422.
- Validation Commands (if visible in the project): `pytest -q`.
- Migration/Rollback (if needed): Rollback by reverting exception-status mapping only.

### ID: EVO-002
- Priority: P1
- Theme: Reliability
- Problem: Oversized request rejection depends on `Content-Length`, so size guard is non-deterministic when header is absent/invalid.
- Evidence: Header-only check in middleware (`hyperquant/api/app.py:152-166`); decode cap happens after body parse (`hyperquant/api/app.py:195,220,239,262,282,343,385`; `hyperquant/utils.py:44-51`).
- Root Cause: Guardrail implemented as declarative-header validation, not authoritative streamed byte counting.
- Impact: Extra memory/CPU pressure from large bodies can reach JSON parsing stage before rejection.
- Fix (single solution): Add ASGI receive-wrapper middleware that cumulatively counts request body bytes and returns 413 once `max_http_body_bytes` is exceeded, independent of header value.
- Steps:
  1. Implement receive wrapper in `create_app` middleware.
  2. Keep existing `Content-Length` fast-fail as optimization.
  3. Add tests for missing/incorrect `Content-Length` with oversized body.
- Acceptance Criteria (verifiable): Oversized payloads return 413 regardless of `Content-Length` presence; normal payloads unaffected.
- Validation Commands (if visible in the project): `pytest -q`.
- Migration/Rollback (if needed): Revert wrapper middleware; retain previous behavior.

### ID: EVO-003
- Priority: P1
- Theme: Domain
- Problem: Resident planner suppresses all context compression exceptions and silently demotes context candidate.
- Evidence: broad catch at `hyperquant/resident_tier.py:883-887` and fallback candidate with string error at `hyperquant/resident_tier.py:930-937` (`ResidentPlanner.plan`).
- Root Cause: `except Exception` used for route qualification control flow.
- Impact: Internal defects can be hidden as “input did not qualify,” reducing correctness and debuggability.
- Fix (single solution): Narrow exception handling to known qualification failures (e.g., `ValueError`, contour/guarantee domain exceptions) and re-raise unexpected exceptions.
- Steps:
  1. Enumerate expected domain exceptions from context route.
  2. Replace broad catch with explicit tuple.
  3. Add test proving unexpected exception propagation.
- Acceptance Criteria (verifiable): Expected non-qualifying inputs still produce context candidate error payload; unexpected runtime failures bubble up.
- Validation Commands (if visible in the project): `pytest -q`.
- Migration/Rollback (if needed): Restore broad catch if compatibility emergency occurs.

### ID: EVO-004
- Priority: P1
- Theme: Platform
- Problem: Route defaults are duplicated across runtime config, API schema, and CLI parser.
- Evidence: context defaults duplicated across `hyperquant/context_codec.py:41-53`, `hyperquant/api/models.py:50-63`, `hyperquant/cli.py:477-488`; resident/vector duplication across `hyperquant/resident_tier.py:42-61`, `hyperquant/api/models.py:127-145`, `hyperquant/cli.py:508-521`.
- Root Cause: Multiple layers define literal defaults independently.
- Impact: Drift risk during tuning; inconsistent behavior between CLI and API.
- Fix (single solution): Create one defaults module (e.g., `hyperquant/config_defaults.py`) and reference constants from dataclasses, Pydantic model fields, and CLI `add_argument` defaults.
- Steps:
  1. Add constants for context/vector/resident defaults.
  2. Replace literals in config dataclasses and API models.
  3. Wire CLI parser defaults to the same constants.
  4. Run tests for parity.
- Acceptance Criteria (verifiable): One source file owns default values; no duplicated numeric literals remain in these three layers.
- Validation Commands (if visible in the project): `pytest -q`; `python -m compileall hyperquant`.
- Migration/Rollback (if needed): Revert references to constants and restore inline defaults.

### ID: EVO-005
- Priority: P2
- Theme: UX
- Problem: `context-decompress-file` exposes compression-tuning flags that do not affect decode behavior.
- Evidence: parser wiring adds all context args (`hyperquant/cli.py:611-615,477-488`), while decode path operates from envelope metadata (`hyperquant/context_codec.py:622-667`).
- Root Cause: Parser reused from compress flow despite decode not depending on tuning thresholds.
- Impact: Operator confusion; unnecessary flags and cognitive overhead.
- Fix (single solution): Simplify `context-decompress-file` CLI to only required decode inputs and instantiate decompressor without irrelevant config flags.
- Steps:
  1. Remove `_add_context_args` call for decompress subcommand.
  2. Adjust command implementation/docs accordingly.
  3. Add/adjust CLI test coverage.
- Acceptance Criteria (verifiable): Help output for `context-decompress-file` contains only decode-relevant parameters.
- Validation Commands (if visible in the project): `pytest -q`.
- Migration/Rollback (if needed): Re-add removed parser options.

### ID: EVO-006
- Priority: P2
- Theme: Platform
- Problem: Synthetic context/random test data builders are duplicated across test modules.
- Evidence: `tests/test_context_codec.py:31-72` and `tests/test_api.py:36-73`.
- Root Cause: Independent local helper definitions.
- Impact: Maintenance overhead and subtle data-shape divergence risk.
- Fix (single solution): Move shared builders to a single test helper module and import from both suites.
- Steps:
  1. Create shared test utility module.
  2. Replace duplicated local helper functions with imports.
  3. Run full test suite.
- Acceptance Criteria (verifiable): No duplicate builder implementations remain in test files; tests remain green.
- Validation Commands (if visible in the project): `pytest -q`.
- Migration/Rollback (if needed): Inline helpers back into each file.

### ID: EVO-007
- Priority: P1
- Theme: Reliability
- Problem: Existing API tests cover happy path and selected 400/422 responses, but not corrected 500 semantics or robust request-size guard edge cases.
- Evidence: Current API tests assert 200/400/422 and metrics (`tests/test_api.py:81-299`) but have no server-error contract test and no no-`Content-Length` oversize test.
- Root Cause: Test suite optimized for route functionality, not error-contract boundaries.
- Impact: Regression risk after any handler/middleware refactor.
- Fix (single solution): Add contract-focused tests for internal 500 and transport-size enforcement scenarios.
- Steps:
  1. Inject controlled internal failure in one endpoint path.
  2. Assert 500 status and error metric increment.
  3. Send oversized request without reliable `Content-Length`; assert 413.
- Acceptance Criteria (verifiable): New tests fail on current buggy behavior and pass after fixes.
- Validation Commands (if visible in the project): `pytest -q`.
- Migration/Rollback (if needed): Remove new tests if contract decision is reversed.

## 4. Explicit Non-Goals
- No changes to compression math/algorithms, quantization fidelity, or benchmark methodology (`hyperquant/vector_codec.py:261-310`, `hyperquant/vector_codec.py#RotatedScalarCodec`; `hyperquant/context_codec.py:291-316`, `hyperquant/context_codec.py#ContextCodec`; `docs/benchmark-protocol.md:11-25`).
- No new external integrations (auth/object-store/distributed cache), consistent with current OSS scope (`docs/deployment.md:20-22`).
- No broad refactors outside the specified reliability/defaults/CLI UX/test-contract targets.
- No dependency/version churn unless strictly required by implementing the tasks (`pyproject.toml:25-35`).
