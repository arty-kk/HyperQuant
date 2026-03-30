# Evolution Plan

## 0. Baseline (from audit)
- Architecture map:
  - Product scope is a 4-route compression portfolio (`conservative_codebook`, `vector_codec`, `context_codec`, `resident_tier`) focused on measurable resident-memory reduction, not universal compression wins (`README.md:18-27`, `README.md:134-149`, `docs/architecture.md:3-18`).
  - Main entry points are CLI commands (via `hyperquant.cli:main`) and FastAPI endpoints (`pyproject.toml:35-36`, `README.md:122-133`, `hyperquant/api/app.py:217-451`).
  - Resident tier business logic is centered in `ResidentPlanner.plan` and `_TieredPageEncoder._encode_pages` and then materialized/read through `ResidentTierStore` (`hyperquant/resident_tier.py:435-675` `_TieredPageEncoder._encode_pages`, `hyperquant/resident_tier.py:860-995` `ResidentPlanner.plan`, `hyperquant/resident_tier.py:677-829` `ResidentTierStore`).
  - API request/response contracts are in Pydantic models and then mapped to route implementations in handlers (`hyperquant/api/models.py:36-197`, `hyperquant/api/app.py:239-451`).
- Critical flows: 
  1. Train bundle → codebook compress/decompress (`hyperquant/cli.py:151-187`, `hyperquant/api/app.py:239-281`).
  2. Vector compress/decompress with configurable FWHT path (`hyperquant/api/models.py:45-52`, `hyperquant/api/app.py:283-324`).
  3. Context compress with contour/guarantee fail-closed behavior (`hyperquant/api/app.py:372-427`, `hyperquant/guarantee.py:11-92`).
  4. Resident planning and route selection (`hyperquant/api/app.py:326-370`, `hyperquant/resident_tier.py:860-995`).
  5. Resident store build/open/verify/slice read (`README.md:57-65`, `hyperquant/resident_tier.py:690-829`).
  6. Route and resident benchmark evidence generation (`README.md:81-85`, `Makefile:25-34`, `hyperquant/route_benchmark.py:121-207`, `hyperquant/resident_tier.py:1008-1071`).
- Current pain points:
  - **P0 data-integrity contract gap:** payload hash verification is optional on read (`hyperquant/resident_tier.py:739-744` `ResidentTierStore._read_verified_payload`), descriptor hash is nullable (`hyperquant/resident_tier.py:142-143`, `hyperquant/resident_tier.py:168-169` `ResidentPageDescriptor`), and manifest validation does not require hashes for materialized pages (`hyperquant/resident_tier.py:242-251` `ResidentTierManifest.validate`). This weakens corruption detection if manifest metadata is modified.
  - **P1 API/state desync:** vector decompress forces `prefer_native_fwht=True` regardless of request/envelope policy (`hyperquant/api/app.py:311-313` `vector_decompress_endpoint`), while compress explicitly accepts caller preference (`hyperquant/api/models.py:51` `VectorCompressRequest`, `hyperquant/api/app.py:289-290`).
  - **P1 contract exposure gap:** resident planning code supports protected indices (`hyperquant/resident_tier.py:868-886` `ResidentPlanner.plan`), but API schema and endpoint mapping for `/v1/resident/plan` do not expose/pass `protected_vector_indices` (`hyperquant/api/models.py:135-160`, `hyperquant/api/app.py:330-359`).
  - **P1 hot-path inefficiency in verification:** `verify_integrity` reads payload once directly and then reads/decodes the same payload again via `get_page` (`hyperquant/resident_tier.py:751-756` `ResidentTierStore.verify_integrity`, `hyperquant/resident_tier.py:780-805` `ResidentTierStore.get_page`).
  - **P2 leaky abstraction/coupling:** resident tier encoder calls ContextCodec private methods (`_protected_mask`, `_hash_page`, `_relative_rms`, `_max_abs_error`, `_top_rank_factors`, `_quantize_page_int8`) (`hyperquant/resident_tier.py:444`, `hyperquant/resident_tier.py:482`, `hyperquant/resident_tier.py:484`, `hyperquant/resident_tier.py:494`, `hyperquant/resident_tier.py:504`, `hyperquant/resident_tier.py:528`), making Context internals de-facto external contracts.
  - **P2 validation contract drift risk:** local default validation path is `pytest -q`/`make test` (`Makefile:9-10`, `docs/deployment.md:13-17`), while CI additionally executes compile/native/build/proof-pack steps (`.github/workflows/ci.yml:23-38`). This split can create local-pass/CI-fail friction and slows feedback loops.
- Constraints: 
  - Python package target is `>=3.10` with numpy/fastapi/pydantic/prometheus stack (`pyproject.toml:6-33`).
  - Existing validation commands are Makefile targets and `pytest -q` (`Makefile:1-34`, `docs/deployment.md:13-17`).
  - Route separation and explicit contract behavior are intentional product constraints and should be preserved (`README.md:20-27`, `docs/architecture.md:21-25`).

## 1. North Star
- UX outcomes:
  - API behavior is deterministic across compress/decompress configs (proxy: identical route config semantics between `/v1/vector/compress` and `/v1/vector/decompress`).
  - Resident planning supports the same protection controls available in core codec paths (proxy: `/v1/resident/plan` request parity with planner inputs).
  - Integrity verification reports become predictable and actionable (proxy: `verify-resident-store` fails on missing/tampered hash metadata).
- Domain outcomes:
  - Resident artifact integrity becomes a hard invariant: every materialized page has a required hash, and verification always enforces it.
  - Resident tier uses stable public interfaces (or adapter facade) instead of private Context internals.
  - Planner/endpoint contracts are aligned to a single source of truth for options that affect protected data.
- Engineering outcomes:
  - Lower regression risk through targeted tests for contract parity and integrity edge cases.
  - Faster maintenance by reducing cross-module private coupling.
  - Better verification throughput for large stores by removing duplicate payload reads in integrity checks.

## 2. Roadmap (incremental)
### Phase 1 (Stabilize Core) - up to 10 highest-impact tasks (prioritize P0/P1)
- Goal
  - Close integrity and API contract gaps that can produce incorrect safety behavior or inconsistent runtime behavior.
- Scope (what we touch / what we don’t)
  - Touch: `resident_tier`, API models/handlers, tests for these paths.
  - Don’t touch: codec math algorithms, benchmark methodology, packaging/dependency versions.
- Deliverables (concrete changes)
  - Enforce mandatory `payload_sha256` for all non-reference resident pages during manifest validation and verification.
  - Align vector decompress compressor construction with envelope/runtime policy instead of hardcoded native preference.
  - Extend resident plan API contract to support protected vector indices and pass through to planner.
- Dependencies
  - None external; all required surfaces are already in repo.
- Risk & Rollback strategy (if migration/contract changes are required)
  - Risk: stricter manifest validation may reject legacy stores lacking hashes.
  - Rollback: provide explicit compatibility gate/version branch in manifest loader for old schema and log “legacy, unverified hash” mode.
- Validation (how to verify: tests/linter/commands from the repo)
  - `pytest -q`
  - `make test`

### Phase 2 (UX & Domain Consolidation) - up to 10 tasks
- Goal
  - Reduce behavioral friction and lock boundaries between planner, encoder, and API contracts.
- Scope (what we touch / what we don’t)
  - Touch: resident verification flow, internal interfaces between `resident_tier` and `context_codec`, docs for behavior contracts.
  - Don’t touch: external API routes list and major CLI UX patterns.
- Deliverables (concrete changes)
  - Refactor integrity pass to avoid duplicate payload reads/decodes per page.
  - Introduce an explicit shared utility/facade for page similarity/int8/low-rank helpers used by resident tier (replace private method calls).
  - Add contract tests for planner/API parity (protected indices, route config consistency).
- Dependencies
  - Phase 1 completed to freeze required invariants first.
- Risk & Rollback strategy (if migration/contract changes are required)
  - Risk: refactor may alter output parity for page mode selection.
  - Rollback: keep golden behavior tests from current mode-count/error metrics and revert to old path if parity breaks.
- Validation (how to verify: tests/linter/commands from the repo)
  - `pytest -q`
  - `make test`

### Phase 3 (Scale & Maintainability)- up to 10 tasks (only if it truly blocks progress)
- Goal
  - Make validation and delivery path reliable at scale.
- Scope (what we touch / what we don’t)
  - Touch: repository automation for existing checks.
  - Don’t touch: product behavior and codec logic.
- Deliverables (concrete changes)
  - Add one repo-native aggregated validation target that mirrors CI core checks.
  - Switch CI to call that shared target to keep local/CI validation in sync.
- Dependencies
  - Phase 1-2 contract tests available to automate.
- Risk & Rollback strategy (if migration/contract changes are required)
  - Risk: CI setup flakiness from missing optional deps.
  - Rollback: scope CI to deterministic unit suite and keep optional heavy benchmarks manual.
- Validation (how to verify: tests/linter/commands from the repo)
  - `make test`

## 3. Task Specs (atomic, single-strategy)
- ID: EVO-001
  - Priority: P0
  - Theme: Reliability
  - Problem: Resident integrity check can be bypassed when `payload_sha256` is absent.
  - Evidence: `hyperquant/resident_tier.py:739-744` (`ResidentTierStore._read_verified_payload`), `hyperquant/resident_tier.py:142-143` and `hyperquant/resident_tier.py:168-169` (`ResidentPageDescriptor`), `hyperquant/resident_tier.py:242-251` (`ResidentTierManifest.validate`).
  - Root Cause: Hash field is optional in model and not validated as required for materialized pages.
  - Impact: Tampered payloads may be accepted if manifest hash metadata is removed/omitted.
  - Fix (single solution): Require non-empty `payload_sha256` for every non-`page_ref` page in manifest validation and fail verification if missing.
  - Steps:
    1. Tighten `ResidentTierManifest.validate` for hash presence on materialized pages.
    2. Tighten `_read_verified_payload` to reject missing hash for materialized pages.
    3. Add tests for “missing hash in manifest” and “hash mismatch” paths.
  - Acceptance Criteria (verifiable):
    - Opening/verifying a store with any non-reference page lacking hash fails with a clear `ValueError`.
    - Existing properly generated stores continue to pass integrity checks.
  - Validation Commands (if visible in the project):
    - `pytest -q`
  - Migration/Rollback (if needed):
    - For legacy manifests, support explicit legacy schema branch if backward compatibility is mandatory.

- ID: EVO-002
  - Priority: P1
  - Theme: Domain
  - Problem: Vector decompress path ignores caller/runtime native FWHT preference.
  - Evidence: `hyperquant/api/app.py:311-313` (`vector_decompress_endpoint`), `hyperquant/api/models.py:45-52` (`VectorCompressRequest`), `hyperquant/api/app.py:289-290` (`vector_compress_endpoint`).
  - Root Cause: Decompress handler hardcodes `prefer_native_fwht=True` instead of deriving consistent policy.
  - Impact: API behavior/performance policy differs between compression and decompression for the same envelope family.
  - Fix (single solution): Build vector decompressor using envelope-derived config and server default preference, not hardcoded `True`.
  - Steps:
    1. Introduce deterministic policy resolution for decompress path.
    2. Update endpoint to pass resolved policy to `get_vector_compressor`.
    3. Add API test asserting policy parity behavior.
  - Acceptance Criteria (verifiable):
    - Decompress endpoint no longer hardcodes native preference.
    - New test covers policy resolution path.
  - Validation Commands (if visible in the project):
    - `pytest -q`
  - Migration/Rollback (if needed):
    - No data migration; rollback is code revert.

- ID: EVO-003
  - Priority: P1
  - Theme: UX
  - Problem: `/v1/resident/plan` cannot express protected vectors although planner supports it.
  - Evidence: `hyperquant/resident_tier.py:868-886` (`ResidentPlanner.plan`), `hyperquant/api/models.py:135-160` (`ResidentPlanRequest`), `hyperquant/api/app.py:330-359` (`resident_plan_endpoint`).
  - Root Cause: API schema omitted `protected_vector_indices` and endpoint pass-through.
  - Impact: Users cannot model resident plans under protection constraints via API, causing planning/runtime mismatch.
  - Fix (single solution): Add `protected_vector_indices` to resident plan request model and pass it into planner call.
  - Steps:
    1. Extend `ResidentPlanRequest` model.
    2. Pass through field in endpoint planner invocation.
    3. Add API test for protected planning path.
  - Acceptance Criteria (verifiable):
    - Request accepts protected indices and returns plan successfully.
    - Planner receives and applies protection mask path.
  - Validation Commands (if visible in the project):
    - `pytest -q`
  - Migration/Rollback (if needed):
    - Backward compatible additive request field.

- ID: EVO-004
  - Priority: P1
  - Theme: Performance
  - Problem: Integrity verification performs duplicate I/O and decode work per materialized page.
  - Evidence: `hyperquant/resident_tier.py:751-756` (`ResidentTierStore.verify_integrity`), `hyperquant/resident_tier.py:780-805` (`ResidentTierStore.get_page`).
  - Root Cause: `verify_integrity` reads payload directly, then `get_page` reads/decodes it again.
  - Impact: Slower verification on large stores; unnecessary disk and CPU load.
  - Fix (single solution): Refactor verification to a single decode path that both validates hash and reconstructs page once.
  - Steps:
    1. Introduce internal helper returning verified decoded page.
    2. Reuse helper from both `verify_integrity` and `get_page`.
    3. Add benchmark-style unit assertion for call-count/path behavior if feasible.
  - Acceptance Criteria (verifiable):
    - Materialized page payload is read once per page during `verify_integrity`.
    - Functional integrity results remain identical.
  - Validation Commands (if visible in the project):
    - `pytest -q`
  - Migration/Rollback (if needed):
    - No migration required.

- ID: EVO-005
  - Priority: P2
  - Theme: Platform
  - Problem: Resident tier depends on private ContextCodec internals, raising change risk.
  - Evidence: `hyperquant/resident_tier.py:444`, `hyperquant/resident_tier.py:482`, `hyperquant/resident_tier.py:484`, `hyperquant/resident_tier.py:494`, `hyperquant/resident_tier.py:504`, `hyperquant/resident_tier.py:528` (`_TieredPageEncoder._encode_pages`).
  - Root Cause: No public/shared helper boundary for reusable page operations.
  - Impact: Any internal Context refactor can silently break resident tier behavior.
  - Fix (single solution): Extract shared page-ops utility module with public functions and migrate resident tier usage to it.
  - Steps:
    1. Create public helper functions for protected mask/hash/error/int8/low-rank page operations.
    2. Replace private method calls in resident tier.
    3. Add focused regression tests around shared helpers.
  - Acceptance Criteria (verifiable):
    - `resident_tier` no longer calls Context private methods.
    - Existing behavior metrics (mode counts/errors) stay stable in tests.
  - Validation Commands (if visible in the project):
    - `pytest -q`
  - Migration/Rollback (if needed):
    - No data migration; fallback is restoring previous direct calls.

- ID: EVO-006
  - Priority: P2
  - Theme: Reliability
  - Problem: Local and CI validation pipelines are not defined from a single source of truth.
  - Evidence: `Makefile:9-10` (`test` target), `docs/deployment.md:13-17` (recommended order), `.github/workflows/ci.yml:23-38` (extra CI-only compile/native/build/proof-pack steps).
  - Root Cause: CI workflow embeds checks directly instead of calling a shared repo-native aggregate command.
  - Impact: Contributors can pass local checks but fail CI for steps not represented in standard local command flow.
  - Fix (single solution): Introduce a single aggregate make target (e.g., `ci-check`) and make CI call it.
  - Steps:
    1. Add aggregate target in `Makefile` that sequences compile/native/test/proof-pack smoke checks.
    2. Update CI workflow to execute the aggregate target instead of duplicating step logic.
  - Acceptance Criteria (verifiable):
    - Running the aggregate target locally executes the same core checks as CI.
    - CI job invokes the same aggregate target.
  - Validation Commands (if visible in the project):
    - `make ci-check`
  - Migration/Rollback (if needed):
    - Roll back to previous CI step list if aggregate target causes instability.

## 4. Explicit Non-Goals
- Rewriting core quantization math or changing route algorithms (`vector_codec`, `context_codec`) without evidence of defects.
- Changing dependency versions or lockfiles (no such necessity shown in audit).
- Adding new product routes or external integrations (auth/object store/multi-node cache) outside current OSS scope (`README.md:142-147`, `docs/deployment.md:20-22`).
- Benchmark score chasing without correctness/contract impact.
