# Checklist — hephaestus

Target version: 0.7.2 (bumped; CHANGELOG synced). Sprint phase: Execution.
Phase 1 COMPLETE. Phase 2 gating ADR ACCEPTED (`docs/adr/0001-cuda-backend.md`
— cuda-oxide device substrate + cutile kernel authoring, SoC boundary,
no-toolkit-to-compile, differential parity vs CPU and wgpu). Next concrete
increment: `hephaestus-cuda` crate, stage 1 — device substrate on cuda-oxide
(acquisition, typed buffers, transfers) with skip-without-driver contract tests.

## 0.7.2 reduction width validation [patch]
- [x] Moved `reduction_with_width` power-of-two validation before empty and
  singleton fast paths.
- [x] Added boundary contract coverage proving invalid widths are rejected for
  empty, singleton, and multi-element inputs.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo nextest run -p hephaestus-wgpu
  reduction_width_is_part_of_dispatch_contract --locked` (1 passed);
  `cargo check --workspace --locked`; `cargo clippy --workspace
  --all-targets --locked -- -D warnings`; `cargo nextest run --workspace
  --locked` (29 passed); `cargo test --doc --workspace --locked`;
  `cargo doc --workspace --no-deps --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo bench --bench reduction_width
  --locked` on real adapter (default 49,945 ns/iter; width-128 55,945
  ns/iter for 65,536 elements, 20 iterations); `git diff --check`. Evidence
  tier: value-semantic contract tests, static diagnostics, and empirical
  benchmark.

## 0.7.1 reduction-width benchmark [patch]
- [x] Added `reduction_width` benchmark target for default vs width-128
  reduction dispatch.
- [x] Benchmark validates both device outputs against an exact host-side `u32`
  sum before reporting timings.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo nextest run --workspace --locked` (29
  passed); `cargo test --doc --workspace --locked`; `cargo doc --workspace
  --no-deps --locked`; `cargo metadata --no-deps --locked --format-version 1`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  250,445 ns/iter; caller-owned 77,795 ns/iter for 1,048,576 elements, 20
  iterations); `cargo bench --bench reduction_width --locked` on real adapter
  (default 40,460 ns/iter; width-128 79,655 ns/iter for 65,536 elements, 20
  iterations); `git diff --check`. Evidence tier: value-semantic benchmark
  validation, value-semantic tests, and empirical benchmark.

## 0.7.0 reduction block-width dispatch [minor]
- [x] Added `reduction_with_width` so reduction WGSL generation, pipeline
  cache keying, intermediate output sizing, and dispatch group counts use a
  caller-selected power-of-two `BlockWidth`.
- [x] Kept `reduction` as the default-width API by delegating to
  `reduction_with_width(..., BlockWidth::DEFAULT)`.
- [x] Added contract coverage for width 128 integer reduction and
  non-power-of-two width rejection.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo nextest run --workspace --locked` (29
  passed); `cargo test --doc --workspace --locked`; `cargo doc --workspace
  --no-deps --locked`; `cargo metadata --no-deps --locked --format-version 1`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  278,195 ns/iter; caller-owned 55,390 ns/iter for 1,048,576 elements, 20
  iterations); `git diff --check`. Deeper gates attempted: `cargo
  semver-checks --workspace --all-features` blocked because the crates are not
  published in the registry; `cargo llvm-cov --workspace --locked` blocked by
  missing `llvm-tools-preview`. Evidence tier: typed API contract,
  value-semantic tests, and empirical benchmark.

## 0.6.9 remaining invariant panic names [patch]
- [x] Replaced the unnamed `BlockWidth::DEFAULT` const panic with an explicit
  invariant message.
- [x] Normalized the strided bind-slot conversion `expect` message to the
  same `invariant:` convention as the other library panic sites.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; invariant-panic scan confirms every non-test panic
  site carries an `invariant:` message; `cargo nextest run --workspace
  --locked` (28 passed); `cargo test --doc --workspace --locked`; `cargo doc
  --workspace --no-deps --locked`; `cargo metadata --no-deps --locked
  --format-version 1`; `cargo bench --bench elementwise_into --locked` on
  real adapter (allocating 244,860 ns/iter; caller-owned 81,235 ns/iter for
  1,048,576 elements, 20 iterations); `git diff --check`. Evidence tier:
  source audit, value-semantic tests, and empirical benchmark.

## 0.6.8 library invariant panic messages [patch]
- [x] Replaced library-code unqualified `unwrap()` sites in reduction internal
  buffer selection, pipeline-cache locking, and transient-pool locking with
  explicit invariant `expect(...)` messages.
- [x] Confirmed remaining `unwrap()` sites in source scan are test-local.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; source `unwrap()` scan confirms remaining hits are
  test-local; `cargo nextest run --workspace --locked` (28 passed); `cargo
  test --doc --workspace --locked`; `cargo doc --workspace --no-deps --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo bench
  --bench elementwise_into --locked` on real adapter (allocating 234,105
  ns/iter; caller-owned 79,575 ns/iter for 1,048,576 elements, 20 iterations);
  `git diff --check`. Evidence tier: source audit, value-semantic tests, and
  empirical benchmark.

## 0.6.7 value-semantic negative assertions [patch]
- [x] Replaced remaining broad absence and variant-only assertions in the
  audited Rust test scope with concrete mapped-value or length comparisons.
- [x] Confirmed no `is_err`, `is_ok`, `is_some`, `is_none`, or
  `assert!(matches!)` assertions remain under the audited source/test paths.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo nextest run --workspace --locked` (28
  passed); assertion-pattern scan with `rg` over audited source/test paths
  returned no matches; `cargo test --doc --workspace --locked`; `cargo doc
  --workspace --no-deps --locked`; `cargo metadata --no-deps --locked
  --format-version 1`; `cargo bench --bench elementwise_into --locked` on
  real adapter (allocating 197,245 ns/iter; caller-owned 58,495 ns/iter for
  1,048,576 elements, 20 iterations); `git diff --check`. Evidence tier:
  value-semantic tests, assertion-pattern scan, and empirical benchmark.

## 0.6.6 negative-path contract assertions [patch]
- [x] Replaced remaining elementwise and strided negative-path `is_err()`
  assertions with typed `HephaestusError` checks.
- [x] Strided rejection tests now assert the zero-stride-output dispatch
  message and the exact layout storage error for backing-buffer overflow.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo nextest run -p hephaestus-wgpu
  strided_rejects_aliasing_output_and_short_buffers --locked` (1 passed);
  `cargo nextest run --workspace --locked` (28 passed); `cargo test --doc
  --workspace --locked`; `cargo doc --workspace --no-deps --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo bench
  --bench elementwise_into --locked` on real adapter (allocating 160,095
  ns/iter; caller-owned 52,375 ns/iter for 1,048,576 elements, 20 iterations);
  `git diff --check`. Evidence tier: value-semantic contract tests and
  empirical benchmark.

## 0.6.5 contiguous elementwise alias guard [patch]
- [x] Added a shared output/input alias guard for caller-owned contiguous
  binary, unary, and scalar elementwise dispatch.
- [x] Added contract coverage for binary left/right aliases plus unary and
  scalar aliases, asserting the typed `DispatchFailed` error message.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo nextest run --workspace --locked` (28
  passed); `cargo test --doc --workspace --locked`; `cargo doc --workspace
  --no-deps --locked`; `cargo metadata --no-deps --locked --format-version 1`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  215,105 ns/iter; caller-owned 84,150 ns/iter for 1,048,576 elements, 20
  iterations); `git diff --check`. Evidence tier: value-semantic contract
  tests and empirical benchmark.

## 0.6.4 transient pool best-fit reuse [patch]
- [x] Changed `BoundedBufferPool::take_at_least` to choose the smallest
  retained buffer that satisfies the requested size.
- [x] Added regression coverage preserving a large retained buffer after a
  small request consumes a smaller sufficient buffer.
- Evidence: `cargo test -p hephaestus-wgpu infrastructure::pool --locked`.
  Full gate: `cargo fmt --check`; `cargo check --workspace --locked`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo nextest run --workspace --locked` (27 passed); `cargo test --doc
  --workspace --locked`; `cargo doc --workspace --no-deps --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo bench
  --bench elementwise_into --locked` on real adapter (allocating 393,935
  ns/iter; caller-owned 85,240 ns/iter for 1,048,576 elements, 20
  iterations); `git diff --check`. Evidence tier: value-semantic unit tests
  and empirical benchmark.

## 0.6.3 transient pool FIFO storage [patch]
- [x] Replaced `BoundedBufferPool` backing storage with `VecDeque` so
  oldest-first count-cap eviction uses `pop_front()` instead of `Vec::remove(0)`.
- Evidence: `cargo test -p hephaestus-wgpu infrastructure::pool --locked`.
  Full gate: `cargo fmt --check`; `cargo check --workspace --locked`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo nextest run --workspace --locked` (26 passed); `cargo test --doc
  --workspace --locked`; `cargo doc --workspace --no-deps --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo bench
  --bench elementwise_into --locked` on real adapter (allocating 269,205
  ns/iter; caller-owned 90,290 ns/iter for 1,048,576 elements, 20
  iterations); `git diff --check`. Evidence tier: value-semantic unit tests
  and empirical benchmark.

## 0.6.2 adaptive transient pools [patch]
- [x] Changed `BoundedBufferPool::recycle` to evict oldest retained buffers
  when the count cap is full, then enforce the byte cap.
- [x] Added regression coverage for full-pool small-buffer pollution and a
  zero-count invariant.
- Evidence: `cargo test -p hephaestus-wgpu infrastructure::pool --locked`.
  Full gate: `cargo fmt --check`; `cargo check --workspace --locked`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo nextest run --workspace --locked` (26 passed); `cargo test --doc
  --workspace --locked`; `cargo doc --workspace --no-deps --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo bench
  --bench elementwise_into --locked` on real adapter (allocating 174,145
  ns/iter; caller-owned 95,335 ns/iter for 1,048,576 elements, 20
  iterations); `git diff --check`. Evidence tier: value-semantic unit tests
  and empirical benchmark.

## 0.6.1 bounded transient pools [patch]
- [x] Added `infrastructure::pool::BoundedBufferPool` with retained-buffer
  count and byte caps.
- [x] Routed staging and uniform pools through the bounded pool while keeping
  existing WGPU buffer reuse semantics.
- Evidence: `cargo test -p hephaestus-wgpu infrastructure::pool --locked`;
  `cargo fmt --check`; `cargo check --workspace --locked`; `cargo clippy
  --workspace --all-targets --locked -- -D warnings`; `cargo nextest run
  --workspace --locked` (24 passed); `cargo test --doc --workspace --locked`;
  `cargo doc --workspace --no-deps --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo bench --bench elementwise_into
  --locked` on real adapter (allocating 272,830 ns/iter; caller-owned
  132,710 ns/iter for 1,048,576 elements, 20 iterations); `git diff --check`.
  Evidence tier: type-level ownership plus value-semantic unit/contract tests
  and empirical benchmark.

## 0.6.0 caller-owned contiguous elementwise [minor]
- [x] Added `binary_elementwise_into`, `unary_elementwise_into`, and
  `scalar_elementwise_into` for caller-owned output buffers and `BlockWidth`
  selection.
- [x] Routed allocating contiguous elementwise APIs through the caller-owned
  implementations; scalar dispatch now uses the uniform pool.
- [x] Consolidated pipeline-cache creation into `application::pipeline` for
  contiguous elementwise, strided elementwise, and reduction kernels.
- Evidence: `cargo fmt --check`; `cargo check --workspace --locked`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo nextest run --workspace --locked` (22 passed);
  `cargo test --doc --workspace --locked`; `cargo doc --workspace --no-deps
  --locked`; `cargo metadata --no-deps --locked --format-version 1`;
  `cargo bench --bench elementwise_into --locked` on real adapter
  (allocating 291,410 ns/iter; caller-owned 66,350 ns/iter for 1,048,576
  elements, 20 iterations). Evidence tier: value-semantic differential tests
  and empirical benchmark, not a stored Criterion regression baseline.

## Default provider feature contract [patch]
- [x] Added default `parallel` and `mnemosyne-memory` feature markers to
  `hephaestus-core` and `hephaestus-wgpu`.
- Evidence: `cargo metadata --no-deps --locked --format-version 1`; full Atlas
  feature-policy metadata audit; `cargo check --workspace --offline`;
  `cargo test --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo doc --workspace --no-deps --locked`;
  `git diff --check`.

## 0.3.1 uniform pooling + CUDA ADR [patch]
- [x] Pooled strided meta uniforms (queue-ordered write_buffer reuse);
  one fewer buffer allocation per dispatch. 17 tests green on hardware.
- [x] ADR 0001 accepted (Phase 2 gate).

## 0.3.0 strided unary/scalar + consolidation [minor]
- [x] Shared strided core (SSOT): `StridedMeta` packing, WGSL Meta/decode
  fragments, `cached_pipeline`, `encode_strided` serve all strided kernels.
- [x] `unary_elementwise_strided_into` (broadcast + caller-owned output).
- [x] `scalar_elementwise_strided_into` — zero new kernels (one-element
  operand at all-zero strides through the binary kernel).
- [x] Tests: unary transposed sqrt, unary broadcast neg, scalar/binary
  equivalence over a transposed view; 17 total on real hardware.
- [x] Gates: fmt, clippy `-D warnings`, test, doc — clean.

## 0.2.0 strided dispatch [minor]
- [x] `binary_elementwise_strided_into` over leto `Layout<N>` (rank ≤ 4
  compile-time cap, leto broadcast semantics, caller-owned output,
  aliasing/short-buffer rejection, packed 80-byte Meta uniform).
- [x] Differential strided suite (5) on real hardware; 14 tests total.
- [x] Gates: fmt, clippy `-D warnings`, test, doc — clean.

Previous sprint (0.1.0 scaffold) below.
In-flight item: none. Next concrete increment: strided-layout-aware dispatch (backlog Phase 1).

## 0.1.0 scaffold [arch]
- [x] Workspace: `hephaestus-core` (no GPU deps, `#![forbid(unsafe_code)]`,
  `#![deny(missing_docs)]`) + `hephaestus-wgpu` (wgpu 26).
- [x] `ComputeDevice` seam with GAT `Buffer<T: Pod>`; `DeviceBuffer<T>`;
  distinct error variants (adapter/device/length/dispatch/transfer).
- [x] `WgpuDevice` acquisition (default + custom limits) — single authoritative
  copy of the logic formerly duplicated in apollo-wgpu-helpers.
- [x] `WgpuBuffer<T>`: PhantomData-typed, padded allocation, `raw()` escape
  hatch for consumer-owned pipelines.
- [x] Upload (`create_buffer_init`), zeroed alloc, download (staging +
  map_async + poll), length-mismatch rejection before transfer.
- [x] `binary_elementwise::<Op, T>`: ZST `BinaryWgslOp` markers (Add/Sub/Mul),
  `WgslScalar` type-token substitution, arrayLength tail guard, partial
  trailing workgroup correct.
- [x] `unary_elementwise::<Op, T>`: ZST `UnaryWgslOp` markers (Exp/Ln/Sin/Cos/Sqrt/Abs/Neg/Recip) and shared WGSL template.
- [x] `scalar_elementwise::<Op, T>`: uniform buffer binding and ZST-wrapped pipeline cache keys.
- [x] `reduction::<Op, T>`: ZST `ReductionWgslOp` markers (Sum/Min/Max), multi-pass tree reduction, and type-safe `ReductionIdentity` mapping.
- [x] Contract tests (9): round-trip values, length rejection (download + dispatch), add/mul/unary/scalar/reduction vs CPU reference.
  Verified on real adapter hardware; environment-gated skip otherwise.
- [x] Gates: `cargo fmt --check`, `clippy --all-targets -- -D warnings`,
  `cargo test`, `cargo doc --no-deps` — all clean.
- [x] Pushed to GitHub; apollo delegation integration (see backlog Phase 4).
