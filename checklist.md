# Checklist — hephaestus

Target version: 0.10.0 (bumped; CHANGELOG synced). Sprint phase: Execution.
Phase 1 COMPLETE. Phase 2 current increment: `hephaestus-wgpu` Leto parity
linalg and comparative benchmarks. Next concrete increment: complete WGPU/Leto
parity audit for remaining operator families and shared Atlas seam usage
(`mnemosyne`, `moirai`, `themis`, `hermes`).

## Unreleased WGPU Leto parity linalg [minor]
- [x] Added GPU-resident allocating `matmul`/`batched_matmul` and caller-owned
  `matmul_into`/`batched_matmul_into`, plus `dot`, `trace`, `norm_l1`,
  `norm_l2`, and `norm_max` over strided operands.
- [x] Added GPU-resident allocating `kron` and caller-owned `kron_into` over
  strided matrix operands, with Leto differential contract coverage and
  comparative benchmark coverage against Leto, `ndarray`, and a
  nalgebra-backed reference implementation.
- [x] Added GPU-resident `matpow` over strided square matrix operands, using
  exponentiation by squaring over WGPU `matmul_into` dispatches. Differential
  tests cover Leto parity for floating-point shear powers, integer `A^0`, and
  non-square rejection; comparative benchmarks cover Leto, an `ndarray`
  repeated-squaring reference, and `nalgebra`.
- [x] Added GPU-resident finite-`f32` `matrix_rank` and
  `matrix_rank_with_tolerance` over strided rank-2 operands. Differential tests
  cover exact finite full-rank, rank-deficient, and zero matrices against Leto,
  plus empty-matrix rejection; comparative benchmarks cover WGPU, Leto,
  `ndarray`-backed, and `nalgebra`-backed references. Residual distinction:
  WGPU uses row-reduction pivots while Leto's rank uses the SVD spectrum.
- [x] Added GPU-resident finite-`f32` `det` over strided square rank-2
  operands using the shared WGPU matrix-property row-reduction dispatch.
  Differential tests cover exact finite nonsingular and singular matrices
  against Leto plus rectangular rejection; comparative benchmarks cover WGPU,
  Leto, `ndarray`, and `nalgebra` references. Residual distinction: WGPU uses
  exact row-reduction pivots with no tolerance for determinant while Leto uses
  its CPU determinant algorithm.
- [x] Added WGPU device-resident Cholesky, LU, and QR decomposition surfaces
  mirroring Leto's decomposition, solve, determinant, and inverse APIs where
  each factorization supports them. Differential tests compare factors and
  solve/inverse outputs against Leto; comparative benchmarks cover WGPU, Leto,
  and `nalgebra`. Residual distinction: factorization currently delegates to
  Leto on the host and uploads factors to the device, so this is API parity and
  measured transfer/host-factorization overhead, not GPU-kernel parity.
- [x] Added blocked WGPU Cholesky entry point with CPU panel factorization and
  triangular solve plus GPU SYRK trailing update. Differential coverage now
  includes a 66x66 SPD matrix crossing the 64-wide block boundary; comparative
  benchmarks now measure 128x128 blocked Cholesky against Leto and `nalgebra`.
- [x] Routed WGPU launch planning through Mnemosyne `KernelResourceBudget` and
  Moirai GPU `plan_launch` while preserving Hephaestus checked overflow
  semantics from `BlockWidth::checked_covering_blocks`.
- [x] Added GPU-resident rank-2 `reduce_axis`, `sum_axis`, `min_axis`,
  `max_axis`, `mean_axis`, and caller-owned `*_axis_into` forms, preserving
  Leto's rank-preserving axis-reduction contract (`[rows, cols] -> [1, cols]`
  or `[rows, 1]`). Differential tests cover caller-owned and allocating sum,
  min, max, and mean against Leto; comparative benchmarks cover axis 0.
- [x] Added GPU-resident rank-2 `scan_axis_into`, `scan_axis`,
  `cumsum_into`, and `cumsum`, with forward/reverse scan direction and
  cumulative sum/product markers. Differential tests cover caller-owned and
  allocating Cumsum plus reverse cumulative product against Leto; comparative
  benchmarks cover Cumsum over axis 1 against Leto, an `ndarray` reference,
  and a nalgebra-backed reference.
- [x] Added allocating strided elementwise wrappers
  `binary_elementwise_strided`, `unary_elementwise_strided`, and
  `scalar_elementwise_strided`, returning C-contiguous GPU buffers while
  delegating to the existing caller-owned strided kernels. Differential tests
  cover allocated binary, unary, and scalar outputs against the same CPU
  references used for caller-owned dispatch.
- [x] Corrected `norm_l2` to return `sqrt(sum(x*x))`, matching Leto's CPU
  contract rather than exposing the squared-sum intermediate.
- [x] Extended `comparative` benchmark coverage to WGPU vs Leto, `ndarray`,
  and `nalgebra`; refreshed `benchmark_results.md` from a real local WGPU run
  including blocked 128x128 Cholesky.
- [x] Added fused WGPU map-reduction dispatch for trace and L1 norm. Dot
  product, L2 norm, and max norm retain the measured faster staged paths after
  the fused variant regressed in the local comparative run.
- Evidence: `cargo fmt -p hephaestus-wgpu -p hephaestus-cuda --check`;
  `cargo clippy -p hephaestus-wgpu -p hephaestus-cuda --all-targets -- -D
  warnings`; `cargo nextest run -p hephaestus-wgpu -p hephaestus-cuda` (90
  passed); `cargo test --doc -p hephaestus-wgpu` (0 doctests); `cargo test
  --doc -p hephaestus-cuda` (0 doctests); `cargo doc -p hephaestus-wgpu
  --no-deps`; `cargo doc -p hephaestus-cuda --no-deps`; `cargo bench -p
  hephaestus-wgpu --bench comparative` (refreshed `benchmark_results.md`,
  including blocked 128x128 Cholesky, matrix rank, determinant, LU, and QR;
  CUDA rows skipped because the WGPU bench depends on `hephaestus-cuda` without
  its `cuda` feature in this environment). Full workspace all-features clippy
  attempted earlier and blocked before this slice by `cuda-bindings` requiring
  `CUDA_TOOLKIT_PATH`. Evidence tier: value-semantic differential tests,
  static diagnostics, and empirical benchmarks.

## Unreleased CUDA Leto parity application surface [minor]
- [x] CUDA exports mirror the current WGPU/Leto core operation and decomposition slice:
  elementwise, strided elementwise, reductions, rank-2 axis reductions,
  rank-2 scans, `cumsum_into`/`cumsum`, matrix multiplication, Kronecker
  product, matrix power, finite-`f32` matrix rank, dot, trace, norms, and
  Cholesky/LU/QR decompositions.
- [x] Renamed CUDA forward cumulative-sum caller-owned API to `cumsum_into`,
  matching Leto and WGPU with no compatibility alias.
- [x] Stub-mode CUDA build validates the operation surface without fabricating
  hardware: unavailable-device tests skip by construction, while contract tests
  still exercise host-visible error paths and CPU-backed semantics available in
  the stub.
- [x] Removed stale default-build CUDA blocked-Cholesky export/test references
  because the CUDA blocked SYRK path is CUDA-feature gated and not verified in
  the default stub build.
- Evidence: `cargo fmt -p hephaestus-cuda --check`; `cargo clippy -p
  hephaestus-cuda --all-targets -- -D warnings`; `cargo test -p
  hephaestus-cuda` (38 passed); `cargo test --doc -p hephaestus-cuda` (0
  doctests); `cargo doc -p hephaestus-cuda --no-deps`. Evidence tier: static
  diagnostics and value-semantic contract tests in the currently available
  stub mode.

## 0.10.0 checked launch grid arithmetic [minor]
- [x] Added `BlockWidth::checked_covering_blocks` as the non-saturating launch
  grid arithmetic API for backends that need typed dispatch errors.
- [x] Routed WGPU `workgroups` through the checked core API, leaving WGPU only
  responsible for converting `None` into `HephaestusError::DispatchFailed`.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-core
  domain::launch::tests --offline` (2 passed); `cargo test -p
  hephaestus-wgpu application::pipeline::tests --offline` (2 passed);
  `cargo check --workspace --offline`; `cargo check --workspace --locked`;
  `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  `cargo nextest run --workspace --locked` (35 passed); `cargo test --doc
  --workspace --locked`; `cargo metadata --no-deps --locked --format-version
  1`; `cargo doc --workspace --no-deps --locked`; `cargo bench --bench
  elementwise_into --locked` on real adapter (allocating 332,480 ns/iter;
  caller-owned 102,150 ns/iter for 1,048,576 elements, 20 iterations);
  `cargo bench --bench reduction_width --locked` on real adapter (default
  42,960 ns/iter; width-128 91,620 ns/iter for 65,536 elements, 20
  iterations). Deeper gate attempted: `cargo semver-checks --workspace
  --all-features` blocked because `hephaestus-core` is not published in the
  registry. Evidence tier: value-semantic unit tests, dispatch contract tests,
  static diagnostics, and empirical benchmarks.

## 0.9.4 WGPU byte-size SSOT [patch]
- [x] Made the checked byte-size helper available to WGPU application modules.
- [x] Replaced the remaining local `size_of::<...>() as u64` buffer-size
  calculations in scalar uniform acquisition, strided metadata uniform
  acquisition, and singleton reduction copy encoding.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  infrastructure::device::tests --offline` (3 passed); `cargo test -p
  hephaestus-wgpu application::reduction::tests --offline` (1 passed);
  `cargo check --workspace --locked`; `cargo clippy --workspace --all-targets
  --locked -- -D warnings`; `cargo nextest run --workspace --locked` (35
  passed); `cargo test --doc --workspace --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo doc --workspace --no-deps --locked`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  206,970 ns/iter; caller-owned 60,640 ns/iter for 1,048,576 elements, 20
  iterations); `cargo bench --bench reduction_width --locked` on real adapter
  (default 47,335 ns/iter; width-128 55,895 ns/iter for 65,536 elements, 20
  iterations); `rg` confirmed no remaining local `size_of::<...>() as u64`
  buffer-size casts in `crates/hephaestus-wgpu/src`. Deeper gate attempted:
  `cargo semver-checks --workspace --all-features` blocked because
  `hephaestus-core` is not published in the registry. Evidence tier:
  value-semantic unit tests, dispatch contract tests, static diagnostics, and
  empirical benchmarks.

## 0.9.3 upload byte-size precheck [patch]
- [x] Routed `WgpuDevice::upload` through the shared checked byte-size helper
  before `create_buffer_init`, so upload, allocation, and download paths use
  the same allocation-overflow boundary.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  infrastructure::device::tests --offline` (3 passed); `cargo check
  --workspace --locked`; `cargo clippy --workspace --all-targets --locked
  -- -D warnings`; `cargo nextest run --workspace --locked` (35 passed);
  `cargo test --doc --workspace --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo doc --workspace --no-deps --locked`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  244,285 ns/iter; caller-owned 59,270 ns/iter for 1,048,576 elements, 20
  iterations); `cargo bench --bench reduction_width --locked` on real adapter
  (default 41,100 ns/iter; width-128 58,235 ns/iter for 65,536 elements, 20
  iterations). Deeper gate attempted: `cargo semver-checks --workspace
  --all-features` blocked because `hephaestus-core` is not published in the
  registry. Evidence tier: value-semantic unit tests, dispatch contract tests,
  static diagnostics, and empirical benchmarks.

## 0.9.2 dispatch precheck completion [patch]
- [x] Hoisted binary and unary dispatch workgroup-range validation before
  pipeline cache lookup, bind-group creation, and command encoding.
- [x] Hoisted reduction workgroup-range validation before intermediate output
  buffer allocation in each reduction pass.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  application::pipeline::tests --offline` (2 passed); `cargo check
  --workspace --locked`; `cargo clippy --workspace --all-targets --locked
  -- -D warnings`; `cargo nextest run --workspace --locked` (35 passed);
  `cargo test --doc --workspace --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo doc --workspace --no-deps --locked`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  239,850 ns/iter; caller-owned 152,440 ns/iter for 1,048,576 elements, 20
  iterations); `cargo bench --bench reduction_width --locked` on real adapter
  (rerun: default 44,580 ns/iter; width-128 155,995 ns/iter for 65,536
  elements, 20 iterations). Deeper gate attempted: `cargo semver-checks
  --workspace --all-features` blocked because `hephaestus-core` is not
  published in the registry. Evidence tier: value-semantic unit tests,
  dispatch contract tests, static diagnostics, and empirical benchmarks.

## 0.9.1 dispatch range precheck [patch]
- [x] Hoisted scalar dispatch workgroup-range validation before transient
  uniform-buffer acquisition.
- [x] Hoisted strided dispatch workgroup-range validation before transient
  metadata uniform-buffer acquisition.
- [x] Added shared `workgroups` boundary coverage for the exact `u32::MAX`
  workgroup limit and one element beyond it.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  application::pipeline::tests --offline` (2 passed); `cargo check
  --workspace --locked`; `cargo clippy --workspace --all-targets --locked
  -- -D warnings`; `cargo nextest run --workspace --locked` (35 passed);
  `cargo test --doc --workspace --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo doc --workspace --no-deps --locked`;
  `cargo bench --bench elementwise_into --locked` on real adapter (allocating
  322,175 ns/iter; caller-owned 89,460 ns/iter for 1,048,576 elements, 20
  iterations); `cargo bench --bench reduction_width --locked` on real adapter
  (default 38,700 ns/iter; width-128 41,930 ns/iter for 65,536 elements, 20
  iterations). Deeper gate attempted: `cargo semver-checks --workspace
  --all-features` blocked because `hephaestus-core` is not published in the
  registry. Evidence tier: value-semantic unit tests, dispatch contract tests,
  static diagnostics, and empirical benchmarks.

## 0.9.0 transient buffer alignment [minor]
- [x] Added a shared checked `aligned_size` helper for WGPU byte alignment.
- [x] Made `get_staging_buffer` and `get_uniform_buffer` return
  `Result<wgpu::Buffer>` and reject alignment overflow with
  `AllocationFailed`.
- [x] Updated scalar, strided, and download call sites to propagate allocation
  failures.
- [x] Added value-semantic unit coverage for alignment overflow.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  infrastructure::device::tests --offline` (3 passed); `cargo check
  --workspace --locked`; `cargo clippy --workspace --all-targets --locked
  -- -D warnings`; `cargo nextest run --workspace --locked` (33 passed);
  `cargo test --doc --workspace --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo doc --workspace --no-deps --locked`;
  `cargo bench --bench elementwise_into --locked` on real adapter (serial
  rerun: allocating 263,335 ns/iter; caller-owned 70,295 ns/iter for
  1,048,576 elements, 20 iterations); `cargo bench --bench reduction_width
  --locked` on real adapter (serial rerun: default 47,740 ns/iter; width-128
  107,070 ns/iter for 65,536 elements, 20 iterations). Deeper gate attempted:
  `cargo semver-checks --workspace --all-features` blocked because
  `hephaestus-core` is not published in the registry. Evidence tier:
  value-semantic unit tests, dispatch contract tests, static diagnostics, and
  empirical benchmarks.

## 0.8.1 pipeline cache critical section [patch]
- [x] Split `cached_pipeline` into a locked cache-hit check, unlocked WGPU
  pipeline compilation, and locked recheck/insert.
- [x] Preserved cache correctness under races by rechecking the key before
  insertion.
- Evidence: `cargo fmt --check`; `cargo check --workspace --offline`;
  `cargo check --workspace --locked`; `cargo clippy --workspace
  --all-targets --locked -- -D warnings`; `cargo nextest run --workspace
  --locked` (32 passed); `cargo test --doc --workspace --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo doc
  --workspace --no-deps --locked`; `cargo bench --bench elementwise_into
  --locked` on real adapter (allocating 383,335 ns/iter; caller-owned 74,275
  ns/iter for 1,048,576 elements, 20 iterations); `cargo bench --bench
  reduction_width --locked` on real adapter (default 56,090 ns/iter;
  width-128 59,900 ns/iter for 65,536 elements, 20 iterations). Evidence tier:
  value-semantic dispatch contract tests, static diagnostics, and empirical
  benchmark.

## 0.8.0 checked allocation sizing [minor]
- [x] Added `HephaestusError::AllocationFailed` as the typed boundary for
  allocation requests rejected before buffer creation.
- [x] Replaced unchecked WGPU byte-size multiplication with checked exact and
  padded size helpers shared by allocation and download sizing.
- [x] Added unit coverage for copy-alignment padding and overflow rejection
  without allocating memory.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  infrastructure::device::tests --offline` (2 passed); `cargo check
  --workspace --locked`; `cargo clippy --workspace --all-targets --locked
  -- -D warnings`; `cargo nextest run --workspace --locked` (32 passed);
  `cargo test --doc --workspace --locked`; `cargo metadata --no-deps
  --locked --format-version 1`; `cargo doc --workspace --no-deps --locked`;
  `cargo bench --bench reduction_width --locked` on real adapter (default
  33,110 ns/iter; width-128 49,195 ns/iter for 65,536 elements, 20
  iterations). Deeper gate attempted: `cargo semver-checks --workspace
  --all-features` blocked because `hephaestus-core` is not published in the
  registry. Evidence tier: value-semantic unit tests, contract tests, static
  diagnostics, and empirical benchmark.

## 0.7.3 reduction pass storage [patch]
- [x] Added a single `reduction_pass_count` helper for the multi-pass tree
  depth calculation.
- [x] Preallocated the intermediate `WgpuBuffer` handle vector with that pass
  count before command encoding.
- [x] Added value-semantic unit coverage for empty, singleton, exact-width,
  trailing-width, and multi-pass depths.
- Evidence: `cargo fmt --check`; `cargo test -p hephaestus-wgpu
  application::reduction::tests::pass_count_matches_tree_depth --offline`
  (1 passed); `cargo check --workspace --locked`; `cargo clippy --workspace
  --all-targets --locked -- -D warnings`; `cargo nextest run --workspace
  --locked` (30 passed); `cargo test --doc --workspace --locked`;
  `cargo metadata --no-deps --locked --format-version 1`; `cargo doc
  --workspace --no-deps --locked`; `cargo bench --bench reduction_width
  --locked` on real adapter (rerun: default 50,330 ns/iter; width-128
  97,035 ns/iter for 65,536 elements, 20 iterations). Evidence tier:
  value-semantic unit tests, contract tests, static diagnostics, and empirical
  benchmark.

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
