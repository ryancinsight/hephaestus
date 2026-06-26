# Checklist — hephaestus

2026-06-26 (CUDA dynamic-rank strided delegation). Added
`hephaestus-cuda` dynamic-rank strided elementwise entry points over borrowed
shape/stride slices so runtime-shaped consumers can delegate strided CUDA
binary/unary kernels without materializing fixed-rank Leto layouts or carrying
duplicate local PTX generators. Static-rank APIs now share the same private
binary/unary launch helpers. Coeus routes rank <= 4 supported strided primitive
ops through this provider surface and retains local kernels only for unsupported
activations, write-aliasing, or wider-rank layouts. Verified:
`cargo check -p hephaestus-cuda`, `cargo fmt -p hephaestus-cuda --check`,
`cargo clippy -p hephaestus-cuda --all-targets -- -D warnings`, `cargo doc -p
hephaestus-cuda --no-deps`, `cargo nextest run -p hephaestus-cuda --test
strided` (11/11), and downstream `cargo nextest run -p coeus-cuda --features
cuda` (69/69).

2026-06-23 (strided-scalar pooled-uniform kernel). Paranoid re-audit of the hot
per-op kernels (reduction, scan, strided, sparse) via two skeptical agents.
Verified clean: the multi-pass reduction correctly encodes all passes in one
submit (the `temp_buffers` retention is *required*, not waste); scan/strided/
sparse meta uniforms are pooled; no `poll(Wait)` over-sync; Pod casts are
size-correct; SpMV/SpMM column-index bounds are guaranteed by Leto's CsrMatrix
construction. Found and fixed one real inconsistency: the strided scalar path
allocated a per-call device storage buffer for the broadcast scalar (the
contiguous path already uses a pooled uniform). Added a dedicated
`StridedScalarKernel` reading the scalar from a pooled uniform — no per-call
storage operand, value-identical (verified). Recorded the storage-pool
examination (deferred: within-call buffers are submit-pinned, cross-call needs
fences) and the CUDA strided-scalar parity follow-on (hardware-blocked).
Verified: `cargo fmt`, `cargo clippy -p hephaestus-wgpu --all-targets -- -D
warnings`, strided/scalar contract tests, full workspace nextest, doctests.

2026-06-23 (blocked-decomposition host-allocation reuse). Paranoid memory/safety
re-audit of the WGPU decomposition modules. Verified the Pod meta structs
(`SyrkMeta`/`GemmMeta`/`HhMeta`/`HhReflectorMeta`/`RegionCopyMeta`) are all
`#[repr(C)]`, padding-free, with true SAFETY comments, and the transfer/pool core
(`stage_and_read`, sub-buffer paths) is bounds/overflow-checked and RAII-pooled —
clean. Found and fixed real memory churn: the blocked Cholesky/LU/QR panel loops
allocated a fresh host `Vec` per panel for each region download plus per-panel
scratch. Added the region-download SSOT `download_matrix_region_compact_into`
(reuses host capacity), removed the dead returning-`Vec` wrapper, and hoisted each
decomposition's per-panel host buffers above the loop. Verified: `cargo fmt`,
`cargo clippy -p hephaestus-wgpu --all-targets -- -D warnings`, blocked LU/QR/
Cholesky cross-boundary contract tests, full workspace nextest, doctests.

2026-06-23 (residual-gap review). Reviewed the residual register and resolved the
two genuinely-actionable gaps: `matrix_rank`/`det` ill-conditioned divergence was
previously documented-but-untested at the threshold boundary. Documented both
contracts on the public APIs (`matrix_rank` relative-threshold pivot criterion;
`det` no-determinant-tolerance pivot product) and added analytically-derived
contract tests (`matrix_rank_relative_tolerance_is_the_discriminator`,
`det_of_near_singular_triangular_is_exact_pivot_product`). Restructured
`gap_audit.md` into an honest SSOT (Resolved / Accepted design / Open future work
/ Environment) so the remaining residuals — host-delegated GPU-kernel/performance
parity ([major], tracked) and CUDA hardware/toolchain blockers — are no longer
conflated with open defects. Verified: `cargo fmt`, `cargo clippy
-p hephaestus-wgpu -p hephaestus-core --all-targets -- -D warnings`, focused
rank/det tests, doctests, full workspace nextest.

2026-06-23 (WGPU staging-registry contention). Audited the staging-allocator
integration in `infrastructure/{device,buffer,pool}.rs`. Found the global
`WGPU_MAPPED_BUFFERS` registry resolved a sub-allocated staging pointer to its
mapped block with an `O(n)` `.iter().find()` scan held inside the global lock at
both HostPinned alloc/upload sites. Replaced the `HashMap` with a base-address
`BTreeMap` and a single `resolve_mapped_buffer` helper doing an `O(log n)`
`range(..=ptr).next_back()` containment query (DRY/SSOT). Tightened the registry
+ descriptor to `pub(crate)` (no external consumers; makes the type change
non-breaking) and removed the dead `usage` field. Verified: `cargo fmt`, `cargo
clippy -p hephaestus-wgpu -p hephaestus-core --all-targets -- -D warnings`,
`cargo nextest run --workspace` (228 passed incl. `test_placement_aware_allocation`,
upload/download round-trip), doctests. Remaining bounded characteristic recorded
in gap_audit: the registry still uses one global `Mutex` (now `O(log n)`); a
sharded registry is deferred until a workload measures the lock as hot.

Target version: 0.10.0 (bumped; CHANGELOG synced). Sprint phase: Execution.
Phase 1 COMPLETE. Phase 2 current increment: `hephaestus-wgpu` Leto parity
linalg and comparative benchmarks. Next concrete increment: complete WGPU/Leto
parity audit for remaining operator families and shared Atlas seam usage
(`mnemosyne`, `moirai`, `themis`, `hermes`).

Latest verification: `cargo fmt --check`,
`cargo clippy --workspace --all-targets --locked -- -D warnings`,
`cargo nextest run --workspace --locked` (229 passed),
`cargo test --doc --workspace --locked`,
`cargo doc --workspace --no-deps --locked`,
`cargo metadata --no-deps --locked --format-version 1`, `git diff --check`,
and `cargo bench -p hephaestus-wgpu --bench sparse_comparative --locked`.
`cargo semver-checks --workspace --all-features` is blocked because
`hephaestus-core` has no crates.io baseline.

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
- [x] Added WGPU device-resident SVD parity coverage. Contract tests cover
  closed-form singular values, thin-SVD reconstruction against Leto, and
  rank-revealing behavior on a rank-deficient matrix; comparative benchmarks
  cover WGPU, Leto, and `nalgebra` SVD on a 32x16 full-column-rank matrix.
- [x] Added WGPU device-resident bidiagonalization parity coverage. Contract
  tests cover orthogonal `U`/`V`, upper-bidiagonal structure, `U B V^T`
  reconstruction, singular-value preservation, and wide-matrix rejection;
  comparative benchmarks cover WGPU, Leto, and `nalgebra` SVD on a 32x16
  matrix.
- [x] Added WGPU device-resident Schur parity coverage. Contract tests cover
  orthogonal `Q`, quasi-upper-triangular `T`, `Q T Q^T` reconstruction,
  spectrum preservation, and rectangular rejection; comparative benchmarks
  cover WGPU, Leto, and `nalgebra` complex eigenvalues on a 32x32 block
  matrix.
- [x] Added WGPU device-resident Hessenberg parity coverage. Contract tests
  cover orthogonal `Q`, upper-Hessenberg `H`, `Q H Q^T` reconstruction,
  trace/norm similarity invariants, and rectangular rejection; comparative
  benchmarks cover WGPU, Leto, and `nalgebra` Hessenberg reduction on a 32x32
  matrix.
- [x] Added WGPU device-resident full-pivot LU parity coverage. Contract tests
  cover packed `L/U` reconstruction of `P A Q`, rank reporting, determinant,
  solve, inverse, rank-deficient inverse rejection, and rectangular rejection;
  comparative benchmarks cover WGPU, Leto, and `nalgebra` full-pivot LU on a
  32x32 matrix.
- [x] Added WGPU device-resident Bunch-Kaufman parity coverage. Contract tests
  cover downloaded `L`/`D` factors, permutation agreement with Leto,
  `L D L^T = P A P^T` reconstruction, rectangular rejection, and
  nonsymmetric rejection; comparative benchmarks cover WGPU and Leto
  Bunch-Kaufman with `nalgebra` determinant as the external CPU comparator.
- [x] Added WGPU device-resident UDU parity coverage. Contract tests cover
  downloaded `U`/diagonal `D` factors, determinant, solve, inverse,
  `U D U^T` reconstruction of a symmetric indefinite matrix, rectangular
  rejection, nonsymmetric rejection, and zero-pivot rejection; comparative
  benchmarks cover WGPU and Leto UDU with `nalgebra` determinant as the
  external CPU comparator.
- [x] Added WGPU device-resident column-pivoted QR, pseudoinverse, and matrix
  exponential baseline parity coverage. Contract tests cover column-pivoted QR
  factor agreement against Leto plus closed-form diagonal pseudoinverse and
  matrix exponential cases; comparative benchmarks cover WGPU and Leto for all
  three, with `nalgebra` comparators for column-pivoted QR and pseudoinverse.
- [x] Strengthened WGPU pseudoinverse and matrix exponential parity coverage.
  Contract tests now cover rank-deficient Moore-Penrose identities,
  rectangular full-rank pseudoinverse, non-finite pseudoinverse rejection,
  nilpotent and skew-symmetric matrix-exponential closed forms, a general
  `nalgebra` matrix-exponential oracle, and rectangular/non-finite matrix
  exponential rejection.
- [x] Added WGPU device-resident symmetric Jacobi eigen decomposition and
  eigenvalues-only surfaces mirroring Leto. Differential tests compare
  eigenvalues and eigenvectors against Leto and reject non-symmetric inputs;
  comparative benchmarks cover WGPU, Leto, and `nalgebra` eigenvalues with
  order-insensitive cross-library eigenvalue comparison. Residual distinction:
  eigensolve currently delegates to Leto on the host and uploads results to the
  device, so this is API parity and measured transfer/host-eigensolve overhead,
  not GPU-kernel eigensolver parity.
- [x] Added WGPU device-resident general eigenvalues over complex output
  buffers for square `f32` matrices. Contract coverage includes a diagonal
  matrix with closed-form real eigenvalues and a nonsymmetric Leto
  differential case; comparative benchmark coverage now measures a 32x32
  block-rotation matrix against Leto and `nalgebra` complex eigenvalues.
- [x] Strengthened WGPU general-eigenvalue contract coverage. Tests now cover
  exact 2x2 and 3x3 complex-pair blocks, triangular spectra, structured real
  nonsymmetric spectra, dense 5x5 `nalgebra` oracle comparison,
  symmetric-input all-real spectra, unordered spectrum matching, and
  rectangular rejection.
- [x] Added blocked WGPU Cholesky entry point with CPU panel factorization and
  triangular solve plus GPU SYRK trailing update. Differential coverage now
  includes a 66x66 SPD matrix crossing the 64-wide block boundary; comparative
  benchmarks now measure 128x128 blocked Cholesky against Leto and `nalgebra`.
- [x] Added comparative benchmark rows for blocked WGPU LU and QR paths,
  validating blocked LU solve parity and blocked QR factor parity before
  timing them against Leto and `nalgebra`.
- [x] Routed WGPU launch planning through Mnemosyne `KernelResourceBudget` and
  Moirai GPU `plan_launch` while preserving Hephaestus checked overflow
  semantics from `BlockWidth::checked_covering_blocks`.
- [x] Documented Atlas compute-boundary integration for Mnemosyne, Moirai,
  Themis, and Hermes in ADR 0002 and README. Hermes is integrated at the
  host-delegated Leto tier via `leto-ops`' `simd` feature; direct WGPU/CUDA
  kernel calls into Hermes are rejected as a boundary violation because Hermes
  owns CPU SIMD over host slices, not GPU shader/PTX execution.
- [x] Switched Hephaestus' `moirai-gpu` dependency to Moirai's planner-only
  feature set, closing the duplicate WGPU runtime dependency while preserving
  Mnemosyne `KernelResourceBudget` + Moirai `plan_launch` dispatch sizing.
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
- [x] Replaced WGPU CSR SpMV and SpMM host-delegated products with real WGSL
  kernels over device-resident CSR values, packed CSR index buffers, and
  device-resident dense operands/results. The `sparse` feature owns the Leto
  CSR upload/download boundary, while dispatch sizing flows through the shared
  Mnemosyne `KernelResourceBudget` and Moirai `plan_launch` helper.
- Evidence: `cargo fmt -p hephaestus-wgpu -p hephaestus-cuda --check`;
  `cargo clippy -p hephaestus-wgpu --all-targets -- -D warnings`; `cargo
  nextest run -p hephaestus-wgpu -j 1` (62 passed); `cargo nextest run -p
  hephaestus-cuda -j 1` (51 passed); `cargo test --doc -p hephaestus-wgpu` (0
  doctests); `cargo test --doc -p hephaestus-cuda` (0 doctests); `cargo doc -p
  hephaestus-wgpu --no-deps`; `cargo doc -p hephaestus-cuda --no-deps`; `cargo
  bench -p hephaestus-wgpu --bench comparative` (refreshed
  `benchmark_results.md`, including blocked 128x128 Cholesky, matrix rank,
  determinant, LU, full-pivot LU, QR, SVD, bidiagonalization, Schur,
  Hessenberg, Bunch-Kaufman, UDU, column-pivoted QR, pseudoinverse, matrix
  exponential, symmetric eigen, and general eigenvalues; CUDA rows skipped because the WGPU bench depends on
  `hephaestus-cuda` without its
  `cuda` feature in this environment); `git diff --check`. Full workspace
  all-features clippy
  attempted earlier and blocked before this slice by `cuda-bindings` requiring
  `CUDA_TOOLKIT_PATH`. Evidence tier: value-semantic differential tests,
  static diagnostics, and empirical benchmarks.
- Additional matrix-function evidence: `cargo nextest run -p hephaestus-wgpu
  linalg_pinv linalg_matexp -j 1 --no-fail-fast --no-capture` (8 passed);
  `cargo clippy -p hephaestus-wgpu --test contract -- -D warnings`;
  `rustfmt --edition 2021 --check crates/hephaestus-wgpu/tests/contract.rs`;
  `git diff --check`. Evidence tier: value-semantic closed-form,
  Moore-Penrose algebraic, differential, and invalid-input tests.
- Additional general-eigenvalue evidence: `cargo nextest run -p
  hephaestus-wgpu eigenvalues -j 1 --no-fail-fast --no-capture` (6 passed);
  `cargo clippy -p hephaestus-wgpu --test contract -- -D warnings`;
  `rustfmt --edition 2021 --check crates/hephaestus-wgpu/tests/contract.rs`;
  `git diff --check`. Evidence tier: closed-form, differential, and
  invalid-input value-semantic tests.
- Atlas compute-boundary evidence: `docs/adr/0002-atlas-compute-boundaries.md`,
  README layer-boundary update, `leto-ops` dependency audit confirming the
  `simd` feature and Hermes dispatch calls, and Hermes README/backlog boundary
  audit. Evidence tier: implementation and documentation audit.
- Additional blocked decomposition benchmark evidence: `cargo nextest run -p
  hephaestus-wgpu blocked_lu blocked_qr -j 1 --no-fail-fast` (8 passed);
  `cargo check -p hephaestus-wgpu --bench comparative`; `cargo bench -p
  hephaestus-wgpu --bench comparative` (added blocked LU/QR timing rows).
- Additional blocked decomposition synchronization evidence: `cargo check -p
  hephaestus-wgpu --bench decomposition_sync`; `cargo bench -p
  hephaestus-wgpu --bench decomposition_sync` (LU 66x66 sync floor 308.2 µs,
  QR 70x35 sync floor 227.8 µs).
- Additional blocked LU transfer evidence: narrowed blocked LU host/device
  transfers to the active diagonal-panel and trailing-submatrix regions.
  `cargo nextest run -p hephaestus-wgpu blocked_lu -j 1 --no-fail-fast` (4
  passed); `cargo clippy -p hephaestus-wgpu --all-targets -- -D warnings`;
  `cargo bench -p hephaestus-wgpu --bench comparative` (blocked LU 66x66
  239.8 µs, Leto 62.1 µs, nalgebra 7.0 µs; blocked QR 70x35 1.13 ms, Leto
  11.4 µs, nalgebra 6.1 µs); `cargo bench -p hephaestus-wgpu --bench
  decomposition_sync` (LU 66x66 sync floor 405.2 µs, QR 70x35 sync floor
  240.7 µs). Evidence tier: value-semantic blocked LU differential tests,
  static diagnostics, and empirical local benchmarks.
- Additional blocked QR transfer evidence: factored LU region transfers into
  a decomposition-local helper and changed blocked QR to upload/download a
  compact trailing-column tile per panel before applying Householder
  reflectors. `cargo fmt -p hephaestus-wgpu --check`; `cargo clippy -p
  hephaestus-wgpu --all-targets -- -D warnings`; `cargo nextest run -p
  hephaestus-wgpu blocked_lu blocked_qr -j 1 --no-fail-fast` (8 passed);
  `cargo bench -p hephaestus-wgpu --bench decomposition_sync` (LU 66x66 sync
  floor 220.7 µs, QR 70x35 sync floor 205.6 µs); `cargo bench -p
  hephaestus-wgpu --bench comparative` (blocked LU 66x66 260.5 µs, Leto 63.5
  µs, nalgebra 7.1 µs; blocked QR 70x35 1.04 ms, Leto 10.4 µs, nalgebra 6.0
  µs). Evidence tier: value-semantic blocked LU/QR differential tests, static
  diagnostics, and empirical local benchmarks.
- Additional blocked QR reflector-upload evidence: packed all panel
  Householder vectors into one device buffer and extended the reflector kernel
  metadata with a vector offset, removing per-reflector vector-buffer creation
  while preserving one GPU launch per reflector. `cargo fmt -p hephaestus-wgpu
  --check`; `cargo clippy -p hephaestus-wgpu --all-targets -- -D warnings`;
  `cargo nextest run -p hephaestus-wgpu blocked_lu blocked_qr -j 1
  --no-fail-fast` (8 passed); `cargo bench -p hephaestus-wgpu --bench
  decomposition_sync` (LU 66x66 sync floor 219.9 µs, QR 70x35 sync floor
  138.5 µs); `cargo bench -p hephaestus-wgpu --bench comparative` (blocked LU
  66x66 273.7 µs, Leto 66.2 µs, nalgebra 7.1 µs; blocked QR 70x35 1.05 ms,
  Leto 10.5 µs, nalgebra 6.1 µs). Evidence tier: value-semantic blocked
  LU/QR differential tests, static diagnostics, and empirical local
  benchmarks.
- Additional blocked QR timestamp evidence: extended `decomposition_sync` with
  a `wgpu::TIMESTAMP_QUERY` path that creates a timestamp-capable device when
  available, wraps 32 minimal compute passes with begin/end timestamps, and
  resolves them to host memory. `rustfmt --edition 2021 --check
  crates/hephaestus-wgpu/benches/decomposition_sync.rs`; `cargo clippy -p
  hephaestus-wgpu --all-targets -- -D warnings`; `cargo nextest run -p
  hephaestus-wgpu blocked_lu blocked_qr -j 1 --no-fail-fast` (8 passed);
  `cargo bench -p hephaestus-wgpu --bench decomposition_sync` (LU 66x66 sync
  floor 338.4 µs, QR 70x35 sync floor 207.6 µs, QR 32-reflector timestamp
  launch total 155.2 µs, median 3.4 µs). Evidence tier: value-semantic blocked
  LU/QR tests, static diagnostics, synthetic transfer benchmark, and
  GPU-timeline timestamp-query measurement.
- Additional blocked QR reflector-batching evidence: changed the WGPU
  Householder panel kernel so one compute pass owns a trailing column and
  applies all panel reflectors sequentially inside that workgroup, eliminating
  per-reflector compute-pass launches while preserving reflector order.
  `rustfmt --edition 2021 --check
  crates/hephaestus-wgpu/src/application/decomposition/qr.rs`; `cargo clippy
  -p hephaestus-wgpu --all-targets -- -D warnings`; `cargo nextest run -p
  hephaestus-wgpu blocked_qr -j 1 --no-fail-fast` (4 passed); `cargo bench -p
  hephaestus-wgpu --bench decomposition_sync` (QR timestamp total 8.4 µs,
  median 160 ns); `cargo bench -p hephaestus-wgpu --bench comparative`
  (blocked QR 70x35 420.8 µs, Leto 10.7 µs, nalgebra 6.1 µs). Evidence tier:
  value-semantic blocked QR tests, static diagnostics, empirical benchmark,
  and GPU-timeline timestamp-query measurement.
- Additional blocked QR component-profile evidence: extended
  `decomposition_sync` to measure the 70x35 CPU panel-factorization lower
  bound and the final Leto recompute paid by `qr_decompose_blocked`.
  `rustfmt --edition 2021 --check
  crates/hephaestus-wgpu/benches/decomposition_sync.rs`; `cargo check -p
  hephaestus-wgpu --bench decomposition_sync`; `cargo clippy -p
  hephaestus-wgpu --bench decomposition_sync -- -D warnings`; `cargo bench -p
  hephaestus-wgpu --bench decomposition_sync` (LU sync floor 359.6 µs, QR sync
  floor 222.6 µs, QR CPU panel lower bound 26.3 µs, QR final Leto recompute
  11.5 µs, QR timestamp total 8.2 µs, median 160 ns). Evidence tier: static
  diagnostics, empirical component benchmark, and GPU-timeline timestamp-query
  measurement.
- Additional blocked QR metadata-transfer evidence: packed the panel
  Householder vector offsets and beta coefficients into one
  `HhReflectorMeta` storage buffer, replacing two per-panel metadata uploads
  and bindings with one while preserving reflector order and arithmetic.
  `rustfmt --edition 2021 --check
  crates/hephaestus-wgpu/src/application/decomposition/qr.rs`; `cargo check
  -p hephaestus-wgpu`; `cargo clippy -p hephaestus-wgpu --all-targets --
  -D warnings`; `cargo nextest run -p hephaestus-wgpu blocked_qr -j 1
  --no-fail-fast` (4 passed); `cargo bench -p hephaestus-wgpu --bench
  decomposition_sync` (QR sync floor 219.9 µs, CPU panel lower bound 25.3 µs,
  final Leto recompute 12.9 µs, timestamp total 8.3 µs, median 192 ns);
  `cargo bench -p hephaestus-wgpu --bench comparative` (blocked QR 70x35
  480.8 µs, Leto 14.9 µs, nalgebra 10.0 µs). Evidence tier:
  value-semantic blocked QR tests, static diagnostics, empirical benchmark,
  and GPU-timeline timestamp-query measurement.

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
- Additional WGPU sparse evidence: `cargo check -p hephaestus-wgpu`; `cargo
  clippy -p hephaestus-wgpu --lib -- -D warnings`; `rustfmt --edition 2021
  --check` over the WGPU sparse/module/export files; `cargo nextest run -p
  hephaestus-wgpu sparse -j 1 --no-fail-fast` (1 passed);
  `cargo check -p hephaestus-wgpu --bench sparse_comparative`;
  `cargo bench -p hephaestus-wgpu --bench sparse_comparative` (SpMV
  1000x1000 CSR: WGPU 122.216 µs, Leto 1.274 µs; SpMM 1000x1000x128:
  WGPU 50.046 µs, Leto 38.346 µs). Evidence tier: static diagnostics,
  value-semantic GPU sparse contract test, value-checked benchmark outputs,
  and empirical local benchmark.
- Evidence: `cargo fmt -p hephaestus-wgpu -p hephaestus-cuda --check`; `cargo
  clippy -p hephaestus-wgpu --all-targets -- -D warnings` (compiles
  `hephaestus-cuda` as the WGPU dev dependency); `cargo nextest run -p
  hephaestus-cuda -j 1` (51 passed); `cargo test --doc -p hephaestus-cuda` (0
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
