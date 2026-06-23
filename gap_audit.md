# Gap Audit - hephaestus

## Residual Risks

- [minor] WGPU/Leto parity is complete for the current core array-operation
  slice: elementwise, strided elementwise, scalar elementwise, reductions,
  rank-2 axis reductions, rank-2 scans, matrix products, Kronecker product,
  matrix power, finite-`f32` matrix rank, finite-`f32` determinant, dot, trace,
  norms, Cholesky/LU/full-pivot LU/QR/column-pivoted-QR/SVD/bidiagonalization/Schur/Hessenberg/Bunch-Kaufman/UDU decomposition APIs,
  pseudoinverse and matrix exponential baseline wrappers, symmetric Jacobi eigen decomposition/eigenvalue APIs, and general eigenvalues for diagonal
  closed-form and nonsymmetric Leto-differential cases. Evidence tier:
  value-semantic contract tests against CPU references and Leto, plus
  comparative benchmark evidence recorded in `benchmark_results.md`.
- [minor] WGPU `matrix_rank` uses GPU row reduction with a relative pivot
  threshold; Leto `matrix_rank` uses singular values. Exact finite full-rank,
  rank-deficient, and zero cases are covered, but ill-conditioned matrices may
  diverge between the algorithms. Evidence tier: documented algorithm audit and
  value-semantic contract tests.
- [minor] WGPU `det` uses exact GPU row-reduction pivots with no determinant
  tolerance; Leto uses its CPU determinant algorithm. Exact finite nonsingular
  and singular cases are covered, but ill-conditioned determinants may diverge.
  Evidence tier: documented algorithm audit and value-semantic contract tests.
- [minor] WGPU Cholesky/LU/QR currently provide device-resident factors and
  Leto-matching solve/inverse/determinant surfaces, but factorization delegates
  to Leto on the host before uploading the factors. This is API parity, not
  GPU-kernel parity. Evidence tier: implementation audit, value-semantic
  differential tests, and comparative benchmark rows.
- [minor] WGPU symmetric Jacobi eigen decomposition currently provides
  device-resident eigenvalues/eigenvectors, but the eigensolve delegates to
  Leto on the host before uploading the outputs. This is API parity, not
  GPU-kernel eigensolver parity. Evidence tier: value-semantic differential
  tests, non-symmetric rejection test, and comparative benchmark row.
- [minor] WGPU general eigenvalues are exported with complex device buffers and
  covered for diagonal, exact complex-pair blocks, triangular, structured
  nonsymmetric real-spectrum, dense `nalgebra` differential, symmetric-real,
  and rectangular-rejection cases. Comparative benchmark coverage includes a
  32x32 block-rotation matrix against Leto and `nalgebra`. Remaining risk is
  API/performance parity only: the wrapper delegates to Leto on the host before
  uploading complex device buffers. Evidence tier: value-semantic closed-form,
  differential, invalid-input tests, and empirical benchmark row.
- [minor] WGPU blocked Cholesky now offloads the trailing SYRK update to a GPU
  kernel, but diagonal panel factorization and triangular panel solves remain
  CPU/Leto-backed. Current empirical row: 128x128 blocked Cholesky is slower
  than Leto and `nalgebra` on the local WGPU run. Evidence tier:
  value-semantic differential test across a block boundary and empirical
  benchmark row in `benchmark_results.md`.
- [minor] WGPU blocked LU and blocked QR now have comparative benchmark rows.
  Blocked LU transfers are narrowed to the active diagonal-panel and
  trailing-submatrix regions. Blocked QR transfers compact trailing-column
  tiles per panel before GPU Householder application and uploads all panel
  Householder vectors in one packed buffer. The measured 66x66 blocked LU row
  remains slower than Leto and `nalgebra`; the 70x35 blocked QR row is much
  slower than Leto and `nalgebra`. Evidence tier: value-semantic blocked LU/QR
  tests plus empirical benchmark rows in `benchmark_results.md`.
- [minor] Blocked decomposition synchronization profiling shows a material,
  noisy transfer/synchronization floor after the blocked LU region-transfer
  reduction, blocked QR compact-tile transfer reduction, and packed reflector
  upload. Timestamp queries now measure the QR launch component directly:
  32 separate reflector-equivalent compute passes previously totaled 155.2 µs
  on the local GPU timeline, with 3.4 µs median pass duration. The WGPU QR
  panel path now applies all panel reflectors in one compute pass per panel;
  the corresponding timestamp profile is 8.2 µs total with 160 ns median pass
  duration, and the 70x35 blocked QR row measures 420.8 µs. The component
  profile measures the 70x35 CPU panel-factorization lower bound at 26.3 µs
  and the duplicated final Leto recompute at 11.5 µs, while the synthetic QR
  host/device synchronization floor remains 222.6 µs. The WGPU QR
  trailing-update kernel now packs Householder vector offsets and beta
  coefficients into one reflector metadata buffer, reducing per-panel metadata
  uploads and storage bindings from two to one. The 70x35 comparative row did
  not improve on the local run after this packing change: WGPU measured
  480.8 µs vs Leto 14.9 µs and `nalgebra` 10.0 µs. Remaining risk: blocked QR
  still trails Leto and `nalgebra`; the next measured lever is reducing the
  host/device synchronization count, not only metadata buffer count or CPU
  panel arithmetic. Evidence tier: value-semantic blocked QR tests, empirical
  synchronization/component-profile benchmarks, comparative benchmark, and
  GPU-timeline timestamp measurement in `benchmark_results.md`.
- [patch] Hephaestus WGPU launch planning uses Mnemosyne
  `KernelResourceBudget` and Moirai GPU `plan_launch` through Moirai's
  planner-only feature set. The prior duplicate-WGPU risk is closed:
  Hephaestus now depends on `moirai-gpu` with default features disabled, so
  `moirai-gpu` no longer pulls `wgpu 0.19` into the Hephaestus graph.
  Evidence tier: dependency-tree verification and package checks.
- [minor] Hermes integration is intentionally host-tier for Hephaestus:
  host-delegated parity wrappers call `leto-ops` with `simd` enabled, and Leto
  routes CPU hot loops through Hermes SIMD before Hephaestus uploads verified
  outputs into device buffers. Direct WGPU/CUDA kernel calls into Hermes are
  out of scope because Hermes owns CPU SIMD over host slices while Hephaestus
  owns GPU resource lifetimes and device-resident kernels. Evidence tier:
  dependency/implementation audit and ADR 0002.
- [minor] WGPU pseudoinverse and matrix exponential now have non-diagonal,
  rank-deficient, rectangular, nilpotent, skew-symmetric, general-matrix, and
  invalid-input contract coverage plus comparative benchmark rows. Remaining
  risk is performance/API parity only: both wrappers still delegate to Leto on
  the host and upload device buffers. Evidence tier: value-semantic
  closed-form, Moore-Penrose algebraic, differential, invalid-input tests, and
  empirical benchmark rows.
- [minor] WGPU CSR sparse storage now uploads Leto CSR matrices into
  device-resident values plus one packed index buffer and executes SpMV/SpMM
  in WGSL without downloading operands to the host. The kernel layout stays
  within WGPU's portable four-storage-buffer limit, and dispatch sizing reuses
  the shared Mnemosyne/Moirai launch-planning helper. The focused sparse
  comparative harness validates WGPU outputs against Leto before timing:
  SpMV 1000x1000 CSR measured WGPU 100.888 µs vs Leto 1.280 µs; SpMM
  1000x1000x128 measured WGPU 73.634 µs vs Leto 32.470 µs. Remaining risk:
  sparse performance parity is not achieved for either SpMV or SpMM on this
  run; no `ndarray`/`nalgebra` sparse comparator is recorded because the
  current Leto sparse API benchmark has no dense-library sparse equivalent in
  this harness. Evidence tier: static diagnostics, value-semantic WGPU sparse
  contract test, value-checked benchmark outputs, and empirical benchmark.
- [patch] Python binding tests previously hung after the RNG binding test body
  completed because transient WGPU staging/uniform buffers could remain
  retained across short-lived host-runtime teardown. `PyDevice` now drains the
  bounded WGPU transient pools on drop. Evidence tier: root-cause diagnostic
  run showing the test body completed before process hang, Python package
  nextest, and full workspace nextest.
- [minor] CUDA mirrors the current core operation and decomposition slice in the
  source tree and passes stub-mode verification. Real CUDA feature verification
  is still required on CUDA hardware/toolchain before claiming device-execution
  parity for the CUDA kernels. CUDA blocked Cholesky remains CUDA-feature gated
  and is not part of the default stub-mode claim. Evidence tier: static
  diagnostics and stub-mode contract tests.
- [patch] Full workspace all-features clippy remains blocked by
  `cuda-bindings` requiring `CUDA_TOOLKIT_PATH`; WGPU package-local gates are
  clean. Evidence tier: static diagnostics from the earlier attempted gate.
- [patch] WGPU staging-allocator registry (`WGPU_MAPPED_BUFFERS`) contention
  audit (2026-06-23). The HostPinned alloc/upload paths resolve a Mnemosyne
  sub-allocated pointer to its containing wgpu mapped block. This previously held
  the global registry lock across an `O(n)` `.iter().find()` linear scan over
  every live mapping (`O(n²)` over a staging-heavy workload). Replaced the
  `HashMap` with a base-address-keyed `BTreeMap` and consolidated both sites into
  one `resolve_mapped_buffer` helper doing an `O(log n)`
  `range(..=ptr).next_back()` containment query, shortening the lock-held section.
  Remaining bounded characteristic (not a defect): a single global `Mutex` still
  serializes the registry across threads; the lookup is now `O(log n)` and the
  `wgpu::Buffer` return is an `Arc` handle clone, so the critical section is
  minimal. The deeper lever, if a future staging-heavy multi-thread workload
  measures the global lock as hot, is a sharded registry keyed by address range —
  deferred until a workload demonstrates the need (no current consumer drives
  concurrent HostPinned staging). The registry and its descriptor are now
  `pub(crate)` (no external consumers) and the dead `usage` field is removed.
  Evidence tier: value-semantic placement/transfer contract tests + full
  workspace gate.

## Next Increment

- Continue the parity audit at the next highest-risk residual: reduce blocked
  QR host/device transfer and synchronization costs before replacing CPU panel
  arithmetic with native GPU kernels.
