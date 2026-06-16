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
  32 minimal reflector-equivalent compute passes total 155.2 µs on the local
  GPU timeline, with 3.4 µs median pass duration. The 70x35 blocked QR row
  still combines that launch cost with real reflector kernel work. Next target:
  assess whether reflector batching can remove per-reflector launch traffic.
  Evidence tier: empirical synchronization-profile benchmark and GPU-timeline
  timestamp measurement in `benchmark_results.md`.
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
- [minor] CUDA mirrors the current core operation and decomposition slice in the
  source tree and passes stub-mode verification. Real CUDA feature verification
  is still required on CUDA hardware/toolchain before claiming device-execution
  parity for the CUDA kernels. CUDA blocked Cholesky remains CUDA-feature gated
  and is not part of the default stub-mode claim. Evidence tier: static
  diagnostics and stub-mode contract tests.
- [patch] Full workspace all-features clippy remains blocked by
  `cuda-bindings` requiring `CUDA_TOOLKIT_PATH`; WGPU package-local gates are
  clean. Evidence tier: static diagnostics from the earlier attempted gate.

## Next Increment

- Continue the parity audit at the next highest-risk residual: profile the
  blocked QR reflector-batching design now that timestamp instrumentation
  confirms per-reflector launch cost is material before adding more native
  decomposition kernels.
