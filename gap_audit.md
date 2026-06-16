# Gap Audit - hephaestus

## Residual Risks

- [minor] WGPU/Leto parity is complete for the current core array-operation
  slice: elementwise, strided elementwise, scalar elementwise, reductions,
  rank-2 axis reductions, rank-2 scans, matrix products, Kronecker product,
  matrix power, finite-`f32` matrix rank, finite-`f32` determinant, dot, trace,
  norms, Cholesky/LU/QR decomposition APIs, and symmetric Jacobi eigen
  decomposition/eigenvalue APIs. Evidence tier: value-semantic contract tests
  against CPU references and Leto, plus comparative benchmark evidence
  recorded in `benchmark_results.md`.
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
- [minor] WGPU blocked Cholesky now offloads the trailing SYRK update to a GPU
  kernel, but diagonal panel factorization and triangular panel solves remain
  CPU/Leto-backed. Current empirical row: 128x128 blocked Cholesky is slower
  than Leto and `nalgebra` on the local WGPU run. Evidence tier:
  value-semantic differential test across a block boundary and empirical
  benchmark row in `benchmark_results.md`.
- [patch] Hephaestus WGPU launch planning now uses Mnemosyne
  `KernelResourceBudget` and Moirai GPU `plan_launch`, but `moirai-gpu`
  currently brings `wgpu 0.19` into the graph while Hephaestus uses `wgpu 26`.
  This is an integration/build-size risk; resolve by aligning Moirai GPU to
  `wgpu 26` or splitting the occupancy planner into a GPU-API-free crate.
  Evidence tier: build output from `cargo bench -p hephaestus-wgpu --bench
  comparative` showing both `wgpu v0.19.4` and `wgpu v26.0.1`.
- [minor] Hermes SIMD is used by Leto CPU ops through `leto-ops`, but
  Hephaestus WGPU does not yet directly consume Hermes in a device-side kernel
  path. Evidence tier: implementation audit.
- [minor] Additional WGPU dense decomposition and matrix-function wrappers are
  present in the source tree for SVD, Schur, bidiagonalization,
  Bunch-Kaufman, full-pivot LU, Hessenberg, UDU, pseudoinverse, and matrix
  exponential, but they do not yet have the same value-semantic contract and
  comparative benchmark coverage as the completed core/symmetric-eigen slice.
  General nonsymmetric complex eigenvalues remain unexported in WGPU until a
  stable device-buffer representation and tests are completed. Evidence tier:
  source/API audit and current test/benchmark coverage audit.
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

- Implement value-semantic WGPU/Leto contract tests and comparative benchmark
  rows for the next already-present dense wrapper family (SVD/Schur/matrix
  functions), or remove any wrapper that cannot be verified without a real
  implementation contract.
