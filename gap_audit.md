# Gap Audit - hephaestus

## Residual Risks

- [minor] WGPU/Leto parity is complete for the current core array-operation
  slice: elementwise, strided elementwise, scalar elementwise, reductions,
  rank-2 axis reductions, rank-2 scans, matrix products, Kronecker product,
  matrix power, finite-`f32` matrix rank, finite-`f32` determinant, dot, trace,
  norms, and Cholesky/LU/QR decomposition APIs. Evidence tier: value-semantic
  contract tests against CPU references and Leto, plus comparative benchmark
  evidence recorded in `benchmark_results.md`.
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
- [minor] Leto's dense decomposition and matrix-property surface is not yet
  mirrored by WGPU: SVD, Schur, eigenvalue/eigenvector,
  pseudoinverse, and matrix exponential. Evidence tier: API
  audit against `leto-ops/src/application/linalg`.
- [minor] CUDA mirrors the current core operation slice in the source tree and
  passes stub-mode verification. Real CUDA feature verification is still
  required on CUDA hardware/toolchain before claiming device-execution parity
  for the CUDA kernels. Evidence tier: static diagnostics and stub-mode
  contract tests.
- [patch] Full workspace all-features clippy remains blocked by
  `cuda-bindings` requiring `CUDA_TOOLKIT_PATH`; WGPU package-local gates are
  clean. Evidence tier: static diagnostics from the earlier attempted gate.

## Next Increment

- Implement the next WGPU/Leto linalg parity slice from the dense algorithms
  above only after selecting a single operation family with value-semantic
  differential tests and comparative benchmark coverage against Leto,
  `ndarray`, and `nalgebra` where those libraries expose the operation.
