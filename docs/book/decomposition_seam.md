# Decomposition Seam

`DecompositionOps<D>` is the device-neutral factorization seam. The current
core contract uses rank-2 `f32` operands and returns typed associated handles;
it does not expose a boxed, scalar-suffixed, or backend-specific result type.

The factorization and spectral entry points are:

- `lu` for partial-pivoted `P·A = L·U`;
- `qr` for `A = Q·R` with `m >= n`;
- `col_piv_qr` for rank-revealing column-pivoted QR; and
- `full_piv_lu` for rank-revealing fully pivoted LU;
- `cholesky` for `A = L·Lᵀ` on symmetric positive-definite input;
- `schur`, `hessenberg`, and `bidiagonalize` for orthogonal reductions;
- `bunch_kaufman` and `udu` for symmetric factorizations; and
- `svd`, `singular_values`, `eigenvalues`, `symmetric_eigen`, and
  `symmetric_eigenvalues` for singular-value and eigenvalue results.

The returned handle traits expose only the factor, shape, pivot, determinant,
rank, and solve capabilities required by consumers and conformance tests.
Each handle documents its own failure contract: LU rejects singular factors,
QR rejects rank-deficient least-squares solves, and handle methods report
length or dispatch failures where applicable. The seam does not claim that
every factorization has the same singularity rule.

The core also owns the `BlockedDecompositionBackend` seam and the
`blocked_lu` orchestration. Backend implementations supply the panel and
trailing-update operations; the host-side blocked algorithm is not copied once
per device.

Consumer crates bind to these traits and keep their own domain semantics. A
consumer must not reconstruct factorization state from raw buffers or teach the
book an API that is absent from `hephaestus-core`.
