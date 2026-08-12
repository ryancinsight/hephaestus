# Decomposition Seam

Hephaestus provides a rich set of dense matrix decompositions through the
`DecompositionOps` and `BlockedDecompositionBackend` traits.

## Decomposition Handles

Each factorization returns a typed handle containing the computed factors:

| Handle | Decomposition |
|--------|--------------|
| `LuHandle` | LU factorization (partial pivoting) |
| `FullPivLuHandle` | LU with full pivoting |
| `CholeskyHandle` | Cholesky factorization (positive-definite) |
| `QrHandle` | QR factorization |
| `ColPivQrHandle` | QR with column pivoting |
| `SvdHandle` | Singular value decomposition |
| `SchurHandle` | Schur decomposition |
| `SymmetricEigenHandle` | Symmetric eigenvalue decomposition |
| `HessenbergHandle` | Hessenberg reduction |
| `BidiagonalHandle` | Bidiagonalization |
| `UduHandle` | UDU factorization (for Kalman filtering) |
| `BunchKaufmanHandle` | Bunch-Kaufman factorization (indefinite matrices) |

## Plan-Then-Dispatch

```rust,ignore
// Factor once
let lu = device.lu_factorize(&matrix_buf, shape)?;

// Solve multiple right-hand sides with the same factorization
for rhs in &right_hand_sides {
    device.lu_solve(&lu, rhs, &mut solution)?;
}
```

## Blocked Panel Algorithms

`BlockedDecompositionBackend` provides blocked panel-based implementations
(`blocked_lu`, `factor_cholesky_panel`, `panel_qr_packed`) that cache
intermediate panels in Mnemosyne scratch pools to maximize cache utilization.

## Consumers

`coeus-tensor` uses decompositions for gradient computations (SVD backward,
Cholesky gradient). `CFDrs` uses LU/Cholesky for implicit timestepping.
