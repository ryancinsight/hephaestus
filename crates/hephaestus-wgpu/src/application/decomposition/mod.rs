//! Dense matrix decompositions for the wgpu backend.
//!
//! Provides Cholesky, LU, and QR decompositions that establish API parity
//! with [`leto_ops`] linear algebra. Panel factorization delegates to the
//! CPU (via leto-ops) while the result is stored on the device for
//! downstream GPU consumers.
//!
//! # Mathematical Foundations
//!
//! ## Theorem 1 — Cholesky Factorization
//!
//! Every symmetric positive-definite matrix **A** ∈ ℝⁿˣⁿ admits a unique
//! factorization **A** = **L** **L**ᵀ where **L** is lower-triangular with
//! strictly positive diagonal entries.
//!
//! **Proof (constructive, by induction on *n*).**
//!
//! *Base case* (*n* = 1): **A** contains scalar `a_11` with `a_11 > 0`,
//! so **L** contains scalar `sqrt(a_11)`.
//!
//! *Inductive step.* Partition
//!
//! ```text
//!     ┌         ┐   ┌             ┐ ┌             ┐
//! A = │ a₁₁  bᵀ │ = │ √a₁₁    0  │ │ √a₁₁   l₂₁ᵀ│
//!     │ b    B   │   │ l₂₁    L₂₂ │ │  0      L₂₂ᵀ│
//!     └         ┘   └             ┘ └             ┘
//! ```
//!
//! where l₂₁ = b / √a₁₁ and **Ã** = **B** − l₂₁ l₂₁ᵀ is SPD.
//! By the inductive hypothesis **Ã** = **L₂₂ L₂₂**ᵀ.
//! Uniqueness: the positive-diagonal requirement forces each diagonal
//! entry, then forward-substitution determines the strictly lower part. ∎
//!
//! ## Theorem 2 — LU Factorization with Partial Pivoting
//!
//! Every nonsingular **A** ∈ ℝⁿˣⁿ admits **P A** = **L U** where **P** is
//! a permutation matrix, **L** is unit lower-triangular, and **U** is
//! upper-triangular.
//!
//! **Proof.** Gaussian elimination with partial pivoting selects, at each
//! step *k*, the row *r* ≥ *k* with the largest |aᵣₖ| and swaps rows *k*
//! and *r*. The elimination multiplier ℓᵢₖ = aᵢₖ/aₖₖ is well-defined
//! because a zero pivot would imply a zero sub-determinant, contradicting
//! nonsingularity. Each step applies an elementary lower-triangular matrix
//! **Mₖ** and a permutation **Pₖ**; the product **Mₙ₋₁ Pₙ₋₁ ⋯ M₁ P₁ A**
//! is upper-triangular. Re-ordering the permutations yields **P A = L U**. ∎
//!
//! ## Theorem 3 — QR Factorization via Householder Reflectors
//!
//! Every **A** ∈ ℝᵐˣⁿ with *m* ≥ *n* factors as **A** = **Q R** where
//! **Q** ∈ ℝᵐˣᵐ is orthogonal and **R** ∈ ℝᵐˣⁿ is upper-triangular.
//!
//! **Proof.** At step *k*, the Householder reflector **Hₖ** = **I** − βₖ
//! **vₖ vₖ**ᵀ zeros the entries below the diagonal of column *k*.
//! **Hₖ** is orthogonal (**Hₖ**ᵀ = **Hₖ** and **Hₖ²** = **I**).
//! After *n* steps, **Hₙ ⋯ H₁ A** = **R** is upper-triangular, so
//! **Q** = **H₁ᵀ ⋯ Hₙᵀ** = **H₁ ⋯ Hₙ** (each reflector is symmetric). ∎
//!
//! # Complexity
//!
//! | Decomposition | Flop count | Dominant kernel |
//! |---|---|---|
//! | Cholesky | n³ / 3 | rank-k update (SYRK) |
//!
//! # Block Algorithms
//!
//! For large matrices, the O(n³) trailing-matrix update dominates.  The
//! blocked variant [`crate::application::decomposition::cholesky_decompose_blocked`] processes the
//! matrix in `BLOCK_SIZE × BLOCK_SIZE` panels:
//!
//! 1. **Panel factorisation** (CPU, O(b³/3)) — the diagonal block is
//!    factored by leto-ops.
//! 2. **Panel solve** (CPU, O(b²(n−k)/2)) — the off-diagonal panel
//!    L₂₁ = A₂₁ L₁₁⁻ᵀ is computed by triangular solve.
//! 3. **Trailing SYRK** (GPU, O(b(n−k)²/2)) — a dedicated WGSL kernel
//!    computes A₂₂ -= L₂₁ L₂₁ᵀ directly in device memory.
//!
//! The SYRK kernel uses 16×16 workgroup tiles with shared-memory
//! cooperative loading of panel rows, analogous to the tiled matmul
//! kernel but specialised for the symmetric rank-k update.  Only the
//! lower triangle of the trailing matrix is touched, halving the
//! compute compared to a general matmul + subtract sequence.
//! | LU | 2n³ / 3 | panel elimination + trailing update |
//! | QR | 2n²(m − n/3) | Householder application |

pub(crate) mod validate;

pub mod bidiagonal;
pub mod bunch_kaufman;
pub mod cholesky;
pub mod col_piv_qr;
pub mod eigen;
pub mod full_piv_lu;
pub mod hessenberg;
pub mod lu;
pub mod qr;
pub mod schur;
pub mod svd;
pub mod udu;

pub use bidiagonal::{bidiagonalize, GpuBidiagonalDecomposition};
pub use bunch_kaufman::{bunch_kaufman, GpuBunchKaufmanDecomposition};
pub use cholesky::{cholesky_decompose, cholesky_decompose_blocked, GpuCholesky};
pub use col_piv_qr::{col_piv_qr, GpuColPivQrDecomposition};
pub use eigen::{
    eigenvalues, symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi,
    GpuSymmetricEigenDecomposition,
};
pub use full_piv_lu::{full_piv_lu, GpuFullPivLuDecomposition};
pub use hessenberg::{hessenberg, GpuHessenbergDecomposition};
pub use lu::{lu_decompose, lu_decompose_blocked, GpuLuDecomposition};
pub use qr::{qr_decompose, qr_decompose_blocked, GpuQrDecomposition};
pub use schur::{schur, GpuRealSchur};
pub use svd::{singular_values, svd_decompose, svd_rank_revealing, GpuSvdDecomposition};
pub use udu::{udu_decompose, GpuUduDecomposition};
