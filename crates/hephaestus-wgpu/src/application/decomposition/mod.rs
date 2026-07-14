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
//! | Decomposition | Flop count | Dominant GPU kernel |
//! |---|---|---|
//! | Cholesky | n³ / 3 | rank-k SYRK update |
//! | LU | 2n³ / 3 | trailing GEMM (C -= L₂₁ U₁₂) |
//! | QR | 2n²(m − n/3) | Householder column application |
//!
//! # Blocked Cholesky — SYRK Trailing Update
//!
//! [`crate::application::decomposition::cholesky::cholesky_decompose_blocked`] processes the matrix in
//! `BLOCK_SIZE × BLOCK_SIZE` panels:
//!
//! 1. **Panel factorisation** (CPU, O(b³/3)) — the diagonal block is
//!    factored by leto-ops.
//! 2. **Panel solve** (CPU, O(b²(n−k)/2)) — the off-diagonal panel
//!    L₂₁ = A₂₁ L₁₁⁻ᵀ is computed by triangular solve.
//! 3. **Trailing SYRK** (GPU, O(b(n−k)²/2)) — a dedicated WGSL kernel
//!    computes A₂₂ -= L₂₁ L₂₁ᵀ directly in device memory.
//!
//! The SYRK kernel uses 16×16 workgroup tiles with shared-memory
//! cooperative loading of panel rows.  Only the lower triangle of the
//! trailing matrix is touched, halving the compute compared to a
//! general matmul.
//!
//! # Blocked LU — Trailing GEMM Update
//!
//! **Theorem (Blocked LU complexity).** For *n × n* with block size *b*,
//! the total flop count is 2n³/3, identical to unblocked LU.
//!
//! **Proof.** Partition **P A = L U** into *b × b* blocks:
//!
//! ```text
//! ┌           ┐   ┌       ┐ ┌       ┐
//! │ A₁₁  A₁₂ │   │ L₁₁ 0 │ │ U₁₁ U₁₂│
//! │ A₂₁  A₂₂ │ = │ L₂₁ I │ │  0  S₂₂│
//! └           ┘   └       ┘ └       ┘
//! ```
//!
//! The Schur complement is **S₂₂ = A₂₂ − L₂₁ U₁₂** and the dominant
//! cost is the rank-b GEMM update.  Each block iteration costs:
//!
//! - Panel factor: 2b³/3
//! - Panel solve (L₂₁): b²(n−k)/2
//! - Panel solve (U₁₂): b²(n−k)/2
//! - **Trailing GEMM: 2b(n−k)²** (GPU)
//!
//! Summing over all ⌈n/b⌉ blocks recovers 2n³/3 total flops.  The
//! key performance gain is that the trailing GEMM, which dominates
//! for large *n*, executes on the GPU's massively parallel compute
//! units. ∎
//!
//! The [`crate::application::decomposition::lu::lu_decompose_blocked`] entry point implements this: the panel
//! factorisation uses the same partial-pivoting rule as leto-ops
//! [`panel_lu_packed`](hephaestus_core::panel_lu_packed), and the
//! trailing GEMM runs via a dedicated 16×16 tiled WGSL kernel with
//! shared-memory cooperative loading.
//!
//! # Blocked QR — Trailing Householder Application
//!
//! **Theorem (Blocked QR complexity).** For *m × n* with block size *b*,
//! the total flop count is 2n²(m − n/3), identical to unblocked QR.
//!
//! **Proof.** At step *k*, the Householder reflector **Hₖ** = **I** −
//! βₖ **vₖ vₖ**ᵀ zeros entries below the diagonal of column *k*.
//! Each block of *b* columns produces *b* reflectors that are
//! applied to the trailing *m × (n−k−b)* submatrix.  Each
//! application costs O((m−k)(n−k)) flops — *b* applications per
//! panel gives O(b·(m−k)·(n−k)) — and is embarrassingly parallel
//! across columns.  Each block iteration costs:
//!
//! - Panel factor: 2b²(m−k) − 2b³/3
//! - **Trailing apply: 2b(m−k)(n−k−b)** (GPU, *b* kernel launches)
//!
//! Summing over all ⌈n/b⌉ blocks recovers 2n²(m − n/3) total
//! flops. ∎
//!
//! The [`crate::application::decomposition::qr::qr_decompose_blocked`] entry point implements this: the panel
//! factorisation uses the same Householder convention as leto-ops
//! [`panel_qr_packed`](hephaestus_core::panel_qr_packed), and each
//! of the *b* reflectors is applied to the trailing columns via a
//! dedicated 256-thread workgroup kernel that computes
//! `A[:, col] -= β · v · (vᵀ · A[:, col])` using a parallel
//! tree reduction for the dot product.

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
pub(crate) mod region;
pub mod schur;
pub mod svd;
pub mod udu;

pub use bidiagonal::{GpuBidiagonalDecomposition, bidiagonalize};
pub use bunch_kaufman::{GpuBunchKaufmanDecomposition, bunch_kaufman};
pub use cholesky::{GpuCholesky, cholesky_decompose, cholesky_decompose_blocked};
pub use col_piv_qr::{GpuColPivQrDecomposition, col_piv_qr};
pub use eigen::{
    GpuSymmetricEigenDecomposition, eigenvalues, symmetric_eigen_jacobi,
    symmetric_eigenvalues_jacobi,
};
pub use full_piv_lu::{GpuFullPivLuDecomposition, full_piv_lu};
pub use hessenberg::{GpuHessenbergDecomposition, hessenberg};
pub use lu::{GpuLuDecomposition, lu_decompose, lu_decompose_blocked};
pub use qr::{GpuQrDecomposition, qr_decompose, qr_decompose_blocked};
pub use schur::{GpuRealSchur, schur};
pub use svd::{GpuSvdDecomposition, singular_values, svd_decompose, svd_rank_revealing};
pub use udu::{GpuUduDecomposition, udu_decompose};
