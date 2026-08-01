//! Device-neutral dense decomposition seam (ADR 0042).
//!
//! One trait covers the core factorization trio — LU with partial pivoting,
//! Householder QR, and Cholesky — over rank-2 strided `f32` operands. The
//! scalar type is fixed at `f32` by every backend's decomposition kernels;
//! the scalar dimension enters here when the kernels ship it, not before.
//!
//! Result handles are exposed through oracle-minimal accessor traits: exactly
//! the capability consumers and the conformance oracles need (factor access,
//! pivots, determinants, solves). Backend-specific extras stay inherent on
//! the concrete handle types. The associated result types carry an `'op`
//! lifetime like the prepared forms, so a backend whose handle borrows
//! operand or workspace state implements without erasing that borrow;
//! today's handle-owning backends simply ignore it.

use crate::domain::device::ComputeDevice;
use crate::domain::error::Result;
use crate::domain::view::StridedView;

/// Access to an LU factorization `P·A = L·U` of a square matrix.
pub trait LuHandle<D: ComputeDevice> {
    /// Matrix order `n`.
    fn order(&self) -> usize;
    /// Device-resident packed `L\U` factors, `n × n` row-major (unit
    /// diagonal of `L` implicit).
    fn factors(&self) -> &D::Buffer<f32>;
    /// Row permutation as a permutation vector: row `k` of `P·A` is row
    /// `pivots()[k]` of `A` (leto's pivot convention, shared by every
    /// backend's factorization machinery).
    fn pivots(&self) -> &[usize];
    /// Determinant of `A`, including the permutation sign.
    fn det(&self) -> f32;
    /// Solve `A·x = rhs` for one right-hand side.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch, a singular factor, or the backend
    /// dispatch failure.
    fn solve(&self, device: &D, rhs: &D::Buffer<f32>) -> Result<D::Buffer<f32>>;
}

/// Access to a Householder QR factorization `A = Q·R` with `m ≥ n`.
pub trait QrHandle<D: ComputeDevice> {
    /// Input shape `(m, n)`.
    fn shape(&self) -> (usize, usize);
    /// Device-resident upper-triangular `R`, `m × n` row-major (leto's
    /// convention — rows beyond `n` vanish). The Householder sign choice is
    /// the backend's own; only entry magnitudes and the triangular structure
    /// are contract.
    fn r_buffer(&self) -> &D::Buffer<f32>;
    /// Solve the least-squares problem `min ‖A·x − rhs‖₂`.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch, a rank-deficient factor, or the backend
    /// dispatch failure.
    fn solve_least_squares(&self, device: &D, rhs: &D::Buffer<f32>) -> Result<D::Buffer<f32>>;
}

/// Access to a Cholesky factorization `A = L·Lᵀ` of an SPD matrix.
pub trait CholeskyHandle<D: ComputeDevice> {
    /// Matrix order `n`.
    fn order(&self) -> usize;
    /// Device-resident lower-triangular factor `L`, `n × n` row-major.
    fn lower(&self) -> &D::Buffer<f32>;
    /// Determinant of `A`.
    fn det(&self) -> f32;
    /// Solve `A·x = rhs` for one right-hand side.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch or the backend dispatch failure.
    fn solve(&self, device: &D, rhs: &D::Buffer<f32>) -> Result<D::Buffer<f32>>;
}

/// Device-neutral dense decompositions over rank-2 `f32` operands.
///
/// Implementors are zero-sized per-backend markers. A bound of
/// `R: DecompositionOps<D>` costs nothing at runtime and every call
/// monomorphizes to the backend's own kernel dispatch.
///
/// # Special values
///
/// Floating-point NaN and infinity behaviour follows the kernel
/// dialect's declared capability: see
/// [`KernelDialect::IEEE_SPECIAL_VALUES`](crate::KernelDialect::IEEE_SPECIAL_VALUES)
/// (ADR 0043) for what is and is not promised per dialect.
pub trait DecompositionOps<D: ComputeDevice> {
    /// LU result handle.
    type Lu<'op>: LuHandle<D>
    where
        Self: 'op,
        D: 'op;
    /// QR result handle.
    type Qr<'op>: QrHandle<D>
    where
        Self: 'op,
        D: 'op;
    /// Cholesky result handle.
    type Cholesky<'op>: CholeskyHandle<D>
    where
        Self: 'op,
        D: 'op;

    /// Factor a square matrix as `P·A = L·U` with partial pivoting.
    ///
    /// # Errors
    ///
    /// Returns a non-square or invalid layout, a singular matrix, or the
    /// backend dispatch failure.
    fn lu<'op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<f32>, 2>,
    ) -> Result<Self::Lu<'op>>;

    /// Factor an `m × n` matrix (`m ≥ n`) as `A = Q·R`.
    ///
    /// # Errors
    ///
    /// Returns an invalid shape or layout, or the backend dispatch failure.
    fn qr<'op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<f32>, 2>,
    ) -> Result<Self::Qr<'op>>;

    /// Column-pivoted QR result bound to this backend.
    type ColPivQr<'op>: ColPivQrHandle<D>
    where
        Self: 'op,
        D: 'op;

    /// Fully pivoted LU result bound to this backend.
    type FullPivLu<'op>: FullPivLuHandle<D>
    where
        Self: 'op,
        D: 'op;

    /// Factor a rank-2 view by column-pivoted QR, revealing rank.
    ///
    /// # Errors
    ///
    /// Returns a shape rejection (`m < n`), a layout validation failure,
    /// or the backend dispatch failure.
    fn col_piv_qr<'op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<f32>, 2>,
    ) -> Result<Self::ColPivQr<'op>>;

    /// Factor a square rank-2 view by fully pivoted LU, revealing rank.
    ///
    /// # Errors
    ///
    /// Returns a non-square rejection, a layout validation failure, or
    /// the backend dispatch failure.
    fn full_piv_lu<'op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<f32>, 2>,
    ) -> Result<Self::FullPivLu<'op>>;

    /// Symmetric eigendecomposition result bound to this backend.
    type SymmetricEigen<'op>: SymmetricEigenHandle<D>
    where
        Self: 'op,
        D: 'op;

    /// Factor a symmetric rank-2 view as `A = V·Λ·Vᵀ`.
    ///
    /// # Errors
    ///
    /// Returns a non-square rejection, a layout validation failure, or
    /// the backend dispatch failure.
    fn symmetric_eigen<'op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<f32>, 2>,
    ) -> Result<Self::SymmetricEigen<'op>>;

    /// Eigenvalues of a symmetric rank-2 view, without eigenvectors.
    ///
    /// # Errors
    ///
    /// Returns a non-square rejection, a layout validation failure, or
    /// the backend dispatch failure.
    fn symmetric_eigenvalues(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<f32>, 2>,
    ) -> Result<D::Buffer<f32>>;

    /// Factor a symmetric positive-definite matrix as `A = L·Lᵀ`.
    ///
    /// # Errors
    ///
    /// Returns a non-square or invalid layout, a non-SPD matrix, or the
    /// backend dispatch failure.
    fn cholesky<'op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<f32>, 2>,
    ) -> Result<Self::Cholesky<'op>>;
}

/// Oracle-minimal accessors on a column-pivoted QR result (ADR 0042
/// staging, stage 1).
pub trait ColPivQrHandle<D: ComputeDevice> {
    /// Numerical rank revealed by the pivoted factorization.
    fn rank(&self) -> usize;

    /// Column permutation as a permutation vector (leto's convention; the
    /// conformance clause pins the gather direction against the host
    /// reference).
    fn permutation(&self) -> &[usize];

    /// Solve `min ‖A·x − rhs‖₂`.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch against the factored shape or the
    /// backend dispatch failure.
    fn solve_least_squares(&self, device: &D, rhs: &D::Buffer<f32>) -> Result<D::Buffer<f32>>;
}

/// Oracle-minimal accessors on a fully pivoted LU result (ADR 0042
/// staging, stage 1).
pub trait FullPivLuHandle<D: ComputeDevice> {
    /// Factored dimension `n`.
    fn order(&self) -> usize;

    /// Numerical rank revealed by full pivoting.
    fn rank(&self) -> usize;

    /// Determinant accumulated during elimination.
    fn det(&self) -> f32;

    /// Device-resident packed `L`/`U` factors, `n × n` row-major (unit
    /// lower diagonal implicit), of the fully pivoted elimination.
    fn factors(&self) -> &D::Buffer<f32>;

    /// Row permutation vector (leto's convention).
    fn row_permutation(&self) -> &[usize];

    /// Column permutation vector (leto's convention).
    fn col_permutation(&self) -> &[usize];

    /// Solve `A·x = rhs` for a full-rank factorization.
    ///
    /// # Errors
    ///
    /// Returns a rank-deficiency rejection, a length mismatch, or the
    /// backend dispatch failure.
    fn solve(&self, device: &D, rhs: &D::Buffer<f32>) -> Result<D::Buffer<f32>>;
}

/// Oracle-minimal accessors on a symmetric eigendecomposition result
/// (ADR 0042 staging, stage 2).
pub trait SymmetricEigenHandle<D: ComputeDevice> {
    /// Dimension of the factored symmetric matrix.
    fn order(&self) -> usize;

    /// Device-resident eigenvalues (ordering is the backend's Jacobi
    /// sweep output; the conformance clause asserts the value multiset,
    /// not an order).
    fn eigenvalues(&self) -> &D::Buffer<f32>;

    /// Device-resident eigenvectors, `n × n` row-major with eigenvector
    /// `k` in column `k`, paired index-for-index with `eigenvalues`.
    fn eigenvectors(&self) -> &D::Buffer<f32>;
}
