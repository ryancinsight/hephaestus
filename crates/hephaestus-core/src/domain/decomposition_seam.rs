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
    /// Device-resident upper-triangular `R`, row-major: either the `n × n`
    /// leading block or the full `m × n` factor (whose rows beyond `n`
    /// vanish) — both shapes occur across backends today, and the length of
    /// the returned buffer discriminates them. Normalizing on `n × n` is a
    /// recorded follow-up.
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
