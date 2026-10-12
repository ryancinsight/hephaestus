use super::handles::{
    BidiagonalHandle, BunchKaufmanHandle, CholeskyHandle, ColPivQrHandle, FullPivLuHandle,
    HessenbergHandle, LuHandle, QrHandle, SchurHandle, SvdHandle, SymmetricEigenHandle, UduHandle,
};
use crate::domain::device::ComputeDevice;
use crate::domain::error::Result;
use crate::domain::view::StridedView;

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

    /// Real Schur result bound to this backend.
    type Schur<'op>: SchurHandle<D>
    where
        Self: 'op,
        D: 'op;

    /// Hessenberg result bound to this backend.
    type Hessenberg<'op>: HessenbergHandle<D>
    where
        Self: 'op,
        D: 'op;

    /// Bidiagonal result bound to this backend.
    type Bidiagonal<'op>: BidiagonalHandle<D>
    where
        Self: 'op,
        D: 'op;

    /// Eigenvalues of a square rank-2 view as complex values.
    ///
    /// # Errors
    ///
    /// Returns a non-square rejection, a layout validation failure, or
    /// the backend dispatch failure.
    fn eigenvalues(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<f32>, 2>,
    ) -> Result<D::Buffer<eunomia::Complex<f32>>>;

    /// Reduce a square rank-2 view to real Schur form `A = Q·T·Qᵀ`.
    ///
    /// # Errors
    ///
    /// Returns a non-square rejection, a layout validation failure, or
    /// the backend dispatch failure.
    fn schur<'op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<f32>, 2>,
    ) -> Result<Self::Schur<'op>>;

    /// Reduce a square rank-2 view to Hessenberg form `A = Q·H·Qᵀ`.
    ///
    /// # Errors
    ///
    /// Returns a non-square rejection, a layout validation failure, or
    /// the backend dispatch failure.
    fn hessenberg<'op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<f32>, 2>,
    ) -> Result<Self::Hessenberg<'op>>;

    /// Reduce a rank-2 view to bidiagonal form `A = U·B·Vᵀ`.
    ///
    /// # Errors
    ///
    /// Returns a layout validation failure or the backend dispatch
    /// failure.
    fn bidiagonalize<'op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<f32>, 2>,
    ) -> Result<Self::Bidiagonal<'op>>;

    /// Bunch–Kaufman result bound to this backend.
    type BunchKaufman<'op>: BunchKaufmanHandle<D>
    where
        Self: 'op,
        D: 'op;

    /// `U·D·Uᵀ` result bound to this backend.
    type Udu<'op>: UduHandle<D>
    where
        Self: 'op,
        D: 'op;

    /// Factor a symmetric (possibly indefinite) rank-2 view as
    /// `P·A·Pᵀ = L·D·Lᵀ` with symmetric pivoting.
    ///
    /// # Errors
    ///
    /// Returns a non-square rejection, a layout validation failure, or
    /// the backend dispatch failure.
    fn bunch_kaufman<'op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<f32>, 2>,
    ) -> Result<Self::BunchKaufman<'op>>;

    /// Factor a symmetric rank-2 view as `A = U·D·Uᵀ`.
    ///
    /// # Errors
    ///
    /// Returns a non-square rejection, a singularity rejection, a layout
    /// validation failure, or the backend dispatch failure.
    fn udu<'op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<f32>, 2>,
    ) -> Result<Self::Udu<'op>>;

    /// Singular value decomposition result bound to this backend.
    type Svd<'op>: SvdHandle<D>
    where
        Self: 'op,
        D: 'op;

    /// Factor a rank-2 view as `A = U·Σ·Vᵀ`.
    ///
    /// The rank-revealing entry points backends expose are algorithm
    /// choices over this same contract, not distinct seam methods.
    ///
    /// # Errors
    ///
    /// Returns a layout validation failure or the backend dispatch
    /// failure.
    fn svd<'op>(
        &self,
        device: &D,
        input: StridedView<'op, D::Buffer<f32>, 2>,
    ) -> Result<Self::Svd<'op>>;

    /// Singular values of a rank-2 view, without singular vectors.
    ///
    /// # Errors
    ///
    /// Returns a layout validation failure or the backend dispatch
    /// failure.
    fn singular_values(
        &self,
        device: &D,
        input: StridedView<'_, D::Buffer<f32>, 2>,
    ) -> Result<D::Buffer<f32>>;

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
