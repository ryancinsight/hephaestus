//! WGPU implementation of the device-neutral decomposition seam (ADR 0042).
//!
//! The kernels live in [`crate::application::decomposition`]; this module
//! only adapts them to [`hephaestus_core::DecompositionOps`] so a consumer —
//! or the conformance suite — can factor matrices without naming
//! `WgpuDevice`, matching the crate's other seam adapters.

use hephaestus_core::{
    BidiagonalHandle, BunchKaufmanHandle, CholeskyHandle, ColPivQrHandle, DecompositionOps,
    FullPivLuHandle, HessenbergHandle, LuHandle, QrHandle, Result, SchurHandle, StridedView,
    SvdHandle, SymmetricEigenHandle, UduHandle,
};

use crate::application::decomposition::{
    GpuBidiagonalDecomposition, GpuBunchKaufmanDecomposition, GpuCholesky,
    GpuColPivQrDecomposition, GpuFullPivLuDecomposition, GpuHessenbergDecomposition,
    GpuLuDecomposition, GpuQrDecomposition, GpuRealSchur, GpuSvdDecomposition,
    GpuSymmetricEigenDecomposition, GpuUduDecomposition, bidiagonalize, bunch_kaufman,
    cholesky_decompose, col_piv_qr, eigenvalues, full_piv_lu, hessenberg, lu_decompose,
    qr_decompose, schur, singular_values, svd_decompose, symmetric_eigen_jacobi,
    symmetric_eigenvalues_jacobi, udu_decompose,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Dense decompositions for one WGPU device.
///
/// Zero-sized: decomposition pipelines are cached on the device, so the seam
/// holds no state of its own, matching [`crate::WgpuSparseOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuDecompositionOps;

impl LuHandle<WgpuDevice> for GpuLuDecomposition {
    fn order(&self) -> usize {
        self.n()
    }
    fn factors(&self) -> &WgpuBuffer<f32> {
        Self::factors(self)
    }
    fn pivots(&self) -> &[usize] {
        Self::pivots(self)
    }
    fn det(&self) -> f32 {
        Self::det(self)
    }
    fn solve(&self, device: &WgpuDevice, rhs: &WgpuBuffer<f32>) -> Result<WgpuBuffer<f32>> {
        Self::solve(self, device, rhs)
    }
}

impl QrHandle<WgpuDevice> for GpuQrDecomposition {
    fn shape(&self) -> (usize, usize) {
        Self::shape(self)
    }
    fn r_buffer(&self) -> &WgpuBuffer<f32> {
        Self::r_buffer(self)
    }
    fn solve_least_squares(
        &self,
        device: &WgpuDevice,
        rhs: &WgpuBuffer<f32>,
    ) -> Result<WgpuBuffer<f32>> {
        Self::solve_least_squares(self, device, rhs)
    }
}

impl CholeskyHandle<WgpuDevice> for GpuCholesky {
    fn order(&self) -> usize {
        self.n()
    }
    fn lower(&self) -> &WgpuBuffer<f32> {
        Self::lower(self)
    }
    fn det(&self) -> f32 {
        Self::det(self)
    }
    fn solve(&self, device: &WgpuDevice, rhs: &WgpuBuffer<f32>) -> Result<WgpuBuffer<f32>> {
        Self::solve(self, device, rhs)
    }
}

impl ColPivQrHandle<WgpuDevice> for GpuColPivQrDecomposition {
    fn shape(&self) -> (usize, usize) {
        Self::shape(self)
    }
    fn rank(&self) -> usize {
        Self::rank(self)
    }
    fn permutation(&self) -> &[usize] {
        Self::permutation(self)
    }
    fn solve_least_squares(
        &self,
        device: &WgpuDevice,
        rhs: &WgpuBuffer<f32>,
    ) -> Result<WgpuBuffer<f32>> {
        Self::solve_least_squares(self, device, rhs)
    }
}

impl FullPivLuHandle<WgpuDevice> for GpuFullPivLuDecomposition {
    fn order(&self) -> usize {
        self.n()
    }
    fn rank(&self) -> usize {
        Self::rank(self)
    }
    fn det(&self) -> f32 {
        Self::det(self)
    }
    fn factors(&self) -> &WgpuBuffer<f32> {
        self.lu_buffer()
    }
    fn row_permutation(&self) -> &[usize] {
        Self::row_permutation(self)
    }
    fn col_permutation(&self) -> &[usize] {
        Self::col_permutation(self)
    }
    fn solve(&self, device: &WgpuDevice, rhs: &WgpuBuffer<f32>) -> Result<WgpuBuffer<f32>> {
        Self::solve(self, device, rhs)
    }
}

impl SymmetricEigenHandle<WgpuDevice> for GpuSymmetricEigenDecomposition {
    fn order(&self) -> usize {
        self.n()
    }
    fn eigenvalues(&self) -> &WgpuBuffer<f32> {
        Self::eigenvalues(self)
    }
    fn eigenvectors(&self) -> &WgpuBuffer<f32> {
        Self::eigenvectors(self)
    }
}

impl SvdHandle<WgpuDevice> for GpuSvdDecomposition {
    fn shape(&self) -> (usize, usize) {
        Self::shape(self)
    }
    fn u(&self) -> &WgpuBuffer<f32> {
        Self::u(self)
    }
    fn v(&self) -> &WgpuBuffer<f32> {
        Self::v(self)
    }
    fn singular_values(&self) -> &WgpuBuffer<f32> {
        Self::singular_values(self)
    }
}

impl BunchKaufmanHandle<WgpuDevice> for GpuBunchKaufmanDecomposition {
    fn order(&self) -> usize {
        self.n()
    }
    fn l_buffer(&self) -> &WgpuBuffer<f32> {
        Self::l_buffer(self)
    }
    fn d_buffer(&self) -> &WgpuBuffer<f32> {
        Self::d_buffer(self)
    }
    fn permutation(&self) -> &[usize] {
        Self::permutation(self)
    }
}

impl UduHandle<WgpuDevice> for GpuUduDecomposition {
    fn order(&self) -> usize {
        self.n()
    }
    fn u_buffer(&self) -> &WgpuBuffer<f32> {
        Self::u_buffer(self)
    }
    fn d_buffer(&self) -> &WgpuBuffer<f32> {
        Self::d_buffer(self)
    }
    fn det(&self) -> f32 {
        Self::det(self)
    }
    fn solve(&self, device: &WgpuDevice, rhs: &WgpuBuffer<f32>) -> Result<WgpuBuffer<f32>> {
        Self::solve(self, device, rhs)
    }
}

impl SchurHandle<WgpuDevice> for GpuRealSchur {
    fn order(&self) -> usize {
        self.n()
    }
    fn q_buffer(&self) -> &WgpuBuffer<f32> {
        Self::q_buffer(self)
    }
    fn t_buffer(&self) -> &WgpuBuffer<f32> {
        Self::t_buffer(self)
    }
}

impl HessenbergHandle<WgpuDevice> for GpuHessenbergDecomposition {
    fn order(&self) -> usize {
        self.n()
    }
    fn q_buffer(&self) -> &WgpuBuffer<f32> {
        Self::q_buffer(self)
    }
    fn h_buffer(&self) -> &WgpuBuffer<f32> {
        Self::h_buffer(self)
    }
}

impl BidiagonalHandle<WgpuDevice> for GpuBidiagonalDecomposition {
    fn shape(&self) -> (usize, usize) {
        Self::shape(self)
    }
    fn u_buffer(&self) -> &WgpuBuffer<f32> {
        Self::u_buffer(self)
    }
    fn b_buffer(&self) -> &WgpuBuffer<f32> {
        Self::b_buffer(self)
    }
    fn v_buffer(&self) -> &WgpuBuffer<f32> {
        Self::v_buffer(self)
    }
}

/// Convert the device-neutral view into this backend's operand pair.
#[inline]
fn operand<'a>(view: StridedView<'a, WgpuBuffer<f32>, 2>) -> StridedOperand<'a, f32, 2> {
    StridedOperand {
        buffer: view.buffer,
        layout: view.layout,
    }
}

impl DecompositionOps<WgpuDevice> for WgpuDecompositionOps {
    type Lu<'op> = GpuLuDecomposition;
    type Qr<'op> = GpuQrDecomposition;
    type Cholesky<'op> = GpuCholesky;

    fn lu<'op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<f32>, 2>,
    ) -> Result<Self::Lu<'op>> {
        lu_decompose(device, operand(input))
    }

    fn qr<'op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<f32>, 2>,
    ) -> Result<Self::Qr<'op>> {
        qr_decompose(device, operand(input))
    }

    type ColPivQr<'op> = GpuColPivQrDecomposition;
    type FullPivLu<'op> = GpuFullPivLuDecomposition;

    fn col_piv_qr<'op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<f32>, 2>,
    ) -> Result<Self::ColPivQr<'op>> {
        col_piv_qr(device, operand(input))
    }

    fn full_piv_lu<'op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<f32>, 2>,
    ) -> Result<Self::FullPivLu<'op>> {
        full_piv_lu(device, operand(input))
    }

    type SymmetricEigen<'op> = GpuSymmetricEigenDecomposition;

    fn symmetric_eigen<'op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<f32>, 2>,
    ) -> Result<Self::SymmetricEigen<'op>> {
        symmetric_eigen_jacobi(device, operand(input))
    }

    fn symmetric_eigenvalues(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<f32>, 2>,
    ) -> Result<WgpuBuffer<f32>> {
        symmetric_eigenvalues_jacobi(device, operand(input))
    }

    type Svd<'op> = GpuSvdDecomposition;

    fn svd<'op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<f32>, 2>,
    ) -> Result<Self::Svd<'op>> {
        svd_decompose(device, operand(input))
    }

    fn singular_values(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<f32>, 2>,
    ) -> Result<WgpuBuffer<f32>> {
        singular_values(device, operand(input))
    }

    type BunchKaufman<'op> = GpuBunchKaufmanDecomposition;
    type Udu<'op> = GpuUduDecomposition;

    fn bunch_kaufman<'op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<f32>, 2>,
    ) -> Result<Self::BunchKaufman<'op>> {
        bunch_kaufman(device, operand(input))
    }

    fn udu<'op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<f32>, 2>,
    ) -> Result<Self::Udu<'op>> {
        udu_decompose(device, operand(input))
    }

    type Schur<'op> = GpuRealSchur;
    type Hessenberg<'op> = GpuHessenbergDecomposition;
    type Bidiagonal<'op> = GpuBidiagonalDecomposition;

    fn eigenvalues(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<f32>, 2>,
    ) -> Result<WgpuBuffer<eunomia::Complex<f32>>> {
        eigenvalues(device, operand(input))
    }

    fn schur<'op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<f32>, 2>,
    ) -> Result<Self::Schur<'op>> {
        schur(device, operand(input))
    }

    fn hessenberg<'op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<f32>, 2>,
    ) -> Result<Self::Hessenberg<'op>> {
        hessenberg(device, operand(input))
    }

    fn bidiagonalize<'op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<f32>, 2>,
    ) -> Result<Self::Bidiagonal<'op>> {
        bidiagonalize(device, operand(input))
    }

    fn cholesky<'op>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'op, WgpuBuffer<f32>, 2>,
    ) -> Result<Self::Cholesky<'op>> {
        cholesky_decompose(device, operand(input))
    }
}
