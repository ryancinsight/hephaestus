//! ROCm/HIP implementation of the device-neutral decomposition seam (ADR 0042).
//!
//! The kernels live in [`crate::application::decomposition`]; this module
//! only adapts them to [`hephaestus_core::DecompositionOps`] so a consumer —
//! or the conformance suite — can factor matrices without naming
//! `RocmDevice`, matching the crate's other seam adapters.

use hephaestus_core::{
    CholeskyHandle, ColPivQrHandle, DecompositionOps, FullPivLuHandle, LuHandle, QrHandle, Result,
    StridedView, SvdHandle, SymmetricEigenHandle,
};

use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::decomposition::{
    GpuCholesky, GpuColPivQrDecomposition, GpuFullPivLuDecomposition, GpuLuDecomposition,
    GpuQrDecomposition, GpuSvdDecomposition, GpuSymmetricEigenDecomposition, cholesky_decompose,
    col_piv_qr, full_piv_lu, lu_decompose, qr_decompose, singular_values, svd_decompose,
    symmetric_eigen_jacobi, symmetric_eigenvalues_jacobi,
};
use crate::application::strided::StridedOperand;

/// Dense decompositions for one ROCm/HIP device.
///
/// Zero-sized: decomposition kernels are cached on the device, so the seam
/// holds no state of its own.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmDecompositionOps;

impl LuHandle<RocmDevice> for GpuLuDecomposition {
    fn order(&self) -> usize {
        self.n()
    }
    fn factors(&self) -> &RocmBuffer<f32> {
        Self::factors(self)
    }
    fn pivots(&self) -> &[usize] {
        Self::pivots(self)
    }
    fn det(&self) -> f32 {
        Self::det(self)
    }
    fn solve(&self, device: &RocmDevice, rhs: &RocmBuffer<f32>) -> Result<RocmBuffer<f32>> {
        Self::solve(self, device, rhs)
    }
}

impl QrHandle<RocmDevice> for GpuQrDecomposition {
    fn shape(&self) -> (usize, usize) {
        Self::shape(self)
    }
    fn r_buffer(&self) -> &RocmBuffer<f32> {
        Self::r_buffer(self)
    }
    fn solve_least_squares(
        &self,
        device: &RocmDevice,
        rhs: &RocmBuffer<f32>,
    ) -> Result<RocmBuffer<f32>> {
        Self::solve_least_squares(self, device, rhs)
    }
}

impl CholeskyHandle<RocmDevice> for GpuCholesky {
    fn order(&self) -> usize {
        self.n()
    }
    fn lower(&self) -> &RocmBuffer<f32> {
        Self::lower(self)
    }
    fn det(&self) -> f32 {
        Self::det(self)
    }
    fn solve(&self, device: &RocmDevice, rhs: &RocmBuffer<f32>) -> Result<RocmBuffer<f32>> {
        Self::solve(self, device, rhs)
    }
}

impl ColPivQrHandle<RocmDevice> for GpuColPivQrDecomposition {
    fn rank(&self) -> usize {
        Self::rank(self)
    }
    fn permutation(&self) -> &[usize] {
        Self::permutation(self)
    }
    fn solve_least_squares(
        &self,
        device: &RocmDevice,
        rhs: &RocmBuffer<f32>,
    ) -> Result<RocmBuffer<f32>> {
        Self::solve_least_squares(self, device, rhs)
    }
}

impl FullPivLuHandle<RocmDevice> for GpuFullPivLuDecomposition {
    fn order(&self) -> usize {
        self.n()
    }
    fn rank(&self) -> usize {
        Self::rank(self)
    }
    fn det(&self) -> f32 {
        Self::det(self)
    }
    fn factors(&self) -> &RocmBuffer<f32> {
        self.lu_buffer()
    }
    fn row_permutation(&self) -> &[usize] {
        Self::row_permutation(self)
    }
    fn col_permutation(&self) -> &[usize] {
        Self::col_permutation(self)
    }
    fn solve(&self, device: &RocmDevice, rhs: &RocmBuffer<f32>) -> Result<RocmBuffer<f32>> {
        Self::solve(self, device, rhs)
    }
}

impl SymmetricEigenHandle<RocmDevice> for GpuSymmetricEigenDecomposition {
    fn order(&self) -> usize {
        self.n()
    }
    fn eigenvalues(&self) -> &RocmBuffer<f32> {
        Self::eigenvalues(self)
    }
    fn eigenvectors(&self) -> &RocmBuffer<f32> {
        Self::eigenvectors(self)
    }
}

impl SvdHandle<RocmDevice> for GpuSvdDecomposition {
    fn shape(&self) -> (usize, usize) {
        Self::shape(self)
    }
    fn u(&self) -> &RocmBuffer<f32> {
        Self::u(self)
    }
    fn v(&self) -> &RocmBuffer<f32> {
        Self::v(self)
    }
    fn singular_values(&self) -> &RocmBuffer<f32> {
        Self::singular_values(self)
    }
}

/// Convert the device-neutral view into this backend's operand pair.
#[inline]
fn operand<'a>(view: StridedView<'a, RocmBuffer<f32>, 2>) -> StridedOperand<'a, f32, 2> {
    StridedOperand {
        buffer: view.buffer,
        layout: view.layout,
    }
}

impl DecompositionOps<RocmDevice> for RocmDecompositionOps {
    type Lu<'op> = GpuLuDecomposition;
    type Qr<'op> = GpuQrDecomposition;
    type Cholesky<'op> = GpuCholesky;

    fn lu<'op>(
        &self,
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<f32>, 2>,
    ) -> Result<Self::Lu<'op>> {
        lu_decompose(device, operand(input))
    }

    fn qr<'op>(
        &self,
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<f32>, 2>,
    ) -> Result<Self::Qr<'op>> {
        qr_decompose(device, operand(input))
    }

    type ColPivQr<'op> = GpuColPivQrDecomposition;
    type FullPivLu<'op> = GpuFullPivLuDecomposition;

    fn col_piv_qr<'op>(
        &self,
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<f32>, 2>,
    ) -> Result<Self::ColPivQr<'op>> {
        col_piv_qr(device, operand(input))
    }

    fn full_piv_lu<'op>(
        &self,
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<f32>, 2>,
    ) -> Result<Self::FullPivLu<'op>> {
        full_piv_lu(device, operand(input))
    }

    type SymmetricEigen<'op> = GpuSymmetricEigenDecomposition;

    fn symmetric_eigen<'op>(
        &self,
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<f32>, 2>,
    ) -> Result<Self::SymmetricEigen<'op>> {
        symmetric_eigen_jacobi(device, operand(input))
    }

    fn symmetric_eigenvalues(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<f32>, 2>,
    ) -> Result<RocmBuffer<f32>> {
        symmetric_eigenvalues_jacobi(device, operand(input))
    }

    type Svd<'op> = GpuSvdDecomposition;

    fn svd<'op>(
        &self,
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<f32>, 2>,
    ) -> Result<Self::Svd<'op>> {
        svd_decompose(device, operand(input))
    }

    fn singular_values(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<f32>, 2>,
    ) -> Result<RocmBuffer<f32>> {
        singular_values(device, operand(input))
    }

    fn cholesky<'op>(
        &self,
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<f32>, 2>,
    ) -> Result<Self::Cholesky<'op>> {
        cholesky_decompose(device, operand(input))
    }
}
