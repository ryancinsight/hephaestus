//! CUDA implementation of the device-neutral decomposition seam (ADR 0042).
//!
//! The kernels live in [`crate::application::decomposition`]; this module
//! only adapts them to [`hephaestus_core::DecompositionOps`] so a consumer —
//! or the conformance suite — can factor matrices without naming
//! `CudaDevice`, matching the crate's other seam adapters.

use hephaestus_core::{
    CholeskyHandle, ColPivQrHandle, DecompositionOps, FullPivLuHandle, LuHandle, QrHandle, Result,
    StridedView,
};

use crate::application::decomposition::{
    GpuCholesky, GpuColPivQrDecomposition, GpuFullPivLuDecomposition, GpuLuDecomposition,
    GpuQrDecomposition, cholesky_decompose, col_piv_qr, full_piv_lu, lu_decompose, qr_decompose,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Dense decompositions for one CUDA device.
///
/// Zero-sized: decomposition kernels are cached on the device, so the seam
/// holds no state of its own.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaDecompositionOps;

impl LuHandle<CudaDevice> for GpuLuDecomposition {
    fn order(&self) -> usize {
        self.n()
    }
    fn factors(&self) -> &CudaBuffer<f32> {
        Self::factors(self)
    }
    fn pivots(&self) -> &[usize] {
        Self::pivots(self)
    }
    fn det(&self) -> f32 {
        Self::det(self)
    }
    fn solve(&self, device: &CudaDevice, rhs: &CudaBuffer<f32>) -> Result<CudaBuffer<f32>> {
        Self::solve(self, device, rhs)
    }
}

impl QrHandle<CudaDevice> for GpuQrDecomposition {
    fn shape(&self) -> (usize, usize) {
        Self::shape(self)
    }
    fn r_buffer(&self) -> &CudaBuffer<f32> {
        Self::r_buffer(self)
    }
    fn solve_least_squares(
        &self,
        device: &CudaDevice,
        rhs: &CudaBuffer<f32>,
    ) -> Result<CudaBuffer<f32>> {
        Self::solve_least_squares(self, device, rhs)
    }
}

impl CholeskyHandle<CudaDevice> for GpuCholesky {
    fn order(&self) -> usize {
        self.n()
    }
    fn lower(&self) -> &CudaBuffer<f32> {
        Self::lower(self)
    }
    fn det(&self) -> f32 {
        Self::det(self)
    }
    fn solve(&self, device: &CudaDevice, rhs: &CudaBuffer<f32>) -> Result<CudaBuffer<f32>> {
        Self::solve(self, device, rhs)
    }
}

impl ColPivQrHandle<CudaDevice> for GpuColPivQrDecomposition {
    fn rank(&self) -> usize {
        Self::rank(self)
    }
    fn permutation(&self) -> &[usize] {
        Self::permutation(self)
    }
    fn solve_least_squares(
        &self,
        device: &CudaDevice,
        rhs: &CudaBuffer<f32>,
    ) -> Result<CudaBuffer<f32>> {
        Self::solve_least_squares(self, device, rhs)
    }
}

impl FullPivLuHandle<CudaDevice> for GpuFullPivLuDecomposition {
    fn order(&self) -> usize {
        self.n()
    }
    fn rank(&self) -> usize {
        Self::rank(self)
    }
    fn det(&self) -> f32 {
        Self::det(self)
    }
    fn row_permutation(&self) -> &[usize] {
        Self::row_permutation(self)
    }
    fn col_permutation(&self) -> &[usize] {
        Self::col_permutation(self)
    }
    fn solve(&self, device: &CudaDevice, rhs: &CudaBuffer<f32>) -> Result<CudaBuffer<f32>> {
        Self::solve(self, device, rhs)
    }
}

/// Convert the device-neutral view into this backend's operand pair.
#[inline]
fn operand<'a>(view: StridedView<'a, CudaBuffer<f32>, 2>) -> StridedOperand<'a, f32, 2> {
    StridedOperand {
        buffer: view.buffer,
        layout: view.layout,
    }
}

impl DecompositionOps<CudaDevice> for CudaDecompositionOps {
    type Lu<'op> = GpuLuDecomposition;
    type Qr<'op> = GpuQrDecomposition;
    type Cholesky<'op> = GpuCholesky;

    fn lu<'op>(
        &self,
        device: &CudaDevice,
        input: StridedView<'op, CudaBuffer<f32>, 2>,
    ) -> Result<Self::Lu<'op>> {
        lu_decompose(device, operand(input))
    }

    fn qr<'op>(
        &self,
        device: &CudaDevice,
        input: StridedView<'op, CudaBuffer<f32>, 2>,
    ) -> Result<Self::Qr<'op>> {
        qr_decompose(device, operand(input))
    }

    type ColPivQr<'op> = GpuColPivQrDecomposition;
    type FullPivLu<'op> = GpuFullPivLuDecomposition;

    fn col_piv_qr<'op>(
        &self,
        device: &CudaDevice,
        input: StridedView<'op, CudaBuffer<f32>, 2>,
    ) -> Result<Self::ColPivQr<'op>> {
        col_piv_qr(device, operand(input))
    }

    fn full_piv_lu<'op>(
        &self,
        device: &CudaDevice,
        input: StridedView<'op, CudaBuffer<f32>, 2>,
    ) -> Result<Self::FullPivLu<'op>> {
        full_piv_lu(device, operand(input))
    }

    fn cholesky<'op>(
        &self,
        device: &CudaDevice,
        input: StridedView<'op, CudaBuffer<f32>, 2>,
    ) -> Result<Self::Cholesky<'op>> {
        cholesky_decompose(device, operand(input))
    }
}
