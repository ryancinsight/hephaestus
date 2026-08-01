//! CUDA implementation of the device-neutral decomposition seam (ADR 0042).
//!
//! The kernels live in [`crate::application::decomposition`]; this module
//! only adapts them to [`hephaestus_core::DecompositionOps`] so a consumer —
//! or the conformance suite — can factor matrices without naming
//! `CudaDevice`, matching the crate's other seam adapters.

use hephaestus_core::{CholeskyHandle, DecompositionOps, LuHandle, QrHandle, Result, StridedView};

use crate::application::decomposition::{
    GpuCholesky, GpuLuDecomposition, GpuQrDecomposition, cholesky_decompose, lu_decompose,
    qr_decompose,
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

    fn cholesky<'op>(
        &self,
        device: &CudaDevice,
        input: StridedView<'op, CudaBuffer<f32>, 2>,
    ) -> Result<Self::Cholesky<'op>> {
        cholesky_decompose(device, operand(input))
    }
}
