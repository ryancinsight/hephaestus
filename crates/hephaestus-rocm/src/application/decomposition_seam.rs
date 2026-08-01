//! ROCm/HIP implementation of the device-neutral decomposition seam (ADR 0042).
//!
//! The kernels live in [`crate::application::decomposition`]; this module
//! only adapts them to [`hephaestus_core::DecompositionOps`] so a consumer —
//! or the conformance suite — can factor matrices without naming
//! `RocmDevice`, matching the crate's other seam adapters.

use hephaestus_core::{CholeskyHandle, DecompositionOps, LuHandle, QrHandle, Result, StridedView};

use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::decomposition::{
    GpuCholesky, GpuLuDecomposition, GpuQrDecomposition, cholesky_decompose, lu_decompose,
    qr_decompose,
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

    fn cholesky<'op>(
        &self,
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<f32>, 2>,
    ) -> Result<Self::Cholesky<'op>> {
        cholesky_decompose(device, operand(input))
    }
}
