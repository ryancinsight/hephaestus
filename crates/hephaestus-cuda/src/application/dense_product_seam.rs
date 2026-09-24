//! Provider-owned dense product and composition seams for CUDA.
//!
//! The kernels live in [`crate::application::linalg`]; this module only
//! adapts them to [`hephaestus_core::DenseProductOps`] and
//! [`hephaestus_core::DenseCompositionOps`] so a consumer — or
//! the conformance suite — can run dense products without naming
//! `CudaDevice`, matching the crate's other seam adapters.

use eunomia::Pod;
use hephaestus_core::{
    CudaC, DenseCompositionOps, DenseProductOps, DialectScalar, Result, StridedView,
};

use crate::application::linalg::{
    batched_matmul_into, det, kron_into, matmul_into, matpow, matrix_rank_with_tolerance,
};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Provider-owned dense product and composition implementation for CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaDenseProductOps;

impl<T> DenseProductOps<CudaDevice, T> for CudaDenseProductOps
where
    T: DialectScalar<CudaC> + Pod,
{
    fn matmul_into(
        &self,
        device: &CudaDevice,
        lhs: StridedView<'_, CudaBuffer<T>, 2>,
        rhs: StridedView<'_, CudaBuffer<T>, 2>,
        output: StridedView<'_, CudaBuffer<T>, 2>,
    ) -> Result<()> {
        matmul_into::<T>(device, operand(lhs), operand(rhs), operand(output))
    }

    fn batched_matmul_into(
        &self,
        device: &CudaDevice,
        lhs: StridedView<'_, CudaBuffer<T>, 3>,
        rhs: StridedView<'_, CudaBuffer<T>, 3>,
        output: StridedView<'_, CudaBuffer<T>, 3>,
    ) -> Result<()> {
        batched_matmul_into::<T>(device, operand(lhs), operand(rhs), operand(output))
    }

    fn kron_into(
        &self,
        device: &CudaDevice,
        lhs: StridedView<'_, CudaBuffer<T>, 2>,
        rhs: StridedView<'_, CudaBuffer<T>, 2>,
        output: StridedView<'_, CudaBuffer<T>, 2>,
    ) -> Result<()> {
        kron_into::<T>(device, operand(lhs), operand(rhs), operand(output))
    }
}

impl DenseCompositionOps<CudaDevice> for CudaDenseProductOps {
    fn matpow(
        &self,
        device: &CudaDevice,
        matrix: StridedView<'_, CudaBuffer<f32>, 2>,
        exponent: u32,
    ) -> Result<CudaBuffer<f32>> {
        matpow(device, operand(matrix), exponent)
    }

    fn det(
        &self,
        device: &CudaDevice,
        matrix: StridedView<'_, CudaBuffer<f32>, 2>,
    ) -> Result<CudaBuffer<f32>> {
        det(device, operand(matrix))
    }

    fn matrix_rank_with_tolerance(
        &self,
        device: &CudaDevice,
        matrix: StridedView<'_, CudaBuffer<f32>, 2>,
        relative_tolerance: f32,
    ) -> Result<usize> {
        matrix_rank_with_tolerance(device, operand(matrix), relative_tolerance)
    }
}

/// Convert the device-neutral view into this backend's operand pair.
#[inline]
fn operand<'a, T, const N: usize>(
    view: StridedView<'a, CudaBuffer<T>, N>,
) -> StridedOperand<'a, T, N> {
    StridedOperand {
        buffer: view.buffer,
        layout: view.layout,
    }
}
