//! Provider-owned dense product seam for ROCm.
//!
//! The kernels live in [`crate::application::linalg`]; this module only
//! adapts them to [`hephaestus_core::DenseProductOps`] so a consumer — or
//! the conformance suite — can run dense products without naming
//! `RocmDevice`, matching the crate's other seam adapters.

use eunomia::Pod;
use hephaestus_core::{
    DenseCompositionOps, DenseProductOps, DialectScalar, HipC, Result, StridedView,
};

use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::linalg::{
    batched_matmul_into, det, kron_into, matmul_into, matpow, matrix_rank_with_tolerance,
};
use crate::application::strided::StridedOperand;

/// Provider-owned implementation of [`DenseProductOps`] for ROCm.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmDenseProductOps;

impl<T> DenseProductOps<RocmDevice, T> for RocmDenseProductOps
where
    T: DialectScalar<HipC> + Pod,
{
    fn matmul_into(
        &self,
        device: &RocmDevice,
        lhs: StridedView<'_, RocmBuffer<T>, 2>,
        rhs: StridedView<'_, RocmBuffer<T>, 2>,
        output: StridedView<'_, RocmBuffer<T>, 2>,
    ) -> Result<()> {
        matmul_into::<T>(device, operand(lhs), operand(rhs), operand(output))
    }

    fn batched_matmul_into(
        &self,
        device: &RocmDevice,
        lhs: StridedView<'_, RocmBuffer<T>, 3>,
        rhs: StridedView<'_, RocmBuffer<T>, 3>,
        output: StridedView<'_, RocmBuffer<T>, 3>,
    ) -> Result<()> {
        batched_matmul_into::<T>(device, operand(lhs), operand(rhs), operand(output))
    }

    fn kron_into(
        &self,
        device: &RocmDevice,
        lhs: StridedView<'_, RocmBuffer<T>, 2>,
        rhs: StridedView<'_, RocmBuffer<T>, 2>,
        output: StridedView<'_, RocmBuffer<T>, 2>,
    ) -> Result<()> {
        kron_into::<T>(device, operand(lhs), operand(rhs), operand(output))
    }
}

impl DenseCompositionOps<RocmDevice> for RocmDenseProductOps {
    fn matpow(
        &self,
        device: &RocmDevice,
        matrix: StridedView<'_, RocmBuffer<f32>, 2>,
        exponent: u32,
    ) -> Result<RocmBuffer<f32>> {
        matpow(device, operand(matrix), exponent)
    }

    fn det(
        &self,
        device: &RocmDevice,
        matrix: StridedView<'_, RocmBuffer<f32>, 2>,
    ) -> Result<RocmBuffer<f32>> {
        det(device, operand(matrix))
    }

    fn matrix_rank_with_tolerance(
        &self,
        device: &RocmDevice,
        matrix: StridedView<'_, RocmBuffer<f32>, 2>,
        relative_tolerance: f32,
    ) -> Result<usize> {
        matrix_rank_with_tolerance(device, operand(matrix), relative_tolerance)
    }
}

/// Convert the device-neutral view into this backend's operand pair.
#[inline]
fn operand<'a, T, const N: usize>(
    view: StridedView<'a, RocmBuffer<T>, N>,
) -> StridedOperand<'a, T, N> {
    StridedOperand {
        buffer: view.buffer,
        layout: view.layout,
    }
}
