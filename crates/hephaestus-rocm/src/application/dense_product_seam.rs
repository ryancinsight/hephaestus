//! Provider-owned dense product seam for ROCm.
//!
//! The kernels live in [`crate::application::linalg`]; this module only
//! adapts them to [`hephaestus_core::DenseProductOps`] so a consumer — or
//! the conformance suite — can run dense products without naming
//! `RocmDevice`, matching the crate's other seam adapters.

use eunomia::Pod;
use hephaestus_core::{DenseProductOps, DialectScalar, HipC, Result, StridedView};

use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::linalg::{batched_matmul_into, kron_into, matmul_into};
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
