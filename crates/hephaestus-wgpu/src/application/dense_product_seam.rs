//! Provider-owned dense product seam for WGPU.
//!
//! The kernels live in [`crate::application::linalg`]; this module only
//! adapts them to [`hephaestus_core::DenseProductOps`] so a consumer — or
//! the conformance suite — can run dense products without naming
//! `WgpuDevice`, matching the crate's other seam adapters.

use eunomia::Pod;
use hephaestus_core::{DenseProductOps, DialectScalar, Result, StridedView, Wgsl};

use crate::MatmulZero;
use crate::application::linalg::{batched_matmul_into, kron_into, matmul_into};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Provider-owned implementation of [`DenseProductOps`] for WGPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuDenseProductOps;

impl<T> DenseProductOps<WgpuDevice, T> for WgpuDenseProductOps
where
    T: DialectScalar<Wgsl> + Pod + MatmulZero,
{
    fn matmul_into(
        &self,
        device: &WgpuDevice,
        lhs: StridedView<'_, WgpuBuffer<T>, 2>,
        rhs: StridedView<'_, WgpuBuffer<T>, 2>,
        output: StridedView<'_, WgpuBuffer<T>, 2>,
    ) -> Result<()> {
        matmul_into::<T>(device, operand(lhs), operand(rhs), operand(output))
    }

    fn batched_matmul_into(
        &self,
        device: &WgpuDevice,
        lhs: StridedView<'_, WgpuBuffer<T>, 3>,
        rhs: StridedView<'_, WgpuBuffer<T>, 3>,
        output: StridedView<'_, WgpuBuffer<T>, 3>,
    ) -> Result<()> {
        batched_matmul_into::<T>(device, operand(lhs), operand(rhs), operand(output))
    }

    fn kron_into(
        &self,
        device: &WgpuDevice,
        lhs: StridedView<'_, WgpuBuffer<T>, 2>,
        rhs: StridedView<'_, WgpuBuffer<T>, 2>,
        output: StridedView<'_, WgpuBuffer<T>, 2>,
    ) -> Result<()> {
        kron_into::<T>(device, operand(lhs), operand(rhs), operand(output))
    }
}

/// Convert the device-neutral view into this backend's operand pair.
#[inline]
fn operand<'a, T, const N: usize>(
    view: StridedView<'a, WgpuBuffer<T>, N>,
) -> StridedOperand<'a, T, N> {
    StridedOperand {
        buffer: view.buffer,
        layout: view.layout,
    }
}
