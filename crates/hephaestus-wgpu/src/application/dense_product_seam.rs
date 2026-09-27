//! Provider-owned dense product, composition, and matrix-function seams for
//! WGPU.
//!
//! The kernels live in [`crate::application::linalg`]; this module only
//! adapts them to [`hephaestus_core::DenseProductOps`],
//! [`hephaestus_core::DenseCompositionOps`], and
//! [`hephaestus_core::DenseMatrixFunctionOps`] so a consumer — or
//! the conformance suite — can run dense products without naming
//! `WgpuDevice`, matching the crate's other seam adapters.

use eunomia::Pod;
#[cfg(any(feature = "decomposition", feature = "sparse"))]
use hephaestus_core::DenseMatrixFunctionOps;
use hephaestus_core::{
    DenseCompositionOps, DenseProductOps, DialectScalar, Result, StridedView, Wgsl,
};

use crate::MatmulZero;
use crate::application::linalg::{
    batched_matmul_into, det, kron_into, matmul_into, matpow, matrix_rank_with_tolerance,
};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Provider-owned dense product, composition, and matrix-function implementation
/// for WGPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuDenseProductOps;

// This device's operand type is `hephaestus_core::StridedOperand` over
// `WgpuBuffer<T>`, so the device-neutral `StridedView` the seam hands in *is*
// the operand the kernel entry points take — no per-seam conversion is needed.

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
        matmul_into::<T>(device, lhs, rhs, output)
    }

    fn batched_matmul_into(
        &self,
        device: &WgpuDevice,
        lhs: StridedView<'_, WgpuBuffer<T>, 3>,
        rhs: StridedView<'_, WgpuBuffer<T>, 3>,
        output: StridedView<'_, WgpuBuffer<T>, 3>,
    ) -> Result<()> {
        batched_matmul_into::<T>(device, lhs, rhs, output)
    }

    fn kron_into(
        &self,
        device: &WgpuDevice,
        lhs: StridedView<'_, WgpuBuffer<T>, 2>,
        rhs: StridedView<'_, WgpuBuffer<T>, 2>,
        output: StridedView<'_, WgpuBuffer<T>, 2>,
    ) -> Result<()> {
        kron_into::<T>(device, lhs, rhs, output)
    }
}

impl DenseCompositionOps<WgpuDevice> for WgpuDenseProductOps {
    fn matpow(
        &self,
        device: &WgpuDevice,
        matrix: StridedView<'_, WgpuBuffer<f32>, 2>,
        exponent: u32,
    ) -> Result<WgpuBuffer<f32>> {
        matpow(device, matrix, exponent)
    }

    fn det(
        &self,
        device: &WgpuDevice,
        matrix: StridedView<'_, WgpuBuffer<f32>, 2>,
    ) -> Result<WgpuBuffer<f32>> {
        det(device, matrix)
    }

    fn matrix_rank_with_tolerance(
        &self,
        device: &WgpuDevice,
        matrix: StridedView<'_, WgpuBuffer<f32>, 2>,
        relative_tolerance: f32,
    ) -> Result<usize> {
        matrix_rank_with_tolerance(device, matrix, relative_tolerance)
    }
}

#[cfg(any(feature = "decomposition", feature = "sparse"))]
impl DenseMatrixFunctionOps<WgpuDevice> for WgpuDenseProductOps {
    fn matexp(
        &self,
        device: &WgpuDevice,
        matrix: StridedView<'_, WgpuBuffer<f32>, 2>,
    ) -> Result<WgpuBuffer<f32>> {
        crate::application::linalg::matexp(device, matrix)
    }

    fn pinv(
        &self,
        device: &WgpuDevice,
        matrix: StridedView<'_, WgpuBuffer<f32>, 2>,
    ) -> Result<WgpuBuffer<f32>> {
        crate::application::linalg::pinv(device, matrix)
    }
}
