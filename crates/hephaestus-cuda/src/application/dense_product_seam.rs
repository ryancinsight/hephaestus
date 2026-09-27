//! Provider-owned dense product, composition, and matrix-function seams for
//! CUDA.
//!
//! The kernels live in [`crate::application::linalg`]; this module only
//! adapts them to [`hephaestus_core::DenseProductOps`],
//! [`hephaestus_core::DenseCompositionOps`], and
//! [`hephaestus_core::DenseMatrixFunctionOps`] so a consumer — or
//! the conformance suite — can run dense products without naming
//! `CudaDevice`, matching the crate's other seam adapters.

use eunomia::Pod;
use hephaestus_core::{
    CudaC, DenseCompositionOps, DenseMatrixFunctionOps, DenseProductOps, DialectScalar, Result,
    StridedView,
};

use crate::application::linalg::{
    batched_matmul_into, det, kron_into, matmul_into, matpow, matrix_rank_with_tolerance,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Provider-owned dense product, composition, and matrix-function implementation
/// for CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaDenseProductOps;

// This device's operand type is `hephaestus_core::StridedOperand` over
// `CudaBuffer<T>` (see `crate::application::strided`), so the device-neutral
// `StridedView` the seam hands in *is* the operand the kernel entry points take
// — no per-seam conversion is needed.

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
        matmul_into::<T>(device, lhs, rhs, output)
    }

    fn batched_matmul_into(
        &self,
        device: &CudaDevice,
        lhs: StridedView<'_, CudaBuffer<T>, 3>,
        rhs: StridedView<'_, CudaBuffer<T>, 3>,
        output: StridedView<'_, CudaBuffer<T>, 3>,
    ) -> Result<()> {
        batched_matmul_into::<T>(device, lhs, rhs, output)
    }

    fn kron_into(
        &self,
        device: &CudaDevice,
        lhs: StridedView<'_, CudaBuffer<T>, 2>,
        rhs: StridedView<'_, CudaBuffer<T>, 2>,
        output: StridedView<'_, CudaBuffer<T>, 2>,
    ) -> Result<()> {
        kron_into::<T>(device, lhs, rhs, output)
    }
}

impl DenseCompositionOps<CudaDevice> for CudaDenseProductOps {
    fn matpow(
        &self,
        device: &CudaDevice,
        matrix: StridedView<'_, CudaBuffer<f32>, 2>,
        exponent: u32,
    ) -> Result<CudaBuffer<f32>> {
        matpow(device, matrix, exponent)
    }

    fn det(
        &self,
        device: &CudaDevice,
        matrix: StridedView<'_, CudaBuffer<f32>, 2>,
    ) -> Result<CudaBuffer<f32>> {
        det(device, matrix)
    }

    fn matrix_rank_with_tolerance(
        &self,
        device: &CudaDevice,
        matrix: StridedView<'_, CudaBuffer<f32>, 2>,
        relative_tolerance: f32,
    ) -> Result<usize> {
        matrix_rank_with_tolerance(device, matrix, relative_tolerance)
    }
}

impl DenseMatrixFunctionOps<CudaDevice> for CudaDenseProductOps {
    fn matexp(
        &self,
        device: &CudaDevice,
        matrix: StridedView<'_, CudaBuffer<f32>, 2>,
    ) -> Result<CudaBuffer<f32>> {
        crate::application::linalg::matexp(device, matrix)
    }

    fn pinv(
        &self,
        device: &CudaDevice,
        matrix: StridedView<'_, CudaBuffer<f32>, 2>,
    ) -> Result<CudaBuffer<f32>> {
        crate::application::linalg::pinv(device, matrix)
    }
}
