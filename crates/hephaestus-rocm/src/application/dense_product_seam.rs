//! Provider-owned dense product, composition, and matrix-function seams for
//! ROCm.
//!
//! The kernels live in [`crate::application::linalg`]; this module only
//! adapts them to [`hephaestus_core::DenseProductOps`],
//! [`hephaestus_core::DenseCompositionOps`], and
//! [`hephaestus_core::DenseMatrixFunctionOps`] so a consumer — or
//! the conformance suite — can run dense products without naming
//! `RocmDevice`, matching the crate's other seam adapters.

use eunomia::Pod;
use hephaestus_core::{
    DenseCompositionOps, DenseMatrixFunctionOps, DenseProductOps, DialectScalar, HipC, Result,
    StridedView,
};

use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::linalg::{
    batched_matmul_into, det, kron_into, matmul_into, matpow, matrix_rank_with_tolerance,
};

/// Provider-owned dense product, composition, and matrix-function implementation
/// for ROCm.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmDenseProductOps;

// This device's operand type is `hephaestus_core::StridedOperand` over
// `RocmBuffer<T>` (see `crate::application::strided`), so the device-neutral
// `StridedView` the seam hands in *is* the operand the kernel entry points take
// — no per-seam conversion is needed.

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
        matmul_into::<T>(device, lhs, rhs, output)
    }

    fn batched_matmul_into(
        &self,
        device: &RocmDevice,
        lhs: StridedView<'_, RocmBuffer<T>, 3>,
        rhs: StridedView<'_, RocmBuffer<T>, 3>,
        output: StridedView<'_, RocmBuffer<T>, 3>,
    ) -> Result<()> {
        batched_matmul_into::<T>(device, lhs, rhs, output)
    }

    fn kron_into(
        &self,
        device: &RocmDevice,
        lhs: StridedView<'_, RocmBuffer<T>, 2>,
        rhs: StridedView<'_, RocmBuffer<T>, 2>,
        output: StridedView<'_, RocmBuffer<T>, 2>,
    ) -> Result<()> {
        kron_into::<T>(device, lhs, rhs, output)
    }
}

impl DenseCompositionOps<RocmDevice> for RocmDenseProductOps {
    fn matpow(
        &self,
        device: &RocmDevice,
        matrix: StridedView<'_, RocmBuffer<f32>, 2>,
        exponent: u32,
    ) -> Result<RocmBuffer<f32>> {
        matpow(device, matrix, exponent)
    }

    fn det(
        &self,
        device: &RocmDevice,
        matrix: StridedView<'_, RocmBuffer<f32>, 2>,
    ) -> Result<RocmBuffer<f32>> {
        det(device, matrix)
    }

    fn matrix_rank_with_tolerance(
        &self,
        device: &RocmDevice,
        matrix: StridedView<'_, RocmBuffer<f32>, 2>,
        relative_tolerance: f32,
    ) -> Result<usize> {
        matrix_rank_with_tolerance(device, matrix, relative_tolerance)
    }
}

impl DenseMatrixFunctionOps<RocmDevice> for RocmDenseProductOps {
    fn matexp(
        &self,
        device: &RocmDevice,
        matrix: StridedView<'_, RocmBuffer<f32>, 2>,
    ) -> Result<RocmBuffer<f32>> {
        crate::application::linalg::matexp(device, matrix)
    }

    fn pinv(
        &self,
        device: &RocmDevice,
        matrix: StridedView<'_, RocmBuffer<f32>, 2>,
    ) -> Result<RocmBuffer<f32>> {
        crate::application::linalg::pinv(device, matrix)
    }
}
