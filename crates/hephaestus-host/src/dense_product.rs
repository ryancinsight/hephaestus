//! Leto as a dense-product-seam implementor (ADR 0046 / ADR 0044).
//!
//! [`HostDenseProductOps`] adapts leto-ops' `matmul`, `batched_matmul`, and
//! `kron` kernels onto [`DenseProductOps<HostDevice, T>`](hephaestus_core::DenseProductOps).
//! The same zero-sized marker implements
//! [`DenseCompositionOps`](hephaestus_core::DenseCompositionOps) for matrix
//! power, determinant, and rank, and
//! [`DenseMatrixFunctionOps`](hephaestus_core::DenseMatrixFunctionOps) for the
//! host-delegated exponential and pseudoinverse. This lets one Leto-backed CPU
//! reference run all three dense-linalg conformance families.
//! Shape and output-aliasing are rejected before any output element is
//! written, matching the validation convention the accelerator seams share.

use eunomia::Pod;
use hephaestus_core::{
    ComputeDevice, DenseCompositionOps, DenseMatrixFunctionOps, DenseProductOps, Result,
    StridedView,
};
use leto::{Array2, ArrayView, ArrayViewMut};
use leto_ops::Scalar;

use crate::operands::{require_disjoint_output, upload_array, with_matrix_view, with_operands};
use crate::{HostBuffer, HostDevice, map_leto_error};

/// Dense linear-algebra operations for the host reference device.
///
/// This remains one zero-sized marker, matching the provider seam bundles and
/// keeping all calls statically dispatched.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostDenseProductOps;

impl<T> DenseProductOps<HostDevice, T> for HostDenseProductOps
where
    T: Pod + Scalar,
{
    fn matmul_into(
        &self,
        _device: &HostDevice,
        lhs: StridedView<'_, HostBuffer<T>, 2>,
        rhs: StridedView<'_, HostBuffer<T>, 2>,
        output: StridedView<'_, HostBuffer<T>, 2>,
    ) -> Result<()> {
        require_disjoint_output(lhs.buffer, rhs.buffer, output.buffer)?;
        let mut out_cells = output.buffer.write();
        let mut out_view = ArrayViewMut::<T, 2>::try_new(*output.layout, &mut out_cells)
            .map_err(map_leto_error)?;
        with_operands(lhs.buffer, rhs.buffer, |lhs_cells, rhs_cells| {
            let lhs_view =
                ArrayView::<T, 2>::try_new(*lhs.layout, lhs_cells).map_err(map_leto_error)?;
            let rhs_view =
                ArrayView::<T, 2>::try_new(*rhs.layout, rhs_cells).map_err(map_leto_error)?;
            leto_ops::matmul(&lhs_view, &rhs_view, &mut out_view).map_err(map_leto_error)
        })
    }

    fn batched_matmul_into(
        &self,
        _device: &HostDevice,
        lhs: StridedView<'_, HostBuffer<T>, 3>,
        rhs: StridedView<'_, HostBuffer<T>, 3>,
        output: StridedView<'_, HostBuffer<T>, 3>,
    ) -> Result<()> {
        require_disjoint_output(lhs.buffer, rhs.buffer, output.buffer)?;
        let mut out_cells = output.buffer.write();
        let mut out_view = ArrayViewMut::<T, 3>::try_new(*output.layout, &mut out_cells)
            .map_err(map_leto_error)?;
        with_operands(lhs.buffer, rhs.buffer, |lhs_cells, rhs_cells| {
            let lhs_view =
                ArrayView::<T, 3>::try_new(*lhs.layout, lhs_cells).map_err(map_leto_error)?;
            let rhs_view =
                ArrayView::<T, 3>::try_new(*rhs.layout, rhs_cells).map_err(map_leto_error)?;
            leto_ops::batched_matmul(&lhs_view, &rhs_view, &mut out_view).map_err(map_leto_error)
        })
    }

    fn kron_into(
        &self,
        _device: &HostDevice,
        lhs: StridedView<'_, HostBuffer<T>, 2>,
        rhs: StridedView<'_, HostBuffer<T>, 2>,
        output: StridedView<'_, HostBuffer<T>, 2>,
    ) -> Result<()> {
        require_disjoint_output(lhs.buffer, rhs.buffer, output.buffer)?;
        // `leto_ops::kron` has no output-view form: it derives the product
        // shape with checked arithmetic and returns a dense `Array2`, which
        // is then assigned into the output view.
        let product: Array2<T> = with_operands(lhs.buffer, rhs.buffer, |lhs_cells, rhs_cells| {
            let lhs_view =
                ArrayView::<T, 2>::try_new(*lhs.layout, lhs_cells).map_err(map_leto_error)?;
            let rhs_view =
                ArrayView::<T, 2>::try_new(*rhs.layout, rhs_cells).map_err(map_leto_error)?;
            leto_ops::kron(&lhs_view, &rhs_view).map_err(map_leto_error)
        })?;
        // `try_assign` validates the destination shape against `product`
        // before writing any element, so a wrong-shaped `output` view is
        // rejected with no partial write (leto's `assign_into` contract).
        let mut out_cells = output.buffer.write();
        let mut out_view = ArrayViewMut::<T, 2>::try_new(*output.layout, &mut out_cells)
            .map_err(map_leto_error)?;
        out_view.try_assign(&product).map_err(map_leto_error)
    }
}

impl DenseCompositionOps<HostDevice> for HostDenseProductOps {
    fn matpow(
        &self,
        device: &HostDevice,
        matrix: StridedView<'_, HostBuffer<f32>, 2>,
        exponent: u32,
    ) -> Result<HostBuffer<f32>> {
        let output = with_matrix_view(&matrix, |view| leto_ops::matpow(&view, exponent))?;
        upload_array(device, &output)
    }

    fn det(
        &self,
        device: &HostDevice,
        matrix: StridedView<'_, HostBuffer<f32>, 2>,
    ) -> Result<HostBuffer<f32>> {
        let determinant = with_matrix_view(&matrix, |view| leto_ops::det(&view))?;
        device.upload(&[determinant])
    }

    fn matrix_rank_with_tolerance(
        &self,
        _device: &HostDevice,
        matrix: StridedView<'_, HostBuffer<f32>, 2>,
        relative_tolerance: f32,
    ) -> Result<usize> {
        with_matrix_view(&matrix, |view| {
            leto_ops::matrix_rank_with_tolerance(&view, relative_tolerance)
        })
    }
}

impl DenseMatrixFunctionOps<HostDevice> for HostDenseProductOps {
    fn matexp(
        &self,
        device: &HostDevice,
        matrix: StridedView<'_, HostBuffer<f32>, 2>,
    ) -> Result<HostBuffer<f32>> {
        let [rows, cols] = matrix.layout.shape();
        if rows != cols {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: format!(
                    "matrix exponential requires square matrix, got shape [{rows}, {cols}]"
                ),
            });
        }
        if rows == 0 {
            return device.alloc_zeroed(0);
        }
        let output = with_matrix_view(&matrix, |view| leto_ops::matexp(&view))?;
        upload_array(device, &output)
    }

    fn pinv(
        &self,
        device: &HostDevice,
        matrix: StridedView<'_, HostBuffer<f32>, 2>,
    ) -> Result<HostBuffer<f32>> {
        let [rows, cols] = matrix.layout.shape();
        if rows == 0 || cols == 0 {
            return device.alloc_zeroed(0);
        }
        let output = with_matrix_view(&matrix, |view| leto_ops::pinv(&view))?;
        upload_array(device, &output)
    }
}
