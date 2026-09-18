//! Leto as a dense-product-seam implementor (ADR 0046 / ADR 0044).
//!
//! [`HostDenseProductOps`] adapts leto-ops' `matmul`, `batched_matmul`, and
//! `kron` kernels onto [`DenseProductOps<HostDevice, T>`](hephaestus_core::DenseProductOps),
//! so the CPU substrate joins the kernel-product family's role trait per
//! ADR 0046 §5 and the conformance suite's dense-product clauses run on the
//! host pair.
//! Shape and output-aliasing are rejected before any output element is
//! written, matching the validation convention the accelerator seams share.

use eunomia::Pod;
use hephaestus_core::{DenseProductOps, Result, StridedView};
use leto::{Array2, ArrayView, ArrayViewMut};
use leto_ops::Scalar;

use crate::operands::{require_disjoint_output, with_operands};
use crate::{HostBuffer, HostDevice, map_leto_error};

/// Dense products (matmul, batched matmul, Kronecker) for the host
/// reference device.
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
