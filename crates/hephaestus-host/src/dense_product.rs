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
use hephaestus_core::{DenseProductOps, HephaestusError, Result, StridedView};
use leto::{Array2, ArrayView, ArrayViewMut};
use leto_ops::Scalar;

use crate::{HostBuffer, HostDevice, map_leto_error};

/// Dense products (matmul, batched matmul, Kronecker) for the host
/// reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostDenseProductOps;

/// Reject an output buffer that is the same underlying allocation as either
/// input.
///
/// The host represents every buffer as `Arc<RwLock<Vec<T>>>`
/// ([`HostBuffer`]); reading an operand and writing the output through the
/// same lock would deadlock rather than alias silently, so this check turns
/// that hang into the typed error the seam contract already documents.
fn require_disjoint_output<T>(
    lhs: &HostBuffer<T>,
    rhs: &HostBuffer<T>,
    output: &HostBuffer<T>,
) -> Result<()> {
    if lhs.aliases(output) || rhs.aliases(output) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias either input buffer".to_string(),
        });
    }
    Ok(())
}

/// Read both operands, under one guard when they are the same allocation.
///
/// `std::sync::RwLock::read` documents that it may panic when the current
/// thread already holds the lock, and `matmul(a, a)` — squaring — names one
/// buffer twice; a second read guard on it would be exactly that.
fn with_operands<T, R>(
    lhs: &HostBuffer<T>,
    rhs: &HostBuffer<T>,
    body: impl FnOnce(&[T], &[T]) -> R,
) -> R {
    let lhs_cells = lhs.read();
    if rhs.aliases(lhs) {
        return body(&lhs_cells, &lhs_cells);
    }
    let rhs_cells = rhs.read();
    body(&lhs_cells, &rhs_cells)
}

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
        let mut out_view = ArrayViewMut::<T, 2>::new(*output.layout, &mut out_cells);
        with_operands(lhs.buffer, rhs.buffer, |lhs_cells, rhs_cells| {
            let lhs_view = ArrayView::<T, 2>::new(*lhs.layout, lhs_cells);
            let rhs_view = ArrayView::<T, 2>::new(*rhs.layout, rhs_cells);
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
        let mut out_view = ArrayViewMut::<T, 3>::new(*output.layout, &mut out_cells);
        with_operands(lhs.buffer, rhs.buffer, |lhs_cells, rhs_cells| {
            let lhs_view = ArrayView::<T, 3>::new(*lhs.layout, lhs_cells);
            let rhs_view = ArrayView::<T, 3>::new(*rhs.layout, rhs_cells);
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
            let lhs_view = ArrayView::<T, 2>::new(*lhs.layout, lhs_cells);
            let rhs_view = ArrayView::<T, 2>::new(*rhs.layout, rhs_cells);
            leto_ops::kron(&lhs_view, &rhs_view).map_err(map_leto_error)
        })?;
        // `try_assign` validates the destination shape against `product`
        // before writing any element, so a wrong-shaped `output` view is
        // rejected with no partial write (leto's `assign_into` contract).
        let mut out_cells = output.buffer.write();
        let mut out_view = ArrayViewMut::<T, 2>::new(*output.layout, &mut out_cells);
        out_view.try_assign(&product).map_err(map_leto_error)
    }
}
