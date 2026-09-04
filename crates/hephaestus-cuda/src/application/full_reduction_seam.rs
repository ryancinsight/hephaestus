//! CUDA implementation of the device-neutral full-reduction seam.
//!
//! The prepared form owns a contiguous staging buffer and the multi-pass
//! reduction plan, and borrows the operand pair: each dispatch re-materialises
//! the (possibly strided) input into the staging buffer, re-runs the passes,
//! and writes the scalar into the bound output — so re-dispatch observes
//! writes to the bound input (the seam's rebind contract).

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, ElementwiseOps,
    FullReductionOps, HephaestusError, IdentityOp, IdentityToken, OpIdentity, Result, StridedView,
};
use leto::Layout;

use crate::application::elementwise_seam::CudaElementwiseOps;
use crate::application::prepared_reduction::PreparedReductionPlan;
use crate::application::strided::map_layout_err;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Provider-owned implementation of [`FullReductionOps`] for CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaFullReductionOps;

/// Prepared full reduction bound to one input/output pair.
pub struct CudaPreparedFullReduction<'op, T, const N: usize> {
    input: &'op CudaBuffer<T>,
    input_layout: Layout<N>,
    output: &'op CudaBuffer<T>,
    output_offset: usize,
    /// Contiguous staging target re-filled from `input` at each dispatch;
    /// `None` when the logical input is empty and the plan's identity output
    /// is written directly.
    staging: Option<(CudaBuffer<T>, Layout<N>)>,
    plan: PreparedReductionPlan<T>,
}

impl<T> FullReductionOps<CudaDevice, T> for CudaFullReductionOps
where
    T: DialectScalar<CudaC> + Pod + Send + Sync,
{
    type Dialect = CudaC;
    type Prepared<'op, const N: usize>
        = CudaPreparedFullReduction<'op, T, N>
    where
        T: 'op;

    fn prepare_reduce_full<'op, Op, const N: usize>(
        &self,
        device: &CudaDevice,
        input: StridedView<'op, CudaBuffer<T>, N>,
        output: StridedView<'op, CudaBuffer<T>, 1>,
    ) -> Result<Self::Prepared<'op, N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        output
            .layout
            .validate_storage_len(output.buffer.len())
            .map_err(map_layout_err)?;
        if output.layout.checked_size().map_err(map_layout_err)? != 1 {
            return Err(HephaestusError::DispatchFailed {
                message: "full reduction output must have exactly 1 element".to_string(),
            });
        }
        let logical_len = input.layout.checked_size().map_err(map_layout_err)?;
        input
            .layout
            .validate_storage_len(input.buffer.len())
            .map_err(map_layout_err)?;

        let staging = if logical_len == 0 {
            None
        } else {
            let staging_layout =
                Layout::c_contiguous(input.layout.shape()).map_err(map_layout_err)?;
            Some((
                device.alloc_uninitialized::<T>(logical_len)?,
                staging_layout,
            ))
        };
        let plan = PreparedReductionPlan::prepare::<Op>(device, logical_len, BlockWidth::DEFAULT)?;

        Ok(CudaPreparedFullReduction {
            input: input.buffer,
            input_layout: *input.layout,
            output: output.buffer,
            output_offset: output.layout.offset(),
            staging,
            plan,
        })
    }

    fn dispatch_full<const N: usize>(
        &self,
        device: &CudaDevice,
        prepared: &Self::Prepared<'_, N>,
    ) -> Result<()> {
        if let Some((staging, staging_layout)) = &prepared.staging {
            <CudaElementwiseOps as ElementwiseOps<CudaDevice, T>>::unary_into::<IdentityOp, N>(
                &CudaElementwiseOps,
                device,
                StridedView::new(prepared.input, &prepared.input_layout),
                StridedView::new(staging, staging_layout),
            )?;
            prepared.plan.dispatch(staging)?;
        }
        let mut host_val = [T::zeroed()];
        device.download_sub_buffer(prepared.plan.output(), 0, &mut host_val)?;
        device.write_sub_buffer(prepared.output, prepared.output_offset, &host_val)
    }
}
