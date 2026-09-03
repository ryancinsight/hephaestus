//! ROCm/HIP implementation of the device-neutral full-reduction seam.
//!
//! The prepared form owns a contiguous staging buffer and the multi-pass
//! reduction plan, and borrows the operand pair: each dispatch re-materialises
//! the (possibly strided) input into the staging buffer, re-runs the passes,
//! and writes the scalar into the bound output — so re-dispatch observes
//! writes to the bound input (the seam's rebind contract).

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, DeviceBuffer, DialectScalar, ElementwiseOps,
    FullReductionOps, HephaestusError, HipC, IdentityOp, IdentityToken, OpIdentity, Result,
    StridedView,
};
use leto::Layout;

use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::elementwise_seam::RocmElementwiseOps;
use crate::application::prepared_reduction::PreparedReductionPlan;

fn map_layout_err(e: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("{e}"),
    }
}

/// Provider-owned implementation of [`FullReductionOps`] for ROCm/HIP.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmFullReductionOps;

/// Prepared full reduction bound to one input/output pair.
pub struct RocmPreparedFullReduction<'op, T, const N: usize> {
    input: &'op RocmBuffer<T>,
    input_layout: Layout<N>,
    output: &'op RocmBuffer<T>,
    output_offset: usize,
    /// Contiguous staging target re-filled from `input` at each dispatch;
    /// `None` when the logical input is empty and the plan's identity output
    /// is written directly.
    staging: Option<(RocmBuffer<T>, Layout<N>)>,
    plan: PreparedReductionPlan<T>,
}

impl<T> FullReductionOps<RocmDevice, T> for RocmFullReductionOps
where
    T: DialectScalar<HipC> + Pod + Send + Sync,
{
    type Dialect = HipC;
    type Prepared<'op, const N: usize>
        = RocmPreparedFullReduction<'op, T, N>
    where
        T: 'op;

    fn prepare_reduce_full<'op, Op, const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<T>, N>,
        output: StridedView<'op, RocmBuffer<T>, 1>,
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

        Ok(RocmPreparedFullReduction {
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
        device: &RocmDevice,
        prepared: &Self::Prepared<'_, N>,
    ) -> Result<()> {
        if let Some((staging, staging_layout)) = &prepared.staging {
            <RocmElementwiseOps as ElementwiseOps<RocmDevice, T>>::unary_into::<IdentityOp, N>(
                &RocmElementwiseOps,
                device,
                StridedView::new(prepared.input, &prepared.input_layout),
                StridedView::new(staging, staging_layout),
            )?;
            prepared.plan.dispatch(staging)?;
        }
        // The final pass output holds exactly one element by construction.
        let mut host_val = [T::zeroed()];
        device.download(prepared.plan.output(), &mut host_val)?;
        device.write_sub_buffer(prepared.output, prepared.output_offset, &host_val)
    }
}
