use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, DialectScalar, HephaestusError, HipC, IdentityToken, OpIdentity,
    Result, ScanDirection, ScanOps, StridedView,
};
use leto::Layout;

use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::scan::{ScanLaunch, launch_planned_scan, plan_scan_launch};
use crate::application::strided::StridedOperand;

/// Provider-owned implementation of [`ScanOps`] for ROCm/HIP.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmScanOps;

/// Prepared scan bound to one input/output pair.
///
/// Holds its operand borrows; dispatch re-reads their device addresses, so
/// writes to the bound operands between dispatches are observed (the seam's
/// rebind contract). An empty scan prepares to a no-op.
pub struct RocmPreparedScan<'op, T> {
    input: &'op RocmBuffer<T>,
    output: &'op RocmBuffer<T>,
    plan: Option<ScanLaunch>,
}

impl<T> ScanOps<RocmDevice, T> for RocmScanOps
where
    T: DialectScalar<HipC> + Pod,
{
    type Dialect = HipC;
    type PreparedScan<'op, const N: usize>
        = RocmPreparedScan<'op, T>
    where
        T: 'op;

    fn prepare_scan_axis<'op, Op, const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'op, RocmBuffer<T>, N>,
        axis: usize,
        direction: ScanDirection,
        output: StridedView<'op, RocmBuffer<T>, N>,
    ) -> Result<Self::PreparedScan<'op, N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        if N != 2 {
            return Err(HephaestusError::DispatchFailed {
                message: format!("ROCm scan supports rank 2 only, got rank {N}"),
            });
        }
        // The rank guard above proves N == 2, so the rank-2 components are
        // total; rebuilding a Layout<2> avoids reinterpreting &Layout<N> and
        // keeps the featureless build under forbid(unsafe_code).
        let input_layout = Layout::new(
            [input.layout.shape[0], input.layout.shape[1]],
            [input.layout.strides[0], input.layout.strides[1]],
            input.layout.offset,
        );
        let output_layout = Layout::new(
            [output.layout.shape[0], output.layout.shape[1]],
            [output.layout.strides[0], output.layout.strides[1]],
            output.layout.offset,
        );
        let plan = plan_scan_launch::<Op, T>(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: &input_layout,
            },
            axis,
            direction,
            StridedOperand {
                buffer: output.buffer,
                layout: &output_layout,
            },
            BlockWidth::DEFAULT,
        )?;
        Ok(RocmPreparedScan {
            input: input.buffer,
            output: output.buffer,
            plan,
        })
    }

    fn dispatch_scan<const N: usize>(
        &self,
        device: &RocmDevice,
        prepared: &Self::PreparedScan<'_, N>,
    ) -> Result<()> {
        let Some(plan) = &prepared.plan else {
            return Ok(());
        };
        launch_planned_scan(device, plan, prepared.input.raw(), prepared.output.raw())
    }
}
