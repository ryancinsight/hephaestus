use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, CudaC, DialectScalar, HephaestusError, IdentityToken, OpIdentity,
    Result, ScanDirection, ScanOps, StridedView,
};
use leto::Layout;

use crate::application::scan::{ScanLaunch, launch_planned_scan, plan_scan_launch};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Provider-owned implementation of [`ScanOps`] for CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaScanOps;

/// Prepared scan bound to one input/output pair.
///
/// Holds its operand borrows; dispatch re-reads their device addresses, so
/// writes to the bound operands between dispatches are observed (the seam's
/// rebind contract). An empty scan prepares to a no-op.
pub struct CudaPreparedScan<'op, T> {
    input: &'op CudaBuffer<T>,
    output: &'op CudaBuffer<T>,
    plan: Option<ScanLaunch>,
}

impl<T> ScanOps<CudaDevice, T> for CudaScanOps
where
    T: DialectScalar<CudaC> + Pod,
{
    type Dialect = CudaC;
    type PreparedScan<'op, const N: usize>
        = CudaPreparedScan<'op, T>
    where
        T: 'op;

    fn prepare_scan_axis<'op, Op, const N: usize>(
        &self,
        device: &CudaDevice,
        input: StridedView<'op, CudaBuffer<T>, N>,
        axis: usize,
        direction: ScanDirection,
        output: StridedView<'op, CudaBuffer<T>, N>,
    ) -> Result<Self::PreparedScan<'op, N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        if N != 2 {
            return Err(HephaestusError::DispatchFailed {
                message: format!("CUDA scan supports rank 2 only, got rank {N}"),
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
        Ok(CudaPreparedScan {
            input: input.buffer,
            output: output.buffer,
            plan,
        })
    }

    fn dispatch_scan<const N: usize>(
        &self,
        device: &CudaDevice,
        prepared: &Self::PreparedScan<'_, N>,
    ) -> Result<()> {
        let Some(plan) = &prepared.plan else {
            return Ok(());
        };
        launch_planned_scan(device, plan, prepared.input.raw(), prepared.output.raw())
    }
}
