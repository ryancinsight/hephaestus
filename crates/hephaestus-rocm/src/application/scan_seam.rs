use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, DialectScalar, HephaestusError, HipC, IdentityToken, OpIdentity,
    Result, ScanDirection, ScanOps, StridedView,
};
use leto::Layout;

use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::scan::scan_axis_into;
use crate::application::strided::StridedOperand;

/// Provider-owned implementation of [`ScanOps`] for ROCm/HIP.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmScanOps;

/// Prepared scan; the operation runs in `prepare` under ROCm's synchronous
/// execution model so `dispatch` is a no-op.
#[derive(Clone, Copy, Debug)]
pub struct RocmPreparedScan;

impl<T> ScanOps<RocmDevice, T> for RocmScanOps
where
    T: DialectScalar<HipC> + Pod,
{
    type Dialect = HipC;
    type PreparedScan<const N: usize> = RocmPreparedScan;

    fn prepare_scan_axis<Op, const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, N>,
        axis: usize,
        direction: ScanDirection,
        output: StridedView<'_, RocmBuffer<T>, N>,
    ) -> Result<Self::PreparedScan<N>>
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
        scan_axis_into::<Op, T>(
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
        Ok(RocmPreparedScan)
    }

    fn dispatch_scan<const N: usize>(
        &self,
        _device: &RocmDevice,
        _prepared: &Self::PreparedScan<N>,
    ) -> Result<()> {
        Ok(())
    }
}
