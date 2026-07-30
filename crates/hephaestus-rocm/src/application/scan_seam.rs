use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, DialectScalar, HephaestusError, HipC, IdentityToken, OpIdentity,
    Result, ScanDirection, ScanOps, StridedView,
};
use leto::Layout;

use crate::application::scan::scan_axis_into;
use crate::application::strided::StridedOperand;
use crate::RocmBuffer;
use crate::RocmDevice;

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
        // SAFETY: N == 2 here, so the layouts are Layout<2>.
        let input_layout: &Layout<2> = unsafe { &*(input.layout as *const Layout<N> as *const Layout<2>) };
        let output_layout: &Layout<2> =
            unsafe { &*(output.layout as *const Layout<N> as *const Layout<2>) };
        scan_axis_into::<Op, T>(
            device,
            StridedOperand {
                buffer: input.buffer,
                layout: input_layout,
            },
            axis,
            direction,
            StridedOperand {
                buffer: output.buffer,
                layout: output_layout,
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
