use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, CudaC, DialectScalar, HephaestusError, IdentityToken, OpIdentity,
    Result, ScanDirection, ScanOps, StridedView,
};
use leto::Layout;

use crate::application::scan::scan_axis_into;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Provider-owned implementation of [`ScanOps`] for CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaScanOps;

/// Prepared scan; the operation runs in `prepare` under CUDA's synchronous
/// execution model so `dispatch` is a no-op.
#[derive(Clone, Copy, Debug)]
pub struct CudaPreparedScan;

impl<T> ScanOps<CudaDevice, T> for CudaScanOps
where
    T: DialectScalar<CudaC> + Pod,
{
    type Dialect = CudaC;
    type PreparedScan<const N: usize> = CudaPreparedScan;

    fn prepare_scan_axis<Op, const N: usize>(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, N>,
        axis: usize,
        direction: ScanDirection,
        output: StridedView<'_, CudaBuffer<T>, N>,
    ) -> Result<Self::PreparedScan<N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        if N != 2 {
            return Err(HephaestusError::DispatchFailed {
                message: format!("CUDA scan supports rank 2 only, got rank {N}"),
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
        Ok(CudaPreparedScan)
    }

    fn dispatch_scan<const N: usize>(
        &self,
        _device: &CudaDevice,
        _prepared: &Self::PreparedScan<N>,
    ) -> Result<()> {
        Ok(())
    }
}
