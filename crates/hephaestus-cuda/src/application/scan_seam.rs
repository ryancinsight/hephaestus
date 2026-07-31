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
