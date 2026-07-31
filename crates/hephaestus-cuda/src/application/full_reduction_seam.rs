use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, ElementwiseOps,
    FullReductionOps, HephaestusError, IdentityOp, IdentityToken, OpIdentity, Result, StridedView,
};
use leto::Layout;

use crate::application::elementwise_seam::CudaElementwiseOps;
use crate::application::reduction::reduction_with_width;
use crate::application::strided::map_layout_err;
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

/// Provider-owned implementation of [`FullReductionOps`] for CUDA.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaFullReductionOps;

/// Prepared full reduction; the operation runs in `prepare` under CUDA's
/// synchronous execution model so `dispatch` is a no-op.
#[derive(Clone, Copy, Debug)]
pub struct CudaPreparedFullReduction;

/// Materialise a strided view into a contiguous buffer.
///
/// Always allocates a new buffer because [`CudaBuffer`] is not [`Clone`]; the
/// fast-path for already-contiguous inputs is evaluated but not taken (the
/// caller already owns the buffer reference).
fn contiguous_reduction_source<T, const N: usize>(
    device: &CudaDevice,
    input: StridedView<'_, CudaBuffer<T>, N>,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod,
{
    let logical_len = input.layout.checked_size().map_err(map_layout_err)?;
    input
        .layout
        .validate_storage_len(input.buffer.len())
        .map_err(map_layout_err)?;
    if logical_len == 0 {
        return device.alloc_uninitialized(0);
    }

    let out_layout = Layout::c_contiguous(input.layout.shape).map_err(map_layout_err)?;
    let contig = device.alloc_uninitialized::<T>(logical_len)?;
    <CudaElementwiseOps as ElementwiseOps<CudaDevice, T>>::unary_into::<IdentityOp, N>(
        &CudaElementwiseOps,
        device,
        input,
        StridedView::new(&contig, &out_layout),
    )?;
    Ok(contig)
}

impl<T> FullReductionOps<CudaDevice, T> for CudaFullReductionOps
where
    T: DialectScalar<CudaC> + Pod + Send + Sync,
{
    type Dialect = CudaC;
    type Prepared<const N: usize> = CudaPreparedFullReduction;

    fn prepare_reduce_full<Op, const N: usize>(
        &self,
        device: &CudaDevice,
        input: StridedView<'_, CudaBuffer<T>, N>,
        output: StridedView<'_, CudaBuffer<T>, 1>,
    ) -> Result<Self::Prepared<N>>
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

        let source = contiguous_reduction_source(device, input)?;
        let result = reduction_with_width::<Op, T>(device, &source, BlockWidth::DEFAULT)?;

        let mut host_val = [T::IDENTITY];
        device.download_sub_buffer(&result, 0, &mut host_val)?;
        device.write_sub_buffer(output.buffer, output.layout.offset, &host_val)?;

        Ok(CudaPreparedFullReduction)
    }

    fn dispatch_full<const N: usize>(
        &self,
        _device: &CudaDevice,
        _prepared: &Self::Prepared<N>,
    ) -> Result<()> {
        Ok(())
    }
}
