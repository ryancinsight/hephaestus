use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, DeviceBuffer, DialectScalar, ElementwiseOps,
    FullReductionOps, HephaestusError, HipC, IdentityOp, IdentityToken, OpIdentity, Result,
    StridedView,
};
use leto::Layout;

use crate::RocmBuffer;
use crate::RocmDevice;
use crate::application::elementwise_seam::RocmElementwiseOps;
use crate::application::reduction::reduction_with_width;

fn map_layout_err(e: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("{e}"),
    }
}

/// Materialise a strided view into a contiguous buffer.
///
/// Always allocates a new buffer because [`RocmBuffer`] is not [`Clone`].
fn contiguous_reduction_source<T, const N: usize>(
    device: &RocmDevice,
    input: StridedView<'_, RocmBuffer<T>, N>,
) -> Result<RocmBuffer<T>>
where
    T: DialectScalar<HipC> + Pod,
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
    <RocmElementwiseOps as ElementwiseOps<RocmDevice, T>>::unary_into::<IdentityOp, N>(
        &RocmElementwiseOps,
        device,
        input,
        StridedView::new(&contig, &out_layout),
    )?;
    Ok(contig)
}

/// Provider-owned implementation of [`FullReductionOps`] for ROCm/HIP.
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmFullReductionOps;

/// Prepared full reduction; the operation runs in `prepare` under ROCm's
/// synchronous execution model so `dispatch` is a no-op.
#[derive(Clone, Copy, Debug)]
pub struct RocmPreparedFullReduction;

impl<T> FullReductionOps<RocmDevice, T> for RocmFullReductionOps
where
    T: DialectScalar<HipC> + Pod + Send + Sync,
{
    type Dialect = HipC;
    type Prepared<const N: usize> = RocmPreparedFullReduction;

    fn prepare_reduce_full<Op, const N: usize>(
        &self,
        device: &RocmDevice,
        input: StridedView<'_, RocmBuffer<T>, N>,
        output: StridedView<'_, RocmBuffer<T>, 1>,
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
        device.download(&result, &mut host_val)?;
        device.write_sub_buffer(output.buffer, output.layout.offset, &host_val)?;

        Ok(RocmPreparedFullReduction)
    }

    fn dispatch_full<const N: usize>(
        &self,
        _device: &RocmDevice,
        _prepared: &Self::Prepared<N>,
    ) -> Result<()> {
        Ok(())
    }
}
