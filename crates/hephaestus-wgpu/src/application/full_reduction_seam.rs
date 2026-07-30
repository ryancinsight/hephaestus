use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, DialectScalar, ElementwiseOps, FullReductionOps,
    HephaestusError, IdentityOp, IdentityToken, OpIdentity, Result, StridedView, Wgsl,
};
use leto::Layout;

use crate::application::elementwise_seam::WgpuElementwiseOps;
use crate::application::prepared::{
    checked_submit, device_owner, validate_buffer_owner, validate_device_owner,
};
use crate::application::reduction::{PreparedReduction, prepare_reduction_with_width};
use crate::application::strided::{map_layout_err, validate_out};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::{PipelineCache, WgpuDevice};

/// Provider-owned implementation of [`FullReductionOps`] over wgpu.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuFullReductionOps;

/// Prepared full reduction over a retained contiguous input handle.
pub struct PreparedFullReduction<T> {
    owner: PipelineCache,
    inner: PreparedReduction<T>,
    output_buffer: WgpuBuffer<T>,
    output_byte_offset: u64,
    _source: WgpuBuffer<T>,
}

fn contiguous_reduction_source<T, const N: usize>(
    device: &WgpuDevice,
    input: StridedView<'_, WgpuBuffer<T>, N>,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod + Send + Sync + 'static,
{
    let logical_len = input.layout.checked_size().map_err(map_layout_err)?;
    input
        .layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    if logical_len == 0 {
        return device.alloc_uninitialized(0);
    }
    if input.layout.is_c_contiguous() && logical_len == input.buffer.len {
        return Ok(input.buffer.clone());
    }

    let out_layout = Layout::c_contiguous(input.layout.shape).map_err(map_layout_err)?;
    let contig = device.alloc_uninitialized::<T>(logical_len)?;
    WgpuElementwiseOps.unary_into::<IdentityOp, N>(
        device,
        input,
        StridedView::new(&contig, &out_layout),
    )?;
    Ok(contig)
}

impl<T> FullReductionOps<WgpuDevice, T> for WgpuFullReductionOps
where
    T: DialectScalar<Wgsl> + Pod + Send + Sync + 'static,
{
    type Dialect = Wgsl;
    type Prepared<const N: usize> = PreparedFullReduction<T>;

    fn prepare_reduce_full<Op, const N: usize>(
        &self,
        device: &WgpuDevice,
        input: StridedView<'_, WgpuBuffer<T>, N>,
        output: StridedView<'_, WgpuBuffer<T>, 1>,
    ) -> Result<Self::Prepared<N>>
    where
        Op: CombineExpr<Self::Dialect>,
        T: OpIdentity<Op> + IdentityToken<Op, Self::Dialect>,
    {
        validate_buffer_owner(input.buffer, device, "full reduction input")?;
        validate_buffer_owner(output.buffer, device, "full reduction output")?;
        if validate_out(output.buffer, output.layout)? != 1 {
            return Err(HephaestusError::DispatchFailed {
                message: "full reduction output must have length 1".to_string(),
            });
        }
        if output.buffer.aliases(input.buffer) {
            return Err(HephaestusError::DispatchFailed {
                message: "full reduction output buffer must not alias input buffer".to_string(),
            });
        }

        let source = contiguous_reduction_source(device, input)?;
        let prepared = prepare_reduction_with_width::<Op, T>(device, &source, BlockWidth::DEFAULT)?;
        Ok(PreparedFullReduction {
            owner: device_owner(device),
            inner: prepared,
            output_buffer: output.buffer.clone(),
            output_byte_offset: WgpuDevice::byte_size::<T>(output.layout.offset)?,
            _source: source,
        })
    }

    fn dispatch_full<const N: usize>(
        &self,
        device: &WgpuDevice,
        prepared: &Self::Prepared<N>,
    ) -> Result<()> {
        validate_device_owner(&prepared.owner, device, "full reduction")?;

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-prepared-full-reduction"),
            });
        prepared.inner.encode(&mut encoder)?;
        let src = prepared.inner.output();
        encoder.copy_buffer_to_buffer(
            src,
            0,
            &prepared.output_buffer,
            prepared.output_byte_offset,
            WgpuDevice::byte_size::<T>(1)?,
        );
        checked_submit(device, "hephaestus-prepared-full-reduction", encoder)
    }
}
