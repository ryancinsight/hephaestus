use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{BlockWidth, DialectScalar, HephaestusError, Result, StridedView, Wgsl};

use crate::application::canary;
use crate::application::pipeline::{try_cached_pipeline, workgroups};
use crate::application::prepared::{checked_bind_group, device_owner};
use crate::application::strided::{map_layout_err, pad_shape, pad_strides, validate_out};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

use super::shaders::binary_shader;
use super::types::PreparedElementwise;
use super::types::SeamBinaryKernel;
use super::validate::validate_rank;

pub(crate) fn prepare_binary_inner<Op, T, const N: usize>(
    device: &WgpuDevice,
    lhs: StridedView<'_, WgpuBuffer<T>, N>,
    rhs: StridedView<'_, WgpuBuffer<T>, N>,
    output: StridedView<'_, WgpuBuffer<T>, N>,
    expr: &'static str,
) -> Result<PreparedElementwise>
where
    Op: 'static,
    T: DialectScalar<Wgsl> + Pod + 'static,
{
    canary::require_template_safe::<T>(device, expr, "prepared binary elementwise")?;
    validate_rank::<N>()?;

    let out_layout = output.layout;
    let a_layout = lhs
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    let b_layout = rhs
        .layout
        .broadcast(out_layout.shape())
        .map_err(map_layout_err)?;
    a_layout
        .validate_storage_len(lhs.buffer.len)
        .map_err(map_layout_err)?;
    b_layout
        .validate_storage_len(rhs.buffer.len)
        .map_err(map_layout_err)?;
    if output.buffer.aliases(lhs.buffer) || output.buffer.aliases(rhs.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias any input buffer".to_string(),
        });
    }
    let len = validate_out(output.buffer, out_layout)?;
    if len == 0 {
        return Ok(PreparedElementwise {
            owner: device_owner(device),
            pipeline: None,
            bind_group: None,
            groups: 0,
            _meta_buffer: None,
            _scalar_buffer: None,
        });
    }

    let meta = crate::application::strided::StridedMeta {
        shape: pad_shape(out_layout.shape())?,
        a_strides: pad_strides(a_layout.strides())?,
        b_strides: pad_strides(b_layout.strides())?,
        out_strides: pad_strides(out_layout.strides())?,
        offsets: [
            crate::application::strided::to_u32(a_layout.offset(), "input offset")?,
            crate::application::strided::to_u32(b_layout.offset(), "input offset")?,
            crate::application::strided::to_u32(out_layout.offset(), "output offset")?,
            crate::application::strided::to_u32(len, "dispatch size")?,
        ],
    };

    let groups = workgroups(len, BlockWidth::DEFAULT)?;
    let operation = TypeId::of::<SeamBinaryKernel<Op>>();
    let pipeline = try_cached_pipeline(
        device,
        (operation, TypeId::of::<T>(), BlockWidth::DEFAULT.get()),
        "hephaestus-seam-binary",
        || binary_shader::<T>(BlockWidth::DEFAULT, expr),
    )?;

    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<
        crate::application::strided::StridedMeta,
    >(1)?)?;

    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&meta));

    let bind_group = checked_bind_group(
        device,
        &pipeline,
        "hephaestus-seam-binary",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: meta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lhs.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rhs.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.buffer.as_entire_binding(),
            },
        ],
    )?;

    Ok(PreparedElementwise {
        owner: device_owner(device),
        pipeline: Some(pipeline),
        bind_group: Some(bind_group),
        groups,
        _meta_buffer: Some(meta_buffer),
        _scalar_buffer: None,
    })
}
