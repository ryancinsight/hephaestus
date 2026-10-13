//! Shared binary-op dispatch helpers.

use super::layout::GpuMatrixLayout;
use crate::application::strided::{StridedOperand, map_layout_err, to_i32, to_u32};
use crate::infrastructure::device::WgpuDevice;
use crate::infrastructure::pool::PooledBuffer;
use hephaestus_core::{HephaestusError, Result};
use leto::Layout;

pub(crate) fn map_layout(layout: &Layout<2>) -> Result<GpuMatrixLayout> {
    Ok(GpuMatrixLayout {
        shape: [
            to_u32(layout.shape()[0], "dimension")?,
            to_u32(layout.shape()[1], "dimension")?,
        ],
        strides: [
            to_i32(layout.strides()[0], "stride")?,
            to_i32(layout.strides()[1], "stride")?,
        ],
        offset: to_u32(layout.offset(), "offset")?,
        _pad: [0; 3],
    })
}

#[inline]
pub(crate) fn upload_binary_op_layouts(
    device: &WgpuDevice,
    lhs: &Layout<2>,
    rhs: &Layout<2>,
    out: &Layout<2>,
) -> Result<[PooledBuffer; 3]> {
    let lhs_meta = map_layout(lhs)?;
    let rhs_meta = map_layout(rhs)?;
    let out_meta = map_layout(out)?;
    let layout_size = WgpuDevice::byte_size::<GpuMatrixLayout>(1)?;
    let lhs_layout = device.get_uniform_buffer(layout_size)?;
    let rhs_layout = device.get_uniform_buffer(layout_size)?;
    let out_layout = device.get_uniform_buffer(layout_size)?;

    device
        .queue()
        .write_buffer(&lhs_layout, 0, eunomia::layout::bytes_of(&lhs_meta));
    device
        .queue()
        .write_buffer(&rhs_layout, 0, eunomia::layout::bytes_of(&rhs_meta));
    device
        .queue()
        .write_buffer(&out_layout, 0, eunomia::layout::bytes_of(&out_meta));

    Ok([lhs_layout, rhs_layout, out_layout])
}

#[inline]
pub(crate) fn binary_op_bind_group(
    device: &WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    label: &'static str,
    lhs: &wgpu::Buffer,
    rhs: &wgpu::Buffer,
    out: &wgpu::Buffer,
    layouts: &[PooledBuffer; 3],
) -> wgpu::BindGroup {
    device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: layouts[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: layouts[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: layouts[2].as_entire_binding(),
                },
            ],
        })
}

#[inline]
pub(crate) fn encode_dispatch_2d(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    label: &'static str,
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

#[inline]
pub(crate) fn validate_binary_op_operands<T>(
    lhs: &StridedOperand<'_, T, 2>,
    rhs: &StridedOperand<'_, T, 2>,
    out: &StridedOperand<'_, T, 2>,
    op_name: &str,
) -> Result<()> {
    if lhs.buffer.aliases(out.buffer) || rhs.buffer.aliases(out.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "output buffer must not alias either input buffer".to_string(),
        });
    }

    lhs.layout
        .validate_storage_len(lhs.buffer.len)
        .map_err(map_layout_err)?;
    rhs.layout
        .validate_storage_len(rhs.buffer.len)
        .map_err(map_layout_err)?;
    out.layout
        .validate_storage_len(out.buffer.len)
        .map_err(map_layout_err)?;

    if out.layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{op_name} output layout must not contain zero-stride aliasing"),
        });
    }

    Ok(())
}
