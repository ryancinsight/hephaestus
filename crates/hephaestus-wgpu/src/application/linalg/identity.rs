//! On-device identity-matrix construction.

use super::dispatch::{encode_dispatch_2d, map_layout};
use super::layout::{GpuMatrixLayout, IdentityKernel};
use super::scalars::MatrixIdentityScalar;
use crate::application::pipeline::try_cached_pipeline;
use crate::application::strided::{map_layout_err, to_u32};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use eunomia::Pod;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use leto::Layout;
use std::any::TypeId;

pub(crate) fn identity_shader_source<T: MatrixIdentityScalar>() -> String {
    format!(
        r#"
struct MatrixLayout {{
    shape: vec2<u32>,
    strides: vec2<i32>,
    offset: u32,
}}

@group(0) @binding(0) var<storage, read_write> output: array<{ty}>;
@group(0) @binding(1) var<uniform> matrix_layout: MatrixLayout;
@group(0) @binding(2) var<uniform> zero_value: {ty};
@group(0) @binding(3) var<uniform> one_value: {ty};

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    if (id.x < matrix_layout.shape.y && id.y < matrix_layout.shape.x) {{
        let index = matrix_layout.offset
            + id.y * u32(matrix_layout.strides.x)
            + id.x * u32(matrix_layout.strides.y);
        output[index] = select(zero_value, one_value, id.x == id.y);
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
    )
}

pub(crate) fn identity_buffer_layout<T: Pod>(alignment: u64) -> Result<(u64, u64)> {
    let value_size = WgpuDevice::byte_size::<T>(1)?;
    let one_offset = value_size
        .checked_add(alignment - 1)
        .map(|bytes| (bytes / alignment) * alignment)
        .ok_or_else(|| HephaestusError::AllocationFailed {
            message: format!(
                "identity uniform byte size {value_size} cannot be aligned to {alignment} bytes"
            ),
        })?;
    let buffer_size =
        one_offset
            .checked_add(value_size)
            .ok_or_else(|| HephaestusError::AllocationFailed {
                message: format!(
                    "identity uniform byte size {value_size} at offset {one_offset} overflows u64"
                ),
            })?;
    Ok((one_offset, buffer_size))
}

pub(crate) fn device_identity<T>(device: &WgpuDevice, layout: &Layout<2>) -> Result<WgpuBuffer<T>>
where
    T: MatrixIdentityScalar,
{
    let len = layout.checked_size().map_err(map_layout_err)?;
    let output = device.alloc_uninitialized::<T>(len)?;
    if len == 0 {
        return Ok(output);
    }
    let metadata = map_layout(layout)?;
    let layout_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<GpuMatrixLayout>(1)?)?;

    let identity_value_size = WgpuDevice::byte_size::<T>(1)?;
    let identity_alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment)
        .max(wgpu::COPY_BUFFER_ALIGNMENT);
    let (one_offset, identity_buffer_size) = identity_buffer_layout::<T>(identity_alignment)?;
    let identity_buffer = device.get_uniform_buffer(identity_buffer_size)?;

    device
        .queue()
        .write_buffer(&layout_buffer, 0, eunomia::layout::bytes_of(&metadata));
    device
        .queue()
        .write_buffer(&identity_buffer, 0, eunomia::layout::bytes_of(&T::ZERO));
    device.queue().write_buffer(
        &identity_buffer,
        one_offset,
        eunomia::layout::bytes_of(&T::ONE),
    );
    let key = (TypeId::of::<IdentityKernel<T>>(), TypeId::of::<T>(), 16);
    let pipeline = try_cached_pipeline(device, key, "hephaestus-matrix-identity", || {
        identity_shader_source::<T>()
    })?;
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-matrix-identity"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: layout_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &identity_buffer,
                        offset: 0,
                        size: core::num::NonZeroU64::new(identity_value_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &identity_buffer,
                        offset: one_offset,
                        size: core::num::NonZeroU64::new(identity_value_size),
                    }),
                },
            ],
        });
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-matrix-identity"),
        });
    encode_dispatch_2d(
        &mut encoder,
        &pipeline,
        &bind_group,
        "hephaestus-matrix-identity",
        to_u32(layout.shape()[1].div_ceil(16), "identity workgroups x")?,
        to_u32(layout.shape()[0].div_ceil(16), "identity workgroups y")?,
    );
    device.queue().submit(Some(encoder.finish()));
    drop(layout_buffer);
    drop(identity_buffer);
    Ok(output)
}
