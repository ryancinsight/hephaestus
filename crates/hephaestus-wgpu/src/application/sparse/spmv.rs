//! Sparse matrix-vector product `y = A * x` on WGPU CSR buffers.

use super::GpuCsrMatrix;
use crate::application::linalg::MatmulZero;
use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use crate::UniformBufferGuard;
use bytemuck::{Pod, Zeroable};
use core::marker::PhantomData;
use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpmvMeta {
    offsets: [u32; 4],
}

struct SparseSpmvKernel<T>(PhantomData<T>);

fn spmv_shader_source<T: MatmulZero>(width: BlockWidth) -> String {
    format!(
        r#"
struct SpmvMeta {{
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<uniform> sparse_meta: SpmvMeta;
@group(0) @binding(1) var<storage, read> values: array<{ty}>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<{ty}>;
@group(0) @binding(4) var<storage, read_write> y: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    if (row >= arrayLength(&y)) {{
        return;
    }}

    let row_ptr = sparse_meta.offsets.x;
    let begin = indices[row_ptr + row];
    let end = indices[row_ptr + row + 1u];
    var acc = {ty}({zero});
    for (var idx = begin; idx < end; idx = idx + 1u) {{
        acc = acc + values[idx] * x[indices[idx]];
    }}
    y[row] = acc;
}}
"#,
        ty = T::WGSL_TYPE,
        wg = width.get(),
        zero = T::WGSL_ZERO,
    )
}

/// Compute `y = A · x` into a pre-allocated output buffer `y` (length `nrows`).
pub fn spmv_into<T: WgslScalar + MatmulZero + Pod>(
    device: &WgpuDevice,
    a: &GpuCsrMatrix<T>,
    x: &WgpuBuffer<T>,
    y: &mut WgpuBuffer<T>,
) -> Result<()> {
    let (nrows, ncols) = a.shape();
    if x.len() != ncols {
        return Err(HephaestusError::LengthMismatch {
            host_len: ncols,
            device_len: x.len(),
        });
    }
    if y.len() != nrows {
        return Err(HephaestusError::LengthMismatch {
            host_len: nrows,
            device_len: y.len(),
        });
    }
    if nrows == 0 {
        return Ok(());
    }

    let width = BlockWidth::DEFAULT;
    let groups = workgroups(nrows, width)?;
    let meta = SpmvMeta {
        offsets: [
            u32::try_from(a.row_ptr_offset()).map_err(|_| HephaestusError::DispatchFailed {
                message: format!(
                    "CSR row pointer offset {} exceeds u32 range",
                    a.row_ptr_offset()
                ),
            })?,
            0,
            0,
            0,
        ],
    };
    let pipeline = cached_pipeline(
        device,
        (
            std::any::TypeId::of::<SparseSpmvKernel<T>>(),
            std::any::TypeId::of::<T>(),
            width.get(),
        ),
        "hephaestus-spmv",
        || spmv_shader_source::<T>(width),
    );
    let raw_meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<SpmvMeta>(1)?)?;
    let meta_buffer = UniformBufferGuard::new(device.clone(), raw_meta_buffer);
    device
        .queue()
        .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(&meta));
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-spmv"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.values().buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: a.indices().buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: y.buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-spmv"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-spmv"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(())
}

/// Compute `y = A · x`, allocating the result buffer.
pub fn spmv<T: WgslScalar + MatmulZero + Pod>(
    device: &WgpuDevice,
    a: &GpuCsrMatrix<T>,
    x: &WgpuBuffer<T>,
) -> Result<WgpuBuffer<T>> {
    let (nrows, _) = a.shape();
    let mut y = device.alloc_zeroed::<T>(nrows)?;
    spmv_into(device, a, x, &mut y)?;
    Ok(y)
}
