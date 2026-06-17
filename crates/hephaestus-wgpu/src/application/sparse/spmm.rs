//! Sparse-dense matrix product `C = A * B` on WGPU CSR buffers.

use super::GpuCsrMatrix;
use crate::application::linalg::AsGpuMatrixOperand;
use crate::application::linalg::MatmulZero;
use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::application::strided::map_layout_err;
use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use bytemuck::{Pod, Zeroable};
use core::marker::PhantomData;
use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpmmMeta {
    matrix_shape: [u32; 2],
    b_shape: [u32; 2],
    b_strides: [i32; 2],
    offsets: [u32; 2],
}

struct SparseSpmmKernel<T>(PhantomData<T>);

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds u32 range"),
    })
}

fn to_i32(value: isize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i32 range"),
    })
}

fn spmm_shader_source<T: MatmulZero>(width: BlockWidth) -> String {
    format!(
        r#"
struct SpmmMeta {{
    matrix_shape: vec2<u32>,
    b_shape: vec2<u32>,
    b_strides: vec2<i32>,
    offsets: vec2<u32>,
}}

@group(0) @binding(0) var<uniform> sparse_meta: SpmmMeta;
@group(0) @binding(1) var<storage, read> values: array<{ty}>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> b: array<{ty}>;
@group(0) @binding(4) var<storage, read_write> c: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let flat = gid.x;
    let rows = sparse_meta.matrix_shape.x;
    let cols = sparse_meta.b_shape.y;
    let len = rows * cols;
    if (flat >= len) {{
        return;
    }}

    let row = flat / cols;
    let col = flat - row * cols;
    let row_ptr = sparse_meta.offsets.y;
    let begin = indices[row_ptr + row];
    let end = indices[row_ptr + row + 1u];
    var acc = {ty}({zero});
    for (var idx = begin; idx < end; idx = idx + 1u) {{
        let b_row = indices[idx];
        let b_offset = i32(sparse_meta.offsets.x)
            + i32(b_row) * sparse_meta.b_strides.x
            + i32(col) * sparse_meta.b_strides.y;
        acc = acc + values[idx] * b[u32(b_offset)];
    }}
    c[flat] = acc;
}}
"#,
        ty = T::WGSL_TYPE,
        wg = width.get(),
        zero = T::WGSL_ZERO,
    )
}

/// Compute `C = A · B` into a pre-allocated output buffer `c`.
pub fn spmm_into<'a, T: WgslScalar + MatmulZero + Pod, B: AsGpuMatrixOperand<'a, T>>(
    device: &WgpuDevice,
    a: &GpuCsrMatrix<T>,
    b: &B,
    c: &mut WgpuBuffer<T>,
) -> Result<()> {
    let (nrows, ncols) = a.shape();
    let b_op = b.as_operand();
    let [b_rows, bcols] = b_op.layout.shape;

    if b_rows != ncols {
        return Err(HephaestusError::LengthMismatch {
            host_len: ncols,
            device_len: b_rows,
        });
    }

    let expected_c_len = nrows * bcols;
    if c.len() != expected_c_len {
        return Err(HephaestusError::LengthMismatch {
            host_len: expected_c_len,
            device_len: c.len(),
        });
    }
    if expected_c_len == 0 {
        return Ok(());
    }

    b_op.layout
        .validate_storage_len(b_op.buffer.len())
        .map_err(map_layout_err)?;
    let meta = SpmmMeta {
        matrix_shape: [
            to_u32(nrows, "CSR row count")?,
            to_u32(ncols, "CSR column count")?,
        ],
        b_shape: [
            to_u32(b_rows, "dense rhs row count")?,
            to_u32(bcols, "dense rhs column count")?,
        ],
        b_strides: [
            to_i32(b_op.layout.strides[0], "dense rhs row stride")?,
            to_i32(b_op.layout.strides[1], "dense rhs column stride")?,
        ],
        offsets: [
            to_u32(b_op.layout.offset, "dense rhs offset")?,
            to_u32(a.row_ptr_offset(), "CSR row pointer offset")?,
        ],
    };

    let width = BlockWidth::DEFAULT;
    let groups = workgroups(expected_c_len, width)?;
    let pipeline = cached_pipeline(
        device,
        (
            std::any::TypeId::of::<SparseSpmmKernel<T>>(),
            std::any::TypeId::of::<T>(),
            width.get(),
        ),
        "hephaestus-spmm",
        || spmm_shader_source::<T>(width),
    );
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<SpmmMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(&meta));
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-spmm"),
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
                    resource: b_op.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: c.buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-spmm"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-spmm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    device.recycle_uniform_buffer(meta_buffer);

    Ok(())
}

/// Compute `C = A · B`, allocating the result buffer.
pub fn spmm<'a, T: WgslScalar + MatmulZero + Pod, B: AsGpuMatrixOperand<'a, T>>(
    device: &WgpuDevice,
    a: &GpuCsrMatrix<T>,
    b: &B,
) -> Result<WgpuBuffer<T>> {
    let (nrows, _) = a.shape();
    let b_op = b.as_operand();
    let [_, bcols] = b_op.layout.shape;

    let mut c = device.alloc_zeroed::<T>(nrows * bcols)?;
    spmm_into(device, a, b, &mut c)?;
    Ok(c)
}
