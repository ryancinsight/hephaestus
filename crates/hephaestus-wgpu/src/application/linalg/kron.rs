//! Kronecker-product dispatch.

use super::dispatch::{
    binary_op_bind_group, encode_dispatch_2d, upload_binary_op_layouts, validate_binary_op_operands,
};
use super::layout::KronKernel;
use crate::application::pipeline::cached_pipeline;
use crate::application::strided::{StridedOperand, map_layout_err, to_u32};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use eunomia::Pod;
use hephaestus_core::{ComputeDevice, DialectScalar, HephaestusError, Result, Wgsl};
use leto::Layout;
use std::any::TypeId;

fn kron_shader_source<T: DialectScalar<Wgsl>>() -> String {
    format!(
        r#"
struct MatrixLayout {{
    shape: vec2<u32>,
    strides: vec2<i32>,
    offset: u32,
}}

@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read> b: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;
@group(0) @binding(3) var<uniform> a_layout: MatrixLayout;
@group(0) @binding(4) var<uniform> b_layout: MatrixLayout;
@group(0) @binding(5) var<uniform> out_layout: MatrixLayout;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let out_col = gid.x;
    let out_row = gid.y;
    let b_rows = b_layout.shape.x;
    let b_cols = b_layout.shape.y;
    let rows = a_layout.shape.x * b_rows;
    let cols = a_layout.shape.y * b_cols;

    if (out_row >= rows || out_col >= cols) {{
        return;
    }}

    let a_row = out_row / b_rows;
    let a_col = out_col / b_cols;
    let b_row = out_row % b_rows;
    let b_col = out_col % b_cols;

    let a_offset = i32(a_layout.offset)
        + i32(a_row) * a_layout.strides.x
        + i32(a_col) * a_layout.strides.y;
    let b_offset = i32(b_layout.offset)
        + i32(b_row) * b_layout.strides.x
        + i32(b_col) * b_layout.strides.y;
    let out_offset = i32(out_layout.offset)
        + i32(out_row) * out_layout.strides.x
        + i32(out_col) * out_layout.strides.y;

    out[u32(out_offset)] = a[u32(a_offset)] * b[u32(b_offset)];
}}
"#,
        ty = T::TYPE_TOKEN
    )
}

fn kron_output_shape(lhs: &Layout<2>, rhs: &Layout<2>) -> Result<[usize; 2]> {
    let [lhs_rows, lhs_cols] = lhs.shape();
    let [rhs_rows, rhs_cols] = rhs.shape();
    let rows = lhs_rows
        .checked_mul(rhs_rows)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("Kronecker row count overflows usize: {lhs_rows} * {rhs_rows}"),
        })?;
    let cols = lhs_cols
        .checked_mul(rhs_cols)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("Kronecker column count overflows usize: {lhs_cols} * {rhs_cols}"),
        })?;
    Ok([rows, cols])
}

/// Perform the Kronecker product `out = lhs ⊗ rhs` on the GPU.
///
/// For `lhs` with shape `[m, n]` and `rhs` with shape `[p, q]`, the output
/// shape must be `[m * p, n * q]`.
pub fn kron_into<T>(
    device: &WgpuDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
    out: StridedOperand<'_, T, 2>,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    let [expected_rows, expected_cols] = kron_output_shape(lhs.layout, rhs.layout)?;

    if out.layout.shape() != [expected_rows, expected_cols] {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "Kronecker output shape mismatch: lhs {:?}, rhs {:?}, out {:?}",
                lhs.layout.shape(),
                rhs.layout.shape(),
                out.layout.shape()
            ),
        });
    }

    validate_binary_op_operands(&lhs, &rhs, &out, "Kronecker")?;

    if expected_rows == 0 || expected_cols == 0 {
        return Ok(());
    }

    let layout_uniforms = upload_binary_op_layouts(device, lhs.layout, rhs.layout, out.layout)?;

    let key = (TypeId::of::<KronKernel<T>>(), TypeId::of::<T>(), 16);
    let pipeline = cached_pipeline(device, key, "hephaestus-kron", || kron_shader_source::<T>());

    let bind_group = binary_op_bind_group(
        device,
        &pipeline,
        "hephaestus-kron",
        &lhs.buffer.buffer,
        &rhs.buffer.buffer,
        &out.buffer.buffer,
        &layout_uniforms,
    );

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-kron"),
        });
    encode_dispatch_2d(
        &mut encoder,
        &pipeline,
        &bind_group,
        "hephaestus-kron",
        to_u32(expected_cols.div_ceil(16), "Kronecker workgroups x")?,
        to_u32(expected_rows.div_ceil(16), "Kronecker workgroups y")?,
    );
    device.queue().submit(Some(encoder.finish()));

    Ok(())
}

/// Allocate and compute the Kronecker product `lhs ⊗ rhs` on the GPU.
///
/// For `lhs` with shape `[m, n]` and `rhs` with shape `[p, q]`, the returned
/// buffer has C-contiguous shape `[m * p, n * q]`.
pub fn kron<T>(
    device: &WgpuDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod,
{
    let shape = kron_output_shape(lhs.layout, rhs.layout)?;
    let layout = Layout::c_contiguous(shape).map_err(map_layout_err)?;
    let out = device.alloc_uninitialized::<T>(layout.checked_size().map_err(map_layout_err)?)?;
    kron_into(
        device,
        lhs,
        rhs,
        StridedOperand {
            buffer: &out,
            layout: &layout,
        },
    )?;
    Ok(out)
}
