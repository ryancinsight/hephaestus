//! Matrix-multiplication dispatch.

use super::dispatch::{
    binary_op_bind_group, encode_dispatch_2d, upload_binary_op_layouts, validate_binary_op_operands,
};
use super::layout::MatmulKernel;
use super::scalars::MatmulZero;
use crate::application::pipeline::cached_pipeline;
use crate::application::strided::{StridedOperand, map_layout_err, to_u32};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use eunomia::Pod;
use hephaestus_core::{ComputeDevice, DialectScalar, HephaestusError, Result, Wgsl};
use leto::Layout;
use std::any::TypeId;

fn matmul_shader_source<T: MatmulZero>() -> String {
    format!(
        r#"
struct MatrixLayout {{
    shape: vec2<u32>,
    strides: vec2<i32>,
    offset: u32,
}}

@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read> b: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> c: array<{ty}>;
@group(0) @binding(3) var<uniform> a_layout: MatrixLayout;
@group(0) @binding(4) var<uniform> b_layout: MatrixLayout;
@group(0) @binding(5) var<uniform> c_layout: MatrixLayout;

var<workgroup> A_shared: array<array<{ty}, 16>, 16>;
var<workgroup> B_shared: array<array<{ty}, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    let m = a_layout.shape.x;
    let k = a_layout.shape.y;
    let n = b_layout.shape.y;

    let stride_a_row = a_layout.strides.x;
    let stride_a_col = a_layout.strides.y;
    let stride_b_row = b_layout.strides.x;
    let stride_b_col = b_layout.strides.y;

    var sum = {ty}({zero});
    let num_tiles = (k + 15u) / 16u;

    for (var tile_idx: u32 = 0u; tile_idx < num_tiles; tile_idx = tile_idx + 1u) {{
        // 1. Load A element into shared memory
        let col_a = tile_idx * 16u + local_col;
        if (row < m && col_a < k) {{
            let offset_a = i32(a_layout.offset) + i32(row) * stride_a_row + i32(col_a) * stride_a_col;
            A_shared[local_row][local_col] = a[offset_a];
        }} else {{
            A_shared[local_row][local_col] = {ty}({zero});
        }}

        // 2. Load B element into shared memory
        let row_b = tile_idx * 16u + local_row;
        if (row_b < k && col < n) {{
            let offset_b = i32(b_layout.offset) + i32(row_b) * stride_b_row + i32(col) * stride_b_col;
            B_shared[local_row][local_col] = b[offset_b];
        }} else {{
            B_shared[local_row][local_col] = {ty}({zero});
        }}

        // Synchronize to ensure all threads have finished loading the current tile
        workgroupBarrier();

        // 3. Accumulate product of the current tile
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {{
            sum = sum + A_shared[local_row][i] * B_shared[i][local_col];
        }}

        // Synchronize before loading the next tile
        workgroupBarrier();
    }}

    if (row < m && col < n) {{
        let stride_c_row = c_layout.strides.x;
        let stride_c_col = c_layout.strides.y;
        let offset_c = i32(c_layout.offset) + i32(row) * stride_c_row + i32(col) * stride_c_col;
        c[offset_c] = sum;
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        zero = T::WGSL_ZERO
    )
}

fn matmul_output_shape(lhs: &Layout<2>, rhs: &Layout<2>) -> Result<[usize; 2]> {
    let [rows, lhs_shared] = lhs.shape();
    let [rhs_shared, cols] = rhs.shape();
    if lhs_shared != rhs_shared {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matmul dimension mismatch: lhs {:?}, rhs {:?}",
                lhs.shape(),
                rhs.shape()
            ),
        });
    }
    Ok([rows, cols])
}

/// Perform matrix multiplication `out = lhs * rhs` on the GPU.
///
/// Output shape must conform to `[lhs.rows, rhs.cols]`, and output buffer
/// must not alias either input buffer.
pub fn matmul_into<T>(
    device: &WgpuDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
    out: StridedOperand<'_, T, 2>,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod + MatmulZero,
{
    let [rows, cols] = matmul_output_shape(lhs.layout, rhs.layout)?;
    let lhs_shared = lhs.layout.shape()[1];
    let [out_rows, out_cols] = out.layout.shape();

    if rows != out_rows || cols != out_cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matmul dimension mismatch: lhs {:?}, rhs {:?}, out {:?}",
                lhs.layout.shape(),
                rhs.layout.shape(),
                out.layout.shape()
            ),
        });
    }

    validate_binary_op_operands(&lhs, &rhs, &out, "matmul")?;

    if rows == 0 || cols == 0 || lhs_shared == 0 {
        return Ok(());
    }

    let layout_uniforms = upload_binary_op_layouts(device, lhs.layout, rhs.layout, out.layout)?;

    let key = (TypeId::of::<MatmulKernel<T>>(), TypeId::of::<T>(), 16);
    let pipeline = cached_pipeline(device, key, "hephaestus-matmul", || {
        matmul_shader_source::<T>()
    });

    let bind_group = binary_op_bind_group(
        device,
        &pipeline,
        "hephaestus-matmul",
        &lhs.buffer.buffer,
        &rhs.buffer.buffer,
        &out.buffer.buffer,
        &layout_uniforms,
    );

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-matmul"),
        });

    encode_dispatch_2d(
        &mut encoder,
        &pipeline,
        &bind_group,
        "hephaestus-matmul",
        to_u32(cols.div_ceil(16), "matmul workgroup_x")?,
        to_u32(rows.div_ceil(16), "matmul workgroup_y")?,
    );

    device.queue().submit(Some(encoder.finish()));

    Ok(())
}

/// Allocate and compute matrix multiplication `lhs * rhs` on the GPU.
///
/// The returned buffer has C-contiguous shape `[lhs.rows, rhs.cols]`.
pub fn matmul<T>(
    device: &WgpuDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod + MatmulZero,
{
    let shape = matmul_output_shape(lhs.layout, rhs.layout)?;
    let layout = Layout::c_contiguous(shape).map_err(map_layout_err)?;
    let out = device.alloc_uninitialized::<T>(layout.checked_size().map_err(map_layout_err)?)?;
    matmul_into(
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

fn batched_matmul_output_shape(lhs: &Layout<3>, rhs: &Layout<3>) -> Result<[usize; 3]> {
    let [lhs_batch, m, lhs_k] = lhs.shape();
    let [rhs_batch, rhs_k, n] = rhs.shape();
    let batch = lhs_batch.max(rhs_batch);
    let lhs_batches_ok = lhs_batch == batch || lhs_batch == 1;
    let rhs_batches_ok = rhs_batch == batch || rhs_batch == 1;
    if !lhs_batches_ok || !rhs_batches_ok || lhs_k != rhs_k {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "batched matmul shape mismatch: lhs {:?}, rhs {:?}",
                lhs.shape(),
                rhs.shape()
            ),
        });
    }
    Ok([batch, m, n])
}

/// Perform batched matrix multiplication `out[i] = lhs[i] * rhs[i]` on the GPU.
pub fn batched_matmul_into<T>(
    device: &WgpuDevice,
    lhs: StridedOperand<'_, T, 3>,
    rhs: StridedOperand<'_, T, 3>,
    out: StridedOperand<'_, T, 3>,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod + MatmulZero,
{
    let expected_shape = batched_matmul_output_shape(lhs.layout, rhs.layout)?;
    let [lhs_batch, m, lhs_k] = lhs.layout.shape();
    let [rhs_batch, rhs_k, n] = rhs.layout.shape();
    let [out_batch, out_m, out_n] = out.layout.shape();

    let batch = out_batch;
    let lhs_batches_ok = lhs_batch == batch || lhs_batch == 1;
    let rhs_batches_ok = rhs_batch == batch || rhs_batch == 1;
    if !lhs_batches_ok
        || !rhs_batches_ok
        || lhs_k != rhs_k
        || [out_batch, out_m, out_n] != expected_shape
    {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "batched matmul shape mismatch: lhs {:?}, rhs {:?}, out {:?}",
                lhs.layout.shape(),
                rhs.layout.shape(),
                out.layout.shape()
            ),
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

    if batch == 0 || m == 0 || n == 0 || lhs_k == 0 {
        return Ok(());
    }

    let lhs_batch_stride = if lhs_batch == 1 {
        0
    } else {
        lhs.layout.strides()[0]
    };
    let rhs_batch_stride = if rhs_batch == 1 {
        0
    } else {
        rhs.layout.strides()[0]
    };
    let out_batch_stride = out.layout.strides()[0];

    let key = (TypeId::of::<MatmulKernel<T>>(), TypeId::of::<T>(), 16);
    let pipeline = cached_pipeline(device, key, "hephaestus-matmul", || {
        matmul_shader_source::<T>()
    });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-batched-matmul"),
        });

    let mut uniform_guards = Vec::with_capacity(3 * batch);

    for b in 0..batch {
        let lhs_mat_layout = Layout::try_new(
            [m, lhs_k],
            [lhs.layout.strides()[1], lhs.layout.strides()[2]],
            (lhs.layout.offset() as isize + b as isize * lhs_batch_stride) as usize,
        )
        .expect("valid test layout");
        let rhs_mat_layout = Layout::try_new(
            [rhs_k, n],
            [rhs.layout.strides()[1], rhs.layout.strides()[2]],
            (rhs.layout.offset() as isize + b as isize * rhs_batch_stride) as usize,
        )
        .expect("valid test layout");
        let out_mat_layout = Layout::try_new(
            [out_m, out_n],
            [out.layout.strides()[1], out.layout.strides()[2]],
            (out.layout.offset() as isize + b as isize * out_batch_stride) as usize,
        )
        .expect("valid test layout");

        lhs_mat_layout
            .validate_storage_len(lhs.buffer.len)
            .map_err(map_layout_err)?;
        rhs_mat_layout
            .validate_storage_len(rhs.buffer.len)
            .map_err(map_layout_err)?;
        out_mat_layout
            .validate_storage_len(out.buffer.len)
            .map_err(map_layout_err)?;

        if out_mat_layout.has_zero_stride_aliasing() {
            return Err(HephaestusError::DispatchFailed {
                message: "matmul output layout must not contain zero-stride aliasing".to_string(),
            });
        }

        let layout_uniforms =
            upload_binary_op_layouts(device, &lhs_mat_layout, &rhs_mat_layout, &out_mat_layout)?;

        let bind_group = binary_op_bind_group(
            device,
            &pipeline,
            "hephaestus-matmul-batched",
            &lhs.buffer.buffer,
            &rhs.buffer.buffer,
            &out.buffer.buffer,
            &layout_uniforms,
        );

        encode_dispatch_2d(
            &mut encoder,
            &pipeline,
            &bind_group,
            "hephaestus-matmul-batched",
            to_u32(n.div_ceil(16), "batched_matmul workgroup_x")?,
            to_u32(m.div_ceil(16), "batched_matmul workgroup_y")?,
        );

        uniform_guards.extend(layout_uniforms);
    }

    device.queue().submit(Some(encoder.finish()));

    drop(uniform_guards);

    Ok(())
}

/// Allocate and compute batched matrix multiplication on the GPU.
///
/// Singleton batches broadcast to the other operand's batch count. The returned
/// buffer has C-contiguous shape `[batch, lhs.rows, rhs.cols]`.
pub fn batched_matmul<T>(
    device: &WgpuDevice,
    lhs: StridedOperand<'_, T, 3>,
    rhs: StridedOperand<'_, T, 3>,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod + MatmulZero,
{
    let shape = batched_matmul_output_shape(lhs.layout, rhs.layout)?;
    let layout = Layout::c_contiguous(shape).map_err(map_layout_err)?;
    let out = device.alloc_uninitialized::<T>(layout.checked_size().map_err(map_layout_err)?)?;
    batched_matmul_into(
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
