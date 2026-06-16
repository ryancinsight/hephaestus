//! GPU-resident linear algebra operations.
//!
//! Provides optimized matrix multiplication, matrix functions, vector dot
//! products, matrix trace, and vector/matrix norms (L1, L2, Max) mapped to GPU
//! compute dispatches.

use bytemuck::{Pod, Zeroable};
use core::marker::PhantomData;
use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError, Result};
use leto::Layout;
use std::any::TypeId;

use crate::application::elementwise::{unary_elementwise_into, AbsOp, MulOp, SqrtOp};
use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::application::reduction::{reduction, MaxOp, ReductionIdentity, ReductionWgslOp, SumOp};
use crate::application::strided::{
    binary_elementwise_strided_into, map_layout_err, pad_shape, pad_strides,
    unary_elementwise_strided_into, StridedMeta, StridedOperand, WGSL_DECODE, WGSL_META,
};
use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Helper trait to substitute the correct zero literal in WGSL for different scalar types.
pub trait MatmulZero: WgslScalar {
    /// The WGSL zero literal (e.g. `"0.0"`, `"0u"`, `"0"`).
    const WGSL_ZERO: &'static str;
}

/// WGPU scalar whose host identity values support matrix-power initialization.
pub trait MatrixIdentityScalar: MatmulZero + Pod {
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
}

impl MatmulZero for f32 {
    const WGSL_ZERO: &'static str = "0.0";
}

impl MatrixIdentityScalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl MatmulZero for u32 {
    const WGSL_ZERO: &'static str = "0u";
}

impl MatrixIdentityScalar for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl MatmulZero for i32 {
    const WGSL_ZERO: &'static str = "0";
}

impl MatrixIdentityScalar for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

/// WGPU scalar whose shader type supports the real-valued square root needed
/// to finish an L2 / Frobenius norm.
///
/// WGPU's portable scalar surface here exposes `f32`, `u32`, and `i32`.
/// Leto's `norm_l2` contract is real-valued, so Hephaestus only implements
/// this marker for the real WGSL scalar currently available through
/// [`WgslScalar`].
pub trait L2NormScalar: WgslScalar + Pod + ReductionIdentity<SumOp> {}

impl L2NormScalar for f32 {}

/// Packed layout metadata matching the WGSL `MatrixLayout` uniform.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMatrixLayout {
    shape: [u32; 2],
    strides: [i32; 2],
    offset: u32,
    _pad: [u32; 3], // pad to 32 bytes (multiple of 16)
}

struct MatmulKernel<T>(PhantomData<T>);
struct KronKernel<T>(PhantomData<T>);
struct MapReductionKernel<Op>(PhantomData<Op>);

trait MapReductionWgslOp: Copy + Send + Sync + 'static {
    type ReduceOp: ReductionWgslOp;
    const WGSL_MAP_EXPR: &'static str;
}

#[derive(Clone, Copy, Debug, Default)]
struct TraceOp;

#[derive(Clone, Copy, Debug, Default)]
struct NormL1Op;

impl MapReductionWgslOp for TraceOp {
    type ReduceOp = SumOp;
    const WGSL_MAP_EXPR: &'static str = "lhs";
}

impl MapReductionWgslOp for NormL1Op {
    type ReduceOp = SumOp;
    const WGSL_MAP_EXPR: &'static str = "abs(lhs)";
}

#[inline]
fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds u32 range"),
    })
}

#[inline]
fn to_i32(value: isize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i32 range"),
    })
}

fn map_layout(layout: &Layout<2>) -> Result<GpuMatrixLayout> {
    Ok(GpuMatrixLayout {
        shape: [
            to_u32(layout.shape[0], "dimension")?,
            to_u32(layout.shape[1], "dimension")?,
        ],
        strides: [
            to_i32(layout.strides[0], "stride")?,
            to_i32(layout.strides[1], "stride")?,
        ],
        offset: to_u32(layout.offset, "offset")?,
        _pad: [0; 3],
    })
}

fn map_reduction_shader_source<Op, T>(width: BlockWidth) -> String
where
    Op: MapReductionWgslOp,
    T: WgslScalar + ReductionIdentity<Op::ReduceOp>,
{
    format!(
        r#"{meta}
@group(0) @binding(0) var<uniform> lmeta: Meta;
@group(0) @binding(1) var<storage, read> a: array<{ty}>;
@group(0) @binding(2) var<storage, read> b: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> out: array<{ty}>;

var<workgroup> shared_data: array<{ty}, {wg}>;

@compute @workgroup_size({wg})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let i = global_id.x;
    if (i < lmeta.offsets.w) {{
{decode}        let lhs = a[u32(a_off)];
        let rhs = b[u32(b_off)];
        shared_data[local_id.x] = {map_expr};
    }} else {{
        shared_data[local_id.x] = {identity};
    }}

    workgroupBarrier();

    for (var stride = {wg}u / 2u; stride > 0u; stride = stride / 2u) {{
        if (local_id.x < stride) {{
            let lhs = shared_data[local_id.x];
            let rhs = shared_data[local_id.x + stride];
            shared_data[local_id.x] = {reduce_expr};
        }}
        workgroupBarrier();
    }}

    if (local_id.x == 0u) {{
        out[workgroup_id.x] = shared_data[0];
    }}
}}
"#,
        meta = WGSL_META,
        ty = T::WGSL_TYPE,
        wg = width.get(),
        decode = WGSL_DECODE,
        identity = T::WGSL_IDENTITY,
        map_expr = Op::WGSL_MAP_EXPR,
        reduce_expr = <Op::ReduceOp as ReductionWgslOp>::WGSL_EXPR,
    )
}

fn map_reduction_first_pass<Op, T, const N: usize>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    Op: MapReductionWgslOp,
    T: WgslScalar + Pod + ReductionIdentity<Op::ReduceOp>,
{
    let len = a.layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return device.upload(&[T::IDENTITY]);
    }

    let b_layout = b.layout.broadcast(a.layout.shape).map_err(map_layout_err)?;
    a.layout
        .validate_storage_len(a.buffer.len)
        .map_err(map_layout_err)?;
    b_layout
        .validate_storage_len(b.buffer.len)
        .map_err(map_layout_err)?;

    let groups = workgroups(len, width)? as usize;
    let out = device.alloc_zeroed::<T>(groups)?;

    let meta = StridedMeta {
        shape: pad_shape(a.layout.shape)?,
        a_strides: pad_strides(a.layout.strides)?,
        b_strides: pad_strides(b_layout.strides)?,
        out_strides: [1, 1, 1, 1],
        offsets: [
            to_u32(a.layout.offset, "input offset")?,
            to_u32(b_layout.offset, "input offset")?,
            0,
            to_u32(len, "dispatch size")?,
        ],
    };

    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<MapReductionKernel<Op>>(),
            TypeId::of::<T>(),
            width.get(),
        ),
        "hephaestus-map-reduction",
        || map_reduction_shader_source::<Op, T>(width),
    );

    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<StridedMeta>(1)?)?;
    device
        .queue()
        .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(&meta));

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-map-reduction"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-map-reduction"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-map-reduction"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups as u32, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    device.recycle_uniform_buffer(meta_buffer);

    Ok(out)
}

fn unary_map_reduction<Op, T, const N: usize>(
    device: &WgpuDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<WgpuBuffer<T>>
where
    Op: MapReductionWgslOp,
    T: WgslScalar + Pod + ReductionIdentity<Op::ReduceOp>,
{
    let partial = map_reduction_first_pass::<Op, T, N>(device, view, view, BlockWidth::DEFAULT)?;
    if partial.len == 1 {
        return Ok(partial);
    }
    reduction::<Op::ReduceOp, T>(device, &partial)
}

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
        ty = T::WGSL_TYPE,
        zero = T::WGSL_ZERO
    )
}

fn kron_shader_source<T: WgslScalar>() -> String {
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
        ty = T::WGSL_TYPE
    )
}

/// Perform the Kronecker product `out = lhs ⊗ rhs` on the GPU.
///
/// For `lhs` with shape `[m, n]` and `rhs` with shape `[p, q]`, the output
/// shape must be `[m * p, n * q]`.
pub fn kron<T>(
    device: &WgpuDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
    out: StridedOperand<'_, T, 2>,
) -> Result<()>
where
    T: WgslScalar + Pod,
{
    let [lhs_rows, lhs_cols] = lhs.layout.shape;
    let [rhs_rows, rhs_cols] = rhs.layout.shape;
    let expected_rows =
        lhs_rows
            .checked_mul(rhs_rows)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("Kronecker row count overflows usize: {lhs_rows} * {rhs_rows}"),
            })?;
    let expected_cols =
        lhs_cols
            .checked_mul(rhs_cols)
            .ok_or_else(|| HephaestusError::DispatchFailed {
                message: format!("Kronecker column count overflows usize: {lhs_cols} * {rhs_cols}"),
            })?;

    if out.layout.shape != [expected_rows, expected_cols] {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "Kronecker output shape mismatch: lhs {:?}, rhs {:?}, out {:?}",
                lhs.layout.shape, rhs.layout.shape, out.layout.shape
            ),
        });
    }

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
            message: "Kronecker output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    if expected_rows == 0 || expected_cols == 0 {
        return Ok(());
    }

    let a_meta = map_layout(lhs.layout)?;
    let b_meta = map_layout(rhs.layout)?;
    let out_meta = map_layout(out.layout)?;

    let size = WgpuDevice::byte_size::<GpuMatrixLayout>(1)?;
    let a_layout_buf = device.get_uniform_buffer(size)?;
    let b_layout_buf = device.get_uniform_buffer(size)?;
    let out_layout_buf = device.get_uniform_buffer(size)?;

    device
        .queue()
        .write_buffer(&a_layout_buf, 0, bytemuck::bytes_of(&a_meta));
    device
        .queue()
        .write_buffer(&b_layout_buf, 0, bytemuck::bytes_of(&b_meta));
    device
        .queue()
        .write_buffer(&out_layout_buf, 0, bytemuck::bytes_of(&out_meta));

    let key = (TypeId::of::<KronKernel<T>>(), TypeId::of::<T>(), 16);
    let pipeline = cached_pipeline(device, key, "hephaestus-kron", || kron_shader_source::<T>());

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-kron"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: a_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_layout_buf.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-kron"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-kron"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            to_u32(expected_cols.div_ceil(16), "Kronecker workgroups x")?,
            to_u32(expected_rows.div_ceil(16), "Kronecker workgroups y")?,
            1,
        );
    }
    device.queue().submit(Some(encoder.finish()));

    device.recycle_uniform_buffer(a_layout_buf);
    device.recycle_uniform_buffer(b_layout_buf);
    device.recycle_uniform_buffer(out_layout_buf);

    Ok(())
}

/// Perform matrix multiplication `out = lhs * rhs` on the GPU.
///
/// Output shape must conform to `[lhs.rows, rhs.cols]`, and output buffer
/// must not alias either input buffer.
pub fn matmul<T>(
    device: &WgpuDevice,
    lhs: StridedOperand<'_, T, 2>,
    rhs: StridedOperand<'_, T, 2>,
    out: StridedOperand<'_, T, 2>,
) -> Result<()>
where
    T: WgslScalar + Pod + MatmulZero,
{
    let [rows, lhs_shared] = lhs.layout.shape;
    let [rhs_shared, cols] = rhs.layout.shape;
    let [out_rows, out_cols] = out.layout.shape;

    if lhs_shared != rhs_shared || rows != out_rows || cols != out_cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matmul dimension mismatch: lhs {:?}, rhs {:?}, out {:?}",
                lhs.layout.shape, rhs.layout.shape, out.layout.shape
            ),
        });
    }

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
            message: "matmul output layout must not contain zero-stride aliasing".to_string(),
        });
    }

    if rows == 0 || cols == 0 || lhs_shared == 0 {
        return Ok(());
    }

    let a_meta = map_layout(lhs.layout)?;
    let b_meta = map_layout(rhs.layout)?;
    let c_meta = map_layout(out.layout)?;

    let size = std::mem::size_of::<GpuMatrixLayout>() as u64;
    let a_layout_buf = device.get_uniform_buffer(size)?;
    let b_layout_buf = device.get_uniform_buffer(size)?;
    let c_layout_buf = device.get_uniform_buffer(size)?;

    device
        .queue()
        .write_buffer(&a_layout_buf, 0, bytemuck::bytes_of(&a_meta));
    device
        .queue()
        .write_buffer(&b_layout_buf, 0, bytemuck::bytes_of(&b_meta));
    device
        .queue()
        .write_buffer(&c_layout_buf, 0, bytemuck::bytes_of(&c_meta));

    let key = (TypeId::of::<MatmulKernel<T>>(), TypeId::of::<T>(), 16);
    let pipeline = cached_pipeline(device, key, "hephaestus-matmul", || {
        matmul_shader_source::<T>()
    });

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-matmul"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: a_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_layout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: c_layout_buf.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-matmul"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-matmul"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups_x = cols.div_ceil(16);
        let workgroups_y = rows.div_ceil(16);
        pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
    }

    device.queue().submit(Some(encoder.finish()));

    device.recycle_uniform_buffer(a_layout_buf);
    device.recycle_uniform_buffer(b_layout_buf);
    device.recycle_uniform_buffer(c_layout_buf);

    Ok(())
}

/// Perform batched matrix multiplication `out[i] = lhs[i] * rhs[i]` on the GPU.
pub fn batched_matmul<T>(
    device: &WgpuDevice,
    lhs: StridedOperand<'_, T, 3>,
    rhs: StridedOperand<'_, T, 3>,
    out: StridedOperand<'_, T, 3>,
) -> Result<()>
where
    T: WgslScalar + Pod + MatmulZero,
{
    let [lhs_batch, m, lhs_k] = lhs.layout.shape;
    let [rhs_batch, rhs_k, n] = rhs.layout.shape;
    let [out_batch, out_m, out_n] = out.layout.shape;

    let batch = out_batch;
    let lhs_batches_ok = lhs_batch == batch || lhs_batch == 1;
    let rhs_batches_ok = rhs_batch == batch || rhs_batch == 1;
    if !lhs_batches_ok || !rhs_batches_ok || lhs_k != rhs_k || m != out_m || n != out_n {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "batched matmul shape mismatch: lhs {:?}, rhs {:?}, out {:?}",
                lhs.layout.shape, rhs.layout.shape, out.layout.shape
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

    let lhs_batch_stride = if lhs_batch == 1 {
        0
    } else {
        lhs.layout.strides[0]
    };
    let rhs_batch_stride = if rhs_batch == 1 {
        0
    } else {
        rhs.layout.strides[0]
    };
    let out_batch_stride = out.layout.strides[0];

    for b in 0..batch {
        let lhs_mat_layout = Layout::new(
            [m, lhs_k],
            [lhs.layout.strides[1], lhs.layout.strides[2]],
            (lhs.layout.offset as isize + b as isize * lhs_batch_stride) as usize,
        );
        let rhs_mat_layout = Layout::new(
            [rhs_k, n],
            [rhs.layout.strides[1], rhs.layout.strides[2]],
            (rhs.layout.offset as isize + b as isize * rhs_batch_stride) as usize,
        );
        let out_mat_layout = Layout::new(
            [out_m, out_n],
            [out.layout.strides[1], out.layout.strides[2]],
            (out.layout.offset as isize + b as isize * out_batch_stride) as usize,
        );

        let lhs_operand = StridedOperand {
            buffer: lhs.buffer,
            layout: &lhs_mat_layout,
        };
        let rhs_operand = StridedOperand {
            buffer: rhs.buffer,
            layout: &rhs_mat_layout,
        };
        let out_operand = StridedOperand {
            buffer: out.buffer,
            layout: &out_mat_layout,
        };

        matmul(device, lhs_operand, rhs_operand, out_operand)?;
    }

    Ok(())
}

fn identity_matrix<T: MatrixIdentityScalar>(n: usize) -> Vec<T> {
    let mut values = vec![T::ZERO; n * n];
    for i in 0..n {
        values[i * n + i] = T::ONE;
    }
    values
}

/// Raise a square matrix to a non-negative integer power on the GPU.
///
/// The algorithm is exponentiation by squaring, matching Leto's `matpow`
/// contract: `A^0` is the identity matrix and non-square inputs are rejected.
/// Matrix products are dispatched through [`matmul`]; the host controls only
/// the exponent bits and buffer rotation.
pub fn matpow<T>(
    device: &WgpuDevice,
    matrix: StridedOperand<'_, T, 2>,
    exponent: u32,
) -> Result<WgpuBuffer<T>>
where
    T: WgslScalar + MatrixIdentityScalar,
{
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "matpow requires a square matrix, got shape {:?}",
                matrix.layout.shape
            ),
        });
    }

    matrix
        .layout
        .validate_storage_len(matrix.buffer.len)
        .map_err(map_layout_err)?;

    let layout = Layout::c_contiguous([rows, rows]).map_err(map_layout_err)?;
    let mut result = device.upload(&identity_matrix::<T>(rows))?;
    if exponent == 0 {
        return Ok(result);
    }

    let mut base = device.alloc_zeroed::<T>(rows * rows)?;
    unary_elementwise_strided_into::<crate::application::elementwise::IdentityOp, T, 2>(
        device,
        matrix,
        StridedOperand {
            buffer: &base,
            layout: &layout,
        },
        BlockWidth::DEFAULT,
    )?;

    let mut result_scratch = device.alloc_zeroed::<T>(rows * rows)?;
    let mut base_scratch = device.alloc_zeroed::<T>(rows * rows)?;
    let mut remaining = exponent;

    loop {
        if remaining & 1 == 1 {
            matmul(
                device,
                StridedOperand {
                    buffer: &result,
                    layout: &layout,
                },
                StridedOperand {
                    buffer: &base,
                    layout: &layout,
                },
                StridedOperand {
                    buffer: &result_scratch,
                    layout: &layout,
                },
            )?;
            core::mem::swap(&mut result, &mut result_scratch);
        }

        remaining >>= 1;
        if remaining == 0 {
            break;
        }

        matmul(
            device,
            StridedOperand {
                buffer: &base,
                layout: &layout,
            },
            StridedOperand {
                buffer: &base,
                layout: &layout,
            },
            StridedOperand {
                buffer: &base_scratch,
                layout: &layout,
            },
        )?;
        core::mem::swap(&mut base, &mut base_scratch);
    }

    Ok(result)
}

/// Compute the vector dot product `Σᵢ a[i] * b[i]` on the GPU.
pub fn dot<T>(
    device: &WgpuDevice,
    a: StridedOperand<'_, T, 1>,
    b: StridedOperand<'_, T, 1>,
) -> Result<WgpuBuffer<T>>
where
    T: WgslScalar + Pod + ReductionIdentity<SumOp>,
{
    if a.layout.shape != b.layout.shape {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "dot product shape mismatch: lhs {:?}, rhs {:?}",
                a.layout.shape, b.layout.shape
            ),
        });
    }

    let len = a.layout.shape[0];
    if len == 0 {
        return device.upload(&[T::IDENTITY]);
    }

    let temp_prod = device.alloc_zeroed::<T>(len)?;
    let temp_prod_layout = Layout::c_contiguous([len]).map_err(map_layout_err)?;
    let temp_prod_operand = StridedOperand {
        buffer: &temp_prod,
        layout: &temp_prod_layout,
    };
    binary_elementwise_strided_into::<MulOp, T, 1>(
        device,
        a,
        b,
        temp_prod_operand,
        BlockWidth::DEFAULT,
    )?;
    reduction::<SumOp, T>(device, &temp_prod)
}

/// Compute the trace `tr(A) = Σᵢ aᵢᵢ` of a square matrix on the GPU.
pub fn trace<T>(device: &WgpuDevice, matrix: StridedOperand<'_, T, 2>) -> Result<WgpuBuffer<T>>
where
    T: WgslScalar + Pod + ReductionIdentity<SumOp>,
{
    let [rows, cols] = matrix.layout.shape;
    if rows != cols {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "trace requires a square matrix, got shape {:?}",
                matrix.layout.shape
            ),
        });
    }

    if rows == 0 {
        return device.upload(&[T::IDENTITY]);
    }

    let s0 = matrix.layout.strides[0];
    let s1 = matrix.layout.strides[1];
    let diag_layout = Layout::new([rows], [s0 + s1], matrix.layout.offset);
    let diag_operand = StridedOperand {
        buffer: matrix.buffer,
        layout: &diag_layout,
    };

    unary_map_reduction::<TraceOp, T, 1>(device, diag_operand)
}

/// Compute the L1 norm `Σ |x|` on the GPU.
pub fn norm_l1<T, const N: usize>(
    device: &WgpuDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<WgpuBuffer<T>>
where
    T: WgslScalar + Pod + ReductionIdentity<SumOp>,
{
    unary_map_reduction::<NormL1Op, T, N>(device, view)
}

/// Compute the L2 / Frobenius norm `sqrt(Σ x²)` on the GPU.
pub fn norm_l2<T, const N: usize>(
    device: &WgpuDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<WgpuBuffer<T>>
where
    T: L2NormScalar,
{
    let len = view.layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return device.upload(&[T::IDENTITY]);
    }

    let temp_sq = device.alloc_zeroed::<T>(len)?;
    let temp_sq_layout = Layout::c_contiguous(view.layout.shape).map_err(map_layout_err)?;
    let temp_sq_operand = StridedOperand {
        buffer: &temp_sq,
        layout: &temp_sq_layout,
    };
    binary_elementwise_strided_into::<MulOp, T, N>(
        device,
        view,
        view,
        temp_sq_operand,
        BlockWidth::DEFAULT,
    )?;
    let squared_sum = reduction::<SumOp, T>(device, &temp_sq)?;
    let out = device.alloc_zeroed::<T>(1)?;
    unary_elementwise_into::<SqrtOp, T>(device, &squared_sum, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}

/// Compute the Max norm `max |x|` on the GPU.
pub fn norm_max<T, const N: usize>(
    device: &WgpuDevice,
    view: StridedOperand<'_, T, N>,
) -> Result<WgpuBuffer<T>>
where
    T: WgslScalar + Pod + ReductionIdentity<MaxOp>,
{
    let len = view.layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return device.upload(&[T::IDENTITY]);
    }

    let temp_abs = device.alloc_zeroed::<T>(len)?;
    let temp_abs_layout = Layout::c_contiguous(view.layout.shape).map_err(map_layout_err)?;
    let temp_abs_operand = StridedOperand {
        buffer: &temp_abs,
        layout: &temp_abs_layout,
    };
    unary_elementwise_strided_into::<AbsOp, T, N>(
        device,
        view,
        temp_abs_operand,
        BlockWidth::DEFAULT,
    )?;
    reduction::<MaxOp, T>(device, &temp_abs)
}
