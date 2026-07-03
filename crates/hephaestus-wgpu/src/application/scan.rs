//! Rank-2 prefix/suffix scan kernels over strided matrix operands.

use std::any::TypeId;
use std::marker::PhantomData;

use bytemuck::Pod;
use hephaestus_core::{
    plan_axis_scan, AxisScanMeta, BlockWidth, CombineExpr, ComputeDevice, DialectScalar,
    IdentityToken, OpIdentity, Result, Wgsl,
};
use leto::Layout;

use crate::application::pipeline::cached_pipeline;
use crate::application::strided::{map_layout_err, StridedOperand};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

pub use hephaestus_core::{CumProdOp, CumSumOp, ScanDirection};

struct AxisScanKernel<Op>(PhantomData<Op>);

fn scan_shader_source<Op, T>(width: BlockWidth) -> String
where
    Op: CombineExpr<Wgsl>,
    T: IdentityToken<Op, Wgsl>,
{
    format!(
        r#"
struct AxisScanMeta {{
    input_shape: vec2<u32>,
    input_strides: vec2<i32>,
    output_strides: vec2<i32>,
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<uniform> scan_meta: AxisScanMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> output: array<{ty}>;

fn source_offset(row: u32, col: u32) -> u32 {{
    let off = i32(scan_meta.offsets.x)
        + i32(row) * scan_meta.input_strides.x
        + i32(col) * scan_meta.input_strides.y;
    return u32(off);
}}

fn dest_offset(row: u32, col: u32) -> u32 {{
    let off = i32(scan_meta.offsets.y)
        + i32(row) * scan_meta.output_strides.x
        + i32(col) * scan_meta.output_strides.y;
    return u32(off);
}}

// One thread owns one full scan line and walks it sequentially, writing every
// prefix: O(L) work per length-L line. The combine order is strictly
// left-to-right (right-to-left when reversed), matching the sequential
// reference exactly (bitwise-identical floating-point results).
@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let line = gid.x;
    if (line >= scan_meta.offsets.w) {{
        return;
    }}

    let rows = scan_meta.input_shape.x;
    let cols = scan_meta.input_shape.y;
    let axis = scan_meta.offsets.z & 1u;
    let reverse = (scan_meta.offsets.z & 2u) != 0u;
    let len = select(cols, rows, axis == 0u);
    var acc: {ty} = {identity};

    // `axis` and `reverse` are uniform across the dispatch, so these selects
    // never diverge within a subgroup.
    for (var s = 0u; s < len; s = s + 1u) {{
        let idx = select(s, len - 1u - s, reverse);
        let row = select(line, idx, axis == 0u);
        let col = select(idx, line, axis == 0u);
        let lhs = acc;
        let rhs = input[source_offset(row, col)];
        acc = {expr};
        output[dest_offset(row, col)] = acc;
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<Op, Wgsl>>::TOKEN,
        expr = <Op as CombineExpr<Wgsl>>::EXPR,
    )
}

/// Scan a rank-2 strided matrix along `axis`, preserving the input shape.
pub fn scan_axis_into<Op, T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    direction: ScanDirection,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    let Some(dispatch) = plan_axis_scan(
        input.layout,
        input.buffer.len,
        output.layout,
        output.buffer.len,
        axis,
        direction,
        width,
        input.buffer.aliases(output.buffer),
    )?
    else {
        return Ok(());
    };
    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<AxisScanKernel<Op>>(),
            TypeId::of::<T>(),
            width.get(),
        ),
        "hephaestus-axis-scan",
        || scan_shader_source::<Op, T>(width),
    );

    let raw_meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<AxisScanMeta>(1)?)?;
    let meta_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta_buffer);
    device
        .queue()
        .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(&dispatch.meta));
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-axis-scan"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer.buffer.as_entire_binding(),
                },
            ],
        });
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-axis-scan"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-axis-scan"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch.groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

/// Scan a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
pub fn scan_axis<Op, T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    direction: ScanDirection,
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    let len = input.layout.checked_size().map_err(map_layout_err)?;
    // Guard before device allocation: an empty layout has no elements to scan.
    // scan_axis_into would return Ok(()) immediately for len==0, but we avoid
    // the unnecessary device-pool call by returning an empty buffer here.
    if len == 0 {
        return device.alloc_zeroed::<T>(0);
    }
    let output = device.alloc_zeroed::<T>(len)?;
    let output_layout = Layout::c_contiguous(input.layout.shape).map_err(map_layout_err)?;
    scan_axis_into::<Op, T>(
        device,
        input,
        axis,
        direction,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        width,
    )?;
    Ok(output)
}

/// Forward cumulative sum over a rank-2 strided matrix along `axis`.
#[inline]
pub fn cumsum_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<CumSumOp> + IdentityToken<CumSumOp, Wgsl>,
{
    scan_axis_into::<CumSumOp, T>(device, input, axis, ScanDirection::Forward, output, width)
}

/// Forward cumulative sum over a rank-2 strided matrix, allocating a C-contiguous output buffer.
#[inline]
pub fn cumsum<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<CumSumOp> + IdentityToken<CumSumOp, Wgsl>,
{
    scan_axis::<CumSumOp, T>(device, input, axis, ScanDirection::Forward, width)
}
