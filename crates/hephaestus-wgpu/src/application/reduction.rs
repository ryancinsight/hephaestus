use std::any::TypeId;
use std::marker::PhantomData;

use bytemuck::Pod;
use hephaestus_core::{
    AxisReductionDispatch, AxisReductionMeta, BlockWidth, CombineExpr, ComputeDevice,
    DialectScalar, HephaestusError, IdentityToken, OpIdentity, Result, Wgsl, plan_axis_reduction,
};
use leto::Layout;

use crate::application::pipeline::cached_pipeline;
use crate::application::strided::{StridedOperand, map_layout_err};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

pub use hephaestus_core::{MaxOp, MinOp, SumOp};

mod prepared;
pub use prepared::{
    PreparedReduction, prepare_reduction, prepare_reduction_with_width,
    submit_prepared_reduction_batch,
};

/// ZST wrapper to generate a unique TypeId in the pipeline cache for reduction operations.
struct ReductionOpWrapper<Op>(PhantomData<Op>);
struct ReductionFinalOpWrapper<Op>(PhantomData<Op>);
struct AxisReductionParallelKernel<Op>(PhantomData<Op>);
struct AxisReductionAxis0TiledKernel<Op>(PhantomData<Op>);
struct MeanAxisParallelKernel<T>(PhantomData<T>);
struct MeanAxis0TiledKernel<T>(PhantomData<T>);

const AXIS0_TILE_COLUMNS: u32 = 32;

/// Prepared rank-2 axis reduction over fixed input/output buffers and layouts.
///
/// This removes repeated metadata-uniform acquisition, metadata upload, and bind
/// group construction when callers repeatedly dispatch the same axis reduction
/// into the same output buffer.
pub struct PreparedAxisReduction<T> {
    pipeline: Option<wgpu::ComputePipeline>,
    bind_group: Option<wgpu::BindGroup>,
    groups: u32,
    /// Pooled metadata uniform held for the prepared reduction's lifetime so
    /// it recycles back to the pool on drop (the bind group also keeps the
    /// underlying buffer alive; without the guard the buffer would escape the
    /// pool permanently).
    _meta_buffer: Option<crate::infrastructure::pool::UniformBufferGuard>,
    _marker: PhantomData<T>,
}

impl<T> PreparedAxisReduction<T> {
    /// Dispatch the prepared axis reduction once.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error if command encoding or submission cannot
    /// be completed by the backend.
    pub fn dispatch(&self, device: &WgpuDevice) -> Result<()> {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(());
        };
        let Some(bind_group) = self.bind_group.as_ref() else {
            return Ok(());
        };
        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-prepared-axis-reduction"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hephaestus-prepared-axis-reduction"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(self.groups, 1, 1);
        }
        device.queue().submit(Some(encoder.finish()));
        Ok(())
    }
}

/// Submit multiple prepared axis reductions in one command buffer.
///
/// Each prepared reduction keeps its own output buffer. This API is intended for
/// repeated independent reductions where the caller can consume results after the
/// whole batch completes, amortizing WGPU submit/poll overhead without sharing
/// scratch state between reductions.
///
/// # Errors
///
/// Returns a typed dispatch error if command encoding or submission cannot be
/// completed by the backend.
pub fn submit_prepared_axis_reduction_batch<T>(
    device: &WgpuDevice,
    reductions: &[&PreparedAxisReduction<T>],
) -> Result<()> {
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-prepared-axis-reduction-batch"),
        });
    for reduction in reductions {
        let Some(pipeline) = reduction.pipeline.as_ref() else {
            continue;
        };
        let Some(bind_group) = reduction.bind_group.as_ref() else {
            continue;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-prepared-axis-reduction-batch-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(reduction.groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

fn shader_source<Op: CombineExpr<Wgsl>, T: IdentityToken<Op, Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> input: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<{ty}>;

var<workgroup> shared_data: array<{ty}, {wg}>;

@compute @workgroup_size({wg})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {{
    let i = global_id.x;
    let num_elements = arrayLength(&input);
    
    if (i < num_elements) {{
        shared_data[local_id.x] = input[i];
    }} else {{
        shared_data[local_id.x] = {identity};
    }}
    
    workgroupBarrier();
    
    for (var stride = {wg}u / 2u; stride > 0u; stride = stride / 2u) {{
        if (local_id.x < stride) {{
            let lhs = shared_data[local_id.x];
            let rhs = shared_data[local_id.x + stride];
            shared_data[local_id.x] = {expr};
        }}
        workgroupBarrier();
    }}
    
    if (local_id.x == 0u) {{
        output[workgroup_id.x] = shared_data[0];
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<Op, Wgsl>>::TOKEN,
        expr = <Op as CombineExpr<Wgsl>>::EXPR,
    )
}

fn final_reduction_shader_source<Op: CombineExpr<Wgsl>, T: IdentityToken<Op, Wgsl>>(
    width: BlockWidth,
) -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> input: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<{ty}>;

var<workgroup> shared_data: array<{ty}, {wg}>;

@compute @workgroup_size({wg})
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {{
    let lane = local_id.x;
    let num_elements = arrayLength(&input);
    var acc: {ty} = {identity};
    var i = lane;
    loop {{
        if (i >= num_elements) {{
            break;
        }}
        let lhs = acc;
        let rhs = input[i];
        acc = {expr};
        i = i + {wg}u;
    }}
    shared_data[lane] = acc;
    workgroupBarrier();

    for (var stride = {wg}u / 2u; stride > 0u; stride = stride / 2u) {{
        if (lane < stride) {{
            let lhs = shared_data[lane];
            let rhs = shared_data[lane + stride];
            shared_data[lane] = {expr};
        }}
        workgroupBarrier();
    }}

    if (lane == 0u) {{
        output[0u] = shared_data[0];
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<Op, Wgsl>>::TOKEN,
        expr = <Op as CombineExpr<Wgsl>>::EXPR,
    )
}

/// Grid-strided workgroup-parallel axis reduction (WG-P5): each of the `wg`
/// lanes accumulates every `axis_len`-th element starting at its lane index
/// (a per-lane sequential fold, not a single load), then the lanes'
/// `wg` partials tree-reduce as before. Correct and fully lane-parallel for
/// any `axis_len` — including `axis_len > wg`, where the previous "parallel"
/// kernel could only read the first `wg` elements (each lane loaded at most
/// one) and dispatch fell back to a genuinely serial one-thread-per-row
/// kernel that did zero cross-lane work. This one kernel now covers both
/// regimes; the serial kernel is dead and removed.
///
/// The per-lane stride-accumulation reassociates the combine relative to a
/// purely sequential fold (elements interleave across lanes before the tree
/// reduction), so exact bitwise equality with a sequential CPU reference is
/// no longer guaranteed for inputs where the combine is not exact under
/// reassociation (e.g. float sums with real rounding). Differential tests
/// against a derived epsilon bound this (`axis_reduction_grid_stride_...`
/// contract tests); small-integer-valued fixtures remain exact because
/// integer-valued f32 sums have no rounding error under any grouping.
fn axis_reduction_parallel_shader_source<Op, T>(width: BlockWidth) -> String
where
    Op: CombineExpr<Wgsl>,
    T: IdentityToken<Op, Wgsl>,
{
    format!(
        r#"
struct AxisReductionMeta {{
    input_shape: vec2<u32>,
    input_strides: vec2<i32>,
    output_strides: vec2<i32>,
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<uniform> axis_meta: AxisReductionMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> output: array<{ty}>;

var<workgroup> partials: array<{ty}, {wg}>;

@compute @workgroup_size({wg})
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let i = workgroup_id.x;
    let lane = local_id.x;
    if (i >= axis_meta.offsets.w) {{
        return;
    }}

    let axis = axis_meta.offsets.z;
    let axis_len = select(axis_meta.input_shape.y, axis_meta.input_shape.x, axis == 0u);
    let out_row = select(i, 0u, axis == 0u);
    let out_col = select(0u, i, axis == 0u);

    var acc: {ty} = {identity};
    var idx = lane;
    loop {{
        if (idx >= axis_len) {{
            break;
        }}
        let in_row = select(out_row, idx, axis == 0u);
        let in_col = select(idx, out_col, axis == 0u);
        let in_off = i32(axis_meta.offsets.x)
            + i32(in_row) * axis_meta.input_strides.x
            + i32(in_col) * axis_meta.input_strides.y;
        let lhs = acc;
        let rhs = input[u32(in_off)];
        acc = {expr};
        idx = idx + {wg}u;
    }}
    partials[lane] = acc;
    workgroupBarrier();

    var stride = {wg}u / 2u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (lane < stride) {{
            let lhs = partials[lane];
            let rhs = partials[lane + stride];
            partials[lane] = {expr};
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (lane == 0u) {{
        let out_off = i32(axis_meta.offsets.y)
            + i32(out_row) * axis_meta.output_strides.x
            + i32(out_col) * axis_meta.output_strides.y;
        output[u32(out_off)] = partials[0u];
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<Op, Wgsl>>::TOKEN,
        expr = <Op as CombineExpr<Wgsl>>::EXPR,
    )
}

/// Grid-strided workgroup-parallel mean-axis reduction (WG-P5); see
/// [`axis_reduction_parallel_shader_source`] for the rationale (same fix,
/// applied to the sum-then-divide mean kernel).
fn mean_axis_parallel_shader_source<T>(width: BlockWidth) -> String
where
    T: IdentityToken<SumOp, Wgsl>,
{
    format!(
        r#"
struct AxisReductionMeta {{
    input_shape: vec2<u32>,
    input_strides: vec2<i32>,
    output_strides: vec2<i32>,
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<uniform> axis_meta: AxisReductionMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> output: array<{ty}>;

var<workgroup> partials: array<{ty}, {wg}>;

@compute @workgroup_size({wg})
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let i = workgroup_id.x;
    let lane = local_id.x;
    if (i >= axis_meta.offsets.w) {{
        return;
    }}

    let axis = axis_meta.offsets.z;
    let axis_len = select(axis_meta.input_shape.y, axis_meta.input_shape.x, axis == 0u);
    let out_row = select(i, 0u, axis == 0u);
    let out_col = select(0u, i, axis == 0u);

    var acc: {ty} = {identity};
    var idx = lane;
    loop {{
        if (idx >= axis_len) {{
            break;
        }}
        let in_row = select(out_row, idx, axis == 0u);
        let in_col = select(idx, out_col, axis == 0u);
        let in_off = i32(axis_meta.offsets.x)
            + i32(in_row) * axis_meta.input_strides.x
            + i32(in_col) * axis_meta.input_strides.y;
        acc = acc + input[u32(in_off)];
        idx = idx + {wg}u;
    }}
    partials[lane] = acc;
    workgroupBarrier();

    var stride = {wg}u / 2u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (lane < stride) {{
            partials[lane] = partials[lane] + partials[lane + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (lane == 0u) {{
        let out_off = i32(axis_meta.offsets.y)
            + i32(out_row) * axis_meta.output_strides.x
            + i32(out_col) * axis_meta.output_strides.y;
        output[u32(out_off)] = partials[0u] / {ty}(axis_len);
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = <T as IdentityToken<SumOp, Wgsl>>::TOKEN,
    )
}

fn axis0_tile_shape(width: BlockWidth) -> (u32, u32) {
    let tile_cols = AXIS0_TILE_COLUMNS.min(width.get());
    (tile_cols, width.get() / tile_cols)
}

fn axis_reduction_axis0_tiled_shader_source<Op, T>(width: BlockWidth) -> String
where
    Op: CombineExpr<Wgsl>,
    T: IdentityToken<Op, Wgsl>,
{
    let (tile_cols, tile_rows) = axis0_tile_shape(width);
    format!(
        r#"
struct AxisReductionMeta {{
    input_shape: vec2<u32>,
    input_strides: vec2<i32>,
    output_strides: vec2<i32>,
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<uniform> axis_meta: AxisReductionMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> output: array<{ty}>;

var<workgroup> partials: array<{ty}, {wg}>;

@compute @workgroup_size({tile_cols}, {tile_rows})
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let out_col = workgroup_id.x * {tile_cols}u + local_id.x;
    let row_lane = local_id.y;
    let partial_idx = row_lane * {tile_cols}u + local_id.x;
    let axis_len = axis_meta.input_shape.x;

    var value: {ty} = {identity};
    if (out_col < axis_meta.offsets.w) {{
        var row = row_lane;
        loop {{
            if (row >= axis_len) {{
                break;
            }}
            let in_off = i32(axis_meta.offsets.x)
                + i32(row) * axis_meta.input_strides.x
                + i32(out_col) * axis_meta.input_strides.y;
            let lhs = value;
            let rhs = input[u32(in_off)];
            value = {expr};
            row = row + {tile_rows}u;
        }}
    }}
    partials[partial_idx] = value;
    workgroupBarrier();

    var stride = {tile_rows}u / 2u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (row_lane < stride) {{
            let lhs = partials[partial_idx];
            let rhs = partials[(row_lane + stride) * {tile_cols}u + local_id.x];
            partials[partial_idx] = {expr};
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (row_lane == 0u && out_col < axis_meta.offsets.w) {{
        let out_off = i32(axis_meta.offsets.y) + i32(out_col) * axis_meta.output_strides.y;
        output[u32(out_off)] = partials[local_id.x];
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        tile_cols = tile_cols,
        tile_rows = tile_rows,
        identity = <T as IdentityToken<Op, Wgsl>>::TOKEN,
        expr = <Op as CombineExpr<Wgsl>>::EXPR,
    )
}

fn mean_axis0_tiled_shader_source<T>(width: BlockWidth) -> String
where
    T: IdentityToken<SumOp, Wgsl>,
{
    let (tile_cols, tile_rows) = axis0_tile_shape(width);
    format!(
        r#"
struct AxisReductionMeta {{
    input_shape: vec2<u32>,
    input_strides: vec2<i32>,
    output_strides: vec2<i32>,
    offsets: vec4<u32>,
}}

@group(0) @binding(0) var<uniform> axis_meta: AxisReductionMeta;
@group(0) @binding(1) var<storage, read> input: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> output: array<{ty}>;

var<workgroup> partials: array<{ty}, {wg}>;

@compute @workgroup_size({tile_cols}, {tile_rows})
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let out_col = workgroup_id.x * {tile_cols}u + local_id.x;
    let row_lane = local_id.y;
    let partial_idx = row_lane * {tile_cols}u + local_id.x;
    let axis_len = axis_meta.input_shape.x;

    var value: {ty} = {identity};
    if (out_col < axis_meta.offsets.w) {{
        var row = row_lane;
        loop {{
            if (row >= axis_len) {{
                break;
            }}
            let in_off = i32(axis_meta.offsets.x)
                + i32(row) * axis_meta.input_strides.x
                + i32(out_col) * axis_meta.input_strides.y;
            value = value + input[u32(in_off)];
            row = row + {tile_rows}u;
        }}
    }}
    partials[partial_idx] = value;
    workgroupBarrier();

    var stride = {tile_rows}u / 2u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (row_lane < stride) {{
            partials[partial_idx] = partials[partial_idx]
                + partials[(row_lane + stride) * {tile_cols}u + local_id.x];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (row_lane == 0u && out_col < axis_meta.offsets.w) {{
        let out_off = i32(axis_meta.offsets.y) + i32(out_col) * axis_meta.output_strides.y;
        output[u32(out_off)] = partials[local_id.x] / {ty}(axis_len);
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        tile_cols = tile_cols,
        tile_rows = tile_rows,
        identity = <T as IdentityToken<SumOp, Wgsl>>::TOKEN,
    )
}

fn axis_len<T>(input: StridedOperand<'_, T, 2>, axis: usize) -> Result<usize> {
    input
        .layout
        .shape
        .get(axis)
        .copied()
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("axis {axis} is out of bounds for rank-2 reduction"),
        })
}

fn plan_axis_reduction_dispatch<T>(
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<Option<AxisReductionDispatch>> {
    plan_axis_reduction(
        input.layout,
        input.buffer.len,
        output.layout,
        output.buffer.len,
        axis,
        width,
        input.buffer.aliases(output.buffer),
    )
}

fn dispatch_axis_reduction<T>(
    device: &WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    input: StridedOperand<'_, T, 2>,
    output: StridedOperand<'_, T, 2>,
    dispatch: AxisReductionDispatch,
) -> Result<()> {
    let raw_meta_buffer =
        device.get_uniform_buffer(WgpuDevice::byte_size::<AxisReductionMeta>(1)?)?;
    let meta_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta_buffer);
    device
        .queue()
        .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(&dispatch.meta));
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-axis-reduction"),
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
            label: Some("hephaestus-axis-reduction"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-axis-reduction"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch.groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

fn prepared_axis_reduction<T>(
    device: &WgpuDevice,
    pipeline: wgpu::ComputePipeline,
    input: StridedOperand<'_, T, 2>,
    output: StridedOperand<'_, T, 2>,
    dispatch: AxisReductionDispatch,
    label: &'static str,
) -> Result<PreparedAxisReduction<T>> {
    let raw_meta_buffer =
        device.get_uniform_buffer(WgpuDevice::byte_size::<AxisReductionMeta>(1)?)?;
    let meta_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_meta_buffer);
    device
        .queue()
        .write_buffer(&meta_buffer, 0, bytemuck::bytes_of(&dispatch.meta));
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
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
    Ok(PreparedAxisReduction {
        pipeline: Some(pipeline),
        bind_group: Some(bind_group),
        groups: dispatch.groups,
        _meta_buffer: Some(meta_buffer),
        _marker: PhantomData,
    })
}

fn empty_prepared_axis_reduction<T>() -> PreparedAxisReduction<T> {
    PreparedAxisReduction {
        pipeline: None,
        bind_group: None,
        groups: 0,
        _meta_buffer: None,
        _marker: PhantomData,
    }
}

fn axis_reduction_pipeline<Op, T>(
    device: &WgpuDevice,
    width: BlockWidth,
    dispatch: &mut AxisReductionDispatch,
) -> wgpu::ComputePipeline
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    if dispatch.meta.offsets[2] == 0 {
        let (tile_cols, _) = axis0_tile_shape(width);
        dispatch.groups = dispatch.meta.offsets[3].div_ceil(tile_cols);
        cached_pipeline(
            device,
            (
                TypeId::of::<AxisReductionAxis0TiledKernel<Op>>(),
                TypeId::of::<T>(),
                width.get(),
            ),
            "hephaestus-axis-reduction-axis0-tiled",
            || axis_reduction_axis0_tiled_shader_source::<Op, T>(width),
        )
    } else {
        // Grid-strided (WG-P5): one workgroup per output row, correct and
        // fully lane-parallel for any axis length.
        dispatch.groups = dispatch.meta.offsets[3];
        cached_pipeline(
            device,
            (
                TypeId::of::<AxisReductionParallelKernel<Op>>(),
                TypeId::of::<T>(),
                width.get(),
            ),
            "hephaestus-axis-reduction-parallel",
            || axis_reduction_parallel_shader_source::<Op, T>(width),
        )
    }
}

fn mean_axis_pipeline<T>(
    device: &WgpuDevice,
    width: BlockWidth,
    dispatch: &mut AxisReductionDispatch,
) -> wgpu::ComputePipeline
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    if dispatch.meta.offsets[2] == 0 {
        let (tile_cols, _) = axis0_tile_shape(width);
        dispatch.groups = dispatch.meta.offsets[3].div_ceil(tile_cols);
        cached_pipeline(
            device,
            (
                TypeId::of::<MeanAxis0TiledKernel<T>>(),
                TypeId::of::<T>(),
                width.get(),
            ),
            "hephaestus-mean-axis0-tiled",
            || mean_axis0_tiled_shader_source::<T>(width),
        )
    } else {
        // Grid-strided (WG-P5): one workgroup per output row, correct and
        // fully lane-parallel for any axis length.
        dispatch.groups = dispatch.meta.offsets[3];
        cached_pipeline(
            device,
            (
                TypeId::of::<MeanAxisParallelKernel<T>>(),
                TypeId::of::<T>(),
                width.get(),
            ),
            "hephaestus-mean-axis-parallel",
            || mean_axis_parallel_shader_source::<T>(width),
        )
    }
}

/// Prepare a rank-2 axis reduction over fixed input/output buffers and layouts.
///
/// # Errors
///
/// Returns a typed dispatch error when axis, shape, output-layout, width, or
/// aliasing validation fails.
pub fn prepare_reduce_axis_into<Op, T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    let Some(mut dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
        return Ok(empty_prepared_axis_reduction());
    };
    let pipeline = axis_reduction_pipeline::<Op, T>(device, width, &mut dispatch);
    prepared_axis_reduction(
        device,
        pipeline,
        input,
        output,
        dispatch,
        "hephaestus-prepared-axis-reduction",
    )
}

/// Prepare a rank-2 axis sum over fixed input/output buffers and layouts.
#[inline]
pub fn prepare_sum_axis_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    prepare_reduce_axis_into::<SumOp, T>(device, input, axis, output, width)
}

/// Prepare a rank-2 axis min over fixed input/output buffers and layouts.
#[inline]
pub fn prepare_min_axis_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, Wgsl>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    prepare_reduce_axis_into::<MinOp, T>(device, input, axis, output, width)
}

/// Prepare a rank-2 axis max over fixed input/output buffers and layouts.
#[inline]
pub fn prepare_max_axis_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, Wgsl>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    prepare_reduce_axis_into::<MaxOp, T>(device, input, axis, output, width)
}

/// Prepare a rank-2 axis mean over fixed input/output buffers and layouts.
///
/// # Errors
///
/// Returns a typed dispatch error when the reduced axis is empty or when axis,
/// shape, output-layout, width, or aliasing validation fails.
pub fn prepare_mean_axis_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<PreparedAxisReduction<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    let reduced_axis_len = axis_len(input, axis)?;
    reject_empty_axis(reduced_axis_len, "mean_axis", axis)?;
    let Some(mut dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
        return Ok(empty_prepared_axis_reduction());
    };
    let pipeline = mean_axis_pipeline::<T>(device, width, &mut dispatch);
    prepared_axis_reduction(
        device,
        pipeline,
        input,
        output,
        dispatch,
        "hephaestus-prepared-mean-axis",
    )
}

/// Reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
///
/// This matches Leto's rank-preserving axis-reduction contract: an input with
/// shape `[rows, cols]` reduced over axis `0` writes shape `[1, cols]`, and
/// axis `1` writes shape `[rows, 1]`.
pub fn reduce_axis_into<Op, T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    let Some(mut dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
        return Ok(());
    };
    let pipeline = axis_reduction_pipeline::<Op, T>(device, width, &mut dispatch);
    dispatch_axis_reduction(device, &pipeline, input, output, dispatch)
}

/// Reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
///
/// The output shape preserves the reduced axis as length one, matching
/// [`reduce_axis_into`].
pub fn reduce_axis<Op, T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    if axis >= 2 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("axis {axis} is out of bounds for rank-2 reduction"),
        });
    }
    let mut output_shape = input.layout.shape;
    output_shape[axis] = 1;
    let output_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let output_len = output_layout.checked_size().map_err(map_layout_err)?;
    let output = device.alloc_zeroed::<T>(output_len)?;
    reduce_axis_into::<Op, T>(
        device,
        input,
        axis,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        width,
    )?;
    Ok(output)
}

/// Sum-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn sum_axis_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    reduce_axis_into::<SumOp, T>(device, input, axis, output, width)
}

/// Sum-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn sum_axis<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    reduce_axis::<SumOp, T>(device, input, axis, width)
}

fn reject_empty_axis(axis_len: usize, op_name: &'static str, axis: usize) -> Result<()> {
    if axis_len == 0 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{op_name} is undefined for empty axis {axis}"),
        });
    }
    Ok(())
}

/// Min-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn min_axis_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, Wgsl>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    reduce_axis_into::<MinOp, T>(device, input, axis, output, width)
}

/// Min-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn min_axis<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, Wgsl>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    reduce_axis::<MinOp, T>(device, input, axis, width)
}

/// Max-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn max_axis_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, Wgsl>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    reduce_axis_into::<MaxOp, T>(device, input, axis, output, width)
}

/// Max-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn max_axis<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, Wgsl>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    reduce_axis::<MaxOp, T>(device, input, axis, width)
}

/// Mean-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
///
/// Empty reduced axes are rejected because the arithmetic mean is undefined.
pub fn mean_axis_into<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    let reduced_axis_len = axis_len(input, axis)?;
    reject_empty_axis(reduced_axis_len, "mean_axis", axis)?;
    let Some(mut dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
        return Ok(());
    };
    let pipeline = mean_axis_pipeline::<T>(device, width, &mut dispatch);
    dispatch_axis_reduction(device, &pipeline, input, output, dispatch)
}

/// Mean-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn mean_axis<T>(
    device: &WgpuDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    reject_empty_axis(axis_len(input, axis)?, "mean_axis", axis)?;
    let mut output_shape = input.layout.shape;
    output_shape[axis] = 1;
    let output_layout = Layout::c_contiguous(output_shape).map_err(map_layout_err)?;
    let output_len = output_layout.checked_size().map_err(map_layout_err)?;
    let output = device.alloc_zeroed::<T>(output_len)?;
    mean_axis_into(
        device,
        input,
        axis,
        StridedOperand {
            buffer: &output,
            layout: &output_layout,
        },
        width,
    )?;
    Ok(output)
}

/// Run reduction on the device, returning a 1-element buffer holding the result.
///
/// If the input buffer is empty, it returns a 1-element buffer containing the operation's identity value.
pub fn reduction<Op, T>(device: &WgpuDevice, input: &WgpuBuffer<T>) -> Result<WgpuBuffer<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    reduction_with_width::<Op, T>(device, input, BlockWidth::DEFAULT)
}

/// Run reduction on the device with a caller-selected power-of-two block width.
///
/// If the input buffer is empty, it returns a 1-element buffer containing the
/// operation's identity value. `width` is part of the monomorphized pipeline
/// cache key and WGSL workgroup size; non-power-of-two widths are rejected
/// because the workgroup tree halves its active lane count every step.
pub fn reduction_with_width<Op, T>(
    device: &WgpuDevice,
    input: &WgpuBuffer<T>,
    width: BlockWidth,
) -> Result<WgpuBuffer<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    let prepared = prepare_reduction_with_width::<Op, T>(device, input, width)?;
    prepared.dispatch(device)?;
    Ok(prepared.into_output())
}
