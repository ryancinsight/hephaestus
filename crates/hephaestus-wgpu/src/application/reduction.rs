use std::any::TypeId;
use std::marker::PhantomData;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError, Result};
use leto::Layout;

use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::application::strided::{map_layout_err, to_u32, StridedOperand};
use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Zero-sized reduction operation marker selecting the WGSL combine expression.
pub trait ReductionWgslOp: Copy + Send + Sync + 'static {
    /// WGSL expression combining `lhs` and `rhs` (e.g. `"lhs + rhs"` or `"min(lhs, rhs)"`).
    const WGSL_EXPR: &'static str;
}

/// Associates a scalar type and reduction operation with the identity value.
pub trait ReductionIdentity<Op>: WgslScalar {
    /// The identity value on the host side.
    const IDENTITY: Self;
    /// The WGSL literal for the identity value.
    const WGSL_IDENTITY: &'static str;
}

/// Sum reduction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SumOp;

/// Minimum reduction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MinOp;

/// Maximum reduction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaxOp;

impl ReductionWgslOp for SumOp {
    const WGSL_EXPR: &'static str = "lhs + rhs";
}

impl ReductionWgslOp for MinOp {
    const WGSL_EXPR: &'static str = "min(lhs, rhs)";
}

impl ReductionWgslOp for MaxOp {
    const WGSL_EXPR: &'static str = "max(lhs, rhs)";
}

// ── SumOp Identity implementations ──
impl ReductionIdentity<SumOp> for f32 {
    const IDENTITY: Self = 0.0;
    const WGSL_IDENTITY: &'static str = "0.0";
}
impl ReductionIdentity<SumOp> for u32 {
    const IDENTITY: Self = 0;
    const WGSL_IDENTITY: &'static str = "0u";
}
impl ReductionIdentity<SumOp> for i32 {
    const IDENTITY: Self = 0;
    const WGSL_IDENTITY: &'static str = "0";
}

// ── MinOp Identity implementations ──
impl ReductionIdentity<MinOp> for f32 {
    const IDENTITY: Self = f32::MAX;
    const WGSL_IDENTITY: &'static str = "3.402823466e+38";
}
impl ReductionIdentity<MinOp> for u32 {
    const IDENTITY: Self = u32::MAX;
    const WGSL_IDENTITY: &'static str = "4294967295u";
}
impl ReductionIdentity<MinOp> for i32 {
    const IDENTITY: Self = i32::MAX;
    const WGSL_IDENTITY: &'static str = "2147483647";
}

// ── MaxOp Identity implementations ──
impl ReductionIdentity<MaxOp> for f32 {
    const IDENTITY: Self = f32::MIN;
    const WGSL_IDENTITY: &'static str = "-3.402823466e+38";
}
impl ReductionIdentity<MaxOp> for u32 {
    const IDENTITY: Self = u32::MIN;
    const WGSL_IDENTITY: &'static str = "0u";
}
impl ReductionIdentity<MaxOp> for i32 {
    const IDENTITY: Self = i32::MIN;
    const WGSL_IDENTITY: &'static str = "-2147483648";
}

/// ZST wrapper to generate a unique TypeId in the pipeline cache for reduction operations.
struct ReductionOpWrapper<Op>(PhantomData<Op>);
struct AxisReductionKernel<Op>(PhantomData<Op>);
struct MeanAxisKernel<T>(PhantomData<T>);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AxisReductionMeta {
    input_shape: [u32; 2],
    input_strides: [i32; 2],
    output_strides: [i32; 2],
    _pre_offsets_pad: [u32; 2],
    offsets: [u32; 4],
}

struct AxisReductionDispatch {
    meta: AxisReductionMeta,
    groups: u32,
}

fn shader_source<Op: ReductionWgslOp, T: ReductionIdentity<Op>>(width: BlockWidth) -> String {
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
        ty = T::WGSL_TYPE,
        wg = width.get(),
        identity = T::WGSL_IDENTITY,
        expr = Op::WGSL_EXPR,
    )
}

fn validate_reduction_width(width: BlockWidth) -> Result<()> {
    if !width.get().is_power_of_two() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "reduction block width {} must be a power of two",
                width.get()
            ),
        });
    }
    Ok(())
}

#[inline]
fn to_i32(value: isize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i32 range"),
    })
}

fn axis_reduction_shader_source<Op, T>(width: BlockWidth) -> String
where
    Op: ReductionWgslOp,
    T: ReductionIdentity<Op>,
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

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= axis_meta.offsets.w) {{
        return;
    }}

    let axis = axis_meta.offsets.z;
    let axis_len = select(axis_meta.input_shape.y, axis_meta.input_shape.x, axis == 0u);
    let out_row = select(i, 0u, axis == 0u);
    let out_col = select(0u, i, axis == 0u);
    var acc: {ty} = {identity};

    for (var r = 0u; r < axis_len; r = r + 1u) {{
        let in_row = select(out_row, r, axis == 0u);
        let in_col = select(r, out_col, axis == 0u);
        let in_off = i32(axis_meta.offsets.x)
            + i32(in_row) * axis_meta.input_strides.x
            + i32(in_col) * axis_meta.input_strides.y;
        let lhs = acc;
        let rhs = input[u32(in_off)];
        acc = {expr};
    }}

    let out_off = i32(axis_meta.offsets.y)
        + i32(out_row) * axis_meta.output_strides.x
        + i32(out_col) * axis_meta.output_strides.y;
    output[u32(out_off)] = acc;
}}
"#,
        ty = T::WGSL_TYPE,
        wg = width.get(),
        identity = T::WGSL_IDENTITY,
        expr = Op::WGSL_EXPR,
    )
}

fn mean_axis_shader_source<T>(width: BlockWidth) -> String
where
    T: ReductionIdentity<SumOp>,
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

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= axis_meta.offsets.w) {{
        return;
    }}

    let axis = axis_meta.offsets.z;
    let axis_len = select(axis_meta.input_shape.y, axis_meta.input_shape.x, axis == 0u);
    let out_row = select(i, 0u, axis == 0u);
    let out_col = select(0u, i, axis == 0u);
    var acc: {ty} = {identity};

    for (var r = 0u; r < axis_len; r = r + 1u) {{
        let in_row = select(out_row, r, axis == 0u);
        let in_col = select(r, out_col, axis == 0u);
        let in_off = i32(axis_meta.offsets.x)
            + i32(in_row) * axis_meta.input_strides.x
            + i32(in_col) * axis_meta.input_strides.y;
        acc = acc + input[u32(in_off)];
    }}

    let out_off = i32(axis_meta.offsets.y)
        + i32(out_row) * axis_meta.output_strides.x
        + i32(out_col) * axis_meta.output_strides.y;
    output[u32(out_off)] = acc / {ty}(axis_len);
}}
"#,
        ty = T::WGSL_TYPE,
        wg = width.get(),
        identity = T::WGSL_IDENTITY,
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

fn validate_axis_reduction<T>(
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<Option<AxisReductionDispatch>> {
    validate_reduction_width(width)?;
    if axis >= 2 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("axis {axis} is out of bounds for rank-2 reduction"),
        });
    }

    let mut expected_shape = input.layout.shape;
    expected_shape[axis] = 1;
    if output.layout.shape != expected_shape {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "axis reduction output shape mismatch: input {:?}, axis {axis}, out {:?}",
                input.layout.shape, output.layout.shape
            ),
        });
    }
    if input.buffer.aliases(output.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "axis reduction output buffer must not alias input buffer".to_string(),
        });
    }
    if output.layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "axis reduction output layout must not contain zero-stride aliasing"
                .to_string(),
        });
    }

    input
        .layout
        .validate_storage_len(input.buffer.len)
        .map_err(map_layout_err)?;
    output
        .layout
        .validate_storage_len(output.buffer.len)
        .map_err(map_layout_err)?;
    let output_len = output.layout.checked_size().map_err(map_layout_err)?;
    if output_len == 0 {
        return Ok(None);
    }

    let groups = workgroups(output_len, width)?;
    let meta = AxisReductionMeta {
        input_shape: [
            to_u32(input.layout.shape[0], "input rows")?,
            to_u32(input.layout.shape[1], "input columns")?,
        ],
        input_strides: [
            to_i32(input.layout.strides[0], "input row stride")?,
            to_i32(input.layout.strides[1], "input column stride")?,
        ],
        output_strides: [
            to_i32(output.layout.strides[0], "output row stride")?,
            to_i32(output.layout.strides[1], "output column stride")?,
        ],
        _pre_offsets_pad: [0; 2],
        offsets: [
            to_u32(input.layout.offset, "input offset")?,
            to_u32(output.layout.offset, "output offset")?,
            to_u32(axis, "axis")?,
            to_u32(output_len, "output length")?,
        ],
    };
    Ok(Some(AxisReductionDispatch { meta, groups }))
}

fn dispatch_axis_reduction<T>(
    device: &WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    input: StridedOperand<'_, T, 2>,
    output: StridedOperand<'_, T, 2>,
    dispatch: AxisReductionDispatch,
) -> Result<()> {
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<AxisReductionMeta>(1)?)?;
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
    device.recycle_uniform_buffer(meta_buffer);
    Ok(())
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
    Op: ReductionWgslOp,
    T: WgslScalar + Pod + ReductionIdentity<Op>,
{
    let Some(dispatch) = validate_axis_reduction(input, axis, output, width)? else {
        return Ok(());
    };
    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<AxisReductionKernel<Op>>(),
            TypeId::of::<T>(),
            width.get(),
        ),
        "hephaestus-axis-reduction",
        || axis_reduction_shader_source::<Op, T>(width),
    );
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
    Op: ReductionWgslOp,
    T: WgslScalar + Pod + ReductionIdentity<Op>,
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
    T: WgslScalar + Pod + ReductionIdentity<SumOp>,
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
    T: WgslScalar + Pod + ReductionIdentity<SumOp>,
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
    T: WgslScalar + Pod + ReductionIdentity<MinOp>,
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
    T: WgslScalar + Pod + ReductionIdentity<MinOp>,
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
    T: WgslScalar + Pod + ReductionIdentity<MaxOp>,
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
    T: WgslScalar + Pod + ReductionIdentity<MaxOp>,
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
    T: WgslScalar + Pod + ReductionIdentity<SumOp>,
{
    reject_empty_axis(axis_len(input, axis)?, "mean_axis", axis)?;
    let Some(dispatch) = validate_axis_reduction(input, axis, output, width)? else {
        return Ok(());
    };
    let pipeline = cached_pipeline(
        device,
        (
            TypeId::of::<MeanAxisKernel<T>>(),
            TypeId::of::<T>(),
            width.get(),
        ),
        "hephaestus-mean-axis",
        || mean_axis_shader_source::<T>(width),
    );
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
    T: WgslScalar + Pod + ReductionIdentity<SumOp>,
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

fn reduction_pass_count(mut len: usize, width: BlockWidth) -> usize {
    let width = width.get() as usize;
    let mut passes = 0;
    while len > 1 {
        len = len.div_ceil(width);
        passes += 1;
    }
    passes
}

/// Run reduction on the device, returning a 1-element buffer holding the result.
///
/// If the input buffer is empty, it returns a 1-element buffer containing the operation's identity value.
pub fn reduction<Op, T>(device: &WgpuDevice, input: &WgpuBuffer<T>) -> Result<WgpuBuffer<T>>
where
    Op: ReductionWgslOp,
    T: WgslScalar + Pod + ReductionIdentity<Op>,
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
    Op: ReductionWgslOp,
    T: WgslScalar + Pod + ReductionIdentity<Op>,
{
    validate_reduction_width(width)?;

    if input.len == 0 {
        return device.upload(&[T::IDENTITY]);
    }
    if input.len == 1 {
        // Create a copy of the buffer
        let out = device.alloc_zeroed::<T>(1)?;
        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-reduction-copy-1"),
            });
        encoder.copy_buffer_to_buffer(
            &input.buffer,
            0,
            &out.buffer,
            0,
            WgpuDevice::byte_size::<T>(1)?,
        );
        device.queue().submit(Some(encoder.finish()));
        return Ok(out);
    }
    let mut current_len = input.len;
    let mut temp_buffers: Vec<WgpuBuffer<T>> =
        Vec::with_capacity(reduction_pass_count(input.len, width));

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-reduction-multi-pass"),
        });

    while current_len > 1 {
        let groups = workgroups(current_len, width)?;
        let out_len = current_len.div_ceil(width.get() as usize);
        let out_buffer = device.alloc_zeroed::<T>(out_len)?;

        let key = (
            TypeId::of::<ReductionOpWrapper<Op>>(),
            TypeId::of::<T>(),
            width.get(),
        );

        let pipeline = cached_pipeline(device, key, "hephaestus-reduction", || {
            shader_source::<Op, T>(width)
        });

        let source_resource = if temp_buffers.is_empty() {
            input.buffer.as_entire_binding()
        } else {
            temp_buffers
                .last()
                .expect("invariant: non-initial reduction pass has a previous buffer")
                .buffer
                .as_entire_binding()
        };

        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hephaestus-reduction"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: source_resource,
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buffer.buffer.as_entire_binding(),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hephaestus-reduction-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        temp_buffers.push(out_buffer);
        current_len = out_len;
    }

    device.queue().submit(Some(encoder.finish()));

    // The final result is in the last allocated buffer.
    Ok(temp_buffers
        .pop()
        .expect("invariant: multi-element reduction allocates a final buffer"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_count_matches_tree_depth() {
        let width = BlockWidth::new(256).expect("invariant: test width is non-zero");
        assert_eq!(reduction_pass_count(0, width), 0);
        assert_eq!(reduction_pass_count(1, width), 0);
        assert_eq!(reduction_pass_count(2, width), 1);
        assert_eq!(reduction_pass_count(256, width), 1);
        assert_eq!(reduction_pass_count(257, width), 2);
        assert_eq!(reduction_pass_count(65_536, width), 2);

        let narrow = BlockWidth::new(128).expect("invariant: test width is non-zero");
        assert_eq!(reduction_pass_count(16_385, narrow), 3);
    }
}
