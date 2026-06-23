//! Rank-2 prefix/suffix scan kernels over strided matrix operands.

use std::any::TypeId;
use std::marker::PhantomData;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError, Result};
use leto::Layout;

use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::application::strided::{map_layout_err, to_i32, to_u32, StridedOperand};
use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Direction of a scan along an axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanDirection {
    /// Accumulate from index 0 upward.
    Forward,
    /// Accumulate from the last index downward.
    Reverse,
}

/// Zero-sized scan operation marker selecting the WGSL combine expression.
pub trait ScanWgslOp: Copy + Send + Sync + 'static {
    /// WGSL expression combining `lhs` and `rhs`.
    const WGSL_EXPR: &'static str;
}

/// Associates a scalar type and scan operation with the identity value.
pub trait ScanIdentity<Op>: WgslScalar {
    /// The WGSL literal for the identity value.
    const WGSL_IDENTITY: &'static str;
}

/// Cumulative sum marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CumSumOp;

/// Cumulative product marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CumProdOp;

impl ScanWgslOp for CumSumOp {
    const WGSL_EXPR: &'static str = "lhs + rhs";
}

impl ScanWgslOp for CumProdOp {
    const WGSL_EXPR: &'static str = "lhs * rhs";
}

impl ScanIdentity<CumSumOp> for f32 {
    const WGSL_IDENTITY: &'static str = "0.0";
}

impl ScanIdentity<CumSumOp> for u32 {
    const WGSL_IDENTITY: &'static str = "0u";
}

impl ScanIdentity<CumSumOp> for i32 {
    const WGSL_IDENTITY: &'static str = "0";
}

impl ScanIdentity<CumProdOp> for f32 {
    const WGSL_IDENTITY: &'static str = "1.0";
}

impl ScanIdentity<CumProdOp> for u32 {
    const WGSL_IDENTITY: &'static str = "1u";
}

impl ScanIdentity<CumProdOp> for i32 {
    const WGSL_IDENTITY: &'static str = "1";
}

struct AxisScanKernel<Op>(PhantomData<Op>);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AxisScanMeta {
    input_shape: [u32; 2],
    input_strides: [i32; 2],
    output_strides: [i32; 2],
    _pre_offsets_pad: [u32; 2],
    offsets: [u32; 4],
}

struct AxisScanDispatch {
    meta: AxisScanMeta,
    groups: u32,
}

fn scan_shader_source<Op, T>(width: BlockWidth) -> String
where
    Op: ScanWgslOp,
    T: ScanIdentity<Op>,
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

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= scan_meta.offsets.w) {{
        return;
    }}

    let rows = scan_meta.input_shape.x;
    let cols = scan_meta.input_shape.y;
    let axis = scan_meta.offsets.z & 1u;
    let reverse = (scan_meta.offsets.z & 2u) != 0u;
    let out_row = i / cols;
    let out_col = i % cols;
    var acc: {ty} = {identity};

    if (axis == 0u) {{
        if (reverse) {{
            let count = rows - out_row;
            for (var s = 0u; s < count; s = s + 1u) {{
                let scan_row = rows - 1u - s;
                let lhs = acc;
                let rhs = input[source_offset(scan_row, out_col)];
                acc = {expr};
            }}
        }} else {{
            for (var scan_row = 0u; scan_row <= out_row; scan_row = scan_row + 1u) {{
                let lhs = acc;
                let rhs = input[source_offset(scan_row, out_col)];
                acc = {expr};
            }}
        }}
    }} else {{
        if (reverse) {{
            let count = cols - out_col;
            for (var s = 0u; s < count; s = s + 1u) {{
                let scan_col = cols - 1u - s;
                let lhs = acc;
                let rhs = input[source_offset(out_row, scan_col)];
                acc = {expr};
            }}
        }} else {{
            for (var scan_col = 0u; scan_col <= out_col; scan_col = scan_col + 1u) {{
                let lhs = acc;
                let rhs = input[source_offset(out_row, scan_col)];
                acc = {expr};
            }}
        }}
    }}

    let out_off = i32(scan_meta.offsets.y)
        + i32(out_row) * scan_meta.output_strides.x
        + i32(out_col) * scan_meta.output_strides.y;
    output[u32(out_off)] = acc;
}}
"#,
        ty = T::WGSL_TYPE,
        wg = width.get(),
        identity = T::WGSL_IDENTITY,
        expr = Op::WGSL_EXPR,
    )
}

fn validate_axis_scan<T>(
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    direction: ScanDirection,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<Option<AxisScanDispatch>> {
    if !width.get().is_power_of_two() {
        return Err(HephaestusError::DispatchFailed {
            message: format!("scan block width {} must be a power of two", width.get()),
        });
    }
    if axis >= 2 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("scan axis {axis} is out of bounds for rank-2 scan"),
        });
    }
    if input.layout.shape != output.layout.shape {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "scan output shape mismatch: input {:?}, out {:?}",
                input.layout.shape, output.layout.shape
            ),
        });
    }
    if input.buffer.aliases(output.buffer) {
        return Err(HephaestusError::DispatchFailed {
            message: "scan output buffer must not alias input buffer".to_string(),
        });
    }
    if output.layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "scan output layout must not contain zero-stride aliasing".to_string(),
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

    let direction_bit = match direction {
        ScanDirection::Forward => 0usize,
        ScanDirection::Reverse => 2usize,
    };
    let meta = AxisScanMeta {
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
            to_u32(axis | direction_bit, "axis and direction")?,
            to_u32(output_len, "output length")?,
        ],
    };

    Ok(Some(AxisScanDispatch {
        meta,
        groups: workgroups(output_len, width)?,
    }))
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
    Op: ScanWgslOp,
    T: WgslScalar + Pod + ScanIdentity<Op>,
{
    let Some(dispatch) = validate_axis_scan(input, axis, direction, output, width)? else {
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
    Op: ScanWgslOp,
    T: WgslScalar + Pod + ScanIdentity<Op>,
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
    T: WgslScalar + Pod + ScanIdentity<CumSumOp>,
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
    T: WgslScalar + Pod + ScanIdentity<CumSumOp>,
{
    scan_axis::<CumSumOp, T>(device, input, axis, ScanDirection::Forward, width)
}
