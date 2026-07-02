//! Rank-2 prefix/suffix scan kernels over strided matrix operands on the CUDA device.

use bytemuck::{Pod, Zeroable};
use core::marker::PhantomData;
use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};
use leto::Layout;

use crate::application::cuda_type::CudaScalar;
use crate::application::pipeline::{cached_kernel, grid_size, launch_kernel, LaunchConfig};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;

/// Direction of a scan along an axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanDirection {
    /// Accumulate from index 0 upward.
    Forward,
    /// Accumulate from the last index downward.
    Reverse,
}

/// Zero-sized scan operation marker selecting the CUDA combine expression.
pub trait ScanCudaOp: Copy + Send + Sync + 'static {
    /// CUDA expression combining `lhs` and `rhs`.
    const CUDA_EXPR: &'static str;
}

/// Associates a scalar type and scan operation with the identity value.
pub trait ScanIdentity<Op>: CudaScalar {
    /// The identity value on the host side.
    const IDENTITY: Self;
    /// The CUDA C++ literal for the identity value.
    const CUDA_IDENTITY: &'static str;
}

/// Cumulative sum marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CumSumOp;

/// Cumulative product marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CumProdOp;

impl ScanCudaOp for CumSumOp {
    const CUDA_EXPR: &'static str = "lhs + rhs";
}

impl ScanCudaOp for CumProdOp {
    const CUDA_EXPR: &'static str = "lhs * rhs";
}

// ── CumSumOp Identity implementations ──
impl ScanIdentity<CumSumOp> for f32 {
    const IDENTITY: Self = 0.0;
    const CUDA_IDENTITY: &'static str = "0.0f";
}
impl ScanIdentity<CumSumOp> for u32 {
    const IDENTITY: Self = 0;
    const CUDA_IDENTITY: &'static str = "0u";
}
impl ScanIdentity<CumSumOp> for i32 {
    const IDENTITY: Self = 0;
    const CUDA_IDENTITY: &'static str = "0";
}

// ── CumProdOp Identity implementations ──
impl ScanIdentity<CumProdOp> for f32 {
    const IDENTITY: Self = 1.0;
    const CUDA_IDENTITY: &'static str = "1.0f";
}
impl ScanIdentity<CumProdOp> for u32 {
    const IDENTITY: Self = 1;
    const CUDA_IDENTITY: &'static str = "1u";
}
impl ScanIdentity<CumProdOp> for i32 {
    const IDENTITY: Self = 1;
    const CUDA_IDENTITY: &'static str = "1";
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
    grid_size: u32,
}

#[inline]
fn to_i32(value: isize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i32 range"),
    })
}

#[inline]
fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds u32 range"),
    })
}

#[inline]
fn map_layout_err(e: leto::LetoError) -> HephaestusError {
    HephaestusError::DispatchFailed {
        message: format!("layout rejected: {e}"),
    }
}

fn scan_shader_source<Op: ScanCudaOp, T: ScanIdentity<Op>>() -> String {
    format!(
        r#"
struct AxisScanMeta {{
    unsigned int input_shape[2];
    int input_strides[2];
    int output_strides[2];
    unsigned int _pre_offsets_pad[2];
    unsigned int offsets[4];
}};

__device__ unsigned int source_offset(AxisScanMeta meta, unsigned int row, unsigned int col) {{
    int off = (int)meta.offsets[0]
        + (int)row * meta.input_strides[0]
        + (int)col * meta.input_strides[1];
    return (unsigned int)off;
}}

extern "C" __global__ void scan_kernel(
    AxisScanMeta meta,
    const {ty}* input,
    {ty}* output
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= meta.offsets[3]) {{
        return;
    }}

    unsigned int rows = meta.input_shape[0];
    unsigned int cols = meta.input_shape[1];
    unsigned int axis = meta.offsets[2] & 1u;
    bool reverse = (meta.offsets[2] & 2u) != 0u;
    unsigned int out_row = i / cols;
    unsigned int out_col = i % cols;
    {ty} acc = {identity};

    if (axis == 0u) {{
        if (reverse) {{
            unsigned int count = rows - out_row;
            for (unsigned int s = 0u; s < count; s++) {{
                unsigned int scan_row = rows - 1u - s;
                {ty} lhs = acc;
                {ty} rhs = input[source_offset(meta, scan_row, out_col)];
                acc = {expr};
            }}
        }} else {{
            for (unsigned int scan_row = 0u; scan_row <= out_row; scan_row++) {{
                {ty} lhs = acc;
                {ty} rhs = input[source_offset(meta, scan_row, out_col)];
                acc = {expr};
            }}
        }}
    }} else {{
        if (reverse) {{
            unsigned int count = cols - out_col;
            for (unsigned int s = 0u; s < count; s++) {{
                unsigned int scan_col = cols - 1u - s;
                {ty} lhs = acc;
                {ty} rhs = input[source_offset(meta, out_row, scan_col)];
                acc = {expr};
            }}
        }} else {{
            for (unsigned int scan_col = 0u; scan_col <= out_col; scan_col++) {{
                {ty} lhs = acc;
                {ty} rhs = input[source_offset(meta, out_row, scan_col)];
                acc = {expr};
            }}
        }}
    }}

    int out_off = (int)meta.offsets[1]
        + (int)out_row * meta.output_strides[0]
        + (int)out_col * meta.output_strides[1];
    output[out_off] = acc;
}}
"#,
        ty = T::CUDA_TYPE,
        identity = T::CUDA_IDENTITY,
        expr = Op::CUDA_EXPR,
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
        .validate_storage_len(input.buffer.len())
        .map_err(map_layout_err)?;
    output
        .layout
        .validate_storage_len(output.buffer.len())
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

    let grid_size_val = grid_size(output_len, width)?;

    Ok(Some(AxisScanDispatch {
        meta,
        grid_size: grid_size_val,
    }))
}

/// Scan a rank-2 strided matrix along `axis`, preserving the input shape.
pub fn scan_axis_into<Op, T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    direction: ScanDirection,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    Op: ScanCudaOp,
    T: CudaScalar + Pod + ScanIdentity<Op>,
{
    let Some(dispatch) = validate_axis_scan(input, axis, direction, output, width)? else {
        return Ok(());
    };

    let key = format!(
        "axis_scan_{}_{}_{:?}_{}_{}",
        std::any::type_name::<AxisScanKernel<Op>>(),
        std::any::type_name::<T>(),
        direction,
        axis,
        width.get()
    );

    let kernel = cached_kernel(device, key, "scan_kernel", || scan_shader_source::<Op, T>())?;

    let mut meta_val = dispatch.meta;
    let mut in_ptr = input.buffer.raw();
    let mut out_ptr = output.buffer.raw();

    // Argument list mirrors `scan_kernel(AxisScanMeta, const T*, T*)`.
    let mut args: [*mut std::ffi::c_void; 3] = [
        &mut meta_val as *mut AxisScanMeta as *mut std::ffi::c_void,
        &mut in_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
    ];

    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(dispatch.grid_size, width),
        &mut args,
    )
}

/// Scan a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
pub fn scan_axis<Op, T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    direction: ScanDirection,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: ScanCudaOp,
    T: CudaScalar + Pod + ScanIdentity<Op>,
{
    let len = input.layout.checked_size().map_err(map_layout_err)?;
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
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: CudaScalar + Pod + ScanIdentity<CumSumOp>,
{
    scan_axis_into::<CumSumOp, T>(device, input, axis, ScanDirection::Forward, output, width)
}

/// Forward cumulative sum over a rank-2 strided matrix, allocating a C-contiguous output buffer.
#[inline]
pub fn cumsum<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    T: CudaScalar + Pod + ScanIdentity<CumSumOp>,
{
    scan_axis::<CumSumOp, T>(device, input, axis, ScanDirection::Forward, width)
}
