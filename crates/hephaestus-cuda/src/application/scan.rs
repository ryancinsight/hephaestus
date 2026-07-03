//! Rank-2 prefix/suffix scan kernels over strided matrix operands on the CUDA device.

use bytemuck::{Pod, Zeroable};
use core::marker::PhantomData;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError,
    IdentityToken, Result,
};
use leto::Layout;

use crate::application::pipeline::{cached_kernel, grid_size, launch_kernel, LaunchConfig};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;

pub use hephaestus_core::{CumProdOp, CumSumOp};

/// Direction of a scan along an axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanDirection {
    /// Accumulate from index 0 upward.
    Forward,
    /// Accumulate from the last index downward.
    Reverse,
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

fn scan_shader_source<Op: CombineExpr<CudaC>, T: IdentityToken<Op, CudaC>>() -> String {
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

__device__ unsigned int dest_offset(AxisScanMeta meta, unsigned int row, unsigned int col) {{
    int off = (int)meta.offsets[1]
        + (int)row * meta.output_strides[0]
        + (int)col * meta.output_strides[1];
    return (unsigned int)off;
}}

// One thread owns one full scan line and walks it sequentially, writing every
// prefix: O(L) work per length-L line. The combine order is strictly
// left-to-right (right-to-left when reversed), matching the sequential
// reference exactly (bitwise-identical floating-point results).
extern "C" __global__ void scan_kernel(
    AxisScanMeta meta,
    const {ty}* input,
    {ty}* output
) {{
    unsigned int line = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= meta.offsets[3]) {{
        return;
    }}

    unsigned int rows = meta.input_shape[0];
    unsigned int cols = meta.input_shape[1];
    unsigned int axis = meta.offsets[2] & 1u;
    bool reverse = (meta.offsets[2] & 2u) != 0u;
    unsigned int len = (axis == 0u) ? rows : cols;
    {ty} acc = {identity};

    // `axis` and `reverse` are uniform across the launch, so these selects
    // never diverge within a warp.
    for (unsigned int s = 0u; s < len; s++) {{
        unsigned int idx = reverse ? (len - 1u - s) : s;
        unsigned int row = (axis == 0u) ? idx : line;
        unsigned int col = (axis == 0u) ? line : idx;
        {ty} lhs = acc;
        {ty} rhs = input[source_offset(meta, row, col)];
        acc = {expr};
        output[dest_offset(meta, row, col)] = acc;
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        identity = T::TOKEN,
        expr = Op::EXPR,
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
    // One thread per scan line: lines run along `axis`, so their count is the
    // orthogonal extent. Non-zero here because output_len > 0.
    let line_count = input.layout.shape[1 - axis];
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
            to_u32(line_count, "scan line count")?,
        ],
    };

    let grid_size_val = grid_size(line_count, width)?;

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
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + IdentityToken<Op, CudaC>,
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
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + IdentityToken<Op, CudaC>,
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
    T: DialectScalar<CudaC> + Pod + IdentityToken<CumSumOp, CudaC>,
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
    T: DialectScalar<CudaC> + Pod + IdentityToken<CumSumOp, CudaC>,
{
    scan_axis::<CumSumOp, T>(device, input, axis, ScanDirection::Forward, width)
}
