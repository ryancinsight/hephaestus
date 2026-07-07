//! Rank-2 prefix/suffix scan kernels over strided matrix operands on the CUDA device.

use bytemuck::Pod;
use core::marker::PhantomData;
use hephaestus_core::{
    plan_axis_scan, AxisScanMeta, BlockWidth, CombineExpr, ComputeDevice, CudaC, DeviceBuffer,
    DialectScalar, HephaestusError, IdentityToken, Result,
};
use leto::Layout;

use crate::application::pipeline::{cached_kernel, launch_kernel, LaunchConfig, PipelineKey};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;

pub use hephaestus_core::{CumProdOp, CumSumOp, ScanDirection};

struct AxisScanKernel<Op>(PhantomData<Op>);

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
    let Some(dispatch) = plan_axis_scan(
        input.layout,
        input.buffer.len(),
        output.layout,
        output.buffer.len(),
        axis,
        direction,
        width,
        input.buffer.aliases(output.buffer),
    )?
    else {
        return Ok(());
    };

    let key = PipelineKey::AxisScan {
        marker: std::any::TypeId::of::<AxisScanKernel<Op>>(),
        scalar: std::any::TypeId::of::<T>(),
        direction,
        axis,
        width: width.get(),
    };

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
        LaunchConfig::linear(dispatch.groups, width),
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
