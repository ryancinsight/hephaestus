use crate::application::cuda_type::CudaScalar;
use crate::application::pipeline::{cached_kernel, grid_size};
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;
use bytemuck::{Pod, Zeroable};
use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};

/// Zero-sized reduction operation marker selecting the CUDA combine expression.
pub trait ReductionCudaOp: Copy + Send + Sync + 'static {
    /// CUDA expression combining `lhs` and `rhs` (e.g. `"lhs + rhs"` or `"min(lhs, rhs)"`).
    const CUDA_EXPR: &'static str;
}

/// Associates a scalar type and reduction operation with the identity value.
pub trait ReductionIdentity<Op>: CudaScalar {
    /// The identity value on the host side.
    const IDENTITY: Self;
    /// The CUDA C++ literal for the identity value.
    const CUDA_IDENTITY: &'static str;
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

impl ReductionCudaOp for SumOp {
    const CUDA_EXPR: &'static str = "lhs + rhs";
}

impl ReductionCudaOp for MinOp {
    const CUDA_EXPR: &'static str = "min(lhs, rhs)";
}

impl ReductionCudaOp for MaxOp {
    const CUDA_EXPR: &'static str = "max(lhs, rhs)";
}

// ── SumOp Identity implementations ──
impl ReductionIdentity<SumOp> for f32 {
    const IDENTITY: Self = 0.0;
    const CUDA_IDENTITY: &'static str = "0.0f";
}
impl ReductionIdentity<SumOp> for u32 {
    const IDENTITY: Self = 0;
    const CUDA_IDENTITY: &'static str = "0u";
}
impl ReductionIdentity<SumOp> for i32 {
    const IDENTITY: Self = 0;
    const CUDA_IDENTITY: &'static str = "0";
}

// ── MinOp Identity implementations ──
impl ReductionIdentity<MinOp> for f32 {
    const IDENTITY: Self = f32::MAX;
    const CUDA_IDENTITY: &'static str = "3.402823466e+38f";
}
impl ReductionIdentity<MinOp> for u32 {
    const IDENTITY: Self = u32::MAX;
    const CUDA_IDENTITY: &'static str = "4294967295u";
}
impl ReductionIdentity<MinOp> for i32 {
    const IDENTITY: Self = i32::MAX;
    const CUDA_IDENTITY: &'static str = "2147483647";
}

// ── MaxOp Identity implementations ──
impl ReductionIdentity<MaxOp> for f32 {
    const IDENTITY: Self = f32::MIN;
    const CUDA_IDENTITY: &'static str = "-3.402823466e+38f";
}
impl ReductionIdentity<MaxOp> for u32 {
    const IDENTITY: Self = u32::MIN;
    const CUDA_IDENTITY: &'static str = "0u";
}
impl ReductionIdentity<MaxOp> for i32 {
    const IDENTITY: Self = i32::MIN;
    const CUDA_IDENTITY: &'static str = "-2147483648";
}

fn shader_source<Op: ReductionCudaOp, T: ReductionIdentity<Op>>(width: BlockWidth) -> String {
    format!(
        r#"
#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))

extern "C" __global__ void reduction_kernel(
    const {ty}* input,
    {ty}* output,
    unsigned int n
) {{
    __shared__ {ty} shared_data[{wg}];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {{
        shared_data[tid] = input[i];
    }} else {{
        shared_data[tid] = {identity};
    }}
    
    __syncthreads();
    
    for (unsigned int stride = {wg}u / 2u; stride > 0u; stride /= 2u) {{
        if (tid < stride) {{
            {ty} lhs = shared_data[tid];
            {ty} rhs = shared_data[tid + stride];
            shared_data[tid] = {expr};
        }}
        __syncthreads();
    }}
    
    if (tid == 0) {{
        output[blockIdx.x] = shared_data[0];
    }}
}}
"#,
        ty = T::CUDA_TYPE,
        wg = width.get(),
        identity = T::CUDA_IDENTITY,
        expr = Op::CUDA_EXPR,
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

fn reduction_pass_count(mut len: usize, width: BlockWidth) -> usize {
    let width = width.get() as usize;
    let mut passes = 0;
    while len > 1 {
        len = len.div_ceil(width);
        passes += 1;
    }
    passes
}

/// Run reduction on the CUDA device, returning a 1-element buffer holding the result.
pub fn reduction<Op, T>(device: &CudaDevice, input: &CudaBuffer<T>) -> Result<CudaBuffer<T>>
where
    Op: ReductionCudaOp,
    T: CudaScalar + Pod + ReductionIdentity<Op>,
{
    reduction_with_width::<Op, T>(device, input, BlockWidth::DEFAULT)
}

/// Run reduction on the CUDA device with a caller-selected power-of-two block width.
pub fn reduction_with_width<Op, T>(
    device: &CudaDevice,
    input: &CudaBuffer<T>,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: ReductionCudaOp,
    T: CudaScalar + Pod + ReductionIdentity<Op>,
{
    validate_reduction_width(width)?;

    if input.is_empty() {
        return device.upload(&[T::IDENTITY]);
    }

    if input.len() == 1 {
        let out = device.alloc_zeroed::<T>(1)?;
        #[cfg(feature = "cuda")]
        unsafe {
            let res =
                cuda_core::sys::cuMemcpyDtoD_v2(out.raw(), input.raw(), core::mem::size_of::<T>());
            if res != 0 {
                return Err(HephaestusError::TransferFailed {
                    message: format!("cuMemcpyDtoD_v2 failed with code: {res}"),
                });
            }
        }
        return Ok(out);
    }

    let mut current_len = input.len();
    let mut temp_buffers: Vec<CudaBuffer<T>> =
        Vec::with_capacity(reduction_pass_count(input.len(), width));

    while current_len > 1 {
        let grid_size_val = grid_size(current_len, width)?;
        let out_len = current_len.div_ceil(width.get() as usize);
        let out_buffer = device.alloc_zeroed::<T>(out_len)?;

        let key = format!(
            "reduction_{}_{}_{}",
            std::any::type_name::<Op>(),
            std::any::type_name::<T>(),
            width.get()
        );

        let kernel = cached_kernel(device, key, "reduction_kernel", || {
            shader_source::<Op, T>(width)
        })?;

        let source_ptr = if temp_buffers.is_empty() {
            input.raw()
        } else {
            temp_buffers
                .last()
                .expect("invariant: non-initial reduction pass has a previous buffer")
                .raw()
        };

        #[cfg(feature = "cuda")]
        {
            let mut src_val = source_ptr;
            let mut out_val = out_buffer.raw();
            let mut n_val = current_len as u32;

            let mut args: [*mut std::ffi::c_void; 3] = [
                &mut src_val as *mut u64 as *mut std::ffi::c_void,
                &mut out_val as *mut u64 as *mut std::ffi::c_void,
                &mut n_val as *mut u32 as *mut std::ffi::c_void,
            ];

            // SAFETY: Buffers are valid, dimensions match.
            unsafe {
                let res = cuda_core::sys::cuLaunchKernel(
                    kernel.func,
                    grid_size_val,
                    1,
                    1,
                    width.get(),
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
                if res != 0 {
                    return Err(HephaestusError::DispatchFailed {
                        message: format!("cuLaunchKernel failed with code: {res}"),
                    });
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (kernel, grid_size_val, source_ptr);
        }

        temp_buffers.push(out_buffer);
        current_len = out_len;
    }

    Ok(temp_buffers
        .pop()
        .expect("invariant: multi-element reduction allocates a final buffer"))
}

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
    grid_size: u32,
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

fn to_i32(value: isize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i32 range"),
    })
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds u32 range"),
    })
}

fn reject_empty_axis(axis_len: usize, op_name: &'static str, axis: usize) -> Result<()> {
    if axis_len == 0 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{op_name} is undefined for empty axis {axis}"),
        });
    }
    Ok(())
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

    let grid_size_val = grid_size(output_len, width)?;
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
    Ok(Some(AxisReductionDispatch { meta, grid_size: grid_size_val }))
}

fn axis_reduction_shader_source<Op: ReductionCudaOp, T: ReductionIdentity<Op>>() -> String {
    format!(
        r#"
struct AxisReductionMeta {{
    unsigned int input_shape[2];
    int input_strides[2];
    int output_strides[2];
    unsigned int _pre_offsets_pad[2];
    unsigned int offsets[4];
}};

extern "C" __global__ void axis_reduction_kernel(
    AxisReductionMeta meta,
    const {ty}* input,
    {ty}* output
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= meta.offsets[3]) {{
        return;
    }}

    unsigned int axis = meta.offsets[2];
    unsigned int axis_len = (axis == 0u) ? meta.input_shape[0] : meta.input_shape[1];
    unsigned int out_row = (axis == 0u) ? 0u : i;
    unsigned int out_col = (axis == 0u) ? i : 0u;
    {ty} acc = {identity};

    for (unsigned int r = 0u; r < axis_len; r++) {{
        unsigned int in_row = (axis == 0u) ? r : out_row;
        unsigned int in_col = (axis == 0u) ? out_col : r;
        int in_off = (int)meta.offsets[0]
            + (int)in_row * meta.input_strides[0]
            + (int)in_col * meta.input_strides[1];
        {ty} lhs = acc;
        {ty} rhs = input[in_off];
        acc = {expr};
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

fn mean_axis_shader_source<T: ReductionIdentity<SumOp>>() -> String {
    format!(
        r#"
struct AxisReductionMeta {{
    unsigned int input_shape[2];
    int input_strides[2];
    int output_strides[2];
    unsigned int _pre_offsets_pad[2];
    unsigned int offsets[4];
}};

extern "C" __global__ void mean_axis_kernel(
    AxisReductionMeta meta,
    const {ty}* input,
    {ty}* output
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= meta.offsets[3]) {{
        return;
    }}

    unsigned int axis = meta.offsets[2];
    unsigned int axis_len = (axis == 0u) ? meta.input_shape[0] : meta.input_shape[1];
    unsigned int out_row = (axis == 0u) ? 0u : i;
    unsigned int out_col = (axis == 0u) ? i : 0u;
    {ty} acc = {identity};

    for (unsigned int r = 0u; r < axis_len; r++) {{
        unsigned int in_row = (axis == 0u) ? r : out_row;
        unsigned int in_col = (axis == 0u) ? out_col : r;
        int in_off = (int)meta.offsets[0]
            + (int)in_row * meta.input_strides[0]
            + (int)in_col * meta.input_strides[1];
        acc = acc + input[in_off];
    }}

    int out_off = (int)meta.offsets[1]
        + (int)out_row * meta.output_strides[0]
        + (int)out_col * meta.output_strides[1];
    output[out_off] = acc / ({ty})axis_len;
}}
"#,
        ty = T::CUDA_TYPE,
        identity = T::CUDA_IDENTITY,
    )
}

/// Reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
pub fn reduce_axis_into<Op, T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    Op: ReductionCudaOp,
    T: CudaScalar + Pod + ReductionIdentity<Op>,
{
    let Some(dispatch) = validate_axis_reduction(input, axis, output, width)? else {
        return Ok(());
    };

    let key = format!(
        "axis_reduction_{}_{}_{}_{}",
        std::any::type_name::<Op>(),
        std::any::type_name::<T>(),
        axis,
        width.get()
    );

    let kernel = cached_kernel(device, key, "axis_reduction_kernel", || {
        axis_reduction_shader_source::<Op, T>()
    })?;

    #[cfg(feature = "cuda")]
    {
        let mut meta_val = dispatch.meta;
        let mut in_ptr = input.buffer.raw();
        let mut out_ptr = output.buffer.raw();

        let mut args: [*mut std::ffi::c_void; 3] = [
            &mut meta_val as *mut AxisReductionMeta as *mut std::ffi::c_void,
            &mut in_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        ];

        unsafe {
            let res = cuda_core::sys::cuLaunchKernel(
                kernel.func,
                dispatch.grid_size,
                1,
                1,
                width.get(),
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if res != 0 {
                return Err(HephaestusError::DispatchFailed {
                    message: format!("cuLaunchKernel failed with code: {res}"),
                });
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (kernel, dispatch);
    }

    Ok(())
}

/// Reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
pub fn reduce_axis<Op, T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: ReductionCudaOp,
    T: CudaScalar + Pod + ReductionIdentity<Op>,
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

/// Mean-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
pub fn mean_axis_into<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: CudaScalar + Pod + ReductionIdentity<SumOp>,
{
    reject_empty_axis(axis_len(input, axis)?, "mean_axis", axis)?;
    let Some(dispatch) = validate_axis_reduction(input, axis, output, width)? else {
        return Ok(());
    };

    let key = format!(
        "mean_axis_{}_{}_{}",
        std::any::type_name::<T>(),
        axis,
        width.get()
    );

    let kernel = cached_kernel(device, key, "mean_axis_kernel", || {
        mean_axis_shader_source::<T>()
    })?;

    #[cfg(feature = "cuda")]
    {
        let mut meta_val = dispatch.meta;
        let mut in_ptr = input.buffer.raw();
        let mut out_ptr = output.buffer.raw();

        let mut args: [*mut std::ffi::c_void; 3] = [
            &mut meta_val as *mut AxisReductionMeta as *mut std::ffi::c_void,
            &mut in_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        ];

        unsafe {
            let res = cuda_core::sys::cuLaunchKernel(
                kernel.func,
                dispatch.grid_size,
                1,
                1,
                width.get(),
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if res != 0 {
                return Err(HephaestusError::DispatchFailed {
                    message: format!("cuLaunchKernel failed with code: {res}"),
                });
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (kernel, dispatch);
    }

    Ok(())
}

/// Mean-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
pub fn mean_axis<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    T: CudaScalar + Pod + ReductionIdentity<SumOp>,
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

/// Sum-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn sum_axis_into<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: CudaScalar + Pod + ReductionIdentity<SumOp>,
{
    reduce_axis_into::<SumOp, T>(device, input, axis, output, width)
}

/// Sum-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn sum_axis<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    T: CudaScalar + Pod + ReductionIdentity<SumOp>,
{
    reduce_axis::<SumOp, T>(device, input, axis, width)
}

/// Min-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn min_axis_into<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: CudaScalar + Pod + ReductionIdentity<MinOp>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    reduce_axis_into::<MinOp, T>(device, input, axis, output, width)
}

/// Min-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn min_axis<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    T: CudaScalar + Pod + ReductionIdentity<MinOp>,
{
    reject_empty_axis(axis_len(input, axis)?, "min_axis", axis)?;
    reduce_axis::<MinOp, T>(device, input, axis, width)
}

/// Max-reduce a rank-2 strided matrix along `axis`, preserving the reduced axis as length one.
#[inline]
pub fn max_axis_into<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<()>
where
    T: CudaScalar + Pod + ReductionIdentity<MaxOp>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    reduce_axis_into::<MaxOp, T>(device, input, axis, output, width)
}

/// Max-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
#[inline]
pub fn max_axis<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    T: CudaScalar + Pod + ReductionIdentity<MaxOp>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    reduce_axis::<MaxOp, T>(device, input, axis, width)
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
