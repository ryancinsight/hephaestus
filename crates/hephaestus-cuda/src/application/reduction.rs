use crate::application::pipeline::{cached_kernel, grid_size, launch_kernel, LaunchConfig};
use crate::application::strided::{map_layout_err, StridedOperand};
use crate::infrastructure::buffer::CudaBuffer;
#[cfg(feature = "cuda")]
use crate::infrastructure::device::cuda_byte_count;
use crate::CudaDevice;
use bytemuck::Pod;
use hephaestus_core::{
    plan_axis_reduction, reduction_pass_count, validate_reduction_width, AxisReductionDispatch,
    AxisReductionMeta, BlockWidth, CombineExpr, ComputeDevice, CudaC, DeviceBuffer, DialectScalar,
    HephaestusError, IdentityToken, OpIdentity, Result,
};
use leto::Layout;

pub use hephaestus_core::{MaxOp, MinOp, SumOp};

fn shader_source<Op: CombineExpr<CudaC>, T: IdentityToken<Op, CudaC>>(width: BlockWidth) -> String {
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
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        identity = T::TOKEN,
        expr = Op::EXPR,
    )
}

/// Run reduction on the CUDA device, returning a 1-element buffer holding the result.
pub fn reduction<Op, T>(device: &CudaDevice, input: &CudaBuffer<T>) -> Result<CudaBuffer<T>>
where
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op> + IdentityToken<Op, CudaC>,
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
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op> + IdentityToken<Op, CudaC>,
{
    validate_reduction_width(width)?;

    if input.is_empty() {
        return device.upload(&[T::IDENTITY]);
    }

    if input.len() == 1 {
        let out = device.alloc_zeroed::<T>(1)?;
        #[cfg(feature = "cuda")]
        // SAFETY: this device's context is current on this thread
        // (`alloc_zeroed` above binds it); `input` and `out` are live
        // one-element device allocations (`input.len() == 1` checked above,
        // `out` freshly allocated with len 1), so the `size_of::<T>()`-byte
        // copy stays within both extents. The copy is asynchronous on the
        // null stream; both allocations outlive it because frees route
        // through synchronizing `cuMemFree`-family calls.
        unsafe {
            let bytes = core::mem::size_of::<T>();
            let byte_count = cuda_byte_count(bytes, "singleton reduction copy byte count")?;
            let res = cuda_oxide::sys::cuMemcpyDtoD_v2(out.raw(), input.raw(), byte_count);
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

        let mut src_val = source_ptr;
        let mut out_val = out_buffer.raw();
        let mut n_val = current_len as u32;

        // Argument list mirrors `reduction_kernel(const T*, T*, unsigned int)`.
        let mut args: [*mut std::ffi::c_void; 3] = [
            &mut src_val as *mut u64 as *mut std::ffi::c_void,
            &mut out_val as *mut u64 as *mut std::ffi::c_void,
            &mut n_val as *mut u32 as *mut std::ffi::c_void,
        ];

        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size_val, width),
            &mut args,
        )?;

        temp_buffers.push(out_buffer);
        current_len = out_len;
    }

    Ok(temp_buffers
        .pop()
        .expect("invariant: multi-element reduction allocates a final buffer"))
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

fn reject_empty_axis(axis_len: usize, op_name: &'static str, axis: usize) -> Result<()> {
    if axis_len == 0 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("{op_name} is undefined for empty axis {axis}"),
        });
    }
    Ok(())
}

fn plan_axis_reduction_dispatch<T>(
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<Option<AxisReductionDispatch>> {
    plan_axis_reduction(
        input.layout,
        input.buffer.len(),
        output.layout,
        output.buffer.len(),
        axis,
        width,
        input.buffer.aliases(output.buffer),
    )
}

fn axis_reduction_shader_source<Op: CombineExpr<CudaC>, T: IdentityToken<Op, CudaC>>() -> String {
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
        ty = T::TYPE_TOKEN,
        identity = T::TOKEN,
        expr = Op::EXPR,
    )
}

fn mean_axis_shader_source<T: IdentityToken<SumOp, CudaC>>() -> String {
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
        ty = T::TYPE_TOKEN,
        identity = T::TOKEN,
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
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op> + IdentityToken<Op, CudaC>,
{
    let Some(dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
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

    let mut meta_val = dispatch.meta;
    let mut in_ptr = input.buffer.raw();
    let mut out_ptr = output.buffer.raw();

    // Argument list mirrors `axis_reduction_kernel(AxisReductionMeta, const T*, T*)`.
    let mut args: [*mut std::ffi::c_void; 3] = [
        &mut meta_val as *mut AxisReductionMeta as *mut std::ffi::c_void,
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

/// Reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
pub fn reduce_axis<Op, T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op> + IdentityToken<Op, CudaC>,
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
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
{
    reject_empty_axis(axis_len(input, axis)?, "mean_axis", axis)?;
    let Some(dispatch) = plan_axis_reduction_dispatch(input, axis, output, width)? else {
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

    let mut meta_val = dispatch.meta;
    let mut in_ptr = input.buffer.raw();
    let mut out_ptr = output.buffer.raw();

    // Argument list mirrors `mean_axis_kernel(AxisReductionMeta, const T*, T*)`.
    let mut args: [*mut std::ffi::c_void; 3] = [
        &mut meta_val as *mut AxisReductionMeta as *mut std::ffi::c_void,
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

/// Mean-reduce a rank-2 strided matrix along `axis`, allocating a C-contiguous output buffer.
pub fn mean_axis<T>(
    device: &CudaDevice,
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
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
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
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
    T: DialectScalar<CudaC> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, CudaC>,
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
    T: DialectScalar<CudaC> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, CudaC>,
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
    T: DialectScalar<CudaC> + Pod + OpIdentity<MinOp> + IdentityToken<MinOp, CudaC>,
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
    T: DialectScalar<CudaC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, CudaC>,
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
    T: DialectScalar<CudaC> + Pod + OpIdentity<MaxOp> + IdentityToken<MaxOp, CudaC>,
{
    reject_empty_axis(axis_len(input, axis)?, "max_axis", axis)?;
    reduce_axis::<MaxOp, T>(device, input, axis, width)
}
