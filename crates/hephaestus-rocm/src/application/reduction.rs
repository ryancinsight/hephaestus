//! ROCm contiguous reductions over typed device buffers.

use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::DevicePtr;
use crate::{RocmBuffer, RocmDevice};
use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC,
    IdentityToken, OpIdentity, Result, reduction_pass_count, validate_reduction_width,
};

pub use hephaestus_core::{MaxOp, MinOp, SumOp};

fn shader_source<Op: CombineExpr<HipC>, T: IdentityToken<Op, HipC>>(width: BlockWidth) -> String {
    format!(
        r#"
extern "C" __global__ void reduction_kernel(
    const {ty}* input,
    {ty}* output,
    unsigned int n
) {{
    extern __shared__ {ty} shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    shared_data[tid] = i < n ? input[i] : {identity};
    __syncthreads();

    for (unsigned int stride = {wg}u / 2u; stride > 0u; stride /= 2u) {{
        if (tid < stride) {{
            {ty} lhs = shared_data[tid];
            {ty} rhs = shared_data[tid + stride];
            shared_data[tid] = {expr};
        }}
        __syncthreads();
    }}

    if (tid == 0u) {{
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

fn checked_work_items(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("ROCm reduction length {len} exceeds the HIP kernel argument range"),
    })
}

fn checked_shared_bytes<T>(width: BlockWidth) -> Result<u32> {
    let width_elements =
        usize::try_from(width.get()).map_err(|_| HephaestusError::DispatchFailed {
            message: format!(
                "ROCm reduction block width {} exceeds usize range",
                width.get()
            ),
        })?;
    width_elements
        .checked_mul(core::mem::size_of::<T>())
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!(
                "ROCm reduction shared-memory size overflows for block width {} and element size {}",
                width.get(),
                core::mem::size_of::<T>()
            ),
        })
}

/// Run a contiguous reduction on the ROCm device.
///
/// The result is a one-element buffer containing the operation identity for an
/// empty input. Non-empty inputs use a HIP workgroup tree and retain each
/// intermediate partial buffer until all asynchronous launches complete.
///
/// # Errors
///
/// Returns a typed dispatch or allocation error when the input length, block
/// width, shared-memory requirement, module compilation, or launch contract is
/// not supported by the device.
pub fn reduction<Op, T>(device: &RocmDevice, input: &RocmBuffer<T>) -> Result<RocmBuffer<T>>
where
    Op: CombineExpr<HipC>,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op> + IdentityToken<Op, HipC>,
{
    reduction_with_width::<Op, T>(device, input, BlockWidth::DEFAULT)
}

/// Run a contiguous reduction with a caller-selected power-of-two block width.
///
/// The width is part of the runtime-compiled module cache key and determines
/// the shared-memory tree shape. The reduction is multi-pass when a pass emits
/// more than one partial result.
///
/// # Errors
///
/// Returns a typed dispatch or allocation error when the input length, block
/// width, shared-memory requirement, module compilation, or launch contract is
/// not supported by the device.
pub fn reduction_with_width<Op, T>(
    device: &RocmDevice,
    input: &RocmBuffer<T>,
    width: BlockWidth,
) -> Result<RocmBuffer<T>>
where
    Op: CombineExpr<HipC>,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op> + IdentityToken<Op, HipC>,
{
    validate_reduction_width(width)?;

    if input.is_empty() {
        return device.upload(&[T::IDENTITY]);
    }

    let mut current_len = input.len();
    let mut source_ptr = input.raw();
    let mut partials = Vec::with_capacity(reduction_pass_count(current_len, width));
    let shared_bytes = checked_shared_bytes::<T>(width)?;
    let mut first_pass = true;

    while first_pass || current_len > 1 {
        let n_val = checked_work_items(current_len)?;
        let groups = grid_size(current_len, width)?;
        let output_len = current_len.div_ceil(width.get() as usize);
        let output = device.alloc_zeroed::<T>(output_len)?;
        let key = PipelineKey::Reduction {
            op: core::any::TypeId::of::<Op>(),
            scalar: core::any::TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "reduction_kernel", || {
            shader_source::<Op, T>(width)
        })?;

        let mut input_ptr: DevicePtr = source_ptr;
        let mut output_ptr: DevicePtr = output.raw();
        let mut n_val = n_val;
        let mut args: [*mut core::ffi::c_void; 3] = [
            (&mut input_ptr as *mut DevicePtr).cast(),
            (&mut output_ptr as *mut DevicePtr).cast(),
            (&mut n_val as *mut u32).cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear_shared(groups, width, shared_bytes),
            &mut args,
        )?;

        source_ptr = output.raw();
        current_len = output_len;
        partials.push(output);
        first_pass = false;
    }

    let Some(result) = partials.pop() else {
        return Err(HephaestusError::DispatchFailed {
            message: "ROCm reduction produced no output buffer".to_string(),
        });
    };
    Ok(result)
}
