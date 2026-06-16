use super::reject_output_alias;
use crate::application::cuda_type::CudaScalar;
use crate::application::elementwise::binary::BinaryCudaOp;
use crate::application::pipeline::{cached_kernel, grid_size};
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;
use bytemuck::Pod;
use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};

fn shader_source<Op: BinaryCudaOp, T: CudaScalar>() -> String {
    format!(
        r#"
extern "C" __global__ void scalar_kernel(
    const {ty}* input_ptr,
    {ty} scalar,
    {ty}* out,
    unsigned int n
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        {ty} a = input_ptr[i];
        {ty} b = scalar;
        out[i] = {expr};
    }}
}}
"#,
        ty = T::CUDA_TYPE,
        expr = Op::CUDA_EXPR,
    )
}

/// Run `out[i] = op(a[i], scalar)` on the CUDA device into distinct caller-owned storage.
pub fn scalar_elementwise_into<Op, T>(
    device: &CudaDevice,
    a: &CudaBuffer<T>,
    scalar: T,
    out: &CudaBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryCudaOp,
    T: CudaScalar + Pod,
{
    if out.len() != a.len() {
        return Err(HephaestusError::LengthMismatch {
            host_len: out.len(),
            device_len: a.len(),
        });
    }
    reject_output_alias("scalar", a, out)?;
    if out.is_empty() {
        return Ok(());
    }

    let grid_size_val = grid_size(out.len(), width)?;

    let key = format!(
        "scalar_{}_{}_{}",
        std::any::type_name::<Op>(),
        std::any::type_name::<T>(),
        width.get()
    );

    let kernel = cached_kernel(device, key, "scalar_kernel", || shader_source::<Op, T>())?;

    #[cfg(feature = "cuda")]
    {
        let mut a_ptr = a.raw();
        let mut val = scalar;
        let mut out_ptr = out.raw();
        let mut n_val = out.len() as u32;

        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut val as *mut T as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut n_val as *mut u32 as *mut std::ffi::c_void,
        ];

        // SAFETY: Buffers are valid, size matches.
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
        let _ = (kernel, grid_size_val, scalar);
    }

    Ok(())
}

/// Run `out[i] = op(a[i], scalar)` on the CUDA device, allocating the output buffer.
pub fn scalar_elementwise<Op, T>(
    device: &CudaDevice,
    a: &CudaBuffer<T>,
    scalar: T,
) -> Result<CudaBuffer<T>>
where
    Op: BinaryCudaOp,
    T: CudaScalar + Pod,
{
    let out = device.alloc_zeroed::<T>(a.len())?;
    scalar_elementwise_into::<Op, T>(device, a, scalar, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
