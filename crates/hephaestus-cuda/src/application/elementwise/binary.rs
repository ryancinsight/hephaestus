use super::reject_output_alias;
use crate::application::cuda_type::CudaScalar;
use crate::application::pipeline::{cached_kernel, grid_size};
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;
use bytemuck::Pod;
use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};

/// Zero-sized binary operation marker selecting the CUDA expression.
pub trait BinaryCudaOp: Copy + Send + Sync + 'static {
    /// CUDA expression mapping `a` and `b` (e.g. `"a + b"`).
    const CUDA_EXPR: &'static str;
}

/// Addition operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AddOp;

/// Subtraction operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SubOp;

/// Multiplication operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MulOp;

/// Division operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct DivOp;

/// Power operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct PowOp;

impl BinaryCudaOp for AddOp {
    const CUDA_EXPR: &'static str = "a + b";
}

impl BinaryCudaOp for SubOp {
    const CUDA_EXPR: &'static str = "a - b";
}

impl BinaryCudaOp for MulOp {
    const CUDA_EXPR: &'static str = "a * b";
}

impl BinaryCudaOp for DivOp {
    const CUDA_EXPR: &'static str = "a / b";
}

impl BinaryCudaOp for PowOp {
    const CUDA_EXPR: &'static str = "pow(a, b)";
}

fn shader_source<Op: BinaryCudaOp, T: CudaScalar>() -> String {
    format!(
        r#"
extern "C" __global__ void binary_kernel(
    const {ty}* lhs,
    const {ty}* rhs,
    {ty}* out,
    unsigned int n
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        {ty} a = lhs[i];
        {ty} b = rhs[i];
        out[i] = {expr};
    }}
}}
"#,
        ty = T::CUDA_TYPE,
        expr = Op::CUDA_EXPR,
    )
}

/// Run `out[i] = op(lhs[i], rhs[i])` on the CUDA device into distinct caller-owned storage.
pub fn binary_elementwise_into<Op, T>(
    device: &CudaDevice,
    lhs: &CudaBuffer<T>,
    rhs: &CudaBuffer<T>,
    out: &CudaBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryCudaOp,
    T: CudaScalar + Pod,
{
    if lhs.len() != rhs.len() {
        return Err(HephaestusError::LengthMismatch {
            host_len: lhs.len(),
            device_len: rhs.len(),
        });
    }
    if out.len() != lhs.len() {
        return Err(HephaestusError::LengthMismatch {
            host_len: out.len(),
            device_len: lhs.len(),
        });
    }
    reject_output_alias("binary left", lhs, out)?;
    reject_output_alias("binary right", rhs, out)?;
    if out.is_empty() {
        return Ok(());
    }

    let grid_size_val = grid_size(out.len(), width)?;

    let key = format!(
        "binary_{}_{}_{}",
        std::any::type_name::<Op>(),
        std::any::type_name::<T>(),
        width.get()
    );

    let kernel = cached_kernel(device, key, "binary_kernel", || shader_source::<Op, T>())?;

    #[cfg(feature = "cuda")]
    {
        let mut lhs_ptr = lhs.raw();
        let mut rhs_ptr = rhs.raw();
        let mut out_ptr = out.raw();
        let mut n_val = out.len() as u32;

        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut lhs_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut rhs_ptr as *mut u64 as *mut std::ffi::c_void,
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
        let _ = (kernel, grid_size_val);
    }

    Ok(())
}

/// Run `out[i] = op(lhs[i], rhs[i])` on the CUDA device, allocating the output buffer.
pub fn binary_elementwise<Op, T>(
    device: &CudaDevice,
    lhs: &CudaBuffer<T>,
    rhs: &CudaBuffer<T>,
) -> Result<CudaBuffer<T>>
where
    Op: BinaryCudaOp,
    T: CudaScalar + Pod,
{
    if lhs.len() != rhs.len() {
        return Err(HephaestusError::LengthMismatch {
            host_len: lhs.len(),
            device_len: rhs.len(),
        });
    }
    let out = device.alloc_zeroed::<T>(lhs.len())?;
    binary_elementwise_into::<Op, T>(device, lhs, rhs, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
