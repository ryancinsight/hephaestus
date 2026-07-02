use super::reject_output_alias;
use crate::application::cuda_type::CudaScalar;
use crate::application::pipeline::{cached_kernel, grid_size, launch_kernel, LaunchConfig};
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;
use bytemuck::Pod;
use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError, Result};

/// Zero-sized unary operation marker selecting the CUDA expression.
pub trait UnaryCudaOp: Copy + Send + Sync + 'static {
    /// CUDA expression mapping `x` (e.g. `"exp(x)"`).
    const CUDA_EXPR: &'static str;
}

/// Exponential operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpOp;

/// Natural logarithm operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct LnOp;

/// Sine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SinOp;

/// Cosine operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct CosOp;

/// Square-root operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SqrtOp;

/// Absolute value operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AbsOp;

/// Negation operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct NegOp;

/// Reciprocal operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecipOp;

/// Identity/copy operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityOp;

impl UnaryCudaOp for ExpOp {
    const CUDA_EXPR: &'static str = "exp(x)";
}

impl UnaryCudaOp for LnOp {
    const CUDA_EXPR: &'static str = "log(x)";
}

impl UnaryCudaOp for SinOp {
    const CUDA_EXPR: &'static str = "sin(x)";
}

impl UnaryCudaOp for CosOp {
    const CUDA_EXPR: &'static str = "cos(x)";
}

impl UnaryCudaOp for SqrtOp {
    const CUDA_EXPR: &'static str = "sqrt(x)";
}

impl UnaryCudaOp for AbsOp {
    const CUDA_EXPR: &'static str = "abs(x)";
}

impl UnaryCudaOp for NegOp {
    const CUDA_EXPR: &'static str = "-x";
}

impl UnaryCudaOp for RecipOp {
    const CUDA_EXPR: &'static str = "1.0 / x";
}

impl UnaryCudaOp for IdentityOp {
    const CUDA_EXPR: &'static str = "x";
}

fn shader_source<Op: UnaryCudaOp, T: CudaScalar>() -> String {
    format!(
        r#"
extern "C" __global__ void unary_kernel(
    const {ty}* a,
    {ty}* out,
    unsigned int n
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        {ty} x = a[i];
        out[i] = {expr};
    }}
}}
"#,
        ty = T::CUDA_TYPE,
        expr = Op::CUDA_EXPR,
    )
}

/// Run `out[i] = op(a[i])` on the CUDA device into distinct caller-owned storage.
pub fn unary_elementwise_into<Op, T>(
    device: &CudaDevice,
    a: &CudaBuffer<T>,
    out: &CudaBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: UnaryCudaOp,
    T: CudaScalar + Pod,
{
    if out.len() != a.len() {
        return Err(HephaestusError::LengthMismatch {
            host_len: out.len(),
            device_len: a.len(),
        });
    }
    reject_output_alias("unary", a, out)?;
    if out.is_empty() {
        return Ok(());
    }

    let grid_size_val = grid_size(out.len(), width)?;

    let key = format!(
        "unary_{}_{}_{}",
        std::any::type_name::<Op>(),
        std::any::type_name::<T>(),
        width.get()
    );

    let kernel = cached_kernel(device, key, "unary_kernel", || shader_source::<Op, T>())?;

    let mut a_ptr = a.raw();
    let mut out_ptr = out.raw();
    let mut n_val = out.len() as u32;

    // Argument list mirrors `unary_kernel(const T*, T*, unsigned int)`.
    let mut args: [*mut std::ffi::c_void; 3] = [
        &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut n_val as *mut u32 as *mut std::ffi::c_void,
    ];

    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid_size_val, width),
        &mut args,
    )
}

/// Run `out[i] = op(a[i])` on the CUDA device, allocating the output buffer.
pub fn unary_elementwise<Op, T>(device: &CudaDevice, a: &CudaBuffer<T>) -> Result<CudaBuffer<T>>
where
    Op: UnaryCudaOp,
    T: CudaScalar + Pod,
{
    let out = device.alloc_zeroed::<T>(a.len())?;
    unary_elementwise_into::<Op, T>(device, a, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
