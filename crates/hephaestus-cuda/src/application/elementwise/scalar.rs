use super::reject_output_alias;
use crate::application::pipeline::{cached_kernel, grid_size, launch_kernel, LaunchConfig};
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;
use bytemuck::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError,
    Result,
};

fn shader_source<Op: BinaryExpr<CudaC>, T: DialectScalar<CudaC>>() -> String {
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
        {ty} lhs = input_ptr[i];
        {ty} rhs = scalar;
        out[i] = {expr};
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        expr = Op::EXPR,
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
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
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

    let mut a_ptr = a.raw();
    let mut val = scalar;
    let mut out_ptr = out.raw();
    let mut n_val = out.len() as u32;

    // Argument list mirrors `scalar_kernel(const T*, T, T*, unsigned int)`.
    let mut args: [*mut std::ffi::c_void; 4] = [
        &mut a_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut val as *mut T as *mut std::ffi::c_void,
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

/// Run `out[i] = op(a[i], scalar)` on the CUDA device, allocating the output buffer.
pub fn scalar_elementwise<Op, T>(
    device: &CudaDevice,
    a: &CudaBuffer<T>,
    scalar: T,
) -> Result<CudaBuffer<T>>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    let out = device.alloc_zeroed::<T>(a.len())?;
    scalar_elementwise_into::<Op, T>(device, a, scalar, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
