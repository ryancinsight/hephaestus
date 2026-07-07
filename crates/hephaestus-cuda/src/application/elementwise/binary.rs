use super::reject_output_alias;
use crate::application::pipeline::{
    cached_kernel, grid_size, launch_kernel, LaunchConfig, PipelineKey,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::CudaDevice;
use bytemuck::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError,
    Result,
};

pub use hephaestus_core::{AddOp, DivOp, MulOp, PowOp, SubOp};

fn shader_source<Op: BinaryExpr<CudaC>, T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"
extern "C" __global__ void binary_kernel(
    const {ty}* lhs_in,
    const {ty}* rhs_in,
    {ty}* out,
    unsigned int n
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        {ty} lhs = lhs_in[i];
        {ty} rhs = rhs_in[i];
        out[i] = {expr};
    }}
}}
"#,
        ty = T::TYPE_TOKEN,
        expr = Op::EXPR,
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
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
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

    let key = PipelineKey::Binary {
        op: std::any::TypeId::of::<Op>(),
        scalar: std::any::TypeId::of::<T>(),
        width: width.get(),
    };

    let kernel = cached_kernel(device, key, "binary_kernel", || shader_source::<Op, T>())?;

    let mut lhs_ptr = lhs.raw();
    let mut rhs_ptr = rhs.raw();
    let mut out_ptr = out.raw();
    let mut n_val = out.len() as u32;

    // Argument list mirrors `binary_kernel(const T*, const T*, T*, unsigned int)`.
    let mut args: [*mut std::ffi::c_void; 4] = [
        &mut lhs_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut rhs_ptr as *mut u64 as *mut std::ffi::c_void,
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

/// Run `out[i] = op(lhs[i], rhs[i])` on the CUDA device, allocating the output buffer.
pub fn binary_elementwise<Op, T>(
    device: &CudaDevice,
    lhs: &CudaBuffer<T>,
    rhs: &CudaBuffer<T>,
) -> Result<CudaBuffer<T>>
where
    Op: BinaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
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
