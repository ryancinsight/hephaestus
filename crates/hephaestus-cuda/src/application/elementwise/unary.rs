use super::reject_output_alias;
use crate::CudaDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result,
    UnaryExpr,
};

pub use hephaestus_core::{
    AbsOp, CosOp, ExpNegOp, ExpOp, IdentityOp, LnOp, NegOp, RecipOp, SinOp, SqrtOp,
};

fn shader_source<Op: UnaryExpr<CudaC>, T: DialectScalar<CudaC>>() -> String {
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
        ty = T::TYPE_TOKEN,
        expr = Op::EXPR,
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
    Op: UnaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
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

    let key = PipelineKey::Unary {
        op: std::any::TypeId::of::<Op>(),
        scalar: std::any::TypeId::of::<T>(),
        width: width.get(),
    };

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
    Op: UnaryExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod,
{
    let out = device.alloc_zeroed::<T>(a.len())?;
    unary_elementwise_into::<Op, T>(device, a, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
