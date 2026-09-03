use super::reject_output_alias;
use crate::CudaDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use eunomia::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError,
    Result, TypedBinaryExpr,
};

pub use hephaestus_core::{AddOp, DivOp, EqOp, GeOp, GtOp, LeOp, LtOp, MulOp, NeOp, PowOp, SubOp};

fn shader_source<T: DialectScalar<CudaC>>(expr: &'static str) -> String {
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
        expr = expr,
    )
}

/// Run `out[i] = op(lhs[i], rhs[i])` on the CUDA device into distinct caller-owned storage.
fn binary_elementwise_into_expression<T>(
    device: &CudaDevice,
    lhs: &CudaBuffer<T>,
    rhs: &CudaBuffer<T>,
    out: &CudaBuffer<T>,
    width: BlockWidth,
    operation: std::any::TypeId,
    expr: &'static str,
) -> Result<()>
where
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
        op: operation,
        scalar: std::any::TypeId::of::<T>(),
        width: width.get(),
    };

    let kernel = cached_kernel(device, key, "binary_kernel", || shader_source::<T>(expr))?;

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

/// Run a scalar-aware binary operation into caller-owned storage.
pub fn binary_elementwise_typed_into<Op, T>(
    device: &CudaDevice,
    lhs: &CudaBuffer<T>,
    rhs: &CudaBuffer<T>,
    out: &CudaBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: TypedBinaryExpr<CudaC, T>,
    T: DialectScalar<CudaC> + Pod,
{
    binary_elementwise_into_expression::<T>(
        device,
        lhs,
        rhs,
        out,
        width,
        std::any::TypeId::of::<Op>(),
        <Op as TypedBinaryExpr<CudaC, T>>::EXPR,
    )
}

/// Run a scalar-aware binary operation, allocating the output buffer.
pub fn binary_elementwise_typed<Op, T>(
    device: &CudaDevice,
    lhs: &CudaBuffer<T>,
    rhs: &CudaBuffer<T>,
) -> Result<CudaBuffer<T>>
where
    Op: TypedBinaryExpr<CudaC, T>,
    T: DialectScalar<CudaC> + Pod,
{
    let out = device.alloc_uninitialized::<T>(lhs.len())?;
    binary_elementwise_typed_into::<Op, T>(device, lhs, rhs, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}

/// Run `out[i] = op(lhs[i], rhs[i])` into caller-owned storage.
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
    binary_elementwise_into_expression::<T>(
        device,
        lhs,
        rhs,
        out,
        width,
        std::any::TypeId::of::<Op>(),
        <Op as BinaryExpr<CudaC>>::EXPR,
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
    let out = device.alloc_uninitialized::<T>(lhs.len())?;
    binary_elementwise_into::<Op, T>(device, lhs, rhs, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
