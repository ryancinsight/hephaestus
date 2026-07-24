use super::{checked_work_items, reject_output_alias};
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::DevicePtr;
use crate::{RocmBuffer, RocmDevice};
use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result,
    UnaryExpr,
};

pub use hephaestus_core::{
    AbsOp, CosOp, ExpNegOp, ExpOp, IdentityOp, LnOp, NegOp, RecipOp, SinOp, SqrtOp,
};

fn shader_source<Op: UnaryExpr<HipC>, T: DialectScalar<HipC>>() -> String {
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

/// Run `out[i] = op(a[i])` into distinct caller-owned storage.
pub fn unary_elementwise_into<Op, T>(
    device: &RocmDevice,
    a: &RocmBuffer<T>,
    out: &RocmBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: UnaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
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
    let mut n_val = checked_work_items(out.len())?;
    let key = PipelineKey::Unary {
        op: std::any::TypeId::of::<Op>(),
        scalar: std::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "unary_kernel", || shader_source::<Op, T>())?;

    let mut a_ptr: DevicePtr = a.raw();
    let mut out_ptr: DevicePtr = out.raw();
    let mut args: [*mut core::ffi::c_void; 3] = [
        (&mut a_ptr as *mut DevicePtr).cast(),
        (&mut out_ptr as *mut DevicePtr).cast(),
        (&mut n_val as *mut u32).cast(),
    ];
    launch_kernel(
        device,
        &kernel,
        LaunchConfig::linear(grid_size_val, width),
        &mut args,
    )
}

/// Run `out[i] = op(a[i])`, allocating the output buffer.
pub fn unary_elementwise<Op, T>(device: &RocmDevice, a: &RocmBuffer<T>) -> Result<RocmBuffer<T>>
where
    Op: UnaryExpr<HipC>,
    T: DialectScalar<HipC> + Pod,
{
    let out = device.alloc_zeroed::<T>(a.len())?;
    unary_elementwise_into::<Op, T>(device, a, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
