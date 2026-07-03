use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DialectScalar, HephaestusError, Result, UnaryExpr, Wgsl,
};

use super::reject_output_alias;
use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

pub use hephaestus_core::{
    AbsOp, CosOp, ExpNegOp, ExpOp, IdentityOp, LnOp, NegOp, RecipOp, SinOp, SqrtOp,
};

fn shader_source<Op: UnaryExpr<Wgsl>, T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&out)) {{
        return;
    }}
    let x = a[i];
    out[i] = {expr};
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        expr = <Op as UnaryExpr<Wgsl>>::EXPR,
    )
}

/// Run `out[i] = op(a[i])` on the device into distinct caller-owned storage.
pub fn unary_elementwise_into<Op, T>(
    device: &WgpuDevice,
    a: &WgpuBuffer<T>,
    out: &WgpuBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: UnaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    if out.len != a.len {
        return Err(HephaestusError::LengthMismatch {
            host_len: out.len,
            device_len: a.len,
        });
    }
    reject_output_alias("unary", a, out)?;
    if out.len == 0 {
        return Ok(());
    }
    let groups = workgroups(out.len, width)?;

    let key = (
        std::any::TypeId::of::<Op>(),
        std::any::TypeId::of::<T>(),
        width.get(),
    );
    let pipeline = cached_pipeline(device, key, "hephaestus-unary", || {
        shader_source::<Op, T>(width)
    });

    super::encode_elementwise(
        device,
        &pipeline,
        "hephaestus-unary",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out.buffer.as_entire_binding(),
            },
        ],
        groups,
    )
}

/// Run `out[i] = op(a[i])` on the device, allocating the output buffer.
pub fn unary_elementwise<Op, T>(device: &WgpuDevice, a: &WgpuBuffer<T>) -> Result<WgpuBuffer<T>>
where
    Op: UnaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    let out = device.alloc_zeroed::<T>(a.len)?;
    unary_elementwise_into::<Op, T>(device, a, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
