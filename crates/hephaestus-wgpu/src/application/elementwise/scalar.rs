use bytemuck::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, ComputeDevice, DialectScalar, HephaestusError, Result, Wgsl,
};

use super::reject_output_alias;
use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// ZST wrapper to generate a unique TypeId in the pipeline cache for scalar operations.
struct ScalarOpWrapper<Op>(core::marker::PhantomData<Op>);

fn shader_source<Op: BinaryExpr<Wgsl>, T: DialectScalar<Wgsl>>(width: BlockWidth) -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<uniform> scalar: {ty};
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&out)) {{
        return;
    }}
    let lhs = a[i];
    let rhs = scalar;
    out[i] = {expr};
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        expr = <Op as BinaryExpr<Wgsl>>::EXPR,
    )
}

/// Run `out[i] = op(a[i], scalar)` on the device into distinct caller-owned storage.
pub fn scalar_elementwise_into<Op, T>(
    device: &WgpuDevice,
    a: &WgpuBuffer<T>,
    scalar: T,
    out: &WgpuBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    if out.len != a.len {
        return Err(HephaestusError::LengthMismatch {
            host_len: out.len,
            device_len: a.len,
        });
    }
    reject_output_alias("scalar", a, out)?;
    if out.len == 0 {
        return Ok(());
    }
    let groups = workgroups(out.len, width)?;

    let raw_scalar_buf = device.get_uniform_buffer(WgpuDevice::byte_size::<T>(1)?)?;
    let scalar_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_scalar_buf);
    device
        .queue()
        .write_buffer(&scalar_buffer, 0, bytemuck::bytes_of(&scalar));

    let key = (
        std::any::TypeId::of::<ScalarOpWrapper<Op>>(),
        std::any::TypeId::of::<T>(),
        width.get(),
    );
    let pipeline = cached_pipeline(device, key, "hephaestus-scalar", || {
        shader_source::<Op, T>(width)
    });

    super::encode_elementwise(
        device,
        &pipeline,
        "hephaestus-scalar",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scalar_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.buffer.as_entire_binding(),
            },
        ],
        groups,
    )
}

/// Run `out[i] = op(a[i], scalar)` on the device, allocating the output buffer.
pub fn scalar_elementwise<Op, T>(
    device: &WgpuDevice,
    a: &WgpuBuffer<T>,
    scalar: T,
) -> Result<WgpuBuffer<T>>
where
    Op: BinaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    let out = device.alloc_zeroed::<T>(a.len)?;
    scalar_elementwise_into::<Op, T>(device, a, scalar, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
