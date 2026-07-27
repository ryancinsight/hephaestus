use bytemuck::Pod;
use hephaestus_core::{
    BinaryExpr, BlockWidth, ComputeDevice, DialectScalar, HephaestusError, Result, TypedBinaryExpr,
    Wgsl,
};

use super::reject_output_alias;
use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

pub use hephaestus_core::{AddOp, DivOp, EqOp, GeOp, GtOp, LeOp, LtOp, MulOp, NeOp, PowOp, SubOp};

fn shader_source<T: DialectScalar<Wgsl>>(width: BlockWidth, expr: &'static str) -> String {
    format!(
        r#"@group(0) @binding(0) var<storage, read> a: array<{ty}>;
@group(0) @binding(1) var<storage, read> b: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{ty}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&out)) {{
        return;
    }}
    let lhs = a[i];
    let rhs = b[i];
    out[i] = {expr};
}}
"#,
        ty = T::TYPE_TOKEN,
        wg = width.get(),
        expr = expr,
    )
}

/// Run `out[i] = op(a[i], b[i])` on the device into caller-owned storage.
///
/// Inputs and output must have equal length. The kernel is generated from the
/// `(Op, T, width)` monomorphization and dispatched in enough workgroups to
/// cover `out.len()`. The output buffer must not alias either input buffer.
fn binary_elementwise_into_expression<T>(
    device: &WgpuDevice,
    a: &WgpuBuffer<T>,
    b: &WgpuBuffer<T>,
    out: &WgpuBuffer<T>,
    width: BlockWidth,
    operation: std::any::TypeId,
    expr: &'static str,
) -> Result<()>
where
    T: DialectScalar<Wgsl> + Pod,
{
    if a.len != b.len {
        return Err(HephaestusError::LengthMismatch {
            host_len: a.len,
            device_len: b.len,
        });
    }
    if out.len != a.len {
        return Err(HephaestusError::LengthMismatch {
            host_len: out.len,
            device_len: a.len,
        });
    }
    reject_output_alias("left", a, out)?;
    reject_output_alias("right", b, out)?;
    if out.len == 0 {
        return Ok(());
    }
    let groups = workgroups(out.len, width)?;

    let key = (operation, std::any::TypeId::of::<T>(), width.get());
    let pipeline = cached_pipeline(device, key, "hephaestus-elementwise", || {
        shader_source::<T>(width, expr)
    });

    super::encode_elementwise(
        device,
        &pipeline,
        "hephaestus-elementwise",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.buffer.as_entire_binding(),
            },
        ],
        groups,
    )
}

/// Run `out[i] = op(a[i], b[i])` for a scalar-aware expression on caller-owned
/// storage. The typed expression selects representation-correct literals for
/// the result mask.
pub fn binary_elementwise_typed_into<Op, T>(
    device: &WgpuDevice,
    a: &WgpuBuffer<T>,
    b: &WgpuBuffer<T>,
    out: &WgpuBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: TypedBinaryExpr<Wgsl, T>,
    T: DialectScalar<Wgsl> + Pod,
{
    binary_elementwise_into_expression::<T>(
        device,
        a,
        b,
        out,
        width,
        std::any::TypeId::of::<Op>(),
        <Op as TypedBinaryExpr<Wgsl, T>>::EXPR,
    )
}

/// Run a scalar-aware binary operation, allocating the output buffer.
pub fn binary_elementwise_typed<Op, T>(
    device: &WgpuDevice,
    a: &WgpuBuffer<T>,
    b: &WgpuBuffer<T>,
) -> Result<WgpuBuffer<T>>
where
    Op: TypedBinaryExpr<Wgsl, T>,
    T: DialectScalar<Wgsl> + Pod,
{
    let out = device.alloc_zeroed::<T>(a.len)?;
    binary_elementwise_typed_into::<Op, T>(device, a, b, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}

/// Run `out[i] = op(a[i], b[i])` on caller-owned storage.
pub fn binary_elementwise_into<Op, T>(
    device: &WgpuDevice,
    a: &WgpuBuffer<T>,
    b: &WgpuBuffer<T>,
    out: &WgpuBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    binary_elementwise_into_expression::<T>(
        device,
        a,
        b,
        out,
        width,
        std::any::TypeId::of::<Op>(),
        <Op as BinaryExpr<Wgsl>>::EXPR,
    )
}

/// Run `out[i] = op(a[i], b[i])` on the device, allocating the output buffer.
///
/// Inputs must have equal length. The kernel is generated from the `(Op, T)`
/// monomorphization and dispatched in `ceil(len / 256)` workgroups.
pub fn binary_elementwise<Op, T>(
    device: &WgpuDevice,
    a: &WgpuBuffer<T>,
    b: &WgpuBuffer<T>,
) -> Result<WgpuBuffer<T>>
where
    Op: BinaryExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod,
{
    // The length check is performed inside binary_elementwise_into; the output
    // buffer is allocated at a.len and into validates out.len == a.len (always
    // true) and a.len == b.len (our actual guard).
    let out = device.alloc_zeroed::<T>(a.len)?;
    binary_elementwise_into::<Op, T>(device, a, b, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
