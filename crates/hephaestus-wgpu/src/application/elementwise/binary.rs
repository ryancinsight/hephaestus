use bytemuck::Pod;
use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError, Result};

use super::reject_output_alias;
use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Zero-sized binary operation marker selecting the WGSL expression.
///
/// Mirrors leto-ops' `BinaryOp` ZST pattern on the device side: one generic
/// [`binary_elementwise`] dispatch monomorphizes per `(Op, T)` pair; the op
/// contributes only its WGSL combine expression.
pub trait BinaryWgslOp: Copy + Send + Sync + 'static {
    /// WGSL expression combining `lhs` and `rhs` (e.g. `"lhs + rhs"`).
    const WGSL_EXPR: &'static str;
}

/// Addition marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct AddOp;

/// Subtraction marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct SubOp;

/// Multiplication marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct MulOp;

/// Division marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct DivOp;

/// Power marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct PowOp;

impl BinaryWgslOp for AddOp {
    const WGSL_EXPR: &'static str = "lhs + rhs";
}

impl BinaryWgslOp for SubOp {
    const WGSL_EXPR: &'static str = "lhs - rhs";
}

impl BinaryWgslOp for MulOp {
    const WGSL_EXPR: &'static str = "lhs * rhs";
}

impl BinaryWgslOp for DivOp {
    const WGSL_EXPR: &'static str = "lhs / rhs";
}

impl BinaryWgslOp for PowOp {
    const WGSL_EXPR: &'static str = "pow(lhs, rhs)";
}


fn shader_source<Op: BinaryWgslOp, T: WgslScalar>(width: BlockWidth) -> String {
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
        ty = T::WGSL_TYPE,
        wg = width.get(),
        expr = Op::WGSL_EXPR,
    )
}

/// Run `out[i] = op(a[i], b[i])` on the device into caller-owned storage.
///
/// Inputs and output must have equal length. The kernel is generated from the
/// `(Op, T, width)` monomorphization and dispatched in enough workgroups to
/// cover `out.len()`. The output buffer must not alias either input buffer.
pub fn binary_elementwise_into<Op, T>(
    device: &WgpuDevice,
    a: &WgpuBuffer<T>,
    b: &WgpuBuffer<T>,
    out: &WgpuBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryWgslOp,
    T: WgslScalar + Pod,
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

    let key = (
        std::any::TypeId::of::<Op>(),
        std::any::TypeId::of::<T>(),
        width.get(),
    );
    let pipeline = cached_pipeline(device, key, "hephaestus-elementwise", || {
        shader_source::<Op, T>(width)
    });

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-elementwise"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
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
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-elementwise"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-elementwise"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(())
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
    Op: BinaryWgslOp,
    T: WgslScalar + Pod,
{
    if a.len != b.len {
        return Err(HephaestusError::LengthMismatch {
            host_len: a.len,
            device_len: b.len,
        });
    }
    let out = device.alloc_zeroed::<T>(a.len)?;
    binary_elementwise_into::<Op, T>(device, a, b, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
