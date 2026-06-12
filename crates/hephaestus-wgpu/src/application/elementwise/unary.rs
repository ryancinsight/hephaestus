use bytemuck::Pod;
use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError, Result};

use super::reject_output_alias;
use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Zero-sized unary operation marker selecting the WGSL expression.
pub trait UnaryWgslOp: Copy + Send + Sync + 'static {
    /// WGSL expression mapping `x` (e.g. `"exp(x)"`).
    const WGSL_EXPR: &'static str;
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

impl UnaryWgslOp for ExpOp {
    const WGSL_EXPR: &'static str = "exp(x)";
}

impl UnaryWgslOp for LnOp {
    const WGSL_EXPR: &'static str = "log(x)";
}

impl UnaryWgslOp for SinOp {
    const WGSL_EXPR: &'static str = "sin(x)";
}

impl UnaryWgslOp for CosOp {
    const WGSL_EXPR: &'static str = "cos(x)";
}

impl UnaryWgslOp for SqrtOp {
    const WGSL_EXPR: &'static str = "sqrt(x)";
}

impl UnaryWgslOp for AbsOp {
    const WGSL_EXPR: &'static str = "abs(x)";
}

impl UnaryWgslOp for NegOp {
    const WGSL_EXPR: &'static str = "-x";
}

impl UnaryWgslOp for RecipOp {
    const WGSL_EXPR: &'static str = "1.0 / x";
}

fn shader_source<Op: UnaryWgslOp, T: WgslScalar>(width: BlockWidth) -> String {
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
        ty = T::WGSL_TYPE,
        wg = width.get(),
        expr = Op::WGSL_EXPR,
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
    Op: UnaryWgslOp,
    T: WgslScalar + Pod,
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

    let key = (
        std::any::TypeId::of::<Op>(),
        std::any::TypeId::of::<T>(),
        width.get(),
    );
    let pipeline = cached_pipeline(device, key, "hephaestus-unary", || {
        shader_source::<Op, T>(width)
    });

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-unary"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-unary"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-unary"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups(out.len, width)?, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(())
}

/// Run `out[i] = op(a[i])` on the device, allocating the output buffer.
pub fn unary_elementwise<Op, T>(device: &WgpuDevice, a: &WgpuBuffer<T>) -> Result<WgpuBuffer<T>>
where
    Op: UnaryWgslOp,
    T: WgslScalar + Pod,
{
    let out = device.alloc_zeroed::<T>(a.len)?;
    unary_elementwise_into::<Op, T>(device, a, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
