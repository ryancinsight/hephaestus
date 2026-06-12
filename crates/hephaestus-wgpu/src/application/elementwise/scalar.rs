use bytemuck::Pod;
use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError, Result};

use crate::application::elementwise::binary::BinaryWgslOp;
use crate::application::pipeline::{cached_pipeline, workgroups};
use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// ZST wrapper to generate a unique TypeId in the pipeline cache for scalar operations.
struct ScalarOpWrapper<Op>(core::marker::PhantomData<Op>);

fn shader_source<Op: BinaryWgslOp, T: WgslScalar>(width: BlockWidth) -> String {
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
        ty = T::WGSL_TYPE,
        wg = width.get(),
        expr = Op::WGSL_EXPR,
    )
}

/// Run `out[i] = op(a[i], scalar)` on the device into caller-owned storage.
pub fn scalar_elementwise_into<Op, T>(
    device: &WgpuDevice,
    a: &WgpuBuffer<T>,
    scalar: T,
    out: &WgpuBuffer<T>,
    width: BlockWidth,
) -> Result<()>
where
    Op: BinaryWgslOp,
    T: WgslScalar + Pod,
{
    if out.len != a.len {
        return Err(HephaestusError::LengthMismatch {
            host_len: out.len,
            device_len: a.len,
        });
    }
    if out.len == 0 {
        return Ok(());
    }

    let scalar_buffer = device.get_uniform_buffer(core::mem::size_of::<T>() as u64);
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

    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-scalar"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
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
        });

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-scalar"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-scalar"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups(out.len, width)?, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    device.recycle_uniform_buffer(scalar_buffer);

    Ok(())
}

/// Run `out[i] = op(a[i], scalar)` on the device, allocating the output buffer.
pub fn scalar_elementwise<Op, T>(
    device: &WgpuDevice,
    a: &WgpuBuffer<T>,
    scalar: T,
) -> Result<WgpuBuffer<T>>
where
    Op: BinaryWgslOp,
    T: WgslScalar + Pod,
{
    let out = device.alloc_zeroed::<T>(a.len)?;
    scalar_elementwise_into::<Op, T>(device, a, scalar, &out, BlockWidth::DEFAULT)?;
    Ok(out)
}
