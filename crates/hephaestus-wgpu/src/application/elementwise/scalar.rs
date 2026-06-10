use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};
use wgpu::util::DeviceExt;

use crate::application::elementwise::binary::BinaryWgslOp;
use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

const WORKGROUP_SIZE: u32 = 256;

/// ZST wrapper to generate a unique TypeId in the pipeline cache for scalar operations.
struct ScalarOpWrapper<Op>(core::marker::PhantomData<Op>);

fn shader_source<Op: BinaryWgslOp, T: WgslScalar>() -> String {
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
        wg = WORKGROUP_SIZE,
        expr = Op::WGSL_EXPR,
    )
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
    if a.len == 0 {
        return Ok(out);
    }

    let scalar_buffer = device
        .inner()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hephaestus-scalar-uniform"),
            contents: bytemuck::bytes_of(&scalar),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let key = (
        std::any::TypeId::of::<ScalarOpWrapper<Op>>(),
        std::any::TypeId::of::<T>(),
    );
    let pipeline = {
        let mut cache = device.pipeline_cache.lock().unwrap();
        if let Some(cached) = cache.get(&key) {
            cached.clone()
        } else {
            let module = device
                .inner()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("hephaestus-scalar"),
                    source: wgpu::ShaderSource::Wgsl(shader_source::<Op, T>().into()),
                });
            let pipeline = device
                .inner()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("hephaestus-scalar"),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
            cache.insert(key, pipeline.clone());
            pipeline
        }
    };

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
        let groups = u32::try_from(a.len.div_ceil(WORKGROUP_SIZE as usize)).map_err(|_| {
            HephaestusError::DispatchFailed {
                message: format!("dispatch size {} exceeds u32 workgroup range", a.len),
            }
        })?;
        pass.dispatch_workgroups(groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(out)
}
