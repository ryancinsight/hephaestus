use bytemuck::Pod;
use hephaestus_core::{ComputeDevice, HephaestusError, Result};

use crate::application::wgsl::WgslScalar;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Workgroup width of the elementwise kernels. One dimension is sufficient
/// for linear buffers; dispatch is `ceil(len / WORKGROUP_SIZE)` groups.
const WORKGROUP_SIZE: u32 = 256;

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

impl BinaryWgslOp for AddOp {
    const WGSL_EXPR: &'static str = "lhs + rhs";
}

impl BinaryWgslOp for SubOp {
    const WGSL_EXPR: &'static str = "lhs - rhs";
}

impl BinaryWgslOp for MulOp {
    const WGSL_EXPR: &'static str = "lhs * rhs";
}

/// WGSL template for binary elementwise kernels. `{ty}` and `{expr}` are
/// substituted per monomorphization; `arrayLength` guards the tail so the
/// dispatch never reads or writes past the logical element count.
fn shader_source<Op: BinaryWgslOp, T: WgslScalar>() -> String {
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
        wg = WORKGROUP_SIZE,
        expr = Op::WGSL_EXPR,
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
    if a.len == 0 {
        return Ok(out);
    }

    let key = (std::any::TypeId::of::<Op>(), std::any::TypeId::of::<T>());
    let pipeline = {
        let mut cache = device.pipeline_cache.lock().unwrap();
        if let Some(cached) = cache.get(&key) {
            cached.clone()
        } else {
            let module = device
                .inner()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("hephaestus-elementwise"),
                    source: wgpu::ShaderSource::Wgsl(shader_source::<Op, T>().into()),
                });
            let pipeline = device
                .inner()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("hephaestus-elementwise"),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn device_or_skip() -> Option<WgpuDevice> {
        match WgpuDevice::try_default("hephaestus-elementwise-test") {
            Ok(device) => Some(device),
            Err(e) => {
                eprintln!("skipping elementwise cache test: {e}");
                None
            }
        }
    }

    #[test]
    fn test_pipeline_cache_reused() {
        let Some(device) = device_or_skip() else {
            return;
        };
        let a = device.upload(&[1.0f32, 2.0]).unwrap();
        let b = device.upload(&[3.0f32, 4.0]).unwrap();

        let initial_size = { device.pipeline_cache.lock().unwrap().len() };

        // Dispatch first time: should compile and cache
        let _out1 = binary_elementwise::<AddOp, f32>(&device, &a, &b).unwrap();
        let size_after_first = { device.pipeline_cache.lock().unwrap().len() };
        assert_eq!(size_after_first, initial_size + 1);

        // Dispatch second time: should hit cache and not compile again
        let _out2 = binary_elementwise::<AddOp, f32>(&device, &a, &b).unwrap();
        let size_after_second = { device.pipeline_cache.lock().unwrap().len() };
        assert_eq!(size_after_second, size_after_first);
    }
}
