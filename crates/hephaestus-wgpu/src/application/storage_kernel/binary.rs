use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use eunomia::Pod;
use hephaestus_core::{BinaryStorageKernel, DeviceBuffer, DispatchGrid, HephaestusError, Result};

/// Compiled WGSL kernel with two input storage buffers, one output storage
/// buffer, and one uniform parameter block.
#[derive(Debug)]
pub struct WgslBinaryStorageKernel {
    pipeline: wgpu::ComputePipeline,
    label: &'static str,
}

impl WgslBinaryStorageKernel {
    /// Compile a WGSL single-pass binary storage kernel.
    ///
    /// The WGSL module must expose storage buffers at `@group(0)` bindings
    /// `0`, `1`, and `2`, where bindings `0` and `1` are read-only inputs and
    /// binding `2` is the output. The POD uniform parameter block must be at
    /// `@group(1) @binding(0)`.
    ///
    /// # Errors
    /// Returns [`HephaestusError::DispatchFailed`] if `entry_point` is empty.
    pub fn new(
        device: &WgpuDevice,
        label: &'static str,
        source: &'static str,
        entry_point: &'static str,
    ) -> Result<Self> {
        if entry_point.is_empty() {
            return Err(HephaestusError::DispatchFailed {
                message: "WGSL binary kernel entry point is empty".to_string(),
            });
        }

        let shader = device
            .inner()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = device
            .inner()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Ok(Self { pipeline, label })
    }
}

impl<T, P> BinaryStorageKernel<WgpuDevice, T, P> for WgslBinaryStorageKernel
where
    T: Pod,
    P: Pod,
{
    fn dispatch(
        &self,
        device: &WgpuDevice,
        left: &WgpuBuffer<T>,
        right: &WgpuBuffer<T>,
        output: &WgpuBuffer<T>,
        params: &P,
        grid: DispatchGrid,
    ) -> Result<()> {
        if left.len() != right.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: left.len(),
                device_len: right.len(),
            });
        }
        if left.len() != output.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: left.len(),
                device_len: output.len(),
            });
        }
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        let params_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<P>(1)?)?;

        device
            .queue()
            .write_buffer(&params_buffer, 0, eunomia::layout::bytes_of(params));

        let storage_bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(self.label),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: left.raw().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: right.raw().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.raw().as_entire_binding(),
                    },
                ],
            });
        let params_bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(self.label),
                layout: &self.pipeline.get_bind_group_layout(1),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                }],
            });

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(self.label),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(self.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &storage_bind_group, &[]);
            pass.set_bind_group(1, &params_bind_group, &[]);
            pass.dispatch_workgroups(grid.x, grid.y, grid.z);
        }
        device.queue().submit(Some(encoder.finish()));
        Ok(())
    }
}
