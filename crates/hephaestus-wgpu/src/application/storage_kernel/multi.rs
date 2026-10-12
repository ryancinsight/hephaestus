use eunomia::Pod;
use hephaestus_core::{DispatchGrid, HephaestusError, MultiStorageKernel, Result};

use super::bindings::{WgslStorageBinding, WgslStorageBindingLayout};
use crate::application::bindings::BindGroupEntries;
use crate::infrastructure::device::WgpuDevice;

/// Compiled WGSL kernel with N storage buffers and one uniform parameter block.
#[derive(Debug)]
pub struct WgslMultiStorageKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    storage_count: usize,
    params_binding: u32,
    label: &'static str,
}

impl WgslMultiStorageKernel {
    /// Compile a WGSL multi-storage-buffer kernel.
    ///
    /// `storage_layouts` describes every storage binding in `@group(0)`;
    /// `params_binding` is the `@group(0)` uniform parameter binding.
    ///
    /// # Errors
    /// Returns [`HephaestusError::DispatchFailed`] when the entry point is
    /// empty, no storage bindings are declared, or binding numbers collide.
    pub fn new(
        device: &WgpuDevice,
        label: &'static str,
        source: &'static str,
        entry_point: &'static str,
        storage_layouts: &[WgslStorageBindingLayout],
        params_binding: u32,
    ) -> Result<Self> {
        if entry_point.is_empty() {
            return Err(HephaestusError::DispatchFailed {
                message: "WGSL multi-storage kernel entry point is empty".to_string(),
            });
        }
        if storage_layouts.is_empty() {
            return Err(HephaestusError::DispatchFailed {
                message: "WGSL multi-storage kernel has no storage bindings".to_string(),
            });
        }
        validate_distinct_bindings(storage_layouts, params_binding)?;

        let shader = device
            .inner()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let mut entries = Vec::with_capacity(storage_layouts.len() + 1);
        for layout in storage_layouts {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: layout.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: layout.access.read_only(),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: params_binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let bind_group_layout =
            device
                .inner()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries: &entries,
                });
        let pipeline_layout =
            device
                .inner()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
        let pipeline = device
            .inner()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Ok(Self {
            pipeline,
            bind_group_layout,
            storage_count: storage_layouts.len(),
            params_binding,
            label,
        })
    }
}

impl<'a, P: Pod, const N: usize> MultiStorageKernel<WgpuDevice, P, [WgslStorageBinding<'a>; N]>
    for WgslMultiStorageKernel
{
    fn dispatch(
        &self,
        device: &WgpuDevice,
        bindings: [WgslStorageBinding<'a>; N],
        params: &P,
        grid: DispatchGrid,
    ) -> Result<()> {
        if N != self.storage_count {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "WGSL multi-storage kernel expected {} storage bindings, got {N}",
                    self.storage_count
                ),
            });
        }
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        let params_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<P>(1)?)?;

        device
            .queue()
            .write_buffer(&params_buffer, 0, eunomia::layout::bytes_of(params));

        let mut entries = BindGroupEntries::with_capacity(N + 1);
        for binding in bindings {
            entries.push(wgpu::BindGroupEntry {
                binding: binding.binding,
                resource: binding.buffer.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: self.params_binding,
            resource: params_buffer.as_entire_binding(),
        });

        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(self.label),
                layout: &self.bind_group_layout,
                entries: &entries,
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
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.x, grid.y, grid.z);
        }
        device.queue().submit(Some(encoder.finish()));
        Ok(())
    }
}

fn validate_distinct_bindings(
    storage_layouts: &[WgslStorageBindingLayout],
    params_binding: u32,
) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for layout in storage_layouts {
        if layout.binding == params_binding {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "storage binding {} collides with parameter binding",
                    layout.binding
                ),
            });
        }
        if !seen.insert(layout.binding) {
            return Err(HephaestusError::DispatchFailed {
                message: format!("duplicate storage binding {}", layout.binding),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_cases_share_process_state() {
        crate::test_support::run_cases(&[
            (
                "multi_storage_layout_rejects_parameter_collision",
                multi_storage_layout_rejects_parameter_collision as fn(),
            ),
            (
                "multi_storage_layout_rejects_duplicate_storage_binding",
                multi_storage_layout_rejects_duplicate_storage_binding as fn(),
            ),
        ]);
    }

    fn multi_storage_layout_rejects_parameter_collision() {
        let err = validate_distinct_bindings(
            &[
                WgslStorageBindingLayout::read_only(0),
                WgslStorageBindingLayout::read_write(2),
            ],
            2,
        )
        .unwrap_err();
        assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
    }

    fn multi_storage_layout_rejects_duplicate_storage_binding() {
        let err = validate_distinct_bindings(
            &[
                WgslStorageBindingLayout::read_only(0),
                WgslStorageBindingLayout::read_write(0),
            ],
            2,
        )
        .unwrap_err();
        assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
    }
}
