//! Generic WGSL storage-kernel dispatch.

use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use bytemuck::Pod;
use hephaestus_core::{
    BinaryStorageKernel, DeviceBuffer, DispatchGrid, HephaestusError, MultiStorageDevice,
    MultiStorageKernel, Result, UnaryStorageKernel,
};

/// Storage-buffer access declared in a WGSL bind-group layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WgslStorageAccess {
    /// Read-only storage buffer.
    ReadOnly,
    /// Read-write storage buffer.
    ReadWrite,
}

impl WgslStorageAccess {
    #[inline]
    const fn read_only(self) -> bool {
        match self {
            Self::ReadOnly => true,
            Self::ReadWrite => false,
        }
    }
}

/// One storage binding in a WGSL kernel layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WgslStorageBindingLayout {
    binding: u32,
    access: WgslStorageAccess,
}

impl WgslStorageBindingLayout {
    /// Construct a read-only storage binding layout entry.
    #[must_use]
    #[inline]
    pub const fn read_only(binding: u32) -> Self {
        Self {
            binding,
            access: WgslStorageAccess::ReadOnly,
        }
    }

    /// Construct a read-write storage binding layout entry.
    #[must_use]
    #[inline]
    pub const fn read_write(binding: u32) -> Self {
        Self {
            binding,
            access: WgslStorageAccess::ReadWrite,
        }
    }
}

/// One typed WGPU storage-buffer binding for a multi-storage WGSL kernel.
#[derive(Clone, Copy, Debug)]
pub struct WgslStorageBinding<'a> {
    binding: u32,
    buffer: &'a wgpu::Buffer,
}

impl<'a> WgslStorageBinding<'a> {
    /// Bind `buffer` to a WGSL storage binding.
    #[must_use]
    #[inline]
    pub fn new<T: Pod>(binding: u32, buffer: &'a WgpuBuffer<T>) -> Self {
        Self {
            binding,
            buffer: buffer.raw(),
        }
    }
}

impl MultiStorageDevice for WgpuDevice {
    type StorageBinding<'a> = WgslStorageBinding<'a>;

    fn storage_binding<T: Pod>(binding: u32, buffer: &Self::Buffer<T>) -> Self::StorageBinding<'_> {
        WgslStorageBinding::new(binding, buffer)
    }
}

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

        let raw_params = device.get_uniform_buffer(WgpuDevice::byte_size::<P>(1)?)?;
        let params_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_params);
        device
            .queue()
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(params));

        let mut entries = Vec::with_capacity(N + 1);
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

    #[test]
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

/// Compiled WGSL kernel with one input storage buffer, one output storage
/// buffer, and one uniform parameter block.
#[derive(Debug)]
pub struct WgslUnaryStorageKernel {
    pipeline: wgpu::ComputePipeline,
    label: &'static str,
}

impl WgslUnaryStorageKernel {
    /// Compile a WGSL single-pass storage kernel.
    ///
    /// The WGSL module must expose `@group(0)` bindings in this order:
    /// read-only storage input at binding 0, read-write storage output at
    /// binding 1, and uniform parameters at binding 2.
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
                message: "WGSL kernel entry point is empty".to_string(),
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

impl<T, P> UnaryStorageKernel<WgpuDevice, T, P> for WgslUnaryStorageKernel
where
    T: Pod,
    P: Pod,
{
    fn dispatch(
        &self,
        device: &WgpuDevice,
        input: &WgpuBuffer<T>,
        output: &WgpuBuffer<T>,
        params: &P,
        grid: DispatchGrid,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: input.len(),
                device_len: output.len(),
            });
        }
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        let raw_params = device.get_uniform_buffer(WgpuDevice::byte_size::<P>(1)?)?;
        let params_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_params);
        device
            .queue()
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(params));

        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(self.label),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.raw().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.raw().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
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

        let raw_params = device.get_uniform_buffer(WgpuDevice::byte_size::<P>(1)?)?;
        let params_buffer = crate::infrastructure::pool::uniform_guard(device.clone(), raw_params);
        device
            .queue()
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(params));

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
