//! Kernel-device seam implementations.

use super::command::WgpuCommandStream;
use super::grouped::grouped_layouts;
use super::prepared::{WgpuGroupedPrepared, WgpuPrepared};
use super::storage_bindings::{binding_index, storage_layout_entries};
use crate::application::bindings::UniformBuffers;
use crate::application::prepared::device_owner;
use crate::infrastructure::device::WgpuDevice;
use eunomia::Pod;
use hephaestus_core::{
    GroupedKernelDevice, GroupedKernelSource, KernelDevice, KernelSource, Result, Wgsl,
};
use std::marker::PhantomData;

impl KernelDevice for WgpuDevice {
    type Dialect = Wgsl;
    type BindingHandle<'a> = &'a wgpu::Buffer;
    type Prepared<K: KernelSource<Wgsl>> = WgpuPrepared<K>;
    type Stream<'d> = WgpuCommandStream<'d>;

    #[inline]
    fn binding_handle<T: Pod>(buffer: &Self::Buffer<T>) -> Self::BindingHandle<'_> {
        buffer.raw()
    }

    fn prepare<K: KernelSource<Wgsl>>(&self, kernel: &K) -> Result<Self::Prepared<K>> {
        let parameter_binding = binding_index(K::LABEL, K::BINDINGS.len())?;

        let source = kernel.source();
        let shader = self
            .inner()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(K::LABEL),
                source: wgpu::ShaderSource::Wgsl(source),
            });

        let entries = storage_layout_entries(K::LABEL, K::BINDINGS, parameter_binding)?;

        let bind_group_layout =
            self.inner()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(K::LABEL),
                    entries: &entries,
                });
        let pipeline_layout =
            self.inner()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(K::LABEL),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
        let pipeline = self
            .inner()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(K::LABEL),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(K::ENTRY),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Ok(WgpuPrepared {
            pipeline,
            bind_group_layout,
            parameter_binding,
            label: K::LABEL,
            marker: PhantomData,
        })
    }

    fn stream(&self) -> Result<Self::Stream<'_>> {
        let encoder = self
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-wgpu-command-stream"),
            });
        Ok(WgpuCommandStream {
            device: self,
            encoder,
            uniform_buffers: UniformBuffers::new(),
        })
    }
}

impl GroupedKernelDevice for WgpuDevice {
    type GroupedPrepared<K: GroupedKernelSource<Wgsl>> = WgpuGroupedPrepared<K>;
    type GroupedStream<'d> = WgpuCommandStream<'d>;

    fn prepare_grouped<K: GroupedKernelSource<Wgsl>>(
        &self,
        kernel: &K,
    ) -> Result<Self::GroupedPrepared<K>> {
        let source = kernel.source();
        let shader = self
            .inner()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(K::LABEL),
                source: wgpu::ShaderSource::Wgsl(source),
            });

        let bind_group_layouts =
            grouped_layouts::<K>(self.inner(), K::PARAM_GROUP, K::PARAM_BINDING)?;
        let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> = bind_group_layouts
            .iter()
            .map(|(_, layout)| Some(layout))
            .collect();
        let pipeline_layout =
            self.inner()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(K::LABEL),
                    bind_group_layouts: &layout_refs,
                    immediate_size: 0,
                });
        let pipeline = self
            .inner()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(K::LABEL),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(K::ENTRY),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Ok(WgpuGroupedPrepared {
            pipeline,
            bind_group_layouts,
            parameter_group: K::PARAM_GROUP,
            parameter_binding: K::PARAM_BINDING,
            owner: device_owner(self),
            label: K::LABEL,
            marker: PhantomData,
        })
    }

    fn grouped_stream(&self) -> Result<Self::GroupedStream<'_>> {
        self.stream()
    }
}
