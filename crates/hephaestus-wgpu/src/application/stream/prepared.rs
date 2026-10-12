use std::marker::PhantomData;

use eunomia::Pod;
use hephaestus_core::{
    GroupedKernelDevice, GroupedKernelSource, KernelDevice, KernelSource, Result, Wgsl,
};

use super::dispatch::{binding_index, grouped_layouts, storage_layout_entries};
use super::sequence::WgpuCommandStream;
use crate::application::bindings::UniformBuffers;
use crate::application::prepared::device_owner;
use crate::infrastructure::device::{PipelineCache, WgpuDevice};

/// Prepared WGPU pipeline for a kernel source `K`.
pub struct WgpuPrepared<K> {
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) parameter_binding: u32,
    pub(super) label: &'static str,
    pub(super) marker: PhantomData<K>,
}

/// Prepared WGPU pipeline for a grouped kernel source `K`.
pub struct WgpuGroupedPrepared<K> {
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) bind_group_layouts: Vec<(u32, wgpu::BindGroupLayout)>,
    pub(super) parameter_group: u32,
    pub(super) parameter_binding: u32,
    pub(super) owner: PipelineCache,
    pub(super) label: &'static str,
    pub(super) marker: PhantomData<K>,
}

macro_rules! impl_prepared_traits {
    ($name:ident { $($field:ident),+ $(,)? }) => {
        impl<K> core::fmt::Debug for $name<K> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let mut debug = f.debug_struct(stringify!($name));
                $(debug.field(stringify!($field), &self.$field);)+
                debug.finish_non_exhaustive()
            }
        }

        impl<K> Clone for $name<K> {
            fn clone(&self) -> Self {
                Self {
                    $($field: self.$field.clone(),)+
                    marker: PhantomData,
                }
            }
        }
    };
}

impl_prepared_traits!(WgpuGroupedPrepared {
    pipeline,
    bind_group_layouts,
    parameter_group,
    parameter_binding,
    owner,
    label
});
impl_prepared_traits!(WgpuPrepared {
    pipeline,
    bind_group_layout,
    parameter_binding,
    label
});
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
