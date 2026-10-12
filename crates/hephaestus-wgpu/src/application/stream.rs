#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet HEPH-UNWRAP-1: pre-existing test-only debt"
    )
)]

//! WGPU implementation of the backend-neutral kernel command stream.

use std::marker::PhantomData;

use eunomia::Pod;
use hephaestus_core::{
    Access, Binding, BindingDecl, CommandStream, DispatchGrid, GroupedBinding,
    GroupedCommandStream, GroupedKernelDevice, GroupedKernelSequence, GroupedKernelSource,
    HephaestusError, KernelDevice, KernelSource, Result, Wgsl, validate_bindings,
    validate_grouped_bindings,
};

use crate::application::bindings::{BindGroupEntries, BindGroups, UniformBuffers};
use crate::application::prepared::{checked_bind_group, checked_submit_with_timeout, device_owner};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::{PipelineCache, WgpuDevice};

mod bound;
mod storage;

pub use bound::WgpuBoundGroupedDispatch;

/// Prepared WGPU pipeline for a kernel source `K`.
pub struct WgpuPrepared<K> {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    parameter_binding: u32,
    label: &'static str,
    marker: PhantomData<K>,
}

/// Prepared WGPU pipeline for a grouped kernel source `K`.
pub struct WgpuGroupedPrepared<K> {
    pipeline: wgpu::ComputePipeline,
    bind_group_layouts: Vec<(u32, wgpu::BindGroupLayout)>,
    parameter_group: u32,
    parameter_binding: u32,
    owner: PipelineCache,
    label: &'static str,
    marker: PhantomData<K>,
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

/// WGPU command stream for ordered kernel dispatch, copies, and fills.
pub struct WgpuCommandStream<'d> {
    device: &'d WgpuDevice,
    encoder: wgpu::CommandEncoder,
    uniform_buffers: UniformBuffers,
}

/// A fully bound WGPU dispatch whose immutable resources are prepared once.
///
/// Device provenance is held and checked once by the enclosing prepared
/// operation before this crate-private command is encoded.
pub(crate) struct WgpuBoundDispatch {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    _parameters: wgpu::Buffer,
    grid: DispatchGrid,
}

impl WgpuBoundDispatch {
    pub(crate) fn encode_in_pass(&self, pass: &mut wgpu::ComputePass<'_>) {
        if self.grid.x == 0 || self.grid.y == 0 || self.grid.z == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.grid.x, self.grid.y, self.grid.z);
    }
}

impl WgpuCommandStream<'_> {
    pub(crate) const fn device(&self) -> &WgpuDevice {
        self.device
    }

    pub(crate) fn encode_bound_sequence(
        &mut self,
        label: &str,
        prepared: &[WgpuBoundDispatch],
    ) -> Result<()> {
        let mut pass = self
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
        for command in prepared {
            command.encode_in_pass(&mut pass);
        }
        Ok(())
    }

    /// Submit this stream and wait at most `timeout` for device completion.
    ///
    /// This bounded form is intended for hosts and measurement harnesses whose
    /// external waits must have an explicit deadline.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::DispatchFailed`] when submission, device
    /// completion, or a scoped WGPU error fails before the deadline.
    pub fn submit_with_timeout(self, timeout: std::time::Duration) -> Result<()> {
        checked_submit_with_timeout(
            self.device,
            "hephaestus-command-stream",
            self.encoder,
            Some(timeout),
        )?;
        drop(self.uniform_buffers);
        Ok(())
    }

    /// Submit this stream and return the submission's queue index, so the
    /// caller can wait on exactly this submission (index-scoped poll)
    /// instead of draining the whole queue.
    pub(crate) fn submit_indexed(self) -> Result<wgpu::SubmissionIndex> {
        let submission_index = checked_submit_with_timeout(
            self.device,
            "hephaestus-command-stream",
            self.encoder,
            None,
        )?;
        drop(self.uniform_buffers);
        Ok(submission_index)
    }
}

impl WgpuDevice {
    pub(crate) fn bind<K: KernelSource<Wgsl>>(
        &self,
        prepared: &WgpuPrepared<K>,
        bindings: &[Binding<'_, WgpuDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<WgpuBoundDispatch> {
        validate_bindings::<WgpuDevice>(K::LABEL, K::BINDINGS, bindings)?;
        let parameter_size = Self::byte_size::<K::Params>(1)?.max(wgpu::COPY_BUFFER_ALIGNMENT);
        let parameters = self.inner().create_buffer(&wgpu::BufferDescriptor {
            label: Some(K::LABEL),
            size: parameter_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue()
            .write_buffer(&parameters, 0, eunomia::layout::bytes_of(params));

        let bind_group = {
            let mut entries = BindGroupEntries::with_capacity(bindings.len() + 1);
            push_storage_bind_group_entries(K::LABEL, &mut entries, bindings)?;
            entries.push(wgpu::BindGroupEntry {
                binding: prepared.parameter_binding,
                resource: parameters.as_entire_binding(),
            });
            checked_bind_group(self, &prepared.pipeline, prepared.label, &entries)?
        };

        Ok(WgpuBoundDispatch {
            pipeline: prepared.pipeline.clone(),
            bind_group,
            _parameters: parameters,
            grid,
        })
    }
}

/// Active WGPU grouped-kernel sequence encoded into one compute pass.
pub struct WgpuGroupedSequence<'s> {
    device: &'s WgpuDevice,
    pass: wgpu::ComputePass<'s>,
    uniform_buffers: &'s mut UniformBuffers,
}

impl<'s> WgpuGroupedSequence<'s> {
    pub(crate) const fn device(&self) -> &WgpuDevice {
        self.device
    }

    /// Borrow the active pass for WGPU-specific consumer commands.
    ///
    /// The sequence retains the device provenance established by its owning
    /// [`WgpuCommandStream`]. Raw commands remain subject to WGPU validation.
    pub fn raw_pass_mut(&mut self) -> &mut wgpu::ComputePass<'s> {
        &mut self.pass
    }
}

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

impl<'d> CommandStream<'d, WgpuDevice> for WgpuCommandStream<'d> {
    fn encode<K: KernelSource<Wgsl>>(
        &mut self,
        prepared: &WgpuPrepared<K>,
        bindings: &[Binding<'_, WgpuDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_bindings::<WgpuDevice>(K::LABEL, K::BINDINGS, bindings)?;
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        let raw_params = self
            .device
            .get_uniform_buffer(WgpuDevice::byte_size::<K::Params>(1)?)?;
        self.device
            .queue()
            .write_buffer(&raw_params, 0, eunomia::layout::bytes_of(params));

        let bind_group = {
            let mut entries = BindGroupEntries::with_capacity(bindings.len() + 1);
            push_storage_bind_group_entries(K::LABEL, &mut entries, bindings)?;
            entries.push(wgpu::BindGroupEntry {
                binding: prepared.parameter_binding,
                resource: raw_params.as_entire_binding(),
            });

            self.device
                .inner()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(prepared.label),
                    layout: &prepared.bind_group_layout,
                    entries: &entries,
                })
        };
        self.uniform_buffers.push(raw_params);

        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(prepared.label),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&prepared.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid.x, grid.y, grid.z);
        }

        Ok(())
    }

    fn copy<T: Pod>(&mut self, src: &WgpuBuffer<T>, dst: &WgpuBuffer<T>) -> Result<()> {
        use hephaestus_core::DeviceBuffer;
        if src.len() != dst.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: src.len(),
                device_len: dst.len(),
            });
        }
        // Whole-buffer copies may include the allocation's physical padding:
        // it contains no logical values belonging to another view.
        if !src.aliases(dst) && src.raw().size() != 0 {
            self.encoder
                .copy_buffer_to_buffer(src.raw(), 0, dst.raw(), 0, src.raw().size());
        }
        Ok(())
    }

    fn copy_prefix<T: Pod>(
        &mut self,
        src: &WgpuBuffer<T>,
        dst: &WgpuBuffer<T>,
        elements: usize,
    ) -> Result<()> {
        use hephaestus_core::DeviceBuffer;
        if elements > src.len() || elements > dst.len() {
            return Err(HephaestusError::LengthMismatch {
                host_len: elements,
                device_len: src.len().min(dst.len()),
            });
        }
        let byte_len = WgpuDevice::byte_size::<T>(elements)?;
        self.copy_prefix_bytes(src, dst, byte_len)
    }

    fn fill_zero<T: Pod>(&mut self, dst: &WgpuBuffer<T>) -> Result<()> {
        // The whole physical allocation is private to this logical buffer.
        // Clearing its padding makes sub-word scalar lengths valid as well.
        if dst.raw().size() != 0 {
            self.encoder.clear_buffer(dst.raw(), 0, None);
        }
        Ok(())
    }

    fn submit(self) -> Result<()> {
        self.submit_indexed().map(|_| ())
    }
}

fn binding_index(label: &str, binding: usize) -> Result<u32> {
    u32::try_from(binding).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{label}: binding index exceeds u32::MAX"),
    })
}

fn push_storage_bind_group_entries<'a, 'b>(
    label: &str,
    entries: &mut BindGroupEntries<'a>,
    bindings: &[Binding<'b, WgpuDevice>],
) -> Result<()>
where
    'b: 'a,
{
    for (binding, bound) in bindings.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: binding_index(label, binding)?,
            resource: bound.handle.as_entire_binding(),
        });
    }
    Ok(())
}

fn storage_layout_entries(
    label: &'static str,
    bindings: &[BindingDecl],
    parameter_binding: u32,
) -> Result<Vec<wgpu::BindGroupLayoutEntry>> {
    let mut entries = Vec::with_capacity(bindings.len() + 1);
    for (binding, decl) in bindings.iter().enumerate() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: binding_index(label, binding)?,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: matches!(decl.access, Access::ReadOnly),
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: parameter_binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    Ok(entries)
}

impl<'d> GroupedCommandStream<'d, WgpuDevice> for WgpuCommandStream<'d> {
    type Sequence<'s> = WgpuGroupedSequence<'s>;

    fn encode_grouped<K: GroupedKernelSource<Wgsl>>(
        &mut self,
        prepared: &WgpuGroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, WgpuDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_grouped_bindings::<WgpuDevice>(K::LABEL, K::BINDINGS, bindings)?;
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }

        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(prepared.label),
                    timestamp_writes: None,
                });
            encode_grouped_on_pass(
                self.device,
                &mut self.uniform_buffers,
                &mut pass,
                prepared,
                bindings,
                params,
                grid,
            )?;
        }

        Ok(())
    }

    fn encode_grouped_sequence<F>(&mut self, label: &str, encode: F) -> Result<()>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<()>,
    {
        let pass = self
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
        let mut sequence = WgpuGroupedSequence {
            device: self.device,
            pass,
            uniform_buffers: &mut self.uniform_buffers,
        };
        encode(&mut sequence)
    }

    fn submit_grouped(self) -> Result<()> {
        CommandStream::submit(self)
    }
}

impl<'s> GroupedKernelSequence<'s, WgpuDevice> for WgpuGroupedSequence<'s> {
    fn encode_grouped<K: GroupedKernelSource<Wgsl>>(
        &mut self,
        prepared: &WgpuGroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, WgpuDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_grouped_bindings::<WgpuDevice>(K::LABEL, K::BINDINGS, bindings)?;
        if grid.x == 0 || grid.y == 0 || grid.z == 0 {
            return Ok(());
        }
        encode_grouped_on_pass(
            self.device,
            self.uniform_buffers,
            &mut self.pass,
            prepared,
            bindings,
            params,
            grid,
        )
    }
}

fn encode_grouped_on_pass<K: GroupedKernelSource<Wgsl>>(
    device: &WgpuDevice,
    uniform_buffers: &mut UniformBuffers,
    pass: &mut wgpu::ComputePass<'_>,
    prepared: &WgpuGroupedPrepared<K>,
    bindings: &[GroupedBinding<'_, WgpuDevice>],
    params: &K::Params,
    grid: DispatchGrid,
) -> Result<()> {
    let raw_params = device.get_uniform_buffer(WgpuDevice::byte_size::<K::Params>(1)?)?;
    device
        .queue()
        .write_buffer(&raw_params, 0, eunomia::layout::bytes_of(params));

    let bind_groups = build_grouped_bind_groups(device.inner(), prepared, bindings, &raw_params)?;
    uniform_buffers.push(raw_params);

    pass.set_pipeline(&prepared.pipeline);
    for (group, bind_group) in &bind_groups {
        pass.set_bind_group(*group, bind_group, &[]);
    }
    pass.dispatch_workgroups(grid.x, grid.y, grid.z);
    Ok(())
}

fn grouped_layouts<K: GroupedKernelSource<Wgsl>>(
    device: &wgpu::Device,
    parameter_group: u32,
    parameter_binding: u32,
) -> Result<Vec<(u32, wgpu::BindGroupLayout)>> {
    let mut groups = std::collections::BTreeMap::<u32, Vec<wgpu::BindGroupLayoutEntry>>::new();
    let mut seen = std::collections::BTreeSet::<(u32, u32)>::new();
    for decl in K::BINDINGS {
        if !seen.insert((decl.group, decl.binding)) {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "{}: duplicate grouped binding ({}, {})",
                    K::LABEL,
                    decl.group,
                    decl.binding
                ),
            });
        }
        if decl.group == parameter_group && decl.binding == parameter_binding {
            return Err(HephaestusError::DispatchFailed {
                message: format!(
                    "{}: storage binding ({parameter_group}, {parameter_binding}) collides with parameter binding",
                    K::LABEL
                ),
            });
        }
        groups
            .entry(decl.group)
            .or_default()
            .push(wgpu::BindGroupLayoutEntry {
                binding: decl.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: matches!(decl.access, hephaestus_core::Access::ReadOnly),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
    }
    groups
        .entry(parameter_group)
        .or_default()
        .push(wgpu::BindGroupLayoutEntry {
            binding: parameter_binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

    if groups.keys().copied().ne(0..groups.len() as u32) {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "{}: WGPU grouped kernels require dense groups starting at 0",
                K::LABEL
            ),
        });
    }

    let mut layouts = Vec::with_capacity(groups.len());
    for (group, entries) in groups {
        layouts.push((
            group,
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(K::LABEL),
                entries: &entries,
            }),
        ));
    }
    Ok(layouts)
}

fn build_grouped_bind_groups<K>(
    device: &wgpu::Device,
    prepared: &WgpuGroupedPrepared<K>,
    bindings: &[GroupedBinding<'_, WgpuDevice>],
    params: &wgpu::Buffer,
) -> Result<BindGroups> {
    let mut entries = std::collections::BTreeMap::<u32, BindGroupEntries<'_>>::new();
    for bound in bindings {
        entries
            .entry(bound.group)
            .or_default()
            .push(wgpu::BindGroupEntry {
                binding: bound.binding,
                resource: bound.handle.as_entire_binding(),
            });
    }
    entries
        .entry(prepared.parameter_group)
        .or_default()
        .push(wgpu::BindGroupEntry {
            binding: prepared.parameter_binding,
            resource: params.as_entire_binding(),
        });

    let mut bind_groups = BindGroups::with_capacity(prepared.bind_group_layouts.len());
    for (group, layout) in &prepared.bind_group_layouts {
        let Some(group_entries) = entries.get(group) else {
            return Err(HephaestusError::DispatchFailed {
                message: format!("{}: missing bindings for group {group}", prepared.label),
            });
        };
        bind_groups.push((
            *group,
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(prepared.label),
                layout,
                entries: group_entries,
            }),
        ));
    }
    Ok(bind_groups)
}

#[cfg(test)]
#[path = "tests_stream.rs"]
mod tests;
