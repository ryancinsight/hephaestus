//! Grouped-kernel streams and sequences.

use super::command::WgpuCommandStream;
use super::prepared::WgpuGroupedPrepared;
use crate::application::bindings::{BindGroupEntries, BindGroups, UniformBuffers};
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{
    CommandStream, DispatchGrid, GroupedBinding, GroupedCommandStream, GroupedKernelSequence,
    GroupedKernelSource, HephaestusError, Result, Wgsl, validate_grouped_bindings,
};

/// Active WGPU grouped-kernel sequence encoded into one compute pass.
pub struct WgpuGroupedSequence<'s> {
    pub(crate) device: &'s WgpuDevice,
    pub(crate) pass: wgpu::ComputePass<'s>,
    pub(crate) uniform_buffers: &'s mut UniformBuffers,
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

pub(crate) fn encode_grouped_on_pass<K: GroupedKernelSource<Wgsl>>(
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

pub(crate) fn grouped_layouts<K: GroupedKernelSource<Wgsl>>(
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

pub(crate) fn build_grouped_bind_groups<K>(
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
