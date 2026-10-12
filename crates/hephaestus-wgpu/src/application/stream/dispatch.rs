use hephaestus_core::{
    Access, Binding, BindingDecl, DispatchGrid, GroupedBinding, GroupedKernelSource,
    HephaestusError, Result, Wgsl,
};

use super::prepared::WgpuGroupedPrepared;
use crate::application::bindings::{BindGroupEntries, BindGroups, UniformBuffers};
use crate::infrastructure::device::WgpuDevice;

pub(super) fn binding_index(label: &str, binding: usize) -> Result<u32> {
    u32::try_from(binding).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{label}: binding index exceeds u32::MAX"),
    })
}

pub(super) fn push_storage_bind_group_entries<'a, 'b>(
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

pub(super) fn storage_layout_entries(
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
pub(super) fn encode_grouped_on_pass<K: GroupedKernelSource<Wgsl>>(
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

pub(super) fn grouped_layouts<K: GroupedKernelSource<Wgsl>>(
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

pub(super) fn build_grouped_bind_groups<K>(
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
