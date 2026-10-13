//! Bind-group layout helpers.

use crate::application::bindings::BindGroupEntries;
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{Access, Binding, BindingDecl, HephaestusError, Result};

pub(crate) fn binding_index(label: &str, binding: usize) -> Result<u32> {
    u32::try_from(binding).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{label}: binding index exceeds u32::MAX"),
    })
}

pub(crate) fn push_storage_bind_group_entries<'a, 'b>(
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

pub(crate) fn storage_layout_entries(
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
