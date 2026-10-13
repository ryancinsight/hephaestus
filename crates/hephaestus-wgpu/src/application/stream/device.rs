//! Device-side stream opening.

use super::command::WgpuBoundDispatch;
use super::prepared::WgpuPrepared;
use super::storage_bindings::push_storage_bind_group_entries;
use crate::application::bindings::BindGroupEntries;
use crate::application::prepared::checked_bind_group;
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{Binding, DispatchGrid, KernelSource, Result, Wgsl, validate_bindings};

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
