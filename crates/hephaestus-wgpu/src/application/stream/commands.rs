//! Command-stream seam implementation.

use super::command::WgpuCommandStream;
use super::helpers::push_storage_bind_group_entries;
use super::prepared::WgpuPrepared;
use crate::application::bindings::BindGroupEntries;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;
use eunomia::Pod;
use hephaestus_core::{
    Binding, CommandStream, DispatchGrid, HephaestusError, KernelSource, Result, Wgsl,
    validate_bindings,
};

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
