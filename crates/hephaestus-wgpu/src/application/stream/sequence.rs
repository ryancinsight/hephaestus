use eunomia::Pod;
use hephaestus_core::{
    Binding, CommandStream, DispatchGrid, GroupedBinding, GroupedCommandStream,
    GroupedKernelSequence, GroupedKernelSource, HephaestusError, KernelSource, Result, Wgsl,
    validate_bindings, validate_grouped_bindings,
};

use super::dispatch::{encode_grouped_on_pass, push_storage_bind_group_entries};
use super::prepared::{WgpuGroupedPrepared, WgpuPrepared};
use crate::application::bindings::{BindGroupEntries, UniformBuffers};
use crate::application::prepared::{checked_bind_group, checked_submit_with_timeout};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// WGPU command stream for ordered kernel dispatch, copies, and fills.
pub struct WgpuCommandStream<'d> {
    pub(super) device: &'d WgpuDevice,
    pub(super) encoder: wgpu::CommandEncoder,
    pub(super) uniform_buffers: UniformBuffers,
}

/// A fully bound WGPU dispatch whose immutable resources are prepared once.
///
/// Device provenance is held and checked once by the enclosing prepared
/// operation before this crate-private command is encoded.
pub(crate) struct WgpuBoundDispatch {
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) bind_group: wgpu::BindGroup,
    pub(super) _parameters: wgpu::Buffer,
    pub(super) grid: DispatchGrid,
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
    pub(super) device: &'s WgpuDevice,
    pub(super) pass: wgpu::ComputePass<'s>,
    pub(super) uniform_buffers: &'s mut UniformBuffers,
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
