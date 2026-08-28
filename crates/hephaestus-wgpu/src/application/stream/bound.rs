//! Retained binding state for repeated grouped dispatch.

use core::marker::PhantomData;

use hephaestus_core::{
    DispatchGrid, GroupedBinding, GroupedKernelSource, Result, Wgsl, validate_grouped_bindings,
};

use crate::application::bindings::BindGroups;
use crate::application::prepared::{device_owner, validate_device_owner};
use crate::infrastructure::device::{PipelineCache, WgpuDevice};

use super::{WgpuGroupedPrepared, WgpuGroupedSequence, build_grouped_bind_groups};

/// A grouped WGPU dispatch with fixed buffers, parameters, and launch geometry.
///
/// Construction owns the bind groups and parameter uniform needed by the
/// dispatch. Repeated [`Self::encode_in_sequence`] calls therefore record only
/// the retained command state and perform no Hephaestus-managed allocation or
/// bind-group construction.
#[must_use]
pub struct WgpuBoundGroupedDispatch<K> {
    pipeline: wgpu::ComputePipeline,
    bind_groups: BindGroups,
    _parameters: wgpu::Buffer,
    owner: PipelineCache,
    label: &'static str,
    grid: DispatchGrid,
    marker: PhantomData<K>,
}

impl<K> core::fmt::Debug for WgpuBoundGroupedDispatch<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuBoundGroupedDispatch")
            .field("pipeline", &self.pipeline)
            .field("bind_groups", &self.bind_groups)
            .field("label", &self.label)
            .field("grid", &self.grid)
            .finish_non_exhaustive()
    }
}

impl<K> WgpuBoundGroupedDispatch<K> {
    /// Encode the retained dispatch into an active grouped sequence.
    ///
    /// # Errors
    ///
    /// Returns a typed ownership failure when `sequence` belongs to a
    /// different WGPU device.
    pub fn encode_in_sequence(&self, sequence: &mut WgpuGroupedSequence<'_>) -> Result<()> {
        validate_device_owner(&self.owner, sequence.device(), "bound grouped dispatch")?;
        if self.grid.x == 0 || self.grid.y == 0 || self.grid.z == 0 {
            return Ok(());
        }
        sequence.pass.set_pipeline(&self.pipeline);
        for (group, bind_group) in &self.bind_groups {
            sequence.pass.set_bind_group(*group, bind_group, &[]);
        }
        sequence
            .pass
            .dispatch_workgroups(self.grid.x, self.grid.y, self.grid.z);
        Ok(())
    }
}

impl WgpuDevice {
    /// Bind fixed grouped-kernel resources for allocation-free repeated encode.
    ///
    /// # Errors
    ///
    /// Returns a typed binding-contract, size, allocation, or WGPU validation
    /// failure.
    pub fn bind_grouped_dispatch<K: GroupedKernelSource<Wgsl>>(
        &self,
        prepared: &WgpuGroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, Self>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<WgpuBoundGroupedDispatch<K>> {
        validate_device_owner(&prepared.owner, self, "grouped pipeline")?;
        validate_grouped_bindings::<Self>(K::LABEL, K::BINDINGS, bindings)?;
        let parameter_size = Self::byte_size::<K::Params>(1)?.max(wgpu::COPY_BUFFER_ALIGNMENT);
        let parameters = self.inner().create_buffer(&wgpu::BufferDescriptor {
            label: Some(K::LABEL),
            size: parameter_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue()
            .write_buffer(&parameters, 0, bytemuck::bytes_of(params));
        let validation = self.inner().push_error_scope(wgpu::ErrorFilter::Validation);
        let bind_groups = build_grouped_bind_groups(self.inner(), prepared, bindings, &parameters);
        let validation_error = moirai::block_on(validation.pop());
        let bind_groups = bind_groups?;
        if let Some(error) = validation_error {
            return Err(hephaestus_core::HephaestusError::DispatchFailed {
                message: format!("{} bind-group creation failed: {error}", prepared.label),
            });
        }
        Ok(WgpuBoundGroupedDispatch {
            pipeline: prepared.pipeline.clone(),
            bind_groups,
            _parameters: parameters,
            owner: device_owner(self),
            label: prepared.label,
            grid,
            marker: PhantomData,
        })
    }
}
