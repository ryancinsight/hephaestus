//! Prepared scalar reductions over fixed device buffers.

use std::any::TypeId;

use bytemuck::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, DialectScalar, IdentityToken, OpIdentity, Result, Wgsl,
    reduction_pass_count, validate_reduction_width,
};

use super::{
    ReductionFinalOpWrapper, ReductionOpWrapper, final_reduction_shader_source, shader_source,
};
use crate::application::pipeline::{cached_pipeline, encode_compute_pass, workgroups};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Prepared scalar reduction over a fixed input buffer.
///
/// This preallocates the reduction tree scratch buffers and bind groups once so
/// repeated reductions only encode the already-selected passes and submit the
/// command buffer.
pub struct PreparedReduction<T> {
    passes: Vec<PreparedPass>,
    temp_buffers: Vec<WgpuBuffer<T>>,
    singleton_source: Option<wgpu::Buffer>,
}

struct PreparedPass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    groups: u32,
}

impl<T> PreparedReduction<T> {
    /// Encode this reduction into an existing command encoder.
    ///
    /// This is the canonical reduction-tree encoding path used by individual,
    /// batched, and fused map-reduction dispatch.
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        if let Some(source) = self.singleton_source.as_ref() {
            encoder.copy_buffer_to_buffer(
                source,
                0,
                &self.output().buffer,
                0,
                WgpuDevice::byte_size::<T>(1)?,
            );
            return Ok(());
        }

        for pass in &self.passes {
            encode_compute_pass(
                encoder,
                &pass.pipeline,
                &pass.bind_group,
                pass.groups,
                "hephaestus-prepared-reduction-pass",
            );
        }
        Ok(())
    }

    /// Dispatch the prepared reduction once.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error if command encoding or submission cannot
    /// be completed by the backend.
    pub fn dispatch(&self, device: &WgpuDevice) -> Result<()> {
        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-prepared-reduction"),
            });
        self.encode(&mut encoder)?;
        device.queue().submit(Some(encoder.finish()));
        Ok(())
    }

    /// Return the 1-element output buffer holding the most recent dispatch result.
    #[must_use]
    pub fn output(&self) -> &WgpuBuffer<T> {
        self.temp_buffers
            .last()
            .expect("invariant: prepared reduction always owns a 1-element output")
    }

    pub(crate) fn into_output(mut self) -> WgpuBuffer<T> {
        self.temp_buffers
            .pop()
            .expect("invariant: prepared reduction always owns a 1-element output")
    }
}

/// Submit multiple prepared scalar reductions in one command buffer.
///
/// Each prepared reduction owns independent scratch/output buffers. This avoids
/// write-after-write hazards while amortizing WGPU submit/poll overhead across a
/// caller-visible batch of reductions.
///
/// # Errors
///
/// Returns a typed dispatch error if command encoding or submission cannot be
/// completed by the backend.
pub fn submit_prepared_reduction_batch<T>(
    device: &WgpuDevice,
    reductions: &[&PreparedReduction<T>],
) -> Result<()> {
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-prepared-reduction-batch"),
        });
    for reduction in reductions {
        reduction.encode(&mut encoder)?;
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

/// Prepare a scalar reduction over a fixed input buffer.
///
/// # Errors
///
/// Returns a typed error when the requested block width is invalid or when
/// scratch/output allocation fails.
pub fn prepare_reduction_with_width<Op, T>(
    device: &WgpuDevice,
    input: &WgpuBuffer<T>,
    width: BlockWidth,
) -> Result<PreparedReduction<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    validate_reduction_width(width)?;

    if input.len == 0 {
        let output = device.upload(&[T::IDENTITY])?;
        return Ok(PreparedReduction {
            passes: Vec::new(),
            temp_buffers: vec![output],
            singleton_source: None,
        });
    }
    if input.len == 1 {
        let output = device.alloc_zeroed::<T>(1)?;
        return Ok(PreparedReduction {
            passes: Vec::new(),
            temp_buffers: vec![output],
            singleton_source: Some(input.buffer.clone()),
        });
    }

    let standard_key = (
        TypeId::of::<ReductionOpWrapper<Op>>(),
        TypeId::of::<T>(),
        width.get(),
    );
    let standard_pipeline = cached_pipeline(device, standard_key, "hephaestus-reduction", || {
        shader_source::<Op, T>(width)
    });
    let final_key = (
        TypeId::of::<ReductionFinalOpWrapper<Op>>(),
        TypeId::of::<T>(),
        width.get(),
    );
    let final_pipeline = cached_pipeline(device, final_key, "hephaestus-reduction-final", || {
        final_reduction_shader_source::<Op, T>(width)
    });

    let mut current_len = input.len;
    let width_usize = usize::try_from(width.get())
        .expect("invariant: supported WGPU targets have at least 32-bit usize");
    let pass_count = reduction_pass_count(input.len, width);
    let mut temp_buffers: Vec<WgpuBuffer<T>> = Vec::with_capacity(pass_count);
    let mut passes = Vec::with_capacity(pass_count);

    while current_len > 1 {
        let final_pass = current_len <= width_usize * width_usize;
        let groups = if final_pass {
            1
        } else {
            workgroups(current_len, width)?
        };
        let out_len = if final_pass {
            1
        } else {
            current_len.div_ceil(width_usize)
        };
        let out_buffer = device.alloc_zeroed::<T>(out_len)?;
        let pipeline = if final_pass {
            &final_pipeline
        } else {
            &standard_pipeline
        };
        let source_resource = if temp_buffers.is_empty() {
            input.buffer.as_entire_binding()
        } else {
            temp_buffers
                .last()
                .expect("invariant: non-initial reduction pass has a previous buffer")
                .buffer
                .as_entire_binding()
        };
        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hephaestus-prepared-reduction"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: source_resource,
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buffer.buffer.as_entire_binding(),
                    },
                ],
            });
        passes.push(PreparedPass {
            pipeline: pipeline.clone(),
            bind_group,
            groups,
        });
        temp_buffers.push(out_buffer);
        current_len = out_len;
    }

    Ok(PreparedReduction {
        passes,
        temp_buffers,
        singleton_source: None,
    })
}

/// Prepare a scalar reduction over a fixed input buffer using the default block width.
#[inline]
pub fn prepare_reduction<Op, T>(
    device: &WgpuDevice,
    input: &WgpuBuffer<T>,
) -> Result<PreparedReduction<T>>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    prepare_reduction_with_width::<Op, T>(device, input, BlockWidth::DEFAULT)
}
