//! Prepared scalar reductions over fixed device buffers.

use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, DialectScalar, IdentityToken, OpIdentity, Result, Wgsl,
    reduction_pass_count, validate_reduction_width,
};

use super::{
    PreparedAxisReduction, ReductionFinalOpWrapper, ReductionOpWrapper,
    final_reduction_shader_source, shader_source,
};
use crate::application::pipeline::{encode_compute_pass, try_cached_pipeline, workgroups};
use crate::application::prepared::{checked_bind_group, validate_buffer_owner};
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

/// Prepared scalar reduction over a fixed input buffer.
///
/// This preallocates the reduction tree scratch buffers and bind groups once so
/// repeated reductions only encode the already-selected passes and submit the
/// command buffer.
pub struct PreparedReduction<T> {
    work: PreparedWork,
    temp_buffers: Vec<WgpuBuffer<T>>,
}

struct PreparedPass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    groups: u32,
}

/// Type-independent command work for one prepared reduction.
///
/// The variants make empty, singleton-copy, and reduction-tree states
/// mutually exclusive. Storing the singleton byte size at preparation time
/// also keeps command encoding independent of the scalar monomorphization.
enum PreparedWork {
    Empty,
    SingletonCopy {
        source: wgpu::Buffer,
        byte_size: u64,
    },
    Tree(Box<[PreparedPass]>),
}

impl PreparedWork {
    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    fn has_singleton_copy(&self) -> bool {
        matches!(self, Self::SingletonCopy { .. })
    }

    fn tree_depth(&self) -> usize {
        match self {
            Self::Tree(passes) => passes.len(),
            Self::Empty | Self::SingletonCopy { .. } => 0,
        }
    }

    fn pass(&self, stage: usize) -> Option<&PreparedPass> {
        match self {
            Self::Tree(passes) => passes.get(stage),
            Self::Empty | Self::SingletonCopy { .. } => None,
        }
    }

    fn encode_singleton_copy(&self, encoder: &mut wgpu::CommandEncoder, output: &wgpu::Buffer) {
        if let Self::SingletonCopy { source, byte_size } = self {
            encoder.copy_buffer_to_buffer(source, 0, output, 0, *byte_size);
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, output: &wgpu::Buffer) {
        match self {
            Self::Empty => {}
            Self::SingletonCopy { .. } => self.encode_singleton_copy(encoder, output),
            Self::Tree(passes) => {
                for pass in passes {
                    encode_compute_pass(
                        encoder,
                        &pass.pipeline,
                        &pass.bind_group,
                        pass.groups,
                        "hephaestus-prepared-reduction-pass",
                    );
                }
            }
        }
    }
}

impl<T> PreparedReduction<T> {
    /// Encode this reduction into an existing command encoder.
    ///
    /// This is the canonical reduction-tree encoding path used by individual,
    /// batched, and fused map-reduction dispatch.
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        self.work.encode(encoder, &self.output().buffer);
        Ok(())
    }

    /// Dispatch the prepared reduction once.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error if command encoding or submission cannot
    /// be completed by the backend.
    pub fn dispatch(&self, device: &WgpuDevice) -> Result<()> {
        if self.work.is_empty() {
            return Ok(());
        }
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
/// Each prepared reduction owns independent scratch/output buffers. The encoder
/// groups equal-depth tree stages into one compute pass, preserving a pass
/// boundary between dependent stages while avoiding one pass per tree stage.
/// Singleton copies remain outside the compute passes. This avoids
/// write-after-write hazards while amortizing pass construction and WGPU
/// submit/poll overhead across a caller-visible batch. A batch containing only
/// prepared empty reductions returns without allocating a command encoder or
/// submitting an empty command buffer.
///
/// # Errors
///
/// Returns a typed dispatch error if command encoding or submission cannot be
/// completed by the backend.
pub fn submit_prepared_reduction_batch<T>(
    device: &WgpuDevice,
    reductions: &[&PreparedReduction<T>],
) -> Result<()> {
    submit_prepared_mixed_reduction_batch::<T, T>(device, reductions, &[])
}

/// Submit prepared scalar and axis reductions in one command buffer.
///
/// All reductions must be independent: no reduction may consume another
/// reduction's output. Singleton scalar copies are encoded before compute
/// work. Active axis reductions share the first compute pass with the first
/// stage of each scalar reduction tree; later scalar stages retain one pass per
/// dependency depth. Scalar and axis element types may differ because each
/// prepared plan owns its typed pipeline and bindings.
///
/// A batch with no singleton copy or active compute dispatch returns without
/// allocating a command encoder or submitting an empty command buffer.
///
/// # Examples
///
/// ```no_run
/// # use hephaestus_wgpu::{
/// #     PreparedAxisReduction, PreparedReduction, Result, WgpuDevice,
/// #     submit_prepared_mixed_reduction_batch,
/// # };
/// # fn submit<ScalarT, AxisT>(
/// #     device: &WgpuDevice,
/// #     scalar: &[&PreparedReduction<ScalarT>],
/// #     axis: &[&PreparedAxisReduction<AxisT>],
/// # ) -> Result<()> {
/// submit_prepared_mixed_reduction_batch(device, scalar, axis)
/// # }
/// ```
///
/// # Errors
///
/// Returns a typed dispatch error if command encoding or submission cannot be
/// completed by the backend.
pub fn submit_prepared_mixed_reduction_batch<ScalarT, AxisT>(
    device: &WgpuDevice,
    scalar_reductions: &[&PreparedReduction<ScalarT>],
    axis_reductions: &[&PreparedAxisReduction<AxisT>],
) -> Result<()> {
    let (has_singleton_copy, scalar_tree_depth) =
        scalar_reductions
            .iter()
            .fold((false, 0), |(has_singleton, max_depth), reduction| {
                (
                    has_singleton || reduction.work.has_singleton_copy(),
                    max_depth.max(reduction.work.tree_depth()),
                )
            });
    let has_axis_dispatch = axis_reductions
        .iter()
        .any(|reduction| reduction.pipeline.is_some());
    if !has_singleton_copy && scalar_tree_depth == 0 && !has_axis_dispatch {
        return Ok(());
    }

    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-prepared-reduction-batch"),
        });

    for reduction in scalar_reductions {
        reduction
            .work
            .encode_singleton_copy(&mut encoder, &reduction.output().buffer);
    }

    let compute_pass_count = scalar_tree_depth.max(usize::from(has_axis_dispatch));
    for stage in 0..compute_pass_count {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-prepared-mixed-reduction-batch-stage"),
            timestamp_writes: None,
        });
        if stage == 0 {
            for reduction in axis_reductions {
                let Some(pipeline) = reduction.pipeline.as_ref() else {
                    continue;
                };
                let Some(bind_group) = reduction.bind_group.as_ref() else {
                    continue;
                };
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(reduction.groups, 1, 1);
            }
        }
        for reduction in scalar_reductions {
            let Some(prepared_pass) = reduction.work.pass(stage) else {
                continue;
            };
            compute_pass.set_pipeline(&prepared_pass.pipeline);
            compute_pass.set_bind_group(0, &prepared_pass.bind_group, &[]);
            compute_pass.dispatch_workgroups(prepared_pass.groups, 1, 1);
        }
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
    validate_buffer_owner(input, device, "reduction input")?;

    if input.len == 0 {
        let output = device.upload(&[T::IDENTITY])?;
        return Ok(PreparedReduction {
            work: PreparedWork::Empty,
            temp_buffers: vec![output],
        });
    }
    if input.len == 1 {
        let output = device.alloc_zeroed::<T>(1)?;
        return Ok(PreparedReduction {
            work: PreparedWork::SingletonCopy {
                source: input.buffer.clone(),
                byte_size: WgpuDevice::byte_size::<T>(1)?,
            },
            temp_buffers: vec![output],
        });
    }

    let standard_key = (
        TypeId::of::<ReductionOpWrapper<Op>>(),
        TypeId::of::<T>(),
        width.get(),
    );
    let standard_pipeline =
        try_cached_pipeline(device, standard_key, "hephaestus-reduction", || {
            shader_source::<Op, T>(width)
        })?;
    let final_key = (
        TypeId::of::<ReductionFinalOpWrapper<Op>>(),
        TypeId::of::<T>(),
        width.get(),
    );
    let final_pipeline =
        try_cached_pipeline(device, final_key, "hephaestus-reduction-final", || {
            final_reduction_shader_source::<Op, T>(width)
        })?;

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
        let out_buffer = if out_len == 1 {
            device.alloc_zeroed::<T>(out_len)?
        } else {
            device.alloc_uninitialized::<T>(out_len)?
        };
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
        let bind_group = checked_bind_group(
            device,
            pipeline,
            "hephaestus-prepared-reduction",
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.buffer.as_entire_binding(),
                },
            ],
        )?;
        passes.push(PreparedPass {
            pipeline: pipeline.clone(),
            bind_group,
            groups,
        });
        temp_buffers.push(out_buffer);
        current_len = out_len;
    }

    Ok(PreparedReduction {
        work: PreparedWork::Tree(passes.into_boxed_slice()),
        temp_buffers,
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
