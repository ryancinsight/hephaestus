//! Axis-reduction dispatch machinery: planning, pipelines, and prepared batches.

use std::any::TypeId;
use std::marker::PhantomData;

use eunomia::Pod;
use hephaestus_core::{
    AxisReductionDispatch, AxisReductionMeta, BlockWidth, CombineExpr, ComputeDevice,
    DialectScalar, HephaestusError, IdentityToken, OpIdentity, Result, SumOp, Wgsl,
    plan_axis_reduction,
};

use crate::application::canary;
use crate::application::pipeline::cached_pipeline;
use crate::application::strided::StridedOperand;
use crate::infrastructure::buffer::WgpuBuffer;
use crate::infrastructure::device::WgpuDevice;

use super::shaders::{
    AxisReductionAxis0TiledKernel, AxisReductionParallelKernel, MeanAxis0TiledKernel,
    MeanAxisParallelKernel, axis_reduction_axis0_tiled_shader_source,
    axis_reduction_parallel_shader_source, axis0_tile_shape, mean_axis_parallel_shader_source,
    mean_axis0_tiled_shader_source,
};

/// Prepared rank-2 axis reduction over fixed input/output buffers and layouts.
///
/// This removes repeated metadata-uniform acquisition, metadata upload, and bind
/// group construction when callers repeatedly dispatch the same axis reduction
/// into the same output buffer.
pub struct PreparedAxisReduction<T> {
    pub(crate) pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) bind_group: Option<wgpu::BindGroup>,
    pub(crate) groups: u32,
    /// Pooled metadata uniform held for the prepared reduction's lifetime so
    /// it recycles back to the pool on drop (the bind group also keeps the
    /// underlying buffer alive; without the guard the buffer would escape the
    /// pool permanently).
    _meta_buffer: Option<crate::infrastructure::pool::PooledBuffer>,
    /// Storage binding used when the reduced input axis is empty. WGPU rejects
    /// zero-byte storage bindings even though the kernel only needs the output
    /// identity for that case.
    _empty_input_buffer: Option<WgpuBuffer<T>>,
    _marker: PhantomData<T>,
}

impl<T> PreparedAxisReduction<T> {
    /// Dispatch the prepared axis reduction once.
    ///
    /// # Errors
    ///
    /// Returns a typed dispatch error if command encoding or submission cannot
    /// be completed by the backend.
    pub fn dispatch(&self, device: &WgpuDevice) -> Result<()> {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(());
        };
        let Some(bind_group) = self.bind_group.as_ref() else {
            return Ok(());
        };
        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hephaestus-prepared-axis-reduction"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hephaestus-prepared-axis-reduction"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(self.groups, 1, 1);
        }
        device.queue().submit(Some(encoder.finish()));
        Ok(())
    }
}

/// Submit multiple prepared axis reductions in one command buffer.
///
/// Each prepared reduction keeps its own output buffer. The reductions must be
/// independent: no reduction in the batch may consume another reduction's
/// output. The encoder records every dispatch in one compute pass and submits
/// one command buffer, amortizing pass construction and WGPU submit/poll
/// overhead without sharing scratch state between reductions. A batch with no
/// active dispatch returns without allocating a command encoder or submitting
/// an empty command buffer.
///
/// # Errors
///
/// Returns a typed dispatch error if command encoding or submission cannot be
/// completed by the backend.
pub fn submit_prepared_axis_reduction_batch<T>(
    device: &WgpuDevice,
    reductions: &[&PreparedAxisReduction<T>],
) -> Result<()> {
    super::prepared::submit_prepared_mixed_reduction_batch::<T, T>(device, &[], reductions)
}

pub(crate) fn axis_len<T>(input: StridedOperand<'_, T, 2>, axis: usize) -> Result<usize> {
    input
        .layout
        .shape()
        .get(axis)
        .copied()
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("axis {axis} is out of bounds for rank-2 reduction"),
        })
}

pub(crate) fn plan_axis_reduction_dispatch<T>(
    input: StridedOperand<'_, T, 2>,
    axis: usize,
    output: StridedOperand<'_, T, 2>,
    width: BlockWidth,
) -> Result<Option<AxisReductionDispatch>> {
    plan_axis_reduction(
        input.layout,
        input.buffer.len,
        output.layout,
        output.buffer.len,
        axis,
        width,
        input.buffer.aliases(output.buffer),
    )
}

pub(crate) fn dispatch_axis_reduction<T>(
    device: &WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    input: StridedOperand<'_, T, 2>,
    output: StridedOperand<'_, T, 2>,
    dispatch: AxisReductionDispatch,
) -> Result<()>
where
    T: Pod,
{
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<AxisReductionMeta>(1)?)?;

    let empty_input_buffer = if input.buffer.len == 0 {
        Some(device.alloc_zeroed::<T>(1)?)
    } else {
        None
    };
    let input_buffer = empty_input_buffer
        .as_ref()
        .map_or(&input.buffer.buffer, |buffer| &buffer.buffer);
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&dispatch.meta));
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hephaestus-axis-reduction"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer.buffer.as_entire_binding(),
                },
            ],
        });
    let mut encoder = device
        .inner()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hephaestus-axis-reduction"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hephaestus-axis-reduction"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch.groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

pub(crate) fn prepared_axis_reduction<T>(
    device: &WgpuDevice,
    pipeline: wgpu::ComputePipeline,
    input: StridedOperand<'_, T, 2>,
    output: StridedOperand<'_, T, 2>,
    dispatch: AxisReductionDispatch,
    label: &'static str,
) -> Result<PreparedAxisReduction<T>>
where
    T: Pod,
{
    let meta_buffer = device.get_uniform_buffer(WgpuDevice::byte_size::<AxisReductionMeta>(1)?)?;

    let empty_input_buffer = if input.buffer.len == 0 {
        Some(device.alloc_zeroed::<T>(1)?)
    } else {
        None
    };
    let input_buffer = empty_input_buffer
        .as_ref()
        .map_or(&input.buffer.buffer, |buffer| &buffer.buffer);
    device
        .queue()
        .write_buffer(&meta_buffer, 0, eunomia::layout::bytes_of(&dispatch.meta));
    let bind_group = device
        .inner()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer.buffer.as_entire_binding(),
                },
            ],
        });
    Ok(PreparedAxisReduction {
        pipeline: Some(pipeline),
        bind_group: Some(bind_group),
        groups: dispatch.groups,
        _meta_buffer: Some(meta_buffer),
        _empty_input_buffer: empty_input_buffer,
        _marker: PhantomData,
    })
}

pub(crate) fn empty_prepared_axis_reduction<T>() -> PreparedAxisReduction<T> {
    PreparedAxisReduction {
        pipeline: None,
        bind_group: None,
        groups: 0,
        _meta_buffer: None,
        _empty_input_buffer: None,
        _marker: PhantomData,
    }
}

pub(crate) fn axis_reduction_pipeline<Op, T>(
    device: &WgpuDevice,
    width: BlockWidth,
    dispatch: &mut AxisReductionDispatch,
) -> Result<wgpu::ComputePipeline>
where
    Op: CombineExpr<Wgsl>,
    T: DialectScalar<Wgsl> + Pod + OpIdentity<Op> + IdentityToken<Op, Wgsl>,
{
    canary::require_template_safe::<T>(device, <Op as CombineExpr<Wgsl>>::EXPR, "axis reduction")?;
    Ok(if dispatch.meta.offsets[2] == 0 {
        let (tile_cols, _) = axis0_tile_shape(width);
        dispatch.groups = dispatch.meta.offsets[3].div_ceil(tile_cols);
        cached_pipeline(
            device,
            (
                TypeId::of::<AxisReductionAxis0TiledKernel<Op>>(),
                TypeId::of::<T>(),
                width.get(),
            ),
            "hephaestus-axis-reduction-axis0-tiled",
            || axis_reduction_axis0_tiled_shader_source::<Op, T>(width),
        )
    } else {
        // Grid-strided (WG-P5): one workgroup per output row, correct and
        // fully lane-parallel for any axis length.
        dispatch.groups = dispatch.meta.offsets[3];
        cached_pipeline(
            device,
            (
                TypeId::of::<AxisReductionParallelKernel<Op>>(),
                TypeId::of::<T>(),
                width.get(),
            ),
            "hephaestus-axis-reduction-parallel",
            || axis_reduction_parallel_shader_source::<Op, T>(width),
        )
    })
}

pub(crate) fn mean_axis_pipeline<T>(
    device: &WgpuDevice,
    width: BlockWidth,
    dispatch: &mut AxisReductionDispatch,
) -> wgpu::ComputePipeline
where
    T: DialectScalar<Wgsl> + Pod + OpIdentity<SumOp> + IdentityToken<SumOp, Wgsl>,
{
    if dispatch.meta.offsets[2] == 0 {
        let (tile_cols, _) = axis0_tile_shape(width);
        dispatch.groups = dispatch.meta.offsets[3].div_ceil(tile_cols);
        cached_pipeline(
            device,
            (
                TypeId::of::<MeanAxis0TiledKernel<T>>(),
                TypeId::of::<T>(),
                width.get(),
            ),
            "hephaestus-mean-axis0-tiled",
            || mean_axis0_tiled_shader_source::<T>(width),
        )
    } else {
        // Grid-strided (WG-P5): one workgroup per output row, correct and
        // fully lane-parallel for any axis length.
        dispatch.groups = dispatch.meta.offsets[3];
        cached_pipeline(
            device,
            (
                TypeId::of::<MeanAxisParallelKernel<T>>(),
                TypeId::of::<T>(),
                width.get(),
            ),
            "hephaestus-mean-axis-parallel",
            || mean_axis_parallel_shader_source::<T>(width),
        )
    }
}
