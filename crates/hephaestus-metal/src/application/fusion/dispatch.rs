//! Runtime-rank expression fusion implemented by the Metal provider.
//!
//! Metal executes through its inner WGPU device, so fusion delegates to
//! [`WgpuFusionOps`](hephaestus_wgpu::WgpuFusionOps) after unwrapping the
//! Metal views. The dialect is WGSL on both sides; no second shader
//! language is involved.

use hephaestus_core::{
    DynamicStridedView, FusedElementwiseOps, FusedExpression, FusedReduction, FusedReductionOps,
    Result, Wgsl,
};
use hephaestus_wgpu::{WgpuBuffer, WgpuFusionOps, WgpuFusionScalar};

use crate::{MetalBuffer, MetalDevice};

fn inner_views<'b, T>(
    views: &[DynamicStridedView<'b, MetalBuffer<T>>],
) -> Vec<DynamicStridedView<'b, WgpuBuffer<T>>> {
    views
        .iter()
        .map(|view| DynamicStridedView {
            buffer: view.buffer.wgpu_buffer(),
            layout: view.layout,
        })
        .collect()
}

fn inner_output<T>(
    output: DynamicStridedView<'_, MetalBuffer<T>>,
) -> DynamicStridedView<'_, WgpuBuffer<T>> {
    DynamicStridedView {
        buffer: output.buffer.wgpu_buffer(),
        layout: output.layout,
    }
}

/// Static provider value implementing both runtime fusion roles.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalFusionOps;

impl<T> FusedElementwiseOps<MetalDevice, T> for MetalFusionOps
where
    T: WgpuFusionScalar,
{
    type Dialect = Wgsl;

    fn fused_elementwise_into<E>(
        &self,
        device: &MetalDevice,
        expression: &E,
        inputs: &[DynamicStridedView<'_, MetalBuffer<T>>],
        output: DynamicStridedView<'_, MetalBuffer<T>>,
    ) -> Result<()>
    where
        E: FusedExpression<Self::Dialect>,
    {
        WgpuFusionOps.fused_elementwise_into(
            &device.inner,
            expression,
            &inner_views(inputs),
            inner_output(output),
        )
    }
}

impl<T> FusedReductionOps<MetalDevice, T> for MetalFusionOps
where
    T: WgpuFusionScalar,
{
    type Dialect = Wgsl;

    fn fused_reduce_into<E>(
        &self,
        device: &MetalDevice,
        expression: &E,
        inputs: &[DynamicStridedView<'_, MetalBuffer<T>>],
        reduction: FusedReduction,
        axis: usize,
        output: DynamicStridedView<'_, MetalBuffer<T>>,
    ) -> Result<()>
    where
        E: FusedExpression<Self::Dialect>,
    {
        WgpuFusionOps.fused_reduce_into(
            &device.inner,
            expression,
            &inner_views(inputs),
            reduction,
            axis,
            inner_output(output),
        )
    }
}
