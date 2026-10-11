//! Provider-owned runtime-rank fusion seam for Metal.
//!
//! Zero-copy delegation to Metal-selected WGPU: views reborrow the wrapped
//! [`WgpuBuffer`](hephaestus_wgpu::WgpuBuffer) handles, so fusion evaluates
//! the same WGSL the WGPU backend compiles. The dialect is therefore
//! [`Wgsl`](hephaestus_core::Wgsl), not a Metal-native language.

use hephaestus_core::{
    DynamicStridedView, FusedElementwiseOps, FusedExpression, FusedReduction, FusedReductionOps,
    FusionScalar, Result, Wgsl,
};
use hephaestus_wgpu::WgpuFusionOps;

use crate::infrastructure::buffer::MetalBuffer;
use crate::infrastructure::device::MetalDevice;

/// Provider-owned implementation of the fusion seams for Metal.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalFusionOps {
    inner: WgpuFusionOps,
}

impl<T> FusedElementwiseOps<MetalDevice, T> for MetalFusionOps
where
    T: FusionScalar<Wgsl>,
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
        let inputs = inputs
            .iter()
            .map(|input| DynamicStridedView::new(&input.buffer.inner, input.layout))
            .collect::<Vec<_>>();
        self.inner.fused_elementwise_into(
            device.wgpu_device(),
            expression,
            &inputs,
            DynamicStridedView::new(&output.buffer.inner, output.layout),
        )
    }
}

impl<T> FusedReductionOps<MetalDevice, T> for MetalFusionOps
where
    T: FusionScalar<Wgsl>,
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
        let inputs = inputs
            .iter()
            .map(|input| DynamicStridedView::new(&input.buffer.inner, input.layout))
            .collect::<Vec<_>>();
        self.inner.fused_reduce_into(
            device.wgpu_device(),
            expression,
            &inputs,
            reduction,
            axis,
            DynamicStridedView::new(&output.buffer.inner, output.layout),
        )
    }
}
