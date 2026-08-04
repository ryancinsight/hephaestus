use core::marker::PhantomData;

use hephaestus_core::{
    CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps, Result,
};
use hephaestus_wgpu::{WgpuCrossEntropyOps, WgpuDevice};

use super::operands::{backward, forward};
use super::prepared::{MetalPreparedCrossEntropyBackward, MetalPreparedCrossEntropyForward};
use crate::{MetalBuffer, MetalDevice};

/// Metal mean cross-entropy delegated to WGPU configured for Metal.
///
/// Operand conversion borrows the existing device buffers, so delegation adds
/// no payload copy, host transfer, or backend selection. WGPU preparation still
/// owns its bounded metadata, status, and dispatch-resource allocations.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalCrossEntropyOps;

impl CrossEntropyOps<MetalDevice, f32> for MetalCrossEntropyOps {
    type PreparedForward<'a>
        = MetalPreparedCrossEntropyForward<'a>
    where
        MetalDevice: 'a;
    type PreparedBackward<'a>
        = MetalPreparedCrossEntropyBackward<'a>
    where
        MetalDevice: 'a;

    fn prepare_cross_entropy_forward<'a>(
        &self,
        device: &'a MetalDevice,
        operands: CrossEntropyForwardOperands<'a, MetalBuffer<f32>, MetalBuffer<u32>>,
    ) -> Result<Self::PreparedForward<'a>> {
        Ok(MetalPreparedCrossEntropyForward {
            inner: wgpu_ops()
                .prepare_cross_entropy_forward(device.wgpu_device(), forward(operands))?,
            _lifetime: PhantomData,
        })
    }

    fn dispatch_cross_entropy_forward(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        <WgpuCrossEntropyOps as CrossEntropyOps<WgpuDevice, f32>>::dispatch_cross_entropy_forward(
            &wgpu_ops(),
            device.wgpu_device(),
            &prepared.inner,
        )
    }

    fn prepare_cross_entropy_backward<'a>(
        &self,
        device: &'a MetalDevice,
        operands: CrossEntropyBackwardOperands<'a, MetalBuffer<f32>, MetalBuffer<u32>>,
    ) -> Result<Self::PreparedBackward<'a>> {
        Ok(MetalPreparedCrossEntropyBackward {
            inner: wgpu_ops()
                .prepare_cross_entropy_backward(device.wgpu_device(), backward(operands))?,
            _lifetime: PhantomData,
        })
    }

    fn dispatch_cross_entropy_backward(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        <WgpuCrossEntropyOps as CrossEntropyOps<WgpuDevice, f32>>::dispatch_cross_entropy_backward(
            &wgpu_ops(),
            device.wgpu_device(),
            &prepared.inner,
        )
    }
}

const fn wgpu_ops() -> WgpuCrossEntropyOps {
    WgpuCrossEntropyOps
}
