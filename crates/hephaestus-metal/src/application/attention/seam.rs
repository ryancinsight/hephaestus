use core::marker::PhantomData;

use hephaestus_core::{AttentionBackwardOperands, AttentionForwardOperands, AttentionOps, Result};
use hephaestus_wgpu::{WgpuAttentionOps, WgpuDevice};

use super::operands::{backward, forward};
use super::prepared::{MetalPreparedAttentionBackward, MetalPreparedAttentionForward};
use crate::{MetalBuffer, MetalDevice};

/// Metal attention operations delegated to WGPU configured for Metal.
///
/// Conversion borrows the existing WGPU device and buffers. Preparation and
/// dispatch therefore preserve device residency without allocation, copy,
/// host transfer, or runtime backend selection.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalAttentionOps;

impl AttentionOps<MetalDevice, f32> for MetalAttentionOps {
    type PreparedForward<'a>
        = MetalPreparedAttentionForward<'a>
    where
        MetalDevice: 'a;
    type PreparedBackward<'a>
        = MetalPreparedAttentionBackward<'a>
    where
        MetalDevice: 'a;

    fn prepare_attention_forward<'a>(
        &self,
        device: &'a MetalDevice,
        operands: AttentionForwardOperands<'a, MetalBuffer<f32>, f32>,
    ) -> Result<Self::PreparedForward<'a>> {
        Ok(MetalPreparedAttentionForward {
            inner: wgpu_ops().prepare_attention_forward(device.wgpu_device(), forward(operands))?,
            _lifetime: PhantomData,
        })
    }

    fn dispatch_attention_forward(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedForward<'_>,
    ) -> Result<()> {
        <WgpuAttentionOps as AttentionOps<WgpuDevice, f32>>::dispatch_attention_forward(
            &wgpu_ops(),
            device.wgpu_device(),
            &prepared.inner,
        )
    }

    fn prepare_attention_backward<'a>(
        &self,
        device: &'a MetalDevice,
        operands: AttentionBackwardOperands<'a, MetalBuffer<f32>, f32>,
    ) -> Result<Self::PreparedBackward<'a>> {
        Ok(MetalPreparedAttentionBackward {
            inner: wgpu_ops()
                .prepare_attention_backward(device.wgpu_device(), backward(operands))?,
            _lifetime: PhantomData,
        })
    }

    fn dispatch_attention_backward(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedBackward<'_>,
    ) -> Result<()> {
        <WgpuAttentionOps as AttentionOps<WgpuDevice, f32>>::dispatch_attention_backward(
            &wgpu_ops(),
            device.wgpu_device(),
            &prepared.inner,
        )
    }
}

const fn wgpu_ops() -> WgpuAttentionOps {
    WgpuAttentionOps
}
