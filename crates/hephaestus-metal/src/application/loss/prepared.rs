use core::marker::PhantomData;

use hephaestus_core::CrossEntropyOps;
use hephaestus_wgpu::{WgpuCrossEntropyOps, WgpuDevice};

use crate::MetalDevice;

/// Prepared Metal forward pass delegated without copying to WGPU.
pub struct MetalPreparedCrossEntropyForward<'a> {
    pub(super) inner:
        <WgpuCrossEntropyOps as CrossEntropyOps<WgpuDevice, f32>>::PreparedForward<'a>,
    pub(super) _lifetime: PhantomData<&'a MetalDevice>,
}

/// Prepared Metal additive backward pass delegated without copying to WGPU.
pub struct MetalPreparedCrossEntropyBackward<'a> {
    pub(super) inner:
        <WgpuCrossEntropyOps as CrossEntropyOps<WgpuDevice, f32>>::PreparedBackward<'a>,
    pub(super) _lifetime: PhantomData<&'a MetalDevice>,
}
