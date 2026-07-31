use core::marker::PhantomData;

use hephaestus_core::AttentionOps;
use hephaestus_wgpu::{WgpuAttentionOps, WgpuDevice};

use crate::MetalDevice;

/// Prepared Metal attention forward pass delegated to WGPU.
pub struct MetalPreparedAttentionForward<'a> {
    pub(super) inner: <WgpuAttentionOps as AttentionOps<WgpuDevice, f32>>::PreparedForward<'a>,
    pub(super) _lifetime: PhantomData<&'a MetalDevice>,
}

/// Prepared Metal additive attention backward pass delegated to WGPU.
pub struct MetalPreparedAttentionBackward<'a> {
    pub(super) inner: <WgpuAttentionOps as AttentionOps<WgpuDevice, f32>>::PreparedBackward<'a>,
    pub(super) _lifetime: PhantomData<&'a MetalDevice>,
}
