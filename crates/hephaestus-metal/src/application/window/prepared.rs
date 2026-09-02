use core::marker::PhantomData;

use hephaestus_core::{PoolingOps, SlidingWindowOps};
use hephaestus_wgpu::{WgpuDevice, WgpuPoolingOps, WgpuSlidingWindowOps};

/// Prepared Metal-selected WGPU pooling-forward resources.
pub struct MetalPreparedPoolingForward<'a, T, const R: usize, const S: usize>
where
    T: bytemuck::Pod,
    WgpuPoolingOps: PoolingOps<WgpuDevice, T>,
{
    pub(super) inner: <WgpuPoolingOps as PoolingOps<WgpuDevice, T>>::PreparedForward<'a, R, S>,
    pub(super) _lifetime: PhantomData<&'a T>,
}

/// Prepared Metal-selected WGPU pooling-backward resources.
pub struct MetalPreparedPoolingBackward<'a, T, const R: usize, const S: usize>
where
    T: bytemuck::Pod,
    WgpuPoolingOps: PoolingOps<WgpuDevice, T>,
{
    pub(super) inner: <WgpuPoolingOps as PoolingOps<WgpuDevice, T>>::PreparedBackward<'a, R, S>,
    pub(super) _lifetime: PhantomData<&'a T>,
}

/// Prepared Metal-selected WGPU unfold resources.
pub struct MetalPreparedUnfold<'a, T, const R: usize, const S: usize>
where
    T: bytemuck::Pod,
    WgpuSlidingWindowOps: SlidingWindowOps<WgpuDevice, T>,
{
    pub(super) inner:
        <WgpuSlidingWindowOps as SlidingWindowOps<WgpuDevice, T>>::PreparedUnfold<'a, R, S>,
    pub(super) _lifetime: PhantomData<&'a T>,
}

/// Prepared Metal-selected WGPU fold resources.
pub struct MetalPreparedFold<'a, T, const R: usize, const S: usize>
where
    T: bytemuck::Pod,
    WgpuSlidingWindowOps: SlidingWindowOps<WgpuDevice, T>,
{
    pub(super) inner:
        <WgpuSlidingWindowOps as SlidingWindowOps<WgpuDevice, T>>::PreparedFold<'a, R, S>,
    pub(super) _lifetime: PhantomData<&'a T>,
}
