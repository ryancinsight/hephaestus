use hephaestus_core::{
    PoolingBackwardOperands, PoolingForwardOperands, PoolingMode, PoolingOps, Result,
    SlidingWindowFoldOperands, SlidingWindowOps, SlidingWindowUnfoldOperands,
};
use hephaestus_wgpu::{WgpuDevice, WgpuPoolingOps, WgpuSlidingWindowOps};
use leto::WindowParameters;

use super::operands;
use super::prepared::{
    MetalPreparedFold, MetalPreparedPoolingBackward, MetalPreparedPoolingForward,
    MetalPreparedUnfold,
};
use crate::{MetalBuffer, MetalDevice};

/// Pooling operations delegated to Metal-selected WGPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalPoolingOps;

impl<T> PoolingOps<MetalDevice, T> for MetalPoolingOps
where
    T: eunomia::Pod,
    WgpuPoolingOps: PoolingOps<WgpuDevice, T>,
{
    type PreparedForward<'a, const R: usize, const S: usize>
        = MetalPreparedPoolingForward<'a, T, R, S>
    where
        MetalDevice: 'a,
        T: 'a;
    type PreparedBackward<'a, const R: usize, const S: usize>
        = MetalPreparedPoolingBackward<'a, T, R, S>
    where
        MetalDevice: 'a,
        T: 'a;

    fn prepare_pooling_forward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a MetalDevice,
        operands: PoolingForwardOperands<'a, MetalBuffer<T>, R>,
        parameters: WindowParameters<S>,
        mode: PoolingMode,
    ) -> Result<Self::PreparedForward<'a, R, S>> {
        Ok(MetalPreparedPoolingForward {
            inner: WgpuPoolingOps.prepare_pooling_forward(
                device.wgpu_device(),
                operands::pooling_forward(operands),
                parameters,
                mode,
            )?,
            _lifetime: core::marker::PhantomData,
        })
    }

    fn dispatch_pooling_forward<const R: usize, const S: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedForward<'_, R, S>,
    ) -> Result<()> {
        WgpuPoolingOps.dispatch_pooling_forward(device.wgpu_device(), &prepared.inner)
    }

    fn prepare_pooling_backward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a MetalDevice,
        operands: PoolingBackwardOperands<'a, MetalBuffer<T>, R>,
        parameters: WindowParameters<S>,
        mode: PoolingMode,
    ) -> Result<Self::PreparedBackward<'a, R, S>> {
        Ok(MetalPreparedPoolingBackward {
            inner: WgpuPoolingOps.prepare_pooling_backward(
                device.wgpu_device(),
                operands::pooling_backward(operands),
                parameters,
                mode,
            )?,
            _lifetime: core::marker::PhantomData,
        })
    }

    fn dispatch_pooling_backward<const R: usize, const S: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedBackward<'_, R, S>,
    ) -> Result<()> {
        WgpuPoolingOps.dispatch_pooling_backward(device.wgpu_device(), &prepared.inner)
    }
}

/// Sliding-window operations delegated to Metal-selected WGPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalSlidingWindowOps;

impl<T> SlidingWindowOps<MetalDevice, T> for MetalSlidingWindowOps
where
    T: eunomia::Pod,
    WgpuSlidingWindowOps: SlidingWindowOps<WgpuDevice, T>,
{
    type PreparedUnfold<'a, const R: usize, const S: usize>
        = MetalPreparedUnfold<'a, T, R, S>
    where
        MetalDevice: 'a,
        T: 'a;
    type PreparedFold<'a, const R: usize, const S: usize>
        = MetalPreparedFold<'a, T, R, S>
    where
        MetalDevice: 'a,
        T: 'a;

    fn prepare_unfold<'a, const R: usize, const S: usize>(
        &self,
        device: &'a MetalDevice,
        operands: SlidingWindowUnfoldOperands<'a, MetalBuffer<T>, R>,
        parameters: WindowParameters<S>,
    ) -> Result<Self::PreparedUnfold<'a, R, S>> {
        Ok(MetalPreparedUnfold {
            inner: WgpuSlidingWindowOps.prepare_unfold(
                device.wgpu_device(),
                operands::unfold(operands),
                parameters,
            )?,
            _lifetime: core::marker::PhantomData,
        })
    }

    fn dispatch_unfold<const R: usize, const S: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedUnfold<'_, R, S>,
    ) -> Result<()> {
        WgpuSlidingWindowOps.dispatch_unfold(device.wgpu_device(), &prepared.inner)
    }

    fn prepare_fold<'a, const R: usize, const S: usize>(
        &self,
        device: &'a MetalDevice,
        operands: SlidingWindowFoldOperands<'a, MetalBuffer<T>, R>,
        output_spatial_shape: [usize; S],
        parameters: WindowParameters<S>,
    ) -> Result<Self::PreparedFold<'a, R, S>> {
        Ok(MetalPreparedFold {
            inner: WgpuSlidingWindowOps.prepare_fold(
                device.wgpu_device(),
                operands::fold(operands),
                output_spatial_shape,
                parameters,
            )?,
            _lifetime: core::marker::PhantomData,
        })
    }

    fn dispatch_fold<const R: usize, const S: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedFold<'_, R, S>,
    ) -> Result<()> {
        WgpuSlidingWindowOps.dispatch_fold(device.wgpu_device(), &prepared.inner)
    }
}
