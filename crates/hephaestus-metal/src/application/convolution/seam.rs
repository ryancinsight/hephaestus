use core::marker::PhantomData;

use hephaestus_core::{
    ConvolutionBackwardOperands, ConvolutionForwardOperands, ConvolutionOps, Result,
};
use hephaestus_wgpu::{WgpuConvolutionOps, WgpuDevice};
use leto::{ConvolutionParameters, TransposedConvolutionParameters};

use super::operands::{backward, forward};
use super::prepared::{
    MetalPreparedConvolutionBackward, MetalPreparedConvolutionForward,
    MetalPreparedTransposedConvolutionBackward, MetalPreparedTransposedConvolutionForward,
};
use crate::{MetalBuffer, MetalDevice};

/// Metal convolution operations delegated to WGPU configured for Metal.
///
/// Operand conversion borrows the existing inner WGPU handles. Preparation
/// and dispatch therefore preserve device residency without allocation, copy,
/// host transfer, or runtime backend selection.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetalConvolutionOps;

impl<T> ConvolutionOps<MetalDevice, T> for MetalConvolutionOps
where
    T: bytemuck::Pod,
    WgpuConvolutionOps: ConvolutionOps<WgpuDevice, T>,
{
    type PreparedForward<'a, const R: usize, const S: usize>
        = MetalPreparedConvolutionForward<'a, T, R, S>
    where
        MetalDevice: 'a;
    type PreparedBackward<'a, const R: usize, const S: usize>
        = MetalPreparedConvolutionBackward<'a, T, R, S>
    where
        MetalDevice: 'a;
    type PreparedTransposedForward<'a, const R: usize, const S: usize>
        = MetalPreparedTransposedConvolutionForward<'a, T, R, S>
    where
        MetalDevice: 'a;
    type PreparedTransposedBackward<'a, const R: usize, const S: usize>
        = MetalPreparedTransposedConvolutionBackward<'a, T, R, S>
    where
        MetalDevice: 'a;

    fn prepare_convolution_forward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a MetalDevice,
        operands: ConvolutionForwardOperands<'a, MetalBuffer<T>, R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<Self::PreparedForward<'a, R, S>> {
        Ok(MetalPreparedConvolutionForward {
            inner: wgpu_ops().prepare_convolution_forward(
                device.wgpu_device(),
                forward(operands),
                parameters,
            )?,
            _lifetime: PhantomData,
        })
    }

    fn dispatch_convolution_forward<const R: usize, const S: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedForward<'_, R, S>,
    ) -> Result<()> {
        <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, T>>::dispatch_convolution_forward::<R, S>(
            &wgpu_ops(),
            device.wgpu_device(),
            &prepared.inner,
        )
    }

    fn prepare_convolution_backward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a MetalDevice,
        operands: ConvolutionBackwardOperands<'a, MetalBuffer<T>, R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<Self::PreparedBackward<'a, R, S>> {
        Ok(MetalPreparedConvolutionBackward {
            inner: wgpu_ops().prepare_convolution_backward(
                device.wgpu_device(),
                backward(operands),
                parameters,
            )?,
            _lifetime: PhantomData,
        })
    }

    fn dispatch_convolution_backward<const R: usize, const S: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedBackward<'_, R, S>,
    ) -> Result<()> {
        <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, T>>::dispatch_convolution_backward::<R, S>(
            &wgpu_ops(),
            device.wgpu_device(),
            &prepared.inner,
        )
    }

    fn prepare_convolution_transposed_forward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a MetalDevice,
        operands: ConvolutionForwardOperands<'a, MetalBuffer<T>, R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<Self::PreparedTransposedForward<'a, R, S>> {
        Ok(MetalPreparedTransposedConvolutionForward {
            inner: wgpu_ops().prepare_convolution_transposed_forward(
                device.wgpu_device(),
                forward(operands),
                parameters,
            )?,
            _lifetime: PhantomData,
        })
    }

    fn dispatch_convolution_transposed_forward<const R: usize, const S: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedTransposedForward<'_, R, S>,
    ) -> Result<()> {
        <WgpuConvolutionOps as ConvolutionOps<
            WgpuDevice,
            T,
        >>::dispatch_convolution_transposed_forward::<R, S>(
            &wgpu_ops(),
            device.wgpu_device(),
            &prepared.inner,
        )
    }

    fn prepare_convolution_transposed_backward<'a, const R: usize, const S: usize>(
        &self,
        device: &'a MetalDevice,
        operands: ConvolutionBackwardOperands<'a, MetalBuffer<T>, R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<Self::PreparedTransposedBackward<'a, R, S>> {
        Ok(MetalPreparedTransposedConvolutionBackward {
            inner: wgpu_ops().prepare_convolution_transposed_backward(
                device.wgpu_device(),
                backward(operands),
                parameters,
            )?,
            _lifetime: PhantomData,
        })
    }

    fn dispatch_convolution_transposed_backward<const R: usize, const S: usize>(
        &self,
        device: &MetalDevice,
        prepared: &Self::PreparedTransposedBackward<'_, R, S>,
    ) -> Result<()> {
        <WgpuConvolutionOps as ConvolutionOps<
            WgpuDevice,
            T,
        >>::dispatch_convolution_transposed_backward::<R, S>(
            &wgpu_ops(),
            device.wgpu_device(),
            &prepared.inner,
        )
    }
}

const fn wgpu_ops() -> WgpuConvolutionOps {
    WgpuConvolutionOps
}
