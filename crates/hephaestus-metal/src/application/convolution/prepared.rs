use core::marker::PhantomData;

use hephaestus_core::ConvolutionOps;
use hephaestus_wgpu::{WgpuConvolutionOps, WgpuDevice};

use crate::MetalDevice;

/// Prepared regular Metal convolution forward pass.
pub struct MetalPreparedConvolutionForward<'a, const R: usize, const S: usize> {
    pub(super) inner:
        <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, f32>>::PreparedForward<'a, R, S>,
    pub(super) _lifetime: PhantomData<&'a MetalDevice>,
}

/// Prepared regular Metal convolution backward pass.
pub struct MetalPreparedConvolutionBackward<'a, const R: usize, const S: usize> {
    pub(super) inner:
        <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, f32>>::PreparedBackward<'a, R, S>,
    pub(super) _lifetime: PhantomData<&'a MetalDevice>,
}

/// Prepared transposed Metal convolution forward pass.
pub struct MetalPreparedTransposedConvolutionForward<'a, const R: usize, const S: usize> {
    pub(super) inner:
        <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, f32>>::PreparedTransposedForward<
            'a,
            R,
            S,
        >,
    pub(super) _lifetime: PhantomData<&'a MetalDevice>,
}

/// Prepared transposed Metal convolution backward pass.
pub struct MetalPreparedTransposedConvolutionBackward<'a, const R: usize, const S: usize> {
    pub(super) inner:
        <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, f32>>::PreparedTransposedBackward<
            'a,
            R,
            S,
        >,
    pub(super) _lifetime: PhantomData<&'a MetalDevice>,
}
