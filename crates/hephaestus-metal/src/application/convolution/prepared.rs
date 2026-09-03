use core::marker::PhantomData;

use eunomia::Pod;
use hephaestus_core::ConvolutionOps;
use hephaestus_wgpu::{WgpuConvolutionOps, WgpuDevice};

use crate::MetalDevice;

/// Prepared regular Metal convolution forward pass.
pub struct MetalPreparedConvolutionForward<'a, T, const R: usize, const S: usize>
where
    T: Pod,
    WgpuConvolutionOps: ConvolutionOps<WgpuDevice, T>,
{
    pub(super) inner:
        <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, T>>::PreparedForward<'a, R, S>,
    pub(super) _lifetime: PhantomData<&'a MetalDevice>,
}

/// Prepared regular Metal convolution backward pass.
pub struct MetalPreparedConvolutionBackward<'a, T, const R: usize, const S: usize>
where
    T: Pod,
    WgpuConvolutionOps: ConvolutionOps<WgpuDevice, T>,
{
    pub(super) inner:
        <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, T>>::PreparedBackward<'a, R, S>,
    pub(super) _lifetime: PhantomData<&'a MetalDevice>,
}

/// Prepared transposed Metal convolution forward pass.
pub struct MetalPreparedTransposedConvolutionForward<'a, T, const R: usize, const S: usize>
where
    T: Pod,
    WgpuConvolutionOps: ConvolutionOps<WgpuDevice, T>,
{
    pub(super) inner:
        <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, T>>::PreparedTransposedForward<'a, R, S>,
    pub(super) _lifetime: PhantomData<&'a MetalDevice>,
}

/// Prepared transposed Metal convolution backward pass.
pub struct MetalPreparedTransposedConvolutionBackward<'a, T, const R: usize, const S: usize>
where
    T: Pod,
    WgpuConvolutionOps: ConvolutionOps<WgpuDevice, T>,
{
    pub(super) inner:
        <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, T>>::PreparedTransposedBackward<'a, R, S>,
    pub(super) _lifetime: PhantomData<&'a MetalDevice>,
}
