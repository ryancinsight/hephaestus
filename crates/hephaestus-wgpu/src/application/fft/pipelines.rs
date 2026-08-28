//! Prepared WGPU pipelines shared by every dispatch of one FFT plan.

use hephaestus_core::{KernelDevice, Result};

use crate::{WgpuDevice, WgpuPrepared};

use super::kernel::{
    BitReverse, Butterfly, ChirpKernel, ChirpNegateImaginary, ChirpPointMultiply,
    ChirpPostmultiply, ChirpPremultiply, ChirpScale, FUSED_WORKGROUP_SIZE,
    FUSED_WORKGROUP_STORAGE_BYTES, FftKernel, FusedKernel, Pack, PackKernel, RadixFourBitReverse,
    RadixFourButterfly, Scale, Unpack,
};
use super::scalar::WgpuFftScalar;

pub(crate) struct FftPipelines<T: WgpuFftScalar> {
    pub(crate) fused: Option<WgpuPrepared<FusedKernel<T>>>,
    pub(crate) pack: WgpuPrepared<PackKernel<T, Pack>>,
    pub(crate) unpack: WgpuPrepared<PackKernel<T, Unpack>>,
    pub(crate) bit_reverse: WgpuPrepared<FftKernel<T, BitReverse>>,
    pub(crate) radix_four_bit_reverse: WgpuPrepared<FftKernel<T, RadixFourBitReverse>>,
    pub(crate) butterfly: WgpuPrepared<FftKernel<T, Butterfly>>,
    pub(crate) radix_four_butterfly: WgpuPrepared<FftKernel<T, RadixFourButterfly>>,
    pub(crate) scale: WgpuPrepared<FftKernel<T, Scale>>,
    pub(crate) chirp_premultiply: WgpuPrepared<ChirpKernel<T, ChirpPremultiply>>,
    pub(crate) chirp_point_multiply: WgpuPrepared<ChirpKernel<T, ChirpPointMultiply>>,
    pub(crate) chirp_postmultiply: WgpuPrepared<ChirpKernel<T, ChirpPostmultiply>>,
    pub(crate) chirp_negate_imaginary: WgpuPrepared<ChirpKernel<T, ChirpNegateImaginary>>,
    pub(crate) chirp_scale: WgpuPrepared<ChirpKernel<T, ChirpScale>>,
}

impl<T: WgpuFftScalar> FftPipelines<T> {
    pub(crate) fn new(device: &WgpuDevice) -> Result<Self> {
        let limits = device.device_limits();
        let fused = (limits.max_compute_workgroup_size_x >= FUSED_WORKGROUP_SIZE
            && limits.max_compute_invocations_per_workgroup >= FUSED_WORKGROUP_SIZE
            && limits.max_compute_workgroup_storage_size >= FUSED_WORKGROUP_STORAGE_BYTES)
            .then(|| device.prepare(&FusedKernel::<T>::new()))
            .transpose()?;
        Ok(Self {
            fused,
            pack: device.prepare(&PackKernel::<T, Pack>::new())?,
            unpack: device.prepare(&PackKernel::<T, Unpack>::new())?,
            bit_reverse: device.prepare(&FftKernel::<T, BitReverse>::new())?,
            radix_four_bit_reverse: device.prepare(&FftKernel::<T, RadixFourBitReverse>::new())?,
            butterfly: device.prepare(&FftKernel::<T, Butterfly>::new())?,
            radix_four_butterfly: device.prepare(&FftKernel::<T, RadixFourButterfly>::new())?,
            scale: device.prepare(&FftKernel::<T, Scale>::new())?,
            chirp_premultiply: device.prepare(&ChirpKernel::<T, ChirpPremultiply>::new())?,
            chirp_point_multiply: device.prepare(&ChirpKernel::<T, ChirpPointMultiply>::new())?,
            chirp_postmultiply: device.prepare(&ChirpKernel::<T, ChirpPostmultiply>::new())?,
            chirp_negate_imaginary: device
                .prepare(&ChirpKernel::<T, ChirpNegateImaginary>::new())?,
            chirp_scale: device.prepare(&ChirpKernel::<T, ChirpScale>::new())?,
        })
    }
}
