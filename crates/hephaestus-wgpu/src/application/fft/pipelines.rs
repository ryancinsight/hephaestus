//! Prepared WGPU pipelines shared by every dispatch of one FFT plan.

use hephaestus_core::{KernelDevice, Result};

use crate::{WgpuDevice, WgpuPrepared};

use super::kernel::{
    BitReverse, Butterfly, ChirpKernel, ChirpNegateImaginary, ChirpPointMultiply,
    ChirpPostmultiply, ChirpPremultiply, ChirpScale, FUSED_WORKGROUP_SIZE,
    FUSED_WORKGROUP_STORAGE_BYTES, FftKernel, FusedKernel, Pack, PackKernel, RadixFourBitReverse,
    RadixFourButterfly, Scale, Unpack,
};

pub(crate) struct FftPipelines {
    pub(crate) fused: Option<WgpuPrepared<FusedKernel>>,
    pub(crate) pack: WgpuPrepared<PackKernel<Pack>>,
    pub(crate) unpack: WgpuPrepared<PackKernel<Unpack>>,
    pub(crate) bit_reverse: WgpuPrepared<FftKernel<BitReverse>>,
    pub(crate) radix_four_bit_reverse: WgpuPrepared<FftKernel<RadixFourBitReverse>>,
    pub(crate) butterfly: WgpuPrepared<FftKernel<Butterfly>>,
    pub(crate) radix_four_butterfly: WgpuPrepared<FftKernel<RadixFourButterfly>>,
    pub(crate) scale: WgpuPrepared<FftKernel<Scale>>,
    pub(crate) chirp_premultiply: WgpuPrepared<ChirpKernel<ChirpPremultiply>>,
    pub(crate) chirp_point_multiply: WgpuPrepared<ChirpKernel<ChirpPointMultiply>>,
    pub(crate) chirp_postmultiply: WgpuPrepared<ChirpKernel<ChirpPostmultiply>>,
    pub(crate) chirp_negate_imaginary: WgpuPrepared<ChirpKernel<ChirpNegateImaginary>>,
    pub(crate) chirp_scale: WgpuPrepared<ChirpKernel<ChirpScale>>,
}

impl FftPipelines {
    pub(crate) fn new(device: &WgpuDevice) -> Result<Self> {
        let limits = device.device_limits();
        let fused = (limits.max_compute_workgroup_size_x >= FUSED_WORKGROUP_SIZE
            && limits.max_compute_invocations_per_workgroup >= FUSED_WORKGROUP_SIZE
            && limits.max_compute_workgroup_storage_size >= FUSED_WORKGROUP_STORAGE_BYTES)
            .then(|| device.prepare(&FusedKernel))
            .transpose()?;
        Ok(Self {
            fused,
            pack: device.prepare(&PackKernel::<Pack>::new())?,
            unpack: device.prepare(&PackKernel::<Unpack>::new())?,
            bit_reverse: device.prepare(&FftKernel::<BitReverse>::new())?,
            radix_four_bit_reverse: device.prepare(&FftKernel::<RadixFourBitReverse>::new())?,
            butterfly: device.prepare(&FftKernel::<Butterfly>::new())?,
            radix_four_butterfly: device.prepare(&FftKernel::<RadixFourButterfly>::new())?,
            scale: device.prepare(&FftKernel::<Scale>::new())?,
            chirp_premultiply: device.prepare(&ChirpKernel::<ChirpPremultiply>::new())?,
            chirp_point_multiply: device.prepare(&ChirpKernel::<ChirpPointMultiply>::new())?,
            chirp_postmultiply: device.prepare(&ChirpKernel::<ChirpPostmultiply>::new())?,
            chirp_negate_imaginary: device.prepare(&ChirpKernel::<ChirpNegateImaginary>::new())?,
            chirp_scale: device.prepare(&ChirpKernel::<ChirpScale>::new())?,
        })
    }
}
