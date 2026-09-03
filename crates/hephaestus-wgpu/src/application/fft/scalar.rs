//! WGPU FFT scalar capability contract.

use eunomia::Pod;
use hephaestus_core::{DeviceFeature, HephaestusError, Result};

use crate::WgpuDevice;

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for eunomia::F16 {}
}

/// A scalar with a native WGPU FFT implementation.
///
/// The sealed implementations preserve one rank-generic plan family while
/// selecting the device capability, WGSL scalar token, and coefficient
/// narrowing required by each physical representation.
pub trait WgpuFftScalar: Pod + Send + Sync + 'static + private::Sealed {
    #[doc(hidden)]
    const TYPE_TOKEN: &'static str;

    #[doc(hidden)]
    const FFT_SOURCE_PREAMBLE: &'static str;

    #[doc(hidden)]
    fn validate_fft_capability(device: &WgpuDevice) -> Result<()>;

    #[doc(hidden)]
    fn from_fft_coefficient(value: f64) -> Self;
}

impl WgpuFftScalar for f32 {
    const TYPE_TOKEN: &'static str = "f32";
    const FFT_SOURCE_PREAMBLE: &'static str = "";

    fn validate_fft_capability(_device: &WgpuDevice) -> Result<()> {
        Ok(())
    }

    fn from_fft_coefficient(value: f64) -> Self {
        value as Self
    }
}

impl WgpuFftScalar for eunomia::F16 {
    const TYPE_TOKEN: &'static str = "f16";
    const FFT_SOURCE_PREAMBLE: &'static str = "enable f16;\n\n";

    fn validate_fft_capability(device: &WgpuDevice) -> Result<()> {
        if device.supports_device_feature(DeviceFeature::ShaderF16) {
            Ok(())
        } else {
            Err(HephaestusError::InvalidConfiguration {
                message: "WGPU FFT requires the ShaderF16 device feature for binary16".to_owned(),
            })
        }
    }

    fn from_fft_coefficient(value: f64) -> Self {
        Self::from_f64(value)
    }
}
