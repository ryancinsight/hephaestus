//! WGPU instantiation of the shared convolution conformance clauses.

use hephaestus_conformance::{assert_convolution_contract, assert_convolution_f64_contract};
use hephaestus_core::DeviceFeature;
use hephaestus_wgpu::{WgpuConvolutionOps, WgpuDevice, wgpu};

pub(super) fn wgpu_satisfies_the_convolution_contract() {
    let device = WgpuDevice::try_default_with_optional_features_and_limits(
        "hephaestus-convolution-conformance",
        wgpu::Features::SHADER_F64,
        wgpu::Limits::downlevel_defaults(),
    )
    .expect("WGPU convolution conformance requires a physical adapter");
    assert_convolution_contract(&device, &WgpuConvolutionOps);
    if device.supports_device_feature(DeviceFeature::ShaderF64) {
        assert_convolution_f64_contract(&device, &WgpuConvolutionOps);
    }
}
