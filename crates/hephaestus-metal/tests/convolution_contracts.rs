//! Metal instantiation of the shared convolution conformance clauses.

#![cfg(target_os = "macos")]

use hephaestus_conformance::{assert_convolution_contract, assert_convolution_f64_contract};
use hephaestus_core::{ComputeDeviceCapabilities, DeviceFeature};
use hephaestus_metal::{MetalConvolutionOps, MetalDevice};

#[test]
fn metal_satisfies_the_convolution_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal convolution conformance requires a physical device");
    assert_convolution_contract(&device, &MetalConvolutionOps);
    if device.supports_device_feature(DeviceFeature::ShaderF64) {
        assert_convolution_f64_contract(&device, &MetalConvolutionOps);
    }
}
