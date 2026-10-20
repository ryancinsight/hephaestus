//! Metal instantiation of the shared retained-reduction conformance clauses.
//!
//! The assertions live once in `hephaestus-conformance`, generic over
//! [`hephaestus_core::ComputeDevice`] and [`hephaestus_core::DenseVectorOps`];
//! this file only supplies the device and the backend's seam value per scalar.
//! Double-precision dispatch is capability-gated like the convolution
//! contracts: most Metal devices have no float64 shader support.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_retained_reduction_contract;
use hephaestus_core::{ComputeDeviceCapabilities, DeviceFeature};
use hephaestus_metal::{MetalDevice, MetalVectorOps};

#[test]
fn metal_satisfies_the_retained_reduction_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal retained-reduction conformance requires a physical device");
    let ops = MetalVectorOps::<f32>::new(&device).expect("vector kernels compile");
    assert_retained_reduction_contract(&device, &ops);
    if device.supports_device_feature(DeviceFeature::ShaderF64) {
        let ops = MetalVectorOps::<f64>::new(&device).expect("vector kernels compile");
        assert_retained_reduction_contract(&device, &ops);
    }
}
