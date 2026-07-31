//! WGPU instantiation of the shared runtime-parameter unary contract.

use hephaestus_conformance::assert_parameterized_unary_contract;
use hephaestus_wgpu::{WgpuDevice, WgpuParameterizedUnaryOps};

#[test]
fn wgpu_satisfies_the_parameterized_unary_contract() {
    let device = match WgpuDevice::try_default("hephaestus-parameterized-unary-test") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU parameterized-unary conformance: device unavailable ({error})");
            return;
        }
    };
    assert_parameterized_unary_contract(&device, &WgpuParameterizedUnaryOps);
}
