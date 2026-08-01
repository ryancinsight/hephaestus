//! WGPU instantiation of the shared transfer conformance clauses.

use hephaestus_conformance::assert_transfer_contract;
use hephaestus_wgpu::WgpuDevice;

#[test]
fn wgpu_satisfies_the_transfer_contract() {
    let device = match WgpuDevice::try_default("hephaestus-transfer-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU transfer conformance: adapter unavailable ({error})");
            return;
        }
    };
    assert_transfer_contract(&device);
}
