//! WGPU instantiation of the shared scaled dot-product attention contract.

use hephaestus_conformance::assert_attention_contract;
use hephaestus_wgpu::{WgpuAttentionOps, WgpuDevice};

#[test]
fn wgpu_satisfies_the_attention_contract() {
    let device = match WgpuDevice::try_default("hephaestus-attention-conformance") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skip WGPU attention conformance: device unavailable ({error})");
            return;
        }
    };
    assert_attention_contract(&device, &WgpuAttentionOps);
}
