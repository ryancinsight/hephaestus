//! Metal instantiation of the shared attention conformance clauses.

#![cfg(target_os = "macos")]

use hephaestus_conformance::assert_attention_contract;
use hephaestus_metal::{MetalAttentionOps, MetalDevice};

#[test]
fn metal_satisfies_the_attention_contract() {
    let device =
        MetalDevice::try_default().expect("Metal attention conformance requires a physical device");
    assert_attention_contract(&device, &MetalAttentionOps);
}
