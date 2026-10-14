//! Metal instantiation of the shared scatter-assignment conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::assert_index_put_contract;
use hephaestus_metal::{MetalDevice, MetalIndexPutOps};

#[test]
fn metal_satisfies_the_index_put_contract() {
    let device =
        MetalDevice::try_default().expect("Metal index_put conformance requires a physical device");
    assert_index_put_contract::<_, _, i32>(&device, &MetalIndexPutOps::default());
    assert_index_put_contract::<_, _, f32>(&device, &MetalIndexPutOps::default());
}
