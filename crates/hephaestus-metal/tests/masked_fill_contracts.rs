//! Metal instantiation of the shared masked-fill conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::{assert_masked_fill_contract, assert_masked_fill_special_mask_values};
use hephaestus_metal::{MetalDevice, MetalMaskedFillOps};

#[test]
fn metal_satisfies_the_masked_fill_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal masked_fill conformance requires a physical device");
    assert_masked_fill_contract::<_, _, i32>(&device, &MetalMaskedFillOps::default());
    assert_masked_fill_contract::<_, _, f32>(&device, &MetalMaskedFillOps::default());
    assert_masked_fill_special_mask_values(&device, &MetalMaskedFillOps::default());
}
