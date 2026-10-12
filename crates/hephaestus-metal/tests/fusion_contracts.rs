//! Metal instantiation of the shared runtime-rank fusion contract.

#![cfg(target_os = "macos")]

use std::borrow::Cow;

use hephaestus_conformance::assert_fusion_contract;
use hephaestus_core::{FusedExpression, Wgsl};
use hephaestus_metal::{MetalDevice, MetalFusionOps};

struct AddExpression;

impl FusedExpression<Wgsl> for AddExpression {
    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed("input_0 + input_1")
    }
}

struct InputExpression;

impl FusedExpression<Wgsl> for InputExpression {
    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed("input_0")
    }
}

#[test]
fn metal_satisfies_the_fusion_contract() {
    let device =
        MetalDevice::try_default().expect("Metal fusion conformance requires a physical device");
    assert_fusion_contract::<_, _, _, f32, _, _>(
        &device,
        &MetalFusionOps,
        &AddExpression,
        &InputExpression,
    );
}
