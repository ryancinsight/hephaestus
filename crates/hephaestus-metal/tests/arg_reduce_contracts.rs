//! Metal instantiation of the shared arg-reduce conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::{
    assert_arg_reduce_contract, assert_arg_reduce_transposed_view_contract,
};
use hephaestus_metal::{MetalArgReduceOps, MetalDevice};

#[test]
fn metal_satisfies_the_arg_reduce_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal arg-reduce conformance requires a physical device");
    assert_arg_reduce_contract(&device, &MetalArgReduceOps);
    assert_arg_reduce_transposed_view_contract(&device, &MetalArgReduceOps);
}
