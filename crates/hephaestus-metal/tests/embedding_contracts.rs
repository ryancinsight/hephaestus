//! Metal instantiation of the shared embedding-gather conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::{
    assert_embedding_gather_contract, assert_embedding_gather_rejects_out_of_range_index,
    assert_embedding_scatter_add_accumulates_in_position_order,
    assert_embedding_scatter_add_contract, assert_embedding_scatter_add_rejects_out_of_range_index,
};
use hephaestus_metal::{MetalDevice, MetalEmbeddingOps};

#[test]
fn metal_satisfies_the_embedding_contracts() {
    let device =
        MetalDevice::try_default().expect("Metal embedding conformance requires a physical device");
    let ops = MetalEmbeddingOps::default();
    assert_embedding_gather_contract(&device, &ops);
    assert_embedding_gather_rejects_out_of_range_index(&device, &ops);
    assert_embedding_scatter_add_contract(&device, &ops);
    assert_embedding_scatter_add_accumulates_in_position_order(&device, &ops);
    assert_embedding_scatter_add_rejects_out_of_range_index(&device, &ops);
}
