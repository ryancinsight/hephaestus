//! Metal instantiation of the shared embedding-gather conformance clauses.

#![cfg(target_os = "macos")]
use hephaestus_conformance::{
    assert_embedding_gather_contract, assert_embedding_gather_rejects_out_of_range_index,
    assert_embedding_scatter_add_contract, assert_embedding_scatter_add_rejects_out_of_range_index,
};
use hephaestus_metal::{MetalDevice, MetalEmbeddingOps};

#[test]
fn metal_satisfies_the_embedding_gather_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal embedding-gather conformance requires a physical device");
    assert_embedding_gather_contract(&device, &MetalEmbeddingOps);
}

#[test]
fn metal_embedding_gather_rejects_an_out_of_range_index() {
    let device = MetalDevice::try_default()
        .expect("Metal embedding-gather conformance requires a physical device");
    assert_embedding_gather_rejects_out_of_range_index(&device, &MetalEmbeddingOps);
}

#[test]
fn metal_satisfies_the_embedding_scatter_add_contract() {
    let device = MetalDevice::try_default()
        .expect("Metal embedding-gather conformance requires a physical device");
    assert_embedding_scatter_add_contract(&device, &MetalEmbeddingOps);
    assert_embedding_scatter_add_rejects_out_of_range_index(&device, &MetalEmbeddingOps);
}
