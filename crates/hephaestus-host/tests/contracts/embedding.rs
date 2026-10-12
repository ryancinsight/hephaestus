//! Host instantiation of the shared embedding-gather conformance clauses.

use hephaestus_conformance::{
    assert_embedding_gather_contract, assert_embedding_gather_rejects_out_of_range_index,
    assert_embedding_scatter_add_contract, assert_embedding_scatter_add_rejects_out_of_range_index,
};
use hephaestus_host::{HostDevice, HostEmbeddingOps};

#[test]
fn host_satisfies_the_embedding_gather_contract() {
    let device = HostDevice::new();
    assert_embedding_gather_contract(&device, &HostEmbeddingOps);
    assert_embedding_gather_rejects_out_of_range_index(&device, &HostEmbeddingOps);
}

#[test]
fn host_satisfies_the_embedding_scatter_add_contract() {
    let device = HostDevice::new();
    assert_embedding_scatter_add_contract(&device, &HostEmbeddingOps);
    assert_embedding_scatter_add_rejects_out_of_range_index(&device, &HostEmbeddingOps);
}
