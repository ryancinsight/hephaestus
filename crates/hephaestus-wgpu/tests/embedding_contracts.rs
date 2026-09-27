//! WGPU instantiation of the shared embedding-gather conformance clauses.

use hephaestus_conformance::{
    assert_embedding_gather_contract, assert_embedding_gather_rejects_out_of_range_index,
};
use hephaestus_wgpu::WgpuEmbeddingOps;

pub(super) fn wgpu_satisfies_the_embedding_gather_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_embedding_gather_contract(&device, &WgpuEmbeddingOps);
    assert_embedding_gather_rejects_out_of_range_index(&device, &WgpuEmbeddingOps);
}
