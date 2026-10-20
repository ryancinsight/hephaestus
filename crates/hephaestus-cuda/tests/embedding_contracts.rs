//! CUDA instantiation of the shared embedding-gather conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::{
    assert_embedding_gather_contract, assert_embedding_gather_rejects_out_of_range_index,
    assert_embedding_scatter_add_accumulates_in_position_order,
    assert_embedding_scatter_add_contract, assert_embedding_scatter_add_rejects_out_of_range_index,
};
use hephaestus_cuda::{CudaDevice, CudaEmbeddingOps};

#[test]
fn cuda_satisfies_the_embedding_contracts() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA embedding conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA embedding conformance requires a physical device: {error}"),
    };
    assert_embedding_gather_contract(&device, &CudaEmbeddingOps);
    assert_embedding_gather_rejects_out_of_range_index(&device, &CudaEmbeddingOps);
    assert_embedding_scatter_add_contract(&device, &CudaEmbeddingOps);
    assert_embedding_scatter_add_accumulates_in_position_order(&device, &CudaEmbeddingOps);
    assert_embedding_scatter_add_rejects_out_of_range_index(&device, &CudaEmbeddingOps);
}
