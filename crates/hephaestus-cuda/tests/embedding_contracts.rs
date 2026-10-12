//! CUDA instantiation of the shared embedding-gather conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::{
    assert_embedding_gather_contract, assert_embedding_gather_rejects_out_of_range_index,
    assert_embedding_scatter_add_contract, assert_embedding_scatter_add_rejects_out_of_range_index,
};
use hephaestus_cuda::{CudaDevice, CudaEmbeddingOps};

fn acquire_device() -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA embedding-gather conformance: device unavailable ({error})");
            None
        }
        Err(error) => {
            panic!("CUDA embedding-gather conformance requires a physical device: {error}")
        }
    }
}

#[test]
fn cuda_satisfies_the_embedding_gather_contract() {
    let Some(device) = acquire_device() else {
        return;
    };
    assert_embedding_gather_contract(&device, &CudaEmbeddingOps);
}

#[test]
fn cuda_embedding_gather_rejects_an_out_of_range_index() {
    let Some(device) = acquire_device() else {
        return;
    };
    assert_embedding_gather_rejects_out_of_range_index(&device, &CudaEmbeddingOps);
}

#[test]
fn cuda_satisfies_the_embedding_scatter_add_contract() {
    let Some(device) = acquire_device() else {
        return;
    };
    assert_embedding_scatter_add_contract(&device, &CudaEmbeddingOps);
    assert_embedding_scatter_add_rejects_out_of_range_index(&device, &CudaEmbeddingOps);
}
