//! ROCm instantiation of the shared embedding-gather conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::{
    assert_embedding_gather_contract, assert_embedding_gather_rejects_out_of_range_index,
    assert_embedding_scatter_add_contract, assert_embedding_scatter_add_rejects_out_of_range_index,
};
use hephaestus_rocm::{RocmDevice, RocmEmbeddingOps};

fn acquire_device() -> Option<RocmDevice> {
    match RocmDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm embedding-gather conformance: device unavailable ({error})");
            None
        }
        Err(error) => {
            panic!("ROCm embedding-gather conformance requires a physical device: {error}")
        }
    }
}

#[test]
fn rocm_satisfies_the_embedding_gather_contract() {
    let Some(device) = acquire_device() else {
        return;
    };
    assert_embedding_gather_contract(&device, &RocmEmbeddingOps);
}

#[test]
fn rocm_embedding_gather_rejects_an_out_of_range_index() {
    let Some(device) = acquire_device() else {
        return;
    };
    assert_embedding_gather_rejects_out_of_range_index(&device, &RocmEmbeddingOps);
}

#[test]
fn rocm_satisfies_the_embedding_scatter_add_contract() {
    let Some(device) = acquire_device() else {
        return;
    };
    assert_embedding_scatter_add_contract(&device, &RocmEmbeddingOps);
    assert_embedding_scatter_add_rejects_out_of_range_index(&device, &RocmEmbeddingOps);
}
