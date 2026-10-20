//! ROCm instantiation of the shared embedding-gather conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::{
    assert_embedding_gather_contract, assert_embedding_gather_rejects_out_of_range_index,
    assert_embedding_scatter_add_accumulates_in_position_order,
    assert_embedding_scatter_add_contract, assert_embedding_scatter_add_rejects_out_of_range_index,
};
use hephaestus_rocm::{RocmDevice, RocmEmbeddingOps};

#[test]
fn rocm_satisfies_the_embedding_contracts() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm embedding conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm embedding conformance requires a physical device: {error}"),
    };
    assert_embedding_gather_contract(&device, &RocmEmbeddingOps);
    assert_embedding_gather_rejects_out_of_range_index(&device, &RocmEmbeddingOps);
    assert_embedding_scatter_add_contract(&device, &RocmEmbeddingOps);
    assert_embedding_scatter_add_accumulates_in_position_order(&device, &RocmEmbeddingOps);
    assert_embedding_scatter_add_rejects_out_of_range_index(&device, &RocmEmbeddingOps);
}
