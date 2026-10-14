//! CUDA instantiation of the shared scatter-assignment conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::assert_index_put_contract;
use hephaestus_cuda::{CudaDevice, CudaIndexPutOps};

#[test]
fn cuda_satisfies_the_index_put_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA index_put conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA index_put conformance requires a physical device: {error}"),
    };
    assert_index_put_contract::<_, _, i32>(&device, &CudaIndexPutOps);
    assert_index_put_contract::<_, _, f32>(&device, &CudaIndexPutOps);
}
