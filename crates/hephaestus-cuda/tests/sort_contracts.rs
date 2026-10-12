//! CUDA instantiation of the shared stable-sort conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::{assert_sort_contract, assert_sort_special_float_values};
use hephaestus_cuda::{CudaDevice, CudaSortOps};

#[test]
fn cuda_satisfies_the_sort_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA sort conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA sort conformance requires a physical device: {error}"),
    };
    assert_sort_contract::<_, _, i32>(&device, &CudaSortOps);
    assert_sort_contract::<_, _, f32>(&device, &CudaSortOps);
    assert_sort_special_float_values(&device, &CudaSortOps);
}
