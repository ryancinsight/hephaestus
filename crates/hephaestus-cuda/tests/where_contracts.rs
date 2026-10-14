//! CUDA instantiation of the shared conditional-selection conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::{assert_where_contract, assert_where_special_cond_values};
use hephaestus_cuda::{CudaDevice, CudaWhereOps};

#[test]
fn cuda_satisfies_the_where_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA where conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA where conformance requires a physical device: {error}"),
    };
    assert_where_contract::<_, _, i32>(&device, &CudaWhereOps);
    assert_where_contract::<_, _, f32>(&device, &CudaWhereOps);
    assert_where_special_cond_values(&device, &CudaWhereOps);
}
