//! CUDA instantiation of the shared inner-product conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::{assert_dot_contract, assert_dot_special_float_values};
use hephaestus_cuda::{CudaDevice, CudaDotOps};

#[test]
fn cuda_satisfies_the_dot_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA dot conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA dot conformance requires a physical device: {error}"),
    };
    assert_dot_contract::<_, _, i32>(&device, &CudaDotOps);
    assert_dot_contract::<_, _, f32>(&device, &CudaDotOps);
    assert_dot_contract::<_, _, f64>(&device, &CudaDotOps);
    assert_dot_special_float_values(&device, &CudaDotOps);
}
