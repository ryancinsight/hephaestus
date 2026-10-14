//! CUDA instantiation of the shared nonzero-coordinate conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::{
    assert_nonzero_contract, assert_nonzero_rejects_aliased_output,
    assert_nonzero_special_float_values,
};
use hephaestus_cuda::{CudaDevice, CudaNonzeroOps};

#[test]
fn cuda_satisfies_the_nonzero_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA nonzero conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA nonzero conformance requires a physical device: {error}"),
    };
    assert_nonzero_contract::<_, _, i32>(&device, &CudaNonzeroOps);
    assert_nonzero_contract::<_, _, f32>(&device, &CudaNonzeroOps);
    assert_nonzero_special_float_values(&device, &CudaNonzeroOps);
    assert_nonzero_rejects_aliased_output(&device, &CudaNonzeroOps);
}
