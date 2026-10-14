//! CUDA instantiation of the shared masked-fill conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::{assert_masked_fill_contract, assert_masked_fill_special_mask_values};
use hephaestus_cuda::{CudaDevice, CudaMaskedFillOps};

#[test]
fn cuda_satisfies_the_masked_fill_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA masked_fill conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA masked_fill conformance requires a physical device: {error}"),
    };
    assert_masked_fill_contract::<_, _, i32>(&device, &CudaMaskedFillOps);
    assert_masked_fill_contract::<_, _, f32>(&device, &CudaMaskedFillOps);
    assert_masked_fill_special_mask_values(&device, &CudaMaskedFillOps);
}
