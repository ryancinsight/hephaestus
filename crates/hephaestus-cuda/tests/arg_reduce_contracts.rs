//! CUDA instantiation of the shared arg-reduce conformance clauses.

#![cfg(feature = "cuda")]
use hephaestus_conformance::{
    assert_arg_reduce_contract, assert_arg_reduce_transposed_view_contract,
};
use hephaestus_cuda::{CudaArgReduceOps, CudaDevice};

#[test]
fn cuda_satisfies_the_arg_reduce_contract() {
    let device = match CudaDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip CUDA arg-reduce conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("CUDA arg-reduce conformance requires a physical device: {error}"),
    };
    assert_arg_reduce_contract(&device, &CudaArgReduceOps);
    assert_arg_reduce_transposed_view_contract(&device, &CudaArgReduceOps);
}
