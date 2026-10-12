//! ROCm instantiation of the shared arg-reduce conformance clauses.

#![cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::{
    assert_arg_reduce_contract, assert_arg_reduce_transposed_view_contract,
};
use hephaestus_rocm::{RocmArgReduceOps, RocmDevice};

#[test]
fn rocm_satisfies_the_arg_reduce_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm arg-reduce conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm arg-reduce conformance requires a physical device: {error}"),
    };
    assert_arg_reduce_contract(&device, &RocmArgReduceOps);
    assert_arg_reduce_transposed_view_contract(&device, &RocmArgReduceOps);
}
