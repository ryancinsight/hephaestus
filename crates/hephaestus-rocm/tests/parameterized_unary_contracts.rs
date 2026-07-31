//! ROCm instantiation of the shared runtime-parameter unary contract.

#[cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_parameterized_unary_contract;
#[cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_rocm::RocmDevice;
use hephaestus_rocm::{HardtanhOp, RocmParameterizedUnaryOps, parameterized_unary_strided_into};

#[test]
fn adapterless_build_exports_the_parameterized_dispatch_surface() {
    let _dispatch = parameterized_unary_strided_into::<HardtanhOp, 1>;
    let provider = RocmParameterizedUnaryOps;
    assert_eq!(core::mem::size_of_val(&provider), 0);
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
#[test]
fn rocm_satisfies_the_parameterized_unary_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm parameterized-unary conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm parameterized-unary conformance requires a device: {error}"),
    };
    assert_parameterized_unary_contract(&device, &RocmParameterizedUnaryOps);
}
