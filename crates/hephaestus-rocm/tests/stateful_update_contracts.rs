//! ROCm instantiation of the shared stateful-update contract.

#[cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_stateful_update_contract;
#[cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_rocm::RocmDevice;
use hephaestus_rocm::RocmStatefulUpdateOps;

#[test]
fn adapterless_build_exports_the_stateful_update_surface() {
    assert_eq!(core::mem::size_of::<RocmStatefulUpdateOps>(), 0);
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
#[test]
fn rocm_satisfies_the_stateful_update_contract() {
    let device = match RocmDevice::try_default() {
        Ok(device) => device,
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm stateful-update conformance: device unavailable ({error})");
            return;
        }
        Err(error) => panic!("ROCm stateful-update conformance requires a device: {error}"),
    };
    assert_stateful_update_contract(&device, &RocmStatefulUpdateOps);
}
