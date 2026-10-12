//! ROCm instantiation of the shared runtime-rank fusion contract.

#[cfg(all(feature = "rocm", target_os = "linux"))]
use std::borrow::Cow;

#[cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_conformance::assert_fusion_contract;
#[cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_core::{FusedExpression, HipC};
#[cfg(all(feature = "rocm", target_os = "linux"))]
use hephaestus_rocm::RocmDevice;
use hephaestus_rocm::RocmFusionOps;

#[test]
fn adapterless_build_exports_the_fusion_surface() {
    assert_eq!(core::mem::size_of::<RocmFusionOps>(), 0);
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
struct AddExpression;

#[cfg(all(feature = "rocm", target_os = "linux"))]
impl FusedExpression<HipC> for AddExpression {
    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed("input_0 + input_1")
    }
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
struct InputExpression;

#[cfg(all(feature = "rocm", target_os = "linux"))]
impl FusedExpression<HipC> for InputExpression {
    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed("input_0")
    }
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
fn device(test: &str) -> Option<RocmDevice> {
    match RocmDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_ROCM_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip ROCm fusion conformance: device unavailable ({error})");
            None
        }
        Err(error) => panic!("ROCm fusion conformance requires a device for {test}: {error}"),
    }
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
#[test]
fn rocm_satisfies_the_fusion_contract() {
    let Some(device) = device("rocm_satisfies_the_fusion_contract") else {
        return;
    };
    assert_fusion_contract::<_, _, _, f32, _, _>(
        &device,
        &RocmFusionOps,
        &AddExpression,
        &InputExpression,
    );
}

#[cfg(all(feature = "rocm", target_os = "linux"))]
#[test]
fn rocm_satisfies_the_f64_fusion_contract() {
    let Some(device) = device("rocm_satisfies_the_f64_fusion_contract") else {
        return;
    };
    assert_fusion_contract::<_, _, _, f64, _, _>(
        &device,
        &RocmFusionOps,
        &AddExpression,
        &InputExpression,
    );
}
