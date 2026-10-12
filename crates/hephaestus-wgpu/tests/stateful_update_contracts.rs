//! WGPU instantiation of the shared stateful-update contract.

use hephaestus_conformance::assert_stateful_update_at;
use hephaestus_core::{
    ComputeDevice, DeviceFeature, DevicePreference, HephaestusError, Sgd, SgdParameters,
    StatefulUpdateOperands, StatefulUpdateOps, StridedView,
};
use hephaestus_wgpu::{WgpuDevice, WgpuStatefulUpdateOps};
use leto::Layout;

fn device(label: &str) -> Option<WgpuDevice> {
    match WgpuDevice::try_default(label) {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip WGPU stateful-update conformance: device unavailable ({error})");
            None
        }
        Err(error) => panic!("WGPU stateful-update conformance requires a device: {error}"),
    }
}

fn shader_f64_device(label: &str) -> Option<WgpuDevice> {
    match WgpuDevice::try_with_device_preference_and_required_device_features(
        label,
        DevicePreference::HighPerformance,
        &[DeviceFeature::ShaderF64],
    ) {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("HEPHAESTUS_WGPU_REQUIRE_DEVICE").is_none() => {
            eprintln!("skip WGPU ShaderF64 stateful-update conformance: device unavailable ({error})");
            None
        }
        Err(error) => {
            panic!("WGPU ShaderF64 stateful-update conformance requires a device: {error}")
        }
    }
}

pub(super) fn wgpu_satisfies_the_stateful_update_contract() {
    let Some(device) = device("hephaestus-stateful-update-test") else {
        return;
    };
    assert_stateful_update_at::<f32, _, _>(&device, &WgpuStatefulUpdateOps);
}

pub(super) fn wgpu_satisfies_the_stateful_update_contract_on_shader_f64_devices() {
    let Some(device) = shader_f64_device("hephaestus-stateful-update-shader-f64-test") else {
        return;
    };
    assert_stateful_update_at::<f64, _, _>(&device, &WgpuStatefulUpdateOps);
}

pub(super) fn foreign_device_buffers_fail_before_mutation() {
    let Some(owner) = device("hephaestus-stateful-update-owner") else {
        return;
    };
    let Some(dispatch) = device("hephaestus-stateful-update-dispatch") else {
        return;
    };
    let parameter = owner.upload(&[1.0_f32, 2.0]).expect("parameter upload");
    let gradient = owner.upload(&[0.1_f32, 0.2]).expect("gradient upload");
    let state = owner.upload(&[0.5_f32, 0.6]).expect("state upload");
    let layout = Layout::c_contiguous([2]).expect("layout");
    let states = [StridedView::new(&state, &layout)];
    let error = WgpuStatefulUpdateOps
        .stateful_update::<f32, Sgd, 1>(
            &dispatch,
            StatefulUpdateOperands {
                parameter: StridedView::new(&parameter, &layout),
                gradient: StridedView::new(&gradient, &layout),
                states: &states,
            },
            SgdParameters::new(0.1, 0.9).expect("parameters"),
        )
        .expect_err("foreign buffers must be rejected");
    assert!(matches!(error, HephaestusError::DispatchFailed { .. }));
    let mut actual = [0.0; 2];
    owner
        .download(&parameter, &mut actual)
        .expect("parameter download");
    assert_eq!(actual, [1.0, 2.0]);
}
