//! WGPU instantiation of the shared stateful-update contract.

use super::device_features::device_with_optional_features;
use hephaestus_conformance::assert_stateful_update_contract;
use hephaestus_core::{
    ComputeDevice, DeviceFeature, HephaestusError, Sgd, SgdParameters, StatefulUpdateOperands,
    StatefulUpdateOps, StridedView,
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

pub(super) fn wgpu_satisfies_the_stateful_update_contract() {
    let Some(device) = device("hephaestus-stateful-update-test") else {
        return;
    };
    assert_stateful_update_contract::<_, _, f32>(&device, &WgpuStatefulUpdateOps);
}

pub(super) fn wgpu_stateful_update_respects_precision_capabilities() {
    let Some(device) = device_with_optional_features(
        "hephaestus-stateful-update-precision-contract",
        &[DeviceFeature::ShaderF64],
    ) else {
        return;
    };
    if !device.supports_device_feature(DeviceFeature::ShaderF64) {
        let original = [2.0_f64, 3.0];
        let gradient_values = [5.0_f64, 7.0];
        let state_values = [11.0_f64, 13.0];
        let parameter = device.upload(&original).expect("parameter");
        let gradient = device.upload(&gradient_values).expect("gradient");
        let state = device.upload(&state_values).expect("state");
        let layout = Layout::c_contiguous([2]).expect("layout");
        let states = [StridedView::new(&state, &layout)];
        let error = WgpuStatefulUpdateOps
            .stateful_update::<Sgd, f64, 1>(
                &device,
                StatefulUpdateOperands {
                    parameter: StridedView::new(&parameter, &layout),
                    gradient: StridedView::new(&gradient, &layout),
                    states: &states,
                },
                SgdParameters::new(0.5, 0.25).expect("parameters"),
            )
            .expect_err("unsupported native precision must reject stateful updates");
        match error {
            HephaestusError::InvalidConfiguration { message } => assert_eq!(
                message,
                "WGPU stateful update requires the ShaderF64 device feature for f64"
            ),
            other => panic!("unexpected precision rejection: {other}"),
        }
        assert_eq!(
            device
                .download_owned(&parameter)
                .expect("parameter readback"),
            original
        );
        assert_eq!(
            device.download_owned(&gradient).expect("gradient readback"),
            gradient_values
        );
        assert_eq!(
            device.download_owned(&state).expect("state readback"),
            state_values
        );
        return;
    }
    assert_stateful_update_contract::<_, _, f64>(&device, &WgpuStatefulUpdateOps);
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
        .stateful_update::<Sgd, f32, 1>(
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
