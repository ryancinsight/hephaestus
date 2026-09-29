//! WGPU instantiation of the shared triangular-masking conformance clause.

use hephaestus_conformance::assert_triangular_contract;
use hephaestus_core::{ComputeDevice, StridedView, TriangularMode, TriangularOps};
use hephaestus_wgpu::{HephaestusError, WgpuDevice, WgpuTriangularOps};
use leto::Layout;

pub(super) fn wgpu_satisfies_the_triangular_contract() {
    let Some(device) = super::device_or_skip() else {
        return;
    };
    assert_triangular_contract(&device, &WgpuTriangularOps);
}

pub(super) fn oversized_storage_binding_fails_before_output_mutation() {
    let Some(probe) = super::device_or_skip() else {
        return;
    };
    let mut limits = probe.limits();
    limits.max_storage_buffer_binding_size = 256;
    drop(probe);
    let device = WgpuDevice::try_default_with_limits("triangular binding limit", limits)
        .expect("limited device");
    let input = device.upload(&[3.0_f32; 65]).expect("input upload");
    let output = device.upload(&[-7.0_f32]).expect("output upload");
    let layout = Layout::c_contiguous([1, 1]).expect("layout");

    let error = WgpuTriangularOps
        .triangular_into(
            &device,
            StridedView::new(&input, &layout),
            TriangularMode::Lower,
            0,
            StridedView::new(&output, &layout),
        )
        .expect_err("whole-buffer binding above the enabled limit must fail");
    match error {
        HephaestusError::DispatchFailed { message } => assert_eq!(
            message,
            "triangular input binding requires 260 bytes, device limit is 256"
        ),
        other => panic!("unexpected binding-limit error: {other}"),
    }
    assert_eq!(
        device.download_owned(&output).expect("output download"),
        vec![-7.0],
        "binding rejection mutated output"
    );
}

pub(super) fn unsupported_workgroup_width_fails_before_output_mutation() {
    let Some(probe) = super::device_or_skip() else {
        return;
    };
    let mut limits = probe.limits();
    limits.max_compute_workgroup_size_x = 128;
    limits.max_compute_invocations_per_workgroup = 128;
    drop(probe);
    let device = WgpuDevice::try_default_with_limits("triangular workgroup limit", limits)
        .expect("limited device");
    let input = device.upload(&[3_i32]).expect("input upload");
    let output = device.upload(&[-11_i32]).expect("output upload");
    let layout = Layout::c_contiguous([1, 1]).expect("layout");

    let error = WgpuTriangularOps
        .triangular_into(
            &device,
            StridedView::new(&input, &layout),
            TriangularMode::Upper,
            0,
            StridedView::new(&output, &layout),
        )
        .expect_err("unsupported fixed workgroup width must fail");
    match error {
        HephaestusError::DispatchFailed { message } => assert_eq!(
            message,
            "triangular workgroup width 256 exceeds device limits: size_x=128, invocations=128"
        ),
        other => panic!("unexpected workgroup-limit error: {other}"),
    }
    assert_eq!(
        device.download_owned(&output).expect("output download"),
        vec![-11],
        "workgroup rejection mutated output"
    );
}
