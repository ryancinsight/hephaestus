//! Metal differential contracts for volume ray integrals.

use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError};
use hephaestus_metal::{
    FieldGeometry, MetalDevice, RAY_STRIDE, ray_line_integrals, ray_line_integrals_into,
};

fn device(test: &str) -> Option<MetalDevice> {
    match MetalDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_METAL_REQUIRE_DEVICE").is_some() {
                panic!("Metal device required for {test}: {error}");
            }
            eprintln!("skipping Metal volume contract {test}: {error}");
            None
        }
    }
}

fn geometry() -> FieldGeometry {
    FieldGeometry {
        dims: [9, 5, 5],
        origin: [0.0, 0.0, 0.0],
        spacing: [2.0, 2.0, 2.0],
    }
}

fn affine_field() -> Vec<f32> {
    let mut field = Vec::with_capacity(9 * 5 * 5);
    for ix in 0..9 {
        for _iy in 0..5 {
            for _iz in 0..5 {
                field.push(0.01 * ix as f32 + 0.02);
            }
        }
    }
    field
}

#[test]
fn uniform_hit_and_miss_match_analytic_chords() {
    let Some(device) = device("uniform_hit_and_miss_match_analytic_chords") else {
        return;
    };
    let field = device
        .upload(&vec![0.25f32; 9 * 5 * 5])
        .expect("field upload");
    let rays = device
        .upload(&[
            -10.0, 4.0, 4.0, 1.0, 0.0, 0.0, -10.0, 100.0, 4.0, 1.0, 0.0, 0.0,
        ])
        .expect("ray upload");
    let output = ray_line_integrals(
        &device,
        &field,
        geometry(),
        &rays,
        2,
        0.5,
        BlockWidth::DEFAULT,
    )
    .expect("volume dispatch");
    let mut values = [0.0; 2];
    device
        .download(&output, &mut values)
        .expect("output download");
    assert!((values[0] - 4.0).abs() <= 1e-4);
    assert_eq!(values[1], 0.0);
}

#[test]
fn affine_field_matches_midpoint_trilinear_reference() {
    let Some(device) = device("affine_field_matches_midpoint_trilinear_reference") else {
        return;
    };
    let field = device.upload(&affine_field()).expect("field upload");
    let rays = device
        .upload(&[-10.0, 4.0, 4.0, 1.0, 0.0, 0.0])
        .expect("ray upload");
    let output = ray_line_integrals(
        &device,
        &field,
        geometry(),
        &rays,
        1,
        1.0,
        BlockWidth::DEFAULT,
    )
    .expect("volume dispatch");
    let mut values = [0.0];
    device
        .download(&output, &mut values)
        .expect("output download");
    assert!((values[0] - 0.96).abs() <= 1e-4);
}

#[test]
fn empty_ray_batch_returns_empty_output() {
    let Some(device) = device("empty_ray_batch_returns_empty_output") else {
        return;
    };
    let field = device
        .upload(&vec![0.25f32; 9 * 5 * 5])
        .expect("field upload");
    let rays = device.upload(&[] as &[f32]).expect("empty ray upload");

    let output = ray_line_integrals(
        &device,
        &field,
        geometry(),
        &rays,
        0,
        0.5,
        BlockWidth::DEFAULT,
    )
    .expect("empty volume dispatch");

    assert_eq!(output.len(), 0);
    device
        .download(&output, &mut [])
        .expect("empty output download");
}

#[test]
fn invalid_volume_contracts_are_rejected_before_launch() {
    let Some(device) = device("invalid_volume_contracts_are_rejected_before_launch") else {
        return;
    };
    let field = device
        .upload(&vec![1.0f32; 9 * 5 * 5])
        .expect("field upload");
    let rays = device.upload(&[0.0f32; 6]).expect("ray upload");
    let output = device.alloc_zeroed::<f32>(1).expect("output allocation");
    let error = ray_line_integrals_into(
        &device,
        &field,
        geometry(),
        &rays,
        0.0,
        &output,
        BlockWidth::DEFAULT,
    )
    .expect_err("zero step must be rejected");
    assert!(matches!(error, HephaestusError::DispatchFailed { .. }));
    assert_eq!(rays.len(), RAY_STRIDE);

    let wrong_rays = device.upload(&[0.0f32; 5]).expect("wrong ray upload");
    let error = ray_line_integrals_into(
        &device,
        &field,
        geometry(),
        &wrong_rays,
        1.0,
        &output,
        BlockWidth::DEFAULT,
    )
    .expect_err("packed ray mismatch must be rejected");
    assert!(matches!(
        error,
        HephaestusError::LengthMismatch {
            host_len: 5,
            device_len: 6
        }
    ));
}
