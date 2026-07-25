//! CUDA differential contracts for volume ray integrals.

use hephaestus_core::{BlockWidth, ComputeDevice, DeviceBuffer, HephaestusError};
use hephaestus_cuda::{
    CudaDevice, FieldGeometry, RAY_STRIDE, ray_line_integrals, ray_line_integrals_into,
};

fn device(test: &str) -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_some() {
                panic!("CUDA device required for {test}: {error}");
            }
            eprintln!("skipping CUDA volume contract {test}: {error}");
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

fn constant_field(value: f32) -> Vec<f32> {
    vec![value; 9 * 5 * 5]
}

fn affine_field() -> Vec<f32> {
    (0..9)
        .flat_map(|ix| {
            (0..5).flat_map(move |iy| {
                (0..5)
                    .map(move |iz| 0.01 * ix as f32 + 0.007 * iy as f32 + 0.003 * iz as f32 + 0.05)
            })
        })
        .collect()
}

fn read(device: &CudaDevice, host: &[f32], rays: &[f32], step: f32) -> Vec<f32> {
    let field = device.upload(host).expect("field upload");
    let rays = device.upload(rays).expect("ray upload");
    let count = rays.len() / RAY_STRIDE;
    let output = ray_line_integrals(
        device,
        &field,
        geometry(),
        &rays,
        count,
        step,
        BlockWidth::DEFAULT,
    )
    .expect("volume dispatch");
    let mut values = vec![0.0; count];
    device
        .download(&output, &mut values)
        .expect("output download");
    values
}

#[test]
fn uniform_hit_and_miss_match_analytic_chords() {
    let Some(device) = device("uniform_hit_and_miss_match_analytic_chords") else {
        return;
    };
    let rays = [
        -10.0, 4.0, 4.0, 1.0, 0.0, 0.0, -10.0, 100.0, 4.0, 1.0, 0.0, 0.0,
    ];
    let values = read(&device, &constant_field(0.25), &rays, 0.5);
    assert!((values[0] - 4.0).abs() <= 1e-4);
    assert_eq!(values[1], 0.0);
}

#[test]
fn affine_field_matches_midpoint_trilinear_reference() {
    let Some(device) = device("affine_field_matches_midpoint_trilinear_reference") else {
        return;
    };
    let rays = [-10.0, 4.0, 4.0, 1.0, 0.0, 0.0];
    let values = read(&device, &affine_field(), &rays, 1.0);
    assert!((values[0] - 0.96).abs() <= 1e-4);
}

#[test]
fn invalid_volume_contracts_are_rejected_before_launch() {
    let Some(device) = device("invalid_volume_contracts_are_rejected_before_launch") else {
        return;
    };
    let field = device.upload(&constant_field(1.0)).expect("field upload");
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
